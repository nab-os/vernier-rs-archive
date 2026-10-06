//! Minimal GDSII stream writer for layout export.
//!
//! The C++ library exports GDS via the external `gdstk` library; here we emit
//! the GDSII binary stream directly (it is a simple big-endian record format),
//! covering exactly what layouts need: one cell of axis-aligned rectangles
//! (`BOUNDARY` elements on layer 0).
//!
//! Units follow the C++ default `Library::init(name, 1e-6, 1e-9)`: user unit
//! 1 µm, database unit 1 nm. Rectangle coordinates (in user units) are stored as
//! integers in database units, i.e. scaled by `unit / precision = 1000`.

use crate::layout::Rectangle;

const UNIT: f64 = 1e-6;
const PRECISION: f64 = 1e-9;

// GDSII record types (high byte) with their data type (low byte).
const HEADER: u16 = 0x0002;
const BGNLIB: u16 = 0x0102;
const LIBNAME: u16 = 0x0206;
const UNITS: u16 = 0x0305;
const BGNSTR: u16 = 0x0502;
const STRNAME: u16 = 0x0606;
const BOUNDARY: u16 = 0x0800;
const LAYER: u16 = 0x0D02;
const DATATYPE: u16 = 0x0E02;
const XY: u16 = 0x1003;
const ENDEL: u16 = 0x1100;
const ENDSTR: u16 = 0x0700;
const ENDLIB: u16 = 0x0400;

/// Writes a full GDSII stream containing a single cell named `name` holding the
/// given rectangles as `BOUNDARY` elements on layer 0.
pub fn write_gds(name: &str, rectangles: &[Rectangle]) -> Vec<u8> {
    let mut out = Vec::new();

    // HEADER: version 600.
    record_i16(&mut out, HEADER, &[600]);
    // BGNLIB: modification + access time (12 × INT2), left zero.
    record_i16(&mut out, BGNLIB, &[0; 12]);
    // LIBNAME.
    record_string(&mut out, LIBNAME, name);
    // UNITS: [db unit in user units, db unit in meters] = [precision/unit, precision].
    record_real8(&mut out, UNITS, &[PRECISION / UNIT, PRECISION]);

    // BGNSTR: structure timestamps (12 × INT2).
    record_i16(&mut out, BGNSTR, &[0; 12]);
    record_string(&mut out, STRNAME, name);

    let scale = UNIT / PRECISION; // user units -> database units
    for r in rectangles {
        let x0 = (r.x * scale).round() as i32;
        let y0 = (r.y * scale).round() as i32;
        let x1 = ((r.x + r.width) * scale).round() as i32;
        let y1 = ((r.y + r.height) * scale).round() as i32;

        record_empty(&mut out, BOUNDARY);
        record_i16(&mut out, LAYER, &[0]);
        record_i16(&mut out, DATATYPE, &[0]);
        // Closed polygon: 5 points (last == first).
        let pts = [x0, y0, x1, y0, x1, y1, x0, y1, x0, y0];
        record_i32(&mut out, XY, &pts);
        record_empty(&mut out, ENDEL);
    }

    record_empty(&mut out, ENDSTR);
    record_empty(&mut out, ENDLIB);
    out
}

/// Writes a record header: total length (data + 4-byte header), big-endian.
fn record_header(out: &mut Vec<u8>, record: u16, data_len: usize) {
    let total = (data_len + 4) as u16;
    out.extend_from_slice(&total.to_be_bytes());
    out.extend_from_slice(&record.to_be_bytes());
}

/// A record with no payload (e.g. BOUNDARY, ENDEL, ENDSTR, ENDLIB).
fn record_empty(out: &mut Vec<u8>, record: u16) {
    record_header(out, record, 0);
}

fn record_i16(out: &mut Vec<u8>, record: u16, values: &[i16]) {
    record_header(out, record, values.len() * 2);
    for v in values {
        out.extend_from_slice(&v.to_be_bytes());
    }
}

fn record_i32(out: &mut Vec<u8>, record: u16, values: &[i32]) {
    record_header(out, record, values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_be_bytes());
    }
}

/// GDSII strings are null-padded to an even length.
fn record_string(out: &mut Vec<u8>, record: u16, s: &str) {
    let mut bytes = s.as_bytes().to_vec();
    if !bytes.len().is_multiple_of(2) {
        bytes.push(0);
    }
    record_header(out, record, bytes.len());
    out.extend_from_slice(&bytes);
}

fn record_real8(out: &mut Vec<u8>, record: u16, values: &[f64]) {
    record_header(out, record, values.len() * 8);
    for &v in values {
        out.extend_from_slice(&gds_real8(v));
    }
}

/// Encodes an `f64` as a GDSII 8-byte real (excess-64, base-16 float):
/// sign bit, 7-bit exponent (excess 64), 56-bit fraction with the binary point
/// to the left of the MSB, so `value = (-1)^s · fraction/2^56 · 16^(exp-64)`.
fn gds_real8(value: f64) -> [u8; 8] {
    if value == 0.0 {
        return [0; 8];
    }
    let sign = value < 0.0;
    let mut v = value.abs();

    // Normalize so that 1/16 <= v < 1 by adjusting the base-16 exponent.
    let mut exp: i32 = 64;
    while v >= 1.0 {
        v /= 16.0;
        exp += 1;
    }
    while v < 1.0 / 16.0 {
        v *= 16.0;
        exp -= 1;
    }

    // 56-bit fraction.
    let mut fraction = (v * (2f64.powi(56))).round() as u64;
    // Rounding can push the fraction to 2^56 (carry out); renormalize.
    if fraction >= (1u64 << 56) {
        fraction >>= 4;
        exp += 1;
    }

    let mut bytes = [0u8; 8];
    bytes[0] = (exp as u8) & 0x7f;
    if sign {
        bytes[0] |= 0x80;
    }
    // Fraction occupies the low 7 bytes, big-endian.
    for i in 0..7 {
        let shift = 8 * (6 - i);
        bytes[1 + i] = ((fraction >> shift) & 0xff) as u8;
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads big-endian records back and returns their (record_type, data) list.
    fn parse_records(bytes: &[u8]) -> Vec<(u16, Vec<u8>)> {
        let mut out = Vec::new();
        let mut i = 0;
        while i + 4 <= bytes.len() {
            let len = u16::from_be_bytes([bytes[i], bytes[i + 1]]) as usize;
            let rec = u16::from_be_bytes([bytes[i + 2], bytes[i + 3]]);
            assert!(len >= 4, "record length must include the 4-byte header");
            out.push((rec, bytes[i + 4..i + len].to_vec()));
            i += len;
        }
        assert_eq!(i, bytes.len(), "records must tile the stream exactly");
        out
    }

    #[test]
    fn structure_and_counts() {
        let rects = vec![
            Rectangle::new(0.0, 0.0, 1.0, 1.0),
            Rectangle::new(2.0, 2.0, 1.0, 1.0),
            Rectangle::new(4.0, 0.0, 1.0, 1.0),
        ];
        let bytes = write_gds("PeriodicPattern", &rects);
        let recs = parse_records(&bytes);

        assert_eq!(recs.first().unwrap().0, HEADER);
        assert_eq!(recs.last().unwrap().0, ENDLIB);
        assert_eq!(recs.iter().filter(|(r, _)| *r == BOUNDARY).count(), 3);
        assert_eq!(recs.iter().filter(|(r, _)| *r == ENDEL).count(), 3);
        // Each XY record is 5 points × 2 coords × 4 bytes = 40 bytes.
        for (r, data) in &recs {
            if *r == XY {
                assert_eq!(data.len(), 40);
            }
        }
    }

    #[test]
    fn real8_roundtrip_known_values() {
        // Decode helper mirroring the encoder.
        fn decode(b: [u8; 8]) -> f64 {
            let sign = b[0] & 0x80 != 0;
            let exp = (b[0] & 0x7f) as i32 - 64;
            let mut frac: u64 = 0;
            for i in 0..7 {
                frac = (frac << 8) | b[1 + i] as u64;
            }
            let mant = frac as f64 / 2f64.powi(56);
            let v = mant * 16f64.powi(exp);
            if sign { -v } else { v }
        }
        for &val in &[1.0, 1e-3, 1e-9, 0.5, 1000.0, 9.0] {
            let d = decode(gds_real8(val));
            assert!(
                (d - val).abs() <= val.abs() * 1e-12 + 1e-18,
                "val={val} got={d}"
            );
        }
        assert_eq!(gds_real8(0.0), [0u8; 8]);
    }

    #[test]
    fn xy_coordinates_scale_to_db_units() {
        // A 1×1 µm rectangle at origin -> integer db coords 0 and 1000.
        let bytes = write_gds("c", &[Rectangle::new(0.0, 0.0, 1.0, 1.0)]);
        let recs = parse_records(&bytes);
        let xy = recs.iter().find(|(r, _)| *r == XY).unwrap();
        // Points are (x, y) pairs: [x0, y0, x1, y0, ...] → x1 is the 3rd i32.
        let x0 = i32::from_be_bytes([xy.1[0], xy.1[1], xy.1[2], xy.1[3]]);
        let x1 = i32::from_be_bytes([xy.1[8], xy.1[9], xy.1[10], xy.1[11]]);
        assert_eq!(x0, 0);
        assert_eq!(x1, 1000);
    }
}
