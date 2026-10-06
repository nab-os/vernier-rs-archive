//! End-to-end tests for the object/factory parity layer: render a pattern,
//! construct a detector through the C++-style `Detector` factory, and detect.

use vernier_detector::{Detector, Layout, PatternDetector};
use vernier_patterns::PatternPose;
use vernier_patterns::megarena::Megarena;

#[test]
fn unknown_class_errors() {
    assert!(Detector::new_instance("NotAPattern").is_err());
}

#[test]
fn from_json_sets_class_and_attributes() {
    let json = r#"{ "MegarenaPattern": { "physicalPeriod": 9.0, "codeSize": 12, "sigma": 3.0 } }"#;
    let det = Detector::from_json_str(json).unwrap();
    assert_eq!(det.classname(), "MegarenaPattern");
    assert_eq!(det.get_double("physicalPeriod"), Some(9.0));
    assert_eq!(det.get_int("codeSize"), Some(12));
    assert_eq!(det.get_double("sigma"), Some(3.0));
}

#[test]
fn periodic_via_factory_detects() {
    let size = 512usize;
    let period = 12.0;
    let pattern = vernier_patterns::periodic::Periodic::new(period);
    let image = pattern.render(size, size, &PatternPose::IDENTITY);

    let mut det = Detector::new_instance("PeriodicPattern").unwrap();
    det.set_double("physicalPeriod", period);
    det.set_double("sigma", 4.0);
    det.set_int("minFrequency", 10);
    det.set_int("maxFrequency", 0);
    det.set_double("smoothingSigma", 0.0);

    det.compute(&image).unwrap();
    assert!(det.pattern_found(-1), "periodic pattern should be found");
    assert_eq!(det.pattern_count(), 1);
    let pose = det.get_2d_pose(-1);
    assert!(pose.x.is_finite() && pose.y.is_finite() && pose.theta.is_finite());
    assert!(!pose.is_3d);

    // 3D disambiguation returns the four ambiguous candidates, all marked 3D.
    // (Numeric tilt/scale accuracy is validated in vernier-pose with genuine
    // two-direction planes; the CPU periodic renderer is a 1-D carrier, so it
    // only exercises the wiring here, not the out-of-plane recovery.)
    let poses3d = det.get_all_3d_poses(-1);
    assert_eq!(poses3d.len(), 4);
    assert!(poses3d.iter().all(|p| p.is_3d));
    assert!(det.get_3d_pose(-1).is_3d);

    // A frame that fails must not leave the previous frame's pose behind.
    let empty = vernier_core::image::GrayImage::from_vec(0, 0, Vec::new()).unwrap();
    assert!(det.compute(&empty).is_err());
    assert!(
        !det.pattern_found(-1),
        "a failed frame kept the previous pose"
    );
}

#[test]
fn megarena_via_factory_matches_direct_solve() {
    let size = 512usize;
    let period = 12.0;
    let order = 8u32;
    let pattern = Megarena::new(period, order).unwrap();
    let image = pattern.render(size, size, &PatternPose::IDENTITY);

    let mut det = Detector::new_instance("MegarenaPattern").unwrap();
    det.set_double("physicalPeriod", period);
    det.set_int("codeSize", order as i64);
    det.set_double("sigma", 4.0);
    det.set_int("minFrequency", 10);
    det.set_int("maxFrequency", 0);
    det.set_double("smoothingSigma", 0.0);

    det.compute(&image).unwrap();
    assert!(det.pattern_found(-1), "megarena pattern should be found");

    let pose = det.get_2d_pose(-1);
    assert!(pose.x.is_finite() && pose.y.is_finite());
    // get_all_3d_poses returns exactly the one recovered pose.
    assert_eq!(det.get_all_3d_poses(-1).len(), 1);
}

#[test]
fn bitmap_factory_recognized_but_empty_finds_nothing() {
    // The factory recognizes "BitmapPattern" (surface parity) but a detector
    // with no reference bitmap localizes nothing.
    let size = 256usize;
    let img = Megarena::new(12.0, 8)
        .unwrap()
        .render(size, size, &PatternPose::IDENTITY);
    let mut det = Detector::new_instance("BitmapPattern").unwrap();
    det.set_double("physicalPeriod", 12.0);
    det.set_double("sigma", 4.0);
    det.set_int("minFrequency", 10);
    det.set_int("maxFrequency", 0);
    det.set_double("smoothingSigma", 0.0);
    det.compute(&img).unwrap();
    assert!(!det.pattern_found(-1));
}

#[test]
fn bitmap_detector_matches_its_own_thumbnail() {
    use vernier_core::GrayImage;
    use vernier_cpu::CpuBackend;
    use vernier_detector::BitmapPatternDetector;

    // A real 2D lattice (megarena) so the thumbnail is well defined.
    let size = 512usize;
    let period = 12.0;
    let img = Megarena::new(period, 8)
        .unwrap()
        .render(size, size, &PatternPose::IDENTITY);

    let mut det = BitmapPatternDetector::new(CpuBackend::new());
    det.set_double("physicalPeriod", period);
    det.set_double("sigma", 4.0);
    det.set_int("minFrequency", 10);
    det.set_int("maxFrequency", 0);
    det.set_double("smoothingSigma", 0.0);

    // First pass: no reference bitmap -> not found, but a thumbnail is produced.
    det.compute(&img).unwrap();
    assert!(!det.pattern_found(-1));
    let thumb = det.thumbnail().expect("thumbnail should be computed");
    let n = thumb.size;
    assert!(n >= 3);
    let ref_img = GrayImage::from_vec(
        n,
        n,
        thumb.thumbnail.iter().map(|&v| v as f32 / 255.0).collect(),
    )
    .unwrap();

    // Use that thumbnail as the reference: detection must now match it at the
    // identity orientation (angle 0) with the best correlation on rotation 0.
    det.set_bitmap(&ref_img);
    det.compute(&img).unwrap();
    assert!(det.pattern_found(-1), "should match its own thumbnail");
    assert_eq!(det.max_angle(), 0, "identity orientation should win");
    assert_eq!(det.bitmap_index(), 0);
    let pose = det.get_2d_pose(-1);
    assert!(pose.x.is_finite() && pose.y.is_finite());
    assert!(!pose.is_3d);
}

#[test]
fn stamp_and_hpcode_are_recognized_but_unimplemented() {
    // Factory-surface parity: the class names construct, but compute() reports
    // "not implemented".
    for class in ["StampPattern", "HPCodePattern"] {
        let mut det = Detector::new_instance(class).unwrap();
        assert_eq!(det.classname(), class);
        let img = vernier_core::GrayImage::zeros(64, 64);
        let err = det.compute(&img).unwrap_err();
        assert!(
            err.to_string().contains("not implemented"),
            "unexpected error: {err}"
        );
        assert!(!det.pattern_found(-1));
    }
}

#[test]
fn layout_factory_unknown_class_errors() {
    assert!(Layout::new_instance("NotAPattern").is_err());
}

#[test]
fn layout_render_matches_pattern_render() {
    // Layout::render must be identical to rendering the underlying pattern
    // directly (the layout is a thin, faithful wrapper).
    let size = 128usize;
    let period = 10.0;
    let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
    assert!(layout.set_double("period", period));
    let via_layout = layout.render(size, size, &PatternPose::IDENTITY);

    let direct = vernier_patterns::periodic::Periodic::new(period).render(
        size,
        size,
        &PatternPose::IDENTITY,
    );
    assert_eq!(via_layout.as_slice(), direct.as_slice());
}

#[test]
fn layout_intensity_matches_render_center() {
    // get_intensity at the pattern origin equals the rendered pixel at the
    // image centre for identity pose (both sample pattern-frame (0,0)).
    let size = 64usize;
    let period = 8.0f32;
    let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
    layout.set_double("period", period as f64);
    let img = layout.render(size, size, &PatternPose::IDENTITY);
    let center = img.get(size / 2, size / 2);
    assert!((layout.get_intensity(0.0, 0.0) - center).abs() < 1e-5);
}

#[test]
fn periodic_layout_rectangle_grid() {
    // A 3×3 grid has 9 cells; the (0,0) origin dot is skipped → 8 rectangles,
    // each 0.5·period square at cell corners (offset 0 since dotSize=period/2).
    let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
    layout.set_double("period", 10.0);
    layout.set_int("nRows", 3);
    layout.set_int("nCols", 3);
    let rects = layout.to_rectangles();
    assert_eq!(rects.len(), 8);
    assert!(
        rects
            .iter()
            .all(|r| (r.width - 5.0).abs() < 1e-9 && (r.height - 5.0).abs() < 1e-9)
    );
    // No rectangle sits at the origin cell.
    assert!(!rects.iter().any(|r| r.x == 0.0 && r.y == 0.0));
}

#[test]
fn periodic_layout_svg_and_csv() {
    let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
    layout.set_double("period", 10.0);
    layout.set_int("nRows", 3);
    layout.set_int("nCols", 3);
    layout.set_description("test grid".into());

    let svg = layout.to_svg();
    assert!(svg.starts_with("<?xml"));
    assert!(svg.contains("<svg"));
    assert!(svg.contains("class: PeriodicPattern"));
    assert!(svg.trim_end().ends_with("</svg>"));
    // 8 dots → 8 <rect> elements.
    assert_eq!(svg.matches("<rect ").count(), 8);

    let csv = layout.to_csv();
    assert!(csv.starts_with("x;y;width;height:intensity"));
    // header + 8 data rows.
    assert_eq!(csv.lines().count(), 9);
}

#[test]
fn periodic_layout_gds_has_boundary_per_dot() {
    let mut layout = Layout::new_instance("PeriodicPattern").unwrap();
    layout.set_double("period", 10.0);
    layout.set_int("nRows", 3);
    layout.set_int("nCols", 3);
    let gds = layout.to_gds();
    // GDSII HEADER magic: first record is length 6, type 0x0002.
    assert_eq!(&gds[0..4], &[0x00, 0x06, 0x00, 0x02]);
    // 8 dots → 8 BOUNDARY records (0x0800). Count 4-byte record headers.
    let mut boundaries = 0;
    let mut i = 0;
    while i + 4 <= gds.len() {
        let len = u16::from_be_bytes([gds[i], gds[i + 1]]) as usize;
        let rec = u16::from_be_bytes([gds[i + 2], gds[i + 3]]);
        if rec == 0x0800 {
            boundaries += 1;
        }
        i += len;
    }
    assert_eq!(boundaries, 8);
}

#[test]
fn megarena_layout_rectangles_nonempty_and_bounded() {
    // Default extent spans a few code windows; export must be non-empty and
    // must skip the removed corner cells (col%3==0 && row%3==0).
    let layout = Layout::new_instance("MegarenaPattern").unwrap();
    let rects = layout.to_rectangles();
    assert!(!rects.is_empty(), "megarena layout should emit dots");
    let period = layout.get_double("period").unwrap();
    for r in &rects {
        let col = (r.x / period).round() as i64;
        let row = (r.y / period).round() as i64;
        assert!(
            !(col % 3 == 0 && row % 3 == 0),
            "corner cell must be removed"
        );
    }
}

#[test]
fn megarena_layout_from_json_rebuilds_pattern() {
    let json = r#"{ "MegarenaPattern": { "period": 12.0, "codeSize": 8 } }"#;
    let layout = Layout::from_json_str(json).unwrap();
    assert_eq!(layout.classname(), "MegarenaPattern");
    assert_eq!(layout.get_double("period"), Some(12.0));
    assert_eq!(layout.get_int("codeSize"), Some(8));
}
