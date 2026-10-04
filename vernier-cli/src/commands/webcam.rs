//! `calibrate-webcam`: grab frames from a camera through ffmpeg until enough
//! distinct views of the board are in, then calibrate.
//!
//! ffmpeg does the device work (V4L2 here, but any input it can open, a video
//! file included) and pipes grayscale PGM frames. A reader thread keeps only
//! the newest one, so a slow measurement never backs up the stream.

use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use vernier_camera::{Model, Target, View, calibrate, measure_view};

use super::calibrate::{code_status, report, save};
use crate::imageio;

/// What `calibrate-webcam` needs, already parsed from the command line.
pub struct WebcamArgs {
    /// Camera device, or any input ffmpeg can open.
    pub device: String,
    /// ffmpeg input format, when not guessed from the device.
    pub format: Option<String>,
    /// Capture size asked of the camera, as `WIDTHxHEIGHT`.
    pub video_size: Option<String>,
    /// Distinct views to collect before calibrating.
    pub views: usize,
    /// Least time between two frames examined.
    pub interval: Duration,
    pub target: Target,
    pub model: Model,
    pub output: PathBuf,
    /// Directory to save the kept frames in, if any.
    pub save_frames: Option<PathBuf>,
}

/// One grayscale frame from the stream, intensities in `0.0..=1.0`.
pub(crate) struct Frame {
    /// Position in the stream, counting from 0.
    pub index: u64,
    pub width: usize,
    pub height: usize,
    /// Row-major pixels.
    pub data: Vec<f32>,
}

impl Frame {
    /// Mean absolute difference from another frame, on every fourth pixel of
    /// every fourth row; infinite if the sizes differ.
    pub fn difference(&self, other: &Frame) -> f32 {
        if (self.width, self.height) != (other.width, other.height) {
            return f32::INFINITY;
        }
        let (mut sum, mut count) = (0.0, 0.0);
        for y in (0..self.height).step_by(4) {
            for x in (0..self.width).step_by(4) {
                let i = y * self.width + x;
                sum += (self.data[i] - other.data[i]).abs();
                count += 1.0;
            }
        }
        sum / count
    }
}

/// What the reader thread hands over: the newest frame not yet taken, and
/// whether the stream has ended.
#[derive(Default)]
struct Latest {
    frame: Option<Frame>,
    ended: bool,
}

/// A running ffmpeg and the thread reading its frames. Dropping it stops both.
pub(crate) struct Capture {
    ffmpeg: Child,
    latest: Arc<Mutex<Latest>>,
    reader: Option<JoinHandle<()>>,
}

impl Capture {
    /// Starts ffmpeg on `device` and a thread that keeps its newest frame.
    /// `format` defaults to V4L2 for `/dev/` paths.
    pub fn open(
        device: &str,
        format: Option<&str>,
        video_size: Option<&str>,
    ) -> Result<Self, String> {
        let mut ffmpeg = ffmpeg_command(device, format, video_size)
            .spawn()
            .map_err(|e| {
                format!("could not start ffmpeg ({e}); it must be installed and on the PATH")
            })?;
        let stdout = ffmpeg.stdout.take().expect("piped stdout");

        let latest = Arc::new(Mutex::new(Latest::default()));
        let shared = Arc::clone(&latest);
        let reader = std::thread::spawn(move || {
            let mut stream = BufReader::new(stdout);
            let mut index = 0;
            // Each new frame replaces the last one if it was not taken in time.
            while let Some((width, height, data)) = read_pgm(&mut stream) {
                shared.lock().unwrap().frame = Some(Frame {
                    index,
                    width,
                    height,
                    data,
                });
                index += 1;
            }
            shared.lock().unwrap().ended = true;
        });
        Ok(Self {
            ffmpeg,
            latest,
            reader: Some(reader),
        })
    }

    /// The newest frame if it is newer than `seen`; `Err` once the stream has
    /// ended and nothing new is left.
    pub fn take_newer(&self, seen: Option<u64>) -> Result<Option<Frame>, ()> {
        let mut latest = self.latest.lock().unwrap();
        match &latest.frame {
            Some(frame) if Some(frame.index) != seen => Ok(latest.frame.take()),
            _ if latest.ended => Err(()),
            _ => Ok(None),
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        // Killing ffmpeg closes the pipe, which ends the reader thread.
        let _ = self.ffmpeg.kill();
        let _ = self.ffmpeg.wait();
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}

/// The ffmpeg invocation that decodes `device` and writes its frames to stdout
/// as a stream of grayscale binary PGM images.
fn ffmpeg_command(device: &str, format: Option<&str>, video_size: Option<&str>) -> Command {
    let is_device = device.starts_with("/dev/");
    let mut command = Command::new("ffmpeg");
    command.args(["-hide_banner", "-loglevel", "error", "-nostdin"]);
    if !is_device {
        // A file plays at its own pace, as a camera would.
        command.arg("-re");
    }
    if let Some(format) = format.or(is_device.then_some("v4l2")) {
        command.args(["-f", format]);
    }
    if let Some(size) = video_size {
        command.args(["-video_size", size]);
    }
    command
        .args(["-i", device])
        .args(["-f", "image2pipe", "-c:v", "pgm", "-pix_fmt", "gray", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    command
}

/// One binary PGM from the stream as width, height and pixels scaled to
/// `0.0..=1.0`; `None` at its end or on anything malformed.
fn read_pgm(stream: &mut impl BufRead) -> Option<(usize, usize, Vec<f32>)> {
    let [magic, width, height, max] = read_pgm_header(stream)?;
    if magic != "P5" {
        return None;
    }
    let width: usize = width.parse().ok()?;
    let height: usize = height.parse().ok()?;
    let max: u32 = max.parse().ok()?;
    // Above 255 every sample takes two bytes, most significant first.
    let two_bytes = max > 255;
    let mut bytes = vec![0u8; width * height * if two_bytes { 2 } else { 1 }];
    stream.read_exact(&mut bytes).ok()?;
    let scale = 1.0 / max as f32;
    let data = if two_bytes {
        bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&pair| u16::from_be_bytes(pair) as f32 * scale)
            .collect()
    } else {
        bytes.iter().map(|&b| b as f32 * scale).collect()
    };
    Some((width, height, data))
}

/// The four whitespace-separated header fields of a PGM (magic, width, height,
/// largest value), skipping `#` comments. Reads through the single whitespace
/// byte after the last field, leaving the stream at the first pixel.
fn read_pgm_header(stream: &mut impl BufRead) -> Option<[String; 4]> {
    let mut field = Vec::new();
    let mut fields = Vec::new();
    while fields.len() < 4 {
        let mut byte = [0u8];
        stream.read_exact(&mut byte).ok()?;
        if byte[0] == b'#' {
            let mut comment = Vec::new();
            stream.read_until(b'\n', &mut comment).ok()?;
        } else if byte[0].is_ascii_whitespace() {
            if !field.is_empty() {
                fields.push(String::from_utf8(std::mem::take(&mut field)).ok()?);
            }
        } else {
            field.push(byte[0]);
        }
    }
    fields.try_into().ok()
}

/// Where the board sits in a frame and how it is foreshortened, as returned by
/// [`signature`]: the pixel centroid, and the upper triangle `[xx, xy, yy]` of
/// the symmetric 2×2 matrix `A·Aᵀ`.
pub(crate) type Signature = ([f64; 2], [f64; 3]);

/// Where the board sits in the frame and how it is foreshortened: the centroid
/// of the points and `A·Aᵀ` of the best affine map `A` from board to image,
/// which a turn of the board's frame leaves alone.
pub(crate) fn signature(view: &View) -> Signature {
    let count = view.points.len() as f64;
    let mean = |coordinate: &dyn Fn(&vernier_camera::PointMatch) -> f64| {
        view.points.iter().map(coordinate).sum::<f64>() / count
    };
    let board_mean = [mean(&|p| p.board[0]), mean(&|p| p.board[1])];
    let pixel_mean = [mean(&|p| p.pixel[0]), mean(&|p| p.pixel[1])];

    // Least squares for pixel = A·board, both centred: A = C·S⁻¹, with S the
    // board points' scatter and C the pixel-board cross terms.
    let (mut s_xx, mut s_xy, mut s_yy) = (0.0, 0.0, 0.0);
    let (mut c_ux, mut c_uy, mut c_vx, mut c_vy) = (0.0, 0.0, 0.0, 0.0);
    for p in &view.points {
        let (x, y) = (p.board[0] - board_mean[0], p.board[1] - board_mean[1]);
        let (u, v) = (p.pixel[0] - pixel_mean[0], p.pixel[1] - pixel_mean[1]);
        s_xx += x * x;
        s_xy += x * y;
        s_yy += y * y;
        c_ux += u * x;
        c_uy += u * y;
        c_vx += v * x;
        c_vy += v * y;
    }
    let det = s_xx * s_yy - s_xy * s_xy;
    // One row of C times S⁻¹ = [[s_yy, −s_xy], [−s_xy, s_xx]] / det.
    let times_inverse =
        |cx: f64, cy: f64| [(cx * s_yy - cy * s_xy) / det, (cy * s_xx - cx * s_xy) / det];
    let (a_u, a_v) = (times_inverse(c_ux, c_uy), times_inverse(c_vx, c_vy));
    (
        pixel_mean,
        [
            a_u[0] * a_u[0] + a_u[1] * a_u[1],
            a_u[0] * a_v[0] + a_u[1] * a_v[1],
            a_v[0] * a_v[0] + a_v[1] * a_v[1],
        ],
    )
}

/// Too close to a view already kept to add anything: the board has neither
/// moved nor turned since, as when it is held still between two captures.
/// Returns the index of that kept view. `diagonal` is the frame's, in pixels.
pub(crate) fn duplicate(new: &Signature, kept: &[Signature], diagonal: f64) -> Option<usize> {
    // Frobenius norm of the symmetric matrix stored as its upper triangle.
    let norm = |m: &[f64; 3]| (m[0] * m[0] + 2.0 * m[1] * m[1] + m[2] * m[2]).sqrt();
    kept.iter().position(|old| {
        let (new_centre, new_shape) = new;
        let (old_centre, old_shape) = old;
        let moved = (new_centre[0] - old_centre[0]).hypot(new_centre[1] - old_centre[1]) / diagonal;
        let shape_change = [
            new_shape[0] - old_shape[0],
            new_shape[1] - old_shape[1],
            new_shape[2] - old_shape[2],
        ];
        moved < 0.05 && norm(&shape_change) < 0.05 * norm(old_shape)
    })
}

/// Mean frame-to-frame difference (intensities in `0.0..=1.0`) above which the
/// board is taken to be moving. Sensor noise and compression stay well under.
pub(crate) const STILL_LIMIT: f32 = 0.01;

/// Waits up to a second for the frame after `index`.
pub(crate) fn next_frame(capture: &Capture, index: u64) -> Option<Frame> {
    let deadline = Instant::now() + Duration::from_secs(1);
    while Instant::now() < deadline {
        match capture.take_newer(Some(index)) {
            Ok(Some(frame)) => return Some(frame),
            Ok(None) => std::thread::sleep(Duration::from_millis(5)),
            Err(()) => return None,
        }
    }
    None
}

/// Opens the camera, collects the views and calibrates from them. If the
/// stream ends early, calibrates from what was collected.
pub fn run(args: &WebcamArgs) -> Result<(), String> {
    if args.views < 2 {
        return Err("at least 2 views are needed".into());
    }
    if let Some(dir) = &args.save_frames {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    }
    let capture = Capture::open(
        &args.device,
        args.format.as_deref(),
        args.video_size.as_deref(),
    )?;
    eprintln!(
        "reading {}; hold the board in view and change its angle and place between captures",
        args.device
    );

    let mut collected = Collected::default();
    let mut seen = None;
    let mut next_examined = Instant::now();
    while collected.views.len() < args.views {
        let now = Instant::now();
        if now < next_examined {
            std::thread::sleep(next_examined - now);
        }
        let frame = match capture.take_newer(seen) {
            Ok(Some(frame)) => frame,
            Ok(None) => {
                std::thread::sleep(Duration::from_millis(10));
                continue;
            }
            Err(()) if seen.is_none() => {
                return Err(format!(
                    "no frames came from {}; is it a camera ffmpeg can open?",
                    args.device
                ));
            }
            Err(()) => break,
        };
        seen = Some(frame.index);
        next_examined = Instant::now() + args.interval;

        // The frame after this one tells whether the board is held still.
        let Some(following) = next_frame(&capture, frame.index) else {
            break;
        };
        collected.consider(&frame, &following, args)?;
    }
    drop(capture);

    if collected.views.len() < args.views {
        eprintln!(
            "the stream ended after {} of {} views",
            collected.views.len(),
            args.views
        );
    }
    let result = calibrate(&collected.views, args.model).map_err(|e| e.to_string())?;
    report(&result, &collected.views, &collected.names);
    save(&args.output, &result)
}

/// The views kept so far, with a name for each in the report and the
/// signature that tells new views from them.
#[derive(Default)]
struct Collected {
    views: Vec<View>,
    names: Vec<String>,
    signatures: Vec<Signature>,
}

impl Collected {
    /// Keeps `frame` as a view if the board was still, is found in it, and is
    /// placed differently from every view kept so far; says on stderr what
    /// became of it. Fails only on what should stop the capture.
    fn consider(
        &mut self,
        frame: &Frame,
        following: &Frame,
        args: &WebcamArgs,
    ) -> Result<(), String> {
        let label = format!("frame {}", frame.index);
        // A camera reads its rows one after another, so a board that moves
        // during the exposure comes out sheared, which no lens model explains.
        let moved = frame.difference(following);
        if moved > STILL_LIMIT {
            eprintln!("{label}: the board is moving ({moved:.3}), hold it still for a moment");
            return Ok(());
        }
        let view = match measure_view(&frame.data, frame.width, frame.height, &args.target) {
            Ok(view) => view,
            Err(e) => {
                eprintln!("{label}: {e}");
                return Ok(());
            }
        };
        if let Some(first) = self.views.first()
            && (first.width, first.height) != (view.width, view.height)
        {
            return Err("the frame size changed mid-stream".into());
        }
        let placement = signature(&view);
        let diagonal = (frame.width as f64).hypot(frame.height as f64);
        if let Some(close) = duplicate(&placement, &self.signatures, diagonal) {
            eprintln!(
                "{label}: too close to view {}, move or tilt the board",
                close + 1
            );
            return Ok(());
        }

        let number = self.views.len() + 1;
        let name = match &args.save_frames {
            Some(dir) => {
                let path = dir.join(format!("view_{number:02}.png"));
                imageio::save_grayscale_png(&path, frame.width, frame.height, &frame.data)?;
                path.display().to_string()
            }
            None => format!("view {number}"),
        };
        eprintln!(
            "{label}: kept as view {number}/{}, {} points, {}",
            args.views,
            view.points.len(),
            code_status(&view)
        );
        self.signatures.push(placement);
        self.views.push(view);
        self.names.push(name);
        Ok(())
    }
}
