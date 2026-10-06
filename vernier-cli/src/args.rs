//! Command-line argument definitions (argh).

use argh::FromArgs;

/// vernier-rs: GPU-accelerated pose measurement of calibrated patterns.
#[derive(FromArgs)]
pub struct TopLevel {
    #[argh(subcommand)]
    pub command: Command,
}

/// The available subcommands.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum Command {
    Bench(BenchArgs),
    Calibrate(CalibrateArgs),
    CalibrateWebcam(CalibrateWebcamArgs),
    CheckerboardFigures(CheckerboardFiguresArgs),
    MakePattern(MakePatternArgs),
    DetectMegarena(DetectMegarenaArgs),
    RenderPattern(RenderPatternArgs),
    RoundtripMegarena(RoundtripMegarenaArgs),
    Phone(PhoneArgs),
    SolvePnp(SolvePnpArgs),
    Track(TrackArgs),
    Undistort(UndistortArgs),
}

/// Time the full two-direction detection pipeline on a synthetic image.
#[derive(FromArgs)]
#[argh(subcommand, name = "bench")]
pub struct BenchArgs {
    /// backend to use: cpu or gpu (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// square image side length (default: 512)
    #[argh(option, default = "512")]
    pub size: usize,

    /// number of timed iterations (default: 20)
    #[argh(option, default = "20")]
    pub iters: usize,

    /// band-pass filter width in bins (default: 3.0)
    #[argh(option, default = "3.0")]
    pub sigma: f32,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default: 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default: 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default: 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f32,
}

/// Detect a megarena pattern in a real image and print its absolute pose (port
/// of detectingMegarenaPattern.cpp).
#[derive(FromArgs)]
#[argh(subcommand, name = "detect-megarena")]
pub struct DetectMegarenaArgs {
    /// backend to use: cpu (or gpu with the gpu feature)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// path to the image file (JPEG/PNG/BMP/TIFF)
    #[argh(option)]
    pub image: String,

    /// megarena pattern file (see make-pattern); its pitch is read in
    /// micrometres, the unit the pose is printed in
    #[argh(option)]
    pub pattern: String,

    /// band-pass filter width in bins (default 3.0, C++ PatternPhase default)
    #[argh(option, default = "3.0")]
    pub sigma: f32,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f32,

    /// path prefix for debug overlay images (writes <prefix>_spectrum.png and _decoded.png)
    #[argh(option)]
    pub debug_image: Option<String>,

    /// print intermediate detection details (carriers, planes, orientation)
    #[argh(switch)]
    pub verbose: bool,
}

/// Render a megarena at a known pose, run the full detection pipeline, and
/// report the error between the recovered pose and the ground truth.
#[derive(FromArgs)]
#[argh(subcommand, name = "roundtrip-megarena")]
pub struct RoundtripMegarenaArgs {
    /// backend to use: cpu or gpu (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,

    /// megarena pattern file (see make-pattern): its image gives the rendered
    /// size and dot period in pixels, its pitch over that period the camera
    /// pixel size, read in micrometres
    #[argh(option)]
    pub pattern: String,

    /// ground-truth X position in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub x: f64,

    /// ground-truth Y position in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub y: f64,

    /// ground-truth orientation in radians (default: 0.0)
    #[argh(option, default = "0.0")]
    pub theta: f64,

    /// band-pass filter width in bins (default: 3.0)
    #[argh(option, default = "3.0")]
    pub sigma: f64,

    /// inner annulus radius for peak search in bins; 0 = no lower limit (default: 20)
    #[argh(option, default = "20")]
    pub min_frequency: usize,

    /// outer annulus radius for peak search in bins; 0 = no upper limit (default: 500)
    #[argh(option, default = "500")]
    pub max_frequency: usize,

    /// gaussian blur sigma applied to magnitude before peak search (default: 0.5)
    #[argh(option, default = "0.5")]
    pub smoothing_sigma: f64,

    /// render the pattern on the GPU via Vulkan instead of the CPU path (requires --features vulkan)
    #[argh(switch)]
    pub render_gpu: bool,
}

/// Render the board of a pattern file (see make-pattern) at given coordinates
/// and save it as PNG, at the image size the file gives.
#[derive(FromArgs)]
#[argh(subcommand, name = "render-pattern")]
pub struct RenderPatternArgs {
    /// pattern file giving the board, checkerboard or megarena
    #[argh(option)]
    pub pattern: String,

    /// output PNG file path
    #[argh(option)]
    pub output: String,

    /// pattern X offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub x: f64,

    /// pattern Y offset in pixels (default: 0.0)
    #[argh(option, default = "0.0")]
    pub y: f64,

    /// pattern orientation in radians (default: 0.0)
    #[argh(option, default = "0.0")]
    pub theta: f64,
}

/// Generate the explainer figures and measurements for the coded checkerboard.
#[derive(FromArgs)]
#[argh(subcommand, name = "checkerboard-figures")]
pub struct CheckerboardFiguresArgs {
    /// directory the PNGs are written to
    #[argh(option)]
    pub out_dir: String,

    /// checkerboard pattern file (see make-pattern): its image gives the
    /// square side in pixels and the figures' side, which must be square
    #[argh(option)]
    pub pattern: String,

    /// number of poses used for the phase-bias measurement (default: 24)
    #[argh(option, default = "24")]
    pub poses: usize,
}

/// Write a pattern file describing a board: what every other command reads
/// with --pattern.
#[derive(FromArgs)]
#[argh(subcommand, name = "make-pattern")]
pub struct MakePatternArgs {
    #[argh(subcommand)]
    pub pattern: PatternArgs,
}

/// The pattern types make-pattern can describe.
#[derive(FromArgs)]
#[argh(subcommand)]
pub enum PatternArgs {
    Checkerboard(CheckerboardPatternArgs),
    Megarena(MegarenaPatternArgs),
}

/// A coded checkerboard, as render-pattern draws it.
#[derive(FromArgs)]
#[argh(subcommand, name = "checkerboard")]
pub struct CheckerboardPatternArgs {
    /// side of one printed square, in the unit poses should come out in (e.g. mm)
    #[argh(option)]
    pub square: f64,

    /// LFSR code size the board was rendered with (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,

    /// the board uses the diamond layout
    #[argh(switch)]
    pub diamonds: bool,

    /// two code bits per axis in each 5x5 supercell
    #[argh(switch)]
    pub two_bits: bool,

    /// corner radius as a fraction of a square side, from 0.0 for square
    /// corners (default) to 0.5 for as round as a square gets
    #[argh(option, default = "0.0")]
    pub corner_radius: f64,

    /// the uncoded checkerboard: the carrier alone, which gives no absolute
    /// position
    #[argh(switch)]
    pub plain: bool,

    /// width of the rendered image in pixels (default: 512)
    #[argh(option, default = "crate::pattern::DEFAULT_IMAGE_SIDE")]
    pub width: usize,

    /// height of the rendered image in pixels (default: 512)
    #[argh(option, default = "crate::pattern::DEFAULT_IMAGE_SIDE")]
    pub height: usize,

    /// side of one square in the rendered image, in pixels (default: 12.0)
    #[argh(option, default = "crate::pattern::DEFAULT_SQUARE_PX")]
    pub square_px: f64,

    /// where to write the pattern file (default: named after the pattern and
    /// its parameters, e.g. megarena-pitch2-code8-512x512-20px.json)
    #[argh(option)]
    pub output: Option<String>,
}

/// A megarena dot grid, as render-pattern draws it.
#[derive(FromArgs)]
#[argh(subcommand, name = "megarena")]
pub struct MegarenaPatternArgs {
    /// distance between neighbouring dots, in the unit poses should come out in
    /// (default: 2.0, e.g. mm)
    #[argh(option, default = "crate::pattern::DEFAULT_PITCH")]
    pub pitch: f64,

    /// LFSR code size the board was rendered with (default: 8)
    #[argh(option, default = "8")]
    pub code_size: u32,

    /// width of the rendered image in pixels (default: 512)
    #[argh(option, default = "crate::pattern::DEFAULT_IMAGE_SIDE")]
    pub width: usize,

    /// height of the rendered image in pixels (default: 512)
    #[argh(option, default = "crate::pattern::DEFAULT_IMAGE_SIDE")]
    pub height: usize,

    /// dot pitch in the rendered image, in pixels (default: 20.0)
    #[argh(option, default = "crate::pattern::DEFAULT_PITCH_PX")]
    pub square_px: f64,

    /// where to write the pattern file (default: named after the pattern and
    /// its parameters, e.g. megarena-pitch2-code8-512x512-20px.json)
    #[argh(option)]
    pub output: Option<String>,
}

/// Calibrate a camera from photos of the coded checkerboard (as printed from
/// render-pattern), taken at varied angles and places in the frame.
#[derive(FromArgs)]
#[argh(subcommand, name = "calibrate")]
pub struct CalibrateArgs {
    /// pattern file (see make-pattern) giving the board, checkerboard or
    /// megarena
    #[argh(option)]
    pub pattern: String,

    /// camera model: pinhole or fisheye (default: pinhole)
    #[argh(option, default = "String::from(\"pinhole\")")]
    pub model: String,

    /// where to write the calibration (default: camera.json)
    #[argh(option, default = "String::from(\"camera.json\")")]
    pub output: String,

    /// the photos
    #[argh(positional)]
    pub images: Vec<String>,
}

/// Calibrate and track with a phone's camera: the phone opens a page served
/// over HTTPS on the local network and streams its camera to it.
#[derive(FromArgs)]
#[argh(subcommand, name = "phone")]
pub struct PhoneArgs {
    /// calibration of the phone's camera: tracked with if the file exists,
    /// otherwise calibrated on the page and written there
    /// (default: phone-camera.json)
    #[argh(option, default = "String::from(\"phone-camera.json\")")]
    pub camera: String,

    /// distinct views to calibrate from (default: 15)
    #[argh(option, default = "15")]
    pub views: usize,

    /// camera model to calibrate: pinhole or fisheye (default: pinhole)
    #[argh(option, default = "String::from(\"pinhole\")")]
    pub model: String,

    /// pattern file (see make-pattern) giving the board, checkerboard or
    /// megarena
    #[argh(option)]
    pub pattern: String,

    /// HTTPS port on the local network (default: 8443)
    #[argh(option, default = "8443")]
    pub port: u16,

    /// also write every pose to this CSV file
    #[argh(option)]
    pub csv: Option<String>,

    /// where to demodulate the frames: cpu (or gpu with the gpu feature)
    /// (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,
}

/// Calibrate a webcam live: frames are read through ffmpeg until enough
/// distinct views of the board are in.
#[derive(FromArgs)]
#[argh(subcommand, name = "calibrate-webcam")]
pub struct CalibrateWebcamArgs {
    /// camera device, or any input ffmpeg can open such as a video file
    /// (default: /dev/video0)
    #[argh(option, default = "String::from(\"/dev/video0\")")]
    pub device: String,

    /// ffmpeg demuxer, e.g. v4l2 or dshow (default: v4l2 for /dev/ paths)
    #[argh(option)]
    pub format: Option<String>,

    /// encoding asked of the camera, e.g. mjpeg or yuyv422; list them with
    /// `ffmpeg -f v4l2 -list_formats all -i /dev/video0` (default: the camera's)
    #[argh(option)]
    pub input_format: Option<String>,

    /// capture size asked of the camera, e.g. 1280x720 (default: the camera's)
    #[argh(option)]
    pub video_size: Option<String>,

    /// frames per second asked of the camera, e.g. 30 (default: the camera's)
    #[argh(option)]
    pub framerate: Option<String>,

    /// distinct views to collect (default: 15)
    #[argh(option, default = "15")]
    pub views: usize,

    /// seconds between frames examined (default: 1.0)
    #[argh(option, default = "1.0")]
    pub interval: f64,

    /// pattern file (see make-pattern) giving the board, checkerboard or
    /// megarena
    #[argh(option)]
    pub pattern: String,

    /// camera model: pinhole or fisheye (default: pinhole)
    #[argh(option, default = "String::from(\"pinhole\")")]
    pub model: String,

    /// where to write the calibration (default: camera.json)
    #[argh(option, default = "String::from(\"camera.json\")")]
    pub output: String,

    /// directory to save the kept frames in, to rerun with calibrate
    #[argh(option)]
    pub save_frames: Option<String>,

    /// where to demodulate the frames: cpu (or gpu with the gpu feature)
    /// (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,
}

/// Find the pose of the board in photos taken with a calibrated camera.
#[derive(FromArgs)]
#[argh(subcommand, name = "solve-pnp")]
pub struct SolvePnpArgs {
    /// calibration written by calibrate or calibrate-webcam
    #[argh(option)]
    pub camera: String,

    /// pattern file (see make-pattern) giving the board, checkerboard or
    /// megarena
    #[argh(option)]
    pub pattern: String,

    /// the photos
    #[argh(positional)]
    pub images: Vec<String>,
}

/// Resample a photo as an ideal pinhole camera would have taken it.
#[derive(FromArgs)]
#[argh(subcommand, name = "undistort")]
pub struct UndistortArgs {
    /// calibration written by calibrate or calibrate-webcam
    #[argh(option)]
    pub camera: String,

    /// output PNG file path
    #[argh(option)]
    pub output: String,

    /// output focal length over the calibrated one; below 1 keeps more of a
    /// fisheye's field (default: 1.0)
    #[argh(option, default = "1.0")]
    pub zoom: f64,

    /// the photo
    #[argh(positional)]
    pub image: String,
}

/// Follow the board's pose live: every frame is solved with a calibrated
/// camera and the pose is traced on a page served on localhost.
#[derive(FromArgs)]
#[argh(subcommand, name = "track")]
pub struct TrackArgs {
    /// calibration written by calibrate or calibrate-webcam
    #[argh(option)]
    pub camera: String,

    /// pattern file (see make-pattern) giving the board, checkerboard or
    /// megarena
    #[argh(option)]
    pub pattern: String,

    /// camera device, or any input ffmpeg can open such as a video file
    /// (default: /dev/video0)
    #[argh(option, default = "String::from(\"/dev/video0\")")]
    pub device: String,

    /// ffmpeg demuxer, e.g. v4l2 or dshow (default: v4l2 for /dev/ paths)
    #[argh(option)]
    pub format: Option<String>,

    /// encoding asked of the camera, e.g. mjpeg or yuyv422; list them with
    /// `ffmpeg -f v4l2 -list_formats all -i /dev/video0` (default: the camera's)
    #[argh(option)]
    pub input_format: Option<String>,

    /// capture size asked of the camera; must match the calibration
    /// (default: the calibration's size)
    #[argh(option)]
    pub video_size: Option<String>,

    /// frames per second asked of the camera, e.g. 30 (default: the camera's)
    #[argh(option)]
    pub framerate: Option<String>,

    /// port of the page on localhost (default: 8080)
    #[argh(option, default = "8080")]
    pub port: u16,

    /// also write every pose to this CSV file
    #[argh(option)]
    pub csv: Option<String>,

    /// where to demodulate the frames: cpu (or gpu with the gpu feature)
    /// (default: cpu)
    #[argh(option, default = "String::from(\"cpu\")")]
    pub backend: String,
}
