//! `phone`: calibrate and track with a phone's camera.
//!
//! Browsers only open the camera on a secure page, so the page is served over
//! HTTPS, with a self-signed certificate made on first use, to the whole local
//! network. The phone draws each camera frame to a canvas and posts it as a
//! JPEG; the next one goes once the server has taken it, so the phone sends at
//! the pace frames are measured. The pose and every debug view come back the
//! way `track` shows them, to the phone and to any other browser watching.
//!
//! Without a calibration, the page collects views as `calibrate-webcam` does:
//! the board held still, each view distinct from those already kept. Once
//! there are enough, the camera is calibrated, written, and tracking starts.

use std::net::{IpAddr, UdpSocket};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use serde_json::{Value, json};
use vernier_camera::{Camera, Model, Target, View, calibrate};

use super::calibrate::{load_camera, report, save};
use super::track::{Request, Response, Tracker, listen};
use super::webcam::{Frame, STILL_LIMIT, Signature, duplicate, signature};
use crate::backend_select::{BackendKind, Demodulator};

/// Least time between two kept calibration views, so a slow sweep of the
/// board does not fill the set with near copies.
const VIEW_INTERVAL: Duration = Duration::from_millis(700);

/// How long an upload waits for the frame before it to be taken.
const UPLOAD_WAIT: Duration = Duration::from_secs(5);

/// Options of `vernier phone`.
pub struct PhoneArgs {
    pub target: Target,
    /// Calibration file: read if it exists, written once the page calibrates.
    pub camera: PathBuf,
    /// Distinct views to collect before calibrating.
    pub views: usize,
    pub model: Model,
    /// Port of the page, on every network interface.
    pub port: u16,
    /// File every pose is also written to, as CSV.
    pub csv: Option<PathBuf>,
    /// Where the frames are demodulated.
    pub backend: BackendKind,
}

/// What the measuring loop does with the frames.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Collects views of the board until there are enough to calibrate.
    Calibrate,
    /// Solves the board's pose in each frame.
    Track,
}

/// What the phone needs to know to send frames, answered to every upload.
struct State {
    mode: Mode,
    /// Frame size to send: the calibration's, or that of the first kept view.
    size: Option<(usize, usize)>,
    /// Calibration views kept so far.
    views: usize,
    /// Calibration views needed.
    needed: usize,
    /// The page asked to calibrate again; the loop acts on it at the next
    /// frame.
    recalibrate: bool,
}

impl State {
    fn json(&self) -> Value {
        json!({
            "phone": true,
            "mode": match self.mode { Mode::Calibrate => "calibrate", Mode::Track => "track" },
            "size": self.size.map(|(w, h)| [w, h]),
            "views": self.views,
            "needed": self.needed,
        })
    }
}

/// Hands uploaded frames from the server's threads to the measuring loop, one
/// at a time.
///
/// An upload waits until the loop has taken its frame before it answers, and
/// the phone sends the next frame only once answered: so the phone sends at
/// the pace frames are measured, and the frame measured is always a fresh
/// one rather than one queued up seconds before.
struct Inbox {
    /// The newest frame uploaded and not yet taken.
    slot: Mutex<Option<Frame>>,
    /// Signalled when a frame is put in the slot.
    arrived: Condvar,
    /// Signalled when the loop takes the frame out.
    taken: Condvar,
    /// Frames uploaded so far, which numbers them.
    uploads: AtomicU64,
    state: Mutex<State>,
}

impl Inbox {
    /// Tracks with `camera` if there is one, or else calibrates first.
    fn new(camera: Option<&Camera>, needed: usize) -> Self {
        Self {
            slot: Mutex::new(None),
            arrived: Condvar::new(),
            taken: Condvar::new(),
            uploads: AtomicU64::new(0),
            state: Mutex::new(State {
                mode: if camera.is_some() {
                    Mode::Track
                } else {
                    Mode::Calibrate
                },
                size: camera.map(|c| (c.width, c.height)),
                views: 0,
                needed,
                recalibrate: false,
            }),
        }
    }

    /// Puts a frame in the slot, replacing any not yet taken, and waits for
    /// the loop to take it, up to `UPLOAD_WAIT`.
    fn put(&self, frame: Frame) {
        let mut slot = self.slot.lock().unwrap();
        *slot = Some(frame);
        self.arrived.notify_all();
        let _ = self
            .taken
            .wait_timeout_while(slot, UPLOAD_WAIT, |slot| slot.is_some())
            .unwrap();
    }

    /// Waits for a frame and takes it, releasing the upload waiting on it.
    fn take(&self) -> Frame {
        let mut slot = self.slot.lock().unwrap();
        loop {
            if let Some(frame) = slot.take() {
                self.taken.notify_all();
                return frame;
            }
            slot = self.arrived.wait(slot).unwrap();
        }
    }

    /// The state the phone is told, as a JSON response.
    fn state_response(&self) -> Response {
        Response::ok(
            "application/json",
            self.state.lock().unwrap().json().to_string(),
        )
    }

    /// The phone's routes, answered before the page's own.
    fn route(&self, request: &Request) -> Option<Response> {
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/phone") => Some(self.state_response()),
            ("POST", "/upload") => Some(self.receive_upload(&request.body)),
            ("POST", "/calibrate") => {
                self.state.lock().unwrap().recalibrate = true;
                Some(self.state_response())
            }
            _ => None,
        }
    }

    /// Decodes an uploaded image to grey and hands it to the loop.
    fn receive_upload(&self, body: &[u8]) -> Response {
        let image = match image::load_from_memory(body) {
            Ok(image) => image.to_luma8(),
            Err(e) => return Response::bad(format!("not an image: {e}")),
        };
        let (width, height) = (image.width() as usize, image.height() as usize);
        let data = image.into_raw().iter().map(|&v| v as f32 / 255.0).collect();
        self.put(Frame {
            index: self.uploads.fetch_add(1, Ordering::Relaxed),
            width,
            height,
            data,
        });
        self.state_response()
    }

    /// Takes back a request to calibrate again, if the page made one, and
    /// starts the collection over. Returns the mode to measure the next frame
    /// in, and whether it was reset.
    fn take_mode(&self) -> (Mode, bool) {
        let mut state = self.state.lock().unwrap();
        let reset = std::mem::take(&mut state.recalibrate);
        if reset {
            state.mode = Mode::Calibrate;
            state.size = None;
            state.views = 0;
        }
        (state.mode, reset)
    }

    /// Tells the phone how far the collection has got, and the frame size
    /// the views so far set.
    fn update_collection(&self, collector: &Collector) {
        let mut state = self.state.lock().unwrap();
        state.views = collector.views.len();
        state.size = collector.size();
    }

    /// Switches to tracking at the camera's size, if calibrating gave one.
    fn update_calibrated(&self, camera: Option<&Camera>, collector: &Collector) {
        let mut state = self.state.lock().unwrap();
        if let Some(camera) = camera {
            state.mode = Mode::Track;
            state.size = Some((camera.width, camera.height));
        }
        state.views = collector.views.len();
    }
}

/// Calibration views collected so far.
struct Collector {
    views: Vec<View>,
    /// Where each kept view puts the board and its shape, to tell a new view
    /// from those already kept.
    signatures: Vec<Signature>,
    /// When the last view was kept.
    last_kept: Option<Instant>,
}

impl Collector {
    fn new() -> Self {
        Self {
            views: Vec::new(),
            signatures: Vec::new(),
            last_kept: None,
        }
    }

    /// The frame size of the views kept, all the same.
    fn size(&self) -> Option<(usize, usize)> {
        self.views.first().map(|v| (v.width, v.height))
    }

    /// Keeps the view if it adds something; says what became of it. `reason`
    /// is why there is no view, and `moved` how much the frame changed since
    /// the one before.
    fn offer(&mut self, view: Option<&View>, reason: Option<&str>, moved: f32) -> String {
        let Some(view) = view else {
            return format!("no board: {}", reason.unwrap_or("not measured"));
        };
        if let Some((w, h)) = self.size()
            && (w, h) != (view.width, view.height)
        {
            return format!(
                "frames are {}x{}, the views so far {w}x{h}",
                view.width, view.height
            );
        }
        // A camera reads its rows one after another, so a board that moves
        // during the exposure comes out sheared, which no lens model explains.
        if moved > STILL_LIMIT {
            return format!("hold the board still ({moved:.3} change since the last frame)");
        }
        if self.last_kept.is_some_and(|t| t.elapsed() < VIEW_INTERVAL) {
            return format!(
                "kept view {}; move the board to the next place",
                self.views.len()
            );
        }
        let new_signature = signature(view);
        let diagonal = (view.width as f64).hypot(view.height as f64);
        if let Some(k) = duplicate(&new_signature, &self.signatures, diagonal) {
            return format!("too close to view {}: move or tilt the board", k + 1);
        }
        self.signatures.push(new_signature);
        self.views.push(view.clone());
        self.last_kept = Some(Instant::now());
        format!("kept view {}", self.views.len())
    }

    /// Every third point of each view kept, in pixels, to show coverage.
    fn coverage(&self) -> Value {
        json!(
            self.views
                .iter()
                .map(|v| {
                    v.points
                        .iter()
                        .step_by(3)
                        .map(|p| [p.pixel[0].round() as i64, p.pixel[1].round() as i64])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>()
        )
    }
}

/// Serves the page to the network and measures the frames the phone sends,
/// calibrating first if there is no calibration yet. Runs until stopped.
pub fn run(args: &PhoneArgs) -> Result<(), String> {
    let mut camera = if args.camera.exists() {
        Some(load_camera(&args.camera)?)
    } else {
        None
    };
    let demodulator = Arc::new(Demodulator::new(args.backend)?);
    let inbox = Arc::new(Inbox::new(camera.as_ref(), args.views));

    let ip = local_ip();
    let tls = tls_config(ip)?;
    let routes = {
        let inbox = Arc::clone(&inbox);
        Box::new(move |request: &Request| inbox.route(request))
    };
    let shared = listen(("0.0.0.0", args.port), args.target, Some(tls), Some(routes))?;
    print_instructions(args, ip, camera.as_ref());

    let mut tracker = Tracker::new(shared, args.target, args.csv.as_deref(), demodulator)?;
    let mut collector = Collector::new();
    // The frame before, to tell whether the board is held still.
    let mut previous_frame: Option<Frame> = None;
    loop {
        let frame = inbox.take();
        let (mode, reset) = inbox.take_mode();
        if reset {
            collector = Collector::new();
            previous_frame = None;
            tracker.announce(json!({ "mode": "calibrate", "message": "calibrating again" }));
        }
        match (mode, &camera) {
            (Mode::Track, Some(camera)) => track_frame(&mut tracker, camera, frame)?,
            _ => {
                let moved = previous_frame
                    .as_ref()
                    .map_or(f32::INFINITY, |previous| frame.difference(previous));
                previous_frame = Some(Frame {
                    data: frame.data.clone(),
                    ..frame
                });
                collect_frame(&mut tracker, &mut collector, frame, moved, args.views)?;
                inbox.update_collection(&collector);
                if collector.views.len() >= args.views {
                    camera = calibrate_views(args, &tracker, &mut collector)?;
                    inbox.update_calibrated(camera.as_ref(), &collector);
                }
            }
        }
    }
}

/// Prints where to open the page, also as a QR code for the phone to scan,
/// and whether it tracks or calibrates first.
fn print_instructions(args: &PhoneArgs, ip: Option<IpAddr>, camera: Option<&Camera>) {
    let host = ip.map_or("<this computer's address>".to_string(), |ip| ip.to_string());
    let url = format!("https://{host}:{}/", args.port);
    if let Ok(code) = qrcode::QrCode::new(url.as_bytes()) {
        use qrcode::render::unicode::Dense1x2;
        // Light modules drawn dark, for the usual light-on-dark terminal.
        let art = code
            .render::<Dense1x2>()
            .dark_color(Dense1x2::Light)
            .light_color(Dense1x2::Dark)
            .build();
        eprintln!("{art}");
    }
    eprintln!("open {url} on the phone, on the same network as this computer;");
    eprintln!("the certificate is self-signed, so the browser warns once: go on to the page.");
    match camera {
        Some(c) => eprintln!(
            "tracking with {} ({}x{})",
            args.camera.display(),
            c.width,
            c.height
        ),
        None => eprintln!(
            "no calibration at {}: the page calibrates first, from {} views",
            args.camera.display(),
            args.views
        ),
    }
}

/// Solves the pose in a frame, or tells the page the frame is not the size
/// the camera was calibrated at.
fn track_frame(tracker: &mut Tracker, camera: &Camera, frame: Frame) -> Result<(), String> {
    if (frame.width, frame.height) != (camera.width, camera.height) {
        tracker.announce(json!({
            "ok": false, "frame": frame.index,
            "reason": format!(
                "frames are {}x{} but the camera was calibrated at {}x{}; keep the phone as it was held then",
                frame.width, frame.height, camera.width, camera.height
            ),
        }));
        return Ok(());
    }
    tracker.step(Some(camera), frame, |_, _, _| {})
}

/// Measures a frame and offers its view to the collection, telling the page
/// what became of it and where the views kept so far cover the frame.
fn collect_frame(
    tracker: &mut Tracker,
    collector: &mut Collector,
    frame: Frame,
    moved: f32,
    needed: usize,
) -> Result<(), String> {
    tracker.step(None, frame, |event, debug, view| {
        let message = collector.offer(view, event["reason"].as_str(), moved);
        for value in [&mut *event, &mut *debug] {
            value["mode"] = json!("calibrate");
            value["message"] = json!(message);
            value["views"] = json!(collector.views.len());
            value["needed"] = json!(needed);
        }
        debug["coverage"] = collector.coverage();
    })
}

/// Calibrates from the views collected, writes the camera and tells the page.
/// On failure the collection starts over.
fn calibrate_views(
    args: &PhoneArgs,
    tracker: &Tracker,
    collector: &mut Collector,
) -> Result<Option<Camera>, String> {
    let n = collector.views.len();
    tracker.announce(
        json!({ "mode": "calibrate", "message": format!("calibrating from {n} views…") }),
    );
    eprintln!("calibrating from {n} views");
    match calibrate(&collector.views, args.model) {
        Ok(result) => {
            let names: Vec<String> = (1..=n).map(|i| format!("view {i}")).collect();
            report(&result, &collector.views, &names);
            save(&args.camera, &result)?;
            let c = &result.camera;
            tracker.announce(json!({
                "mode": "calibrated",
                "message": format!("calibrated, rms {:.3} px; written to {}", result.rms, args.camera.display()),
                "rms": result.rms,
                "camera": {
                    "model": c.model.name(), "width": c.width, "height": c.height,
                    "fx": c.fx, "fy": c.fy, "cx": c.cx, "cy": c.cy, "distortion": c.distortion,
                },
                "per_view_rms": result.views.iter().map(|v| v.rms).collect::<Vec<_>>(),
            }));
            Ok(Some(result.camera))
        }
        Err(e) => {
            tracker.announce(json!({
                "mode": "calibrate",
                "message": format!("calibration failed ({e}); collecting again"),
            }));
            eprintln!("calibration failed: {e}");
            *collector = Collector::new();
            Ok(None)
        }
    }
}

/// The address other machines on the network reach this one at: the source
/// address of a route outwards. Connecting a UDP socket sends nothing.
fn local_ip() -> Option<IpAddr> {
    let socket = UdpSocket::bind("0.0.0.0:0").ok()?;
    socket.connect("192.0.2.1:9").ok()?;
    Some(socket.local_addr().ok()?.ip())
}

/// Where the certificate is kept, so the phone's browser only has to accept
/// it once.
fn config_dir() -> PathBuf {
    std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".config")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("vernier")
}

/// The server's TLS configuration, from the self-signed certificate kept in
/// the configuration directory, made first if there is none. It names
/// `localhost` and the machine's network address, those the phone opens.
/// Writes a file only its owner can read, as a private key should be.
fn write_private(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create(true).truncate(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    std::io::Write::write_all(&mut options.open(path)?, contents)
}

fn tls_config(ip: Option<IpAddr>) -> Result<Arc<rustls::ServerConfig>, String> {
    let dir = config_dir();
    let (cert_path, key_path) = (dir.join("phone-cert.pem"), dir.join("phone-key.pem"));
    if !(cert_path.exists() && key_path.exists()) {
        let mut names = vec!["localhost".to_string()];
        names.extend(ip.map(|ip| ip.to_string()));
        let made = rcgen::generate_simple_self_signed(names).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
        std::fs::write(&cert_path, made.cert.pem())
            .map_err(|e| format!("could not write {}: {e}", cert_path.display()))?;
        write_private(&key_path, made.signing_key.serialize_pem().as_bytes())
            .map_err(|e| format!("could not write {}: {e}", key_path.display()))?;
        eprintln!("made a self-signed certificate in {}", dir.display());
    }
    let read_file = |path: &Path| {
        std::fs::read(path).map_err(|e| format!("could not read {}: {e}", path.display()))
    };
    let certs = CertificateDer::pem_slice_iter(&read_file(&cert_path)?)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| format!("{}: {e}", cert_path.display()))?;
    let key = PrivateKeyDer::from_pem_slice(&read_file(&key_path)?)
        .map_err(|e| format!("{}: {e}", key_path.display()))?;
    let config = rustls::ServerConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .map_err(|e| e.to_string())?
    .with_no_client_auth()
    .with_single_cert(certs, key)
    .map_err(|e| e.to_string())?;
    Ok(Arc::new(config))
}

#[cfg(test)]
mod tests {
    use nalgebra::Vector3;
    use vernier_camera::{RigidPose, Scene};

    use super::*;

    /// Writes synthetic views of the board as JPEGs, as a phone would send
    /// them, for driving `vernier phone` from a script:
    /// `cargo test -p vernier-cli --release -- --ignored phone_frames`.
    #[test]
    #[ignore]
    fn phone_frames() {
        let mut camera = Camera::ideal(Model::Pinhole, 480, 360, 450.0, 452.0, 243.0, 177.0);
        camera.distortion = vec![-0.22, 0.07, 0.0006, -0.0004, 0.0];
        let target = Target::new(5.0, 6);
        let dir = std::env::temp_dir().join("vernier-phone-frames");
        std::fs::create_dir_all(&dir).unwrap();
        let mut poses = Vec::new();
        // Still views at varied tilts, then a slow sweep.
        for i in 0..12 {
            let a = i as f64 * 0.9;
            poses.push((
                format!("still_{i:02}"),
                Vector3::new(0.45 * a.sin(), 0.4 * (1.3 * a).cos(), 0.5 * a),
                Vector3::new(
                    12.0 * (0.7 * a).cos(),
                    8.0 * a.sin(),
                    190.0 + 25.0 * (0.5 * a).sin(),
                ),
            ));
        }
        for i in 0..60 {
            let t = i as f64 / 15.0;
            poses.push((
                format!("move_{i:02}"),
                Vector3::new(0.3 * (0.7 * t).sin(), 0.25 * (0.5 * t).cos(), 0.3 + 0.2 * t),
                Vector3::new(10.0 * (0.3 * t).sin(), 0.0, 200.0 + 20.0 * (0.4 * t).sin()),
            ));
        }
        for (name, r, t) in poses {
            let data = Scene {
                camera: &camera,
                target: &target,
                half_size: [200.0, 160.0],
                background: 0.45,
                supersample: 2,
            }
            .render(&RigidPose::from_vectors(r, t));
            let bytes: Vec<u8> = data.iter().map(|&v| (v * 255.0).round() as u8).collect();
            image::GrayImage::from_raw(480, 360, bytes)
                .unwrap()
                .save(dir.join(format!("{name}.jpg")))
                .unwrap();
        }
        eprintln!("wrote {}", dir.display());
    }
}
