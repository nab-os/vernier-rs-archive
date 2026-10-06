//! `track`: the pose of the board, frame after frame, traced live in a
//! browser.
//!
//! Frames come through ffmpeg as in `calibrate-webcam`; each one is measured
//! and solved as fast as that goes, older frames being dropped. A small HTTP
//! server on localhost serves the page, the poses as server-sent events, and
//! for the last few frames everything the measurement went through: the
//! frame, its spectrum, phase maps and restored board as images, the points
//! with their residuals, and a probe of any pixel.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use nalgebra::{Matrix3, Vector3};
use serde_json::{Value, json};
use vernier_camera::{
    Attempt, Camera, Code, Field, PnpSolution, Target, Trace, View, demodulated_field,
    measure_view_traced, solve_pnp,
};
use vernier_core::Real;

use super::calibrate::load_camera;
use super::webcam::{Capture, Frame};

/// The page, served at `/`.
const PAGE: &str = include_str!("track.html");

/// Frames whose debug data stays available, besides a pinned one.
const KEPT_FRAMES: usize = 6;

/// Largest request body taken, a frame's JPEG with room to spare.
const MAX_BODY: usize = 64 << 20;

/// How long the event stream stays silent before it sends a comment line,
/// so that a closed page shows up as a failed write.
const KEEP_ALIVE: Duration = Duration::from_secs(10);

/// Field amplitude drawn at full brightness, a clean board's.
const FULL_AMPLITUDE: f32 = 0.45;

/// Lines of constant first and second square coordinate (the palette's
/// orange and blue).
const LINE_I: [f32; 3] = [0.922, 0.408, 0.204];
const LINE_J: [f32; 3] = [0.165, 0.471, 0.839];

/// Options of `vernier track`.
pub struct TrackArgs {
    /// Calibration file of the camera.
    pub camera: PathBuf,
    pub target: Target,
    /// Device ffmpeg reads frames from.
    pub device: String,
    /// Pixel format asked of the device, if any.
    pub format: Option<String>,
    /// Frame size asked of the device, as `WIDTHxHEIGHT`.
    pub video_size: Option<String>,
    /// Port of the page on localhost.
    pub port: u16,
    /// File every pose is also written to, as CSV.
    pub csv: Option<PathBuf>,
}

/// Reads frames from the camera and tracks the board until the camera stops.
pub fn run(args: &TrackArgs) -> Result<(), String> {
    let camera = load_camera(&args.camera)?;
    let shared = listen(("127.0.0.1", args.port), args.target, None, None)?;
    let capture = Capture::open(
        &args.device,
        args.format.as_deref(),
        args.video_size.as_deref(),
    )?;
    eprintln!("open http://localhost:{}/ to watch the pose", args.port);
    let mut tracker = Tracker::new(shared, args.target, args.csv.as_deref())?;
    let mut last_index: Option<u64> = None;
    loop {
        let frame = match capture.take_newer(last_index) {
            Ok(Some(frame)) => frame,
            // Nothing newer yet: measuring is faster than the camera.
            Ok(None) => {
                std::thread::sleep(Duration::from_millis(5));
                continue;
            }
            Err(()) if last_index.is_none() => {
                return Err(format!(
                    "no frames came from {}; is it a camera ffmpeg can open?",
                    args.device
                ));
            }
            // The stream ended after some frames: done.
            Err(()) => return Ok(()),
        };
        last_index = Some(frame.index);
        if (frame.width, frame.height) != (camera.width, camera.height) {
            return Err(format!(
                "frames are {}x{} but the camera was calibrated at {}x{}; pass --video-size {}x{}",
                frame.width, frame.height, camera.width, camera.height, camera.width, camera.height
            ));
        }
        tracker.step(Some(&camera), frame, |_, _, _| {})?;
    }
}

// ---------- what the server keeps ----------

/// The log magnitude of a frame's spectrum, zero frequency at the centre.
struct Spectrum {
    /// Index of the frame it was computed on.
    frame: u64,
    magnitude: Vec<f32>,
}

/// One processed frame and what the measurement left behind.
struct Snapshot {
    index: u64,
    width: usize,
    height: usize,
    /// Intensity per pixel, 0 to 1, row after row.
    pixels: Vec<f32>,
    /// The frame as a JPEG, encoded once since every page shows it.
    jpeg: Vec<u8>,
    /// The debug data of the frame, already serialised.
    debug_json: String,
    /// The newest spectrum computed, at this frame or before. Frames that
    /// reuse the previous carriers compute none.
    spectrum: Option<Arc<Spectrum>>,
    /// The attempt the view came from, or the last one tried.
    attempt: Option<Attempt>,
    /// The demodulated field, worked out the first time a view needs it.
    field: OnceLock<Option<Field>>,
}

impl Snapshot {
    /// The demodulated field of the frame, computed on first use: it is slow,
    /// and most frames are never looked at that closely.
    fn field(&self) -> Option<&Field> {
        self.field
            .get_or_init(|| {
                let attempt = self.attempt.as_ref()?;
                demodulated_field(&self.pixels, self.width, self.height, attempt)
            })
            .as_ref()
    }
}

/// What the page polls and streams: every event so far, and the newest
/// frames.
#[derive(Default)]
struct Feed {
    /// Every event, serialised, in order. Each page's stream remembers how
    /// many it has sent.
    events: Vec<String>,
    /// The last `KEPT_FRAMES` frames, oldest first.
    recent: VecDeque<Arc<Snapshot>>,
    /// Held while the page is paused on it, so it outlives `recent`.
    pinned: Option<Arc<Snapshot>>,
}

impl Feed {
    /// The frame with the given index, if still kept, or else the newest.
    fn find(&self, index: Option<u64>) -> Option<Arc<Snapshot>> {
        match index {
            None => self.recent.back().cloned(),
            Some(index) => self
                .recent
                .iter()
                .chain(self.pinned.iter())
                .find(|snapshot| snapshot.index == index)
                .cloned(),
        }
    }
}

/// State shared by the measuring loop and the server's threads.
pub(crate) struct Shared {
    feed: Mutex<Feed>,
    /// Signalled whenever an event is added, to wake the event streams.
    changed: Condvar,
    target: Target,
    /// Routes answered before the server's own, as the phone's upload.
    extra_routes: Option<Routes>,
}

impl Shared {
    /// Adds an event, and a frame if it comes with one, then wakes every
    /// event stream.
    fn publish(&self, event: &Value, snapshot: Option<Arc<Snapshot>>) {
        let mut feed = self.feed.lock().unwrap();
        if let Some(snapshot) = snapshot {
            feed.recent.push_back(snapshot);
            while feed.recent.len() > KEPT_FRAMES {
                feed.recent.pop_front();
            }
        }
        feed.events.push(event.to_string());
        // Unlock before waking the streams, which lock the feed at once.
        drop(feed);
        self.changed.notify_all();
    }
}

// ---------- HTTP ----------

/// An HTTP request, as far as the server reads it.
pub(crate) struct Request {
    pub method: String,
    /// The path without its query.
    pub path: String,
    /// What follows the `?`, undecoded.
    query: String,
    pub body: Vec<u8>,
}

impl Request {
    /// The value of a `name=value` pair of the query.
    fn query_param(&self, name: &str) -> Option<&str> {
        self.query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value)
    }

    /// A query parameter read as a whole number.
    fn query_number(&self, name: &str) -> Option<u64> {
        self.query_param(name).and_then(|v| v.parse().ok())
    }
}

/// An HTTP response: status line, content type and body.
pub(crate) struct Response {
    pub status: &'static str,
    /// The `Content-Type`.
    pub kind: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    /// A 200 with a body of the given type.
    pub fn ok(kind: &'static str, body: impl Into<Vec<u8>>) -> Self {
        Self {
            status: "200 OK",
            kind,
            body: body.into(),
        }
    }

    /// A 404 with a message in plain text.
    pub fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: "404 Not Found",
            kind: "text/plain; charset=utf-8",
            body: message.into().into_bytes(),
        }
    }

    /// A 400 with a message in plain text.
    pub fn bad(message: impl Into<String>) -> Self {
        Self {
            status: "400 Bad Request",
            kind: "text/plain; charset=utf-8",
            body: message.into().into_bytes(),
        }
    }
}

/// Requests the server answers besides its own, as the phone's upload; `None`
/// passes the request on to the server's routes.
pub(crate) type Routes = Box<dyn Fn(&Request) -> Option<Response> + Send + Sync>;

/// Starts the page's server in the background, over TLS when given a
/// configuration.
pub(crate) fn listen(
    address: impl std::net::ToSocketAddrs + std::fmt::Debug,
    target: Target,
    tls: Option<Arc<rustls::ServerConfig>>,
    routes: Option<Routes>,
) -> Result<Arc<Shared>, String> {
    let listener =
        TcpListener::bind(&address).map_err(|e| format!("could not listen on {address:?}: {e}"))?;
    let shared = Arc::new(Shared {
        feed: Mutex::new(Feed::default()),
        changed: Condvar::new(),
        target,
        extra_routes: routes,
    });
    let server = Arc::clone(&shared);
    // A thread per connection: an event stream holds its connection open for
    // as long as the page is, and a phone's upload blocks until the loop
    // takes the frame, so neither may hold up the others.
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let shared = Arc::clone(&server);
            let tls = tls.clone();
            std::thread::spawn(move || {
                // Events and small replies go out at once rather than batched.
                let _ = stream.set_nodelay(true);
                match tls {
                    Some(config) => {
                        if let Ok(connection) = rustls::ServerConnection::new(config) {
                            serve(rustls::StreamOwned::new(connection, stream), &shared);
                        }
                    }
                    None => serve(stream, &shared),
                }
            });
        }
    });
    Ok(shared)
}

/// Answers one request on a connection, then closes it; or, for `/events`,
/// streams events until the page goes.
fn serve<S: Read + Write>(stream: S, shared: &Shared) {
    let mut reader = BufReader::new(stream);
    let Some(request) = read_request(&mut reader) else {
        return;
    };
    let stream = reader.get_mut();
    if request.path == "/events" {
        let _ = stream_events(stream, shared);
        return;
    }
    let response = shared
        .extra_routes
        .as_ref()
        .and_then(|routes| routes(&request))
        .unwrap_or_else(|| route(&request, shared));
    let _ = write_response(stream, &response);
}

/// Reads the request line, the headers (only `Content-Length` matters) and
/// the body. `None` when the connection breaks off or the body is too large.
fn read_request(reader: &mut impl BufRead) -> Option<Request> {
    let mut request_line = String::new();
    reader.read_line(&mut request_line).ok()?;
    let mut words = request_line.split_whitespace();
    let method = words.next().unwrap_or("GET").to_string();
    let target = words.next().unwrap_or("/");
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let (path, query) = (path.to_string(), query.to_string());

    let mut length = 0;
    loop {
        let mut header = String::new();
        match reader.read_line(&mut header) {
            Ok(0) | Err(_) => return None,
            // A blank line ends the headers.
            Ok(_) if header == "\r\n" || header == "\n" => break,
            Ok(_) => {
                if let Some((name, value)) = header.split_once(':')
                    && name.trim().eq_ignore_ascii_case("content-length")
                {
                    length = value.trim().parse().unwrap_or(0);
                }
            }
        }
    }
    if length > MAX_BODY {
        return None;
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    Some(Request {
        method,
        path,
        query,
        body,
    })
}

/// Writes a response and its body. Nothing is cached: every frame is new.
fn write_response(stream: &mut impl Write, response: &Response) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",
        response.status,
        response.kind,
        response.body.len()
    )?;
    stream.write_all(&response.body)?;
    stream.flush()
}

/// The server's own routes. Most take `frame=<index>`, the newest frame
/// without it.
fn route(request: &Request, shared: &Shared) -> Response {
    let snapshot = || {
        shared
            .feed
            .lock()
            .unwrap()
            .find(request.query_number("frame"))
    };
    let gone = || Response::not_found("frame no longer kept");
    match request.path.as_str() {
        "/" => Response::ok("text/html; charset=utf-8", PAGE),
        "/frame.jpg" | "/image.jpg" => match snapshot() {
            Some(snapshot) => {
                let kind = request.query_param("kind").unwrap_or("frame");
                match render(&snapshot, kind, &shared.target) {
                    Ok(jpeg) => Response::ok("image/jpeg", jpeg),
                    Err(e) => Response::not_found(e),
                }
            }
            None => gone(),
        },
        "/frame.png" => {
            let png = snapshot().map(|s| encode_png(&gray_bytes(&s.pixels), s.width, s.height));
            match png {
                Some(Ok(png)) => Response::ok("image/png", png),
                _ => gone(),
            }
        }
        "/debug.json" => match snapshot() {
            Some(snapshot) => Response::ok("application/json", snapshot.debug_json.as_bytes()),
            None => gone(),
        },
        "/probe" => match (
            snapshot(),
            request.query_number("x"),
            request.query_number("y"),
        ) {
            (Some(snapshot), Some(x), Some(y)) => Response::ok(
                "application/json",
                probe(&snapshot, x as usize, y as usize).to_string(),
            ),
            _ => gone(),
        },
        // Pins the frame given, or unpins without one.
        "/pin" => {
            let mut feed = shared.feed.lock().unwrap();
            let pinned = request
                .query_number("frame")
                .and_then(|index| feed.find(Some(index)));
            feed.pinned = pinned;
            Response::ok("text/plain", "ok")
        }
        _ => Response::not_found("not found"),
    }
}

/// Every event so far, then each new one as it comes, as server-sent events.
/// Returns once the page has gone and a write fails.
fn stream_events(stream: &mut impl Write, shared: &Shared) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: keep-alive\r\n\r\n"
    )?;
    stream.flush()?;
    let mut sent = 0;
    loop {
        // Copy the new events out so the feed is not locked while writing.
        let batch: Vec<String> = {
            let feed = shared.feed.lock().unwrap();
            let (feed, _) = shared
                .changed
                .wait_timeout_while(feed, KEEP_ALIVE, |feed| feed.events.len() == sent)
                .unwrap();
            feed.events[sent..].to_vec()
        };
        if batch.is_empty() {
            // A comment, ignored by the page, to find out whether it is still there.
            stream.write_all(b": still here\n\n")?;
        }
        for event in batch {
            write!(stream, "data: {event}\n\n")?;
            sent += 1;
        }
        stream.flush()?;
    }
}

// ---------- measuring ----------

/// Measures frames one after another and publishes each to the page.
pub(crate) struct Tracker {
    shared: Arc<Shared>,
    target: Target,
    csv: Option<std::fs::File>,
    start: Instant,
    /// Index of the last frame measured, to count the frames skipped.
    last_index: Option<u64>,
    /// The last view whose code was read: its carriers lead the next frame.
    previous_view: Option<View>,
    /// The newest spectrum, kept for frames that skip the search.
    latest_spectrum: Option<Arc<Spectrum>>,
    /// The search the spectrum came from, for its peaks.
    latest_search: Value,
}

impl Tracker {
    /// A tracker publishing to `shared`, writing poses to `csv` if given.
    pub fn new(
        shared: Arc<Shared>,
        target: Target,
        csv: Option<&std::path::Path>,
    ) -> Result<Self, String> {
        let csv = match csv {
            Some(path) => {
                let mut file = std::fs::File::create(path)
                    .map_err(|e| format!("could not create {}: {e}", path.display()))?;
                writeln!(file, "t,x,y,z,alpha,beta,gamma,rms,points").map_err(|e| e.to_string())?;
                Some(file)
            }
            None => None,
        };
        Ok(Self {
            shared,
            target,
            csv,
            start: Instant::now(),
            last_index: None,
            previous_view: None,
            latest_spectrum: None,
            latest_search: Value::Null,
        })
    }

    /// Seconds since the tracker started.
    pub fn time(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Measures one frame, solves its pose given a camera, and publishes it.
    /// `annotate` sees the event, the debug data and the view, if the frame
    /// measured, before they go out.
    pub fn step(
        &mut self,
        camera: Option<&Camera>,
        frame: Frame,
        annotate: impl FnOnce(&mut Value, &mut Value, Option<&View>),
    ) -> Result<(), String> {
        // Frames the camera gave between this one and the last measured.
        let skipped = self
            .last_index
            .map_or(frame.index, |last| frame.index.saturating_sub(last + 1));
        self.last_index = Some(frame.index);
        let t = self.time();

        let began = Instant::now();
        let mut processed = process(
            camera,
            &self.target,
            &frame.data,
            frame.width,
            frame.height,
            self.previous_view.as_ref(),
        );
        let view = processed.view.take();
        if let Some(magnitude) = processed.spectrum.take() {
            self.latest_spectrum = Some(Arc::new(Spectrum {
                frame: frame.index,
                magnitude,
            }));
            self.latest_search = processed.debug["search"].clone();
        }
        let ms = began.elapsed().as_secs_f64() * 1000.0;

        let mut event = processed.event;
        let mut debug = processed.debug;
        match &processed.pose {
            Some(pose) => self.record_pose(t, pose)?,
            None => {
                if let Some(reason) = event["reason"].as_str() {
                    eprintln!("{t:8.3} s  frame {}: {reason}", frame.index);
                }
            }
        }
        for value in [&mut event, &mut debug] {
            value["t"] = json!(t);
            value["frame"] = json!(frame.index);
            value["skipped"] = json!(skipped);
            value["ms"] = json!(ms);
            value["width"] = json!(frame.width);
            value["height"] = json!(frame.height);
        }
        annotate(&mut event, &mut debug, view.as_ref());
        // Only a view whose code was read places the next frame's carriers.
        self.previous_view = view.filter(View::is_absolute);

        let encoding = Instant::now();
        let jpeg = encode_jpeg(&gray_bytes(&frame.data), frame.width, frame.height, false)?;
        debug["timings"]["jpeg"] = json!(encoding.elapsed().as_secs_f64() * 1000.0);
        event["jpeg_ms"] = debug["timings"]["jpeg"].clone();
        debug["spectrum_frame"] = json!(self.latest_spectrum.as_ref().map(|s| s.frame));
        debug["spectrum_search"] = self.latest_search.clone();

        let snapshot = Arc::new(Snapshot {
            index: frame.index,
            width: frame.width,
            height: frame.height,
            pixels: frame.data,
            jpeg,
            debug_json: debug.to_string(),
            spectrum: self.latest_spectrum.clone(),
            attempt: processed.attempt,
            field: OnceLock::new(),
        });
        self.shared.publish(&event, Some(snapshot));
        Ok(())
    }

    /// Prints a pose to the terminal and writes it to the CSV file.
    fn record_pose(&mut self, t: f64, pose: &PoseSample) -> Result<(), String> {
        let ([x, y, z], [alpha, beta, gamma]) = (pose.xyz, pose.angles);
        let rms = pose.fit.rms;
        println!(
            "{t:8.3} s  x {x:9.3}  y {y:9.3}  z {z:9.3}  α {alpha:8.3}°  β {beta:8.3}°  γ {gamma:8.3}°  rms {rms:.3} px"
        );
        if let Some(file) = self.csv.as_mut() {
            writeln!(
                file,
                "{t:.4},{x:.5},{y:.5},{z:.5},{alpha:.6},{beta:.6},{gamma:.6},{rms:.4},{}",
                pose.fit.used
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// Sends the page an event that comes with no frame.
    pub fn announce(&self, mut event: Value) {
        event["t"] = json!(self.time());
        self.shared.publish(&event, None);
    }
}

/// A solved pose, as printed and written to the CSV file.
struct PoseSample {
    /// Board origin in the camera frame, in millimetres.
    xyz: [f64; 3],
    /// `α, β, γ` in degrees: board to camera as `rotz(α)·roty(β)·rotx(γ)`, the
    /// rotation part of the C++ library's pose.
    angles: [f64; 3],
    fit: PnpSolution,
}

/// One frame measured and solved: the event every page gets, the debug data
/// of this frame, and what to keep for its images.
struct Processed {
    event: Value,
    debug: Value,
    pose: Option<PoseSample>,
    attempt: Option<Attempt>,
    spectrum: Option<Vec<f32>>,
    /// The view, when the frame measured.
    view: Option<View>,
}

impl Processed {
    /// Marks the frame as giving no pose, for `reason`.
    fn failed(mut self, reason: String, view: Option<View>) -> Self {
        self.event["reason"] = json!(reason);
        self.debug["reason"] = json!(reason);
        self.debug["ok"] = json!(false);
        self.view = view;
        self
    }

    /// Carrier quality and window offset per point of the view.
    fn point_quality(&self) -> &[(Real, Real)] {
        self.attempt
            .as_ref()
            .map(|a| a.point_quality.as_slice())
            .unwrap_or_default()
    }
}

/// Measures a frame and, given a camera, solves the board's pose. Without a
/// camera the frame is only measured.
fn process(
    camera: Option<&Camera>,
    target: &Target,
    data: &[f32],
    width: usize,
    height: usize,
    previous: Option<&View>,
) -> Processed {
    let clock = Instant::now();
    let (measured, mut trace) = measure_view_traced(data, width, height, target, previous, true);
    let measure_ms = clock.elapsed().as_secs_f64() * 1000.0;
    let spectrum = trace.spectrum.take();
    let mut debug = json!({
        "camera": camera.map_or(Value::Null, camera_json),
        "target": { "square": target.square, "order": target.order, "period": target.period_squares() },
        "search": search_json(&trace),
        "attempts": trace.attempts.iter().map(attempt_json).collect::<Vec<_>>(),
        "chosen": trace.chosen,
        "timings": { "measure": measure_ms },
    });
    let mut event = json!({
        "ok": false,
        "searched": trace.searched,
        "attempts": trace.attempts.len(),
        "measure_ms": measure_ms,
    });
    // The attempt the view came from or, when none gave one, the last tried.
    let shown_attempt = trace.chosen.or(trace.attempts.len().checked_sub(1));
    let attempt = shown_attempt.map(|i| trace.attempts.swap_remove(i));
    if let Some(attempt) = &attempt {
        describe_attempt(attempt, &mut event, &mut debug);
    }
    let mut result = Processed {
        event,
        debug,
        pose: None,
        attempt,
        spectrum,
        view: None,
    };

    let view = match measured {
        Ok(view) => view,
        Err(e) => return result.failed(e.to_string(), None),
    };
    if let Err(e) = &view.code {
        result.debug["points"] = points_json(&view, result.point_quality(), None);
        let reason = format!("code not read ({e})");
        return result.failed(reason, Some(view));
    }
    let Some(camera) = camera else {
        result.debug["points"] = points_json(&view, result.point_quality(), None);
        result.debug["ok"] = json!(false);
        result.view = Some(view);
        return result;
    };

    let clock = Instant::now();
    let fit = match solve_pnp(camera, &view) {
        Ok(fit) => fit,
        Err(e) => {
            result.debug["points"] = points_json(&view, result.point_quality(), None);
            return result.failed(e.to_string(), Some(view));
        }
    };
    let pnp_ms = clock.elapsed().as_secs_f64() * 1000.0;
    result.debug["timings"]["pnp"] = json!(pnp_ms);
    result.event["pnp_ms"] = json!(pnp_ms);

    let rotation = fit.pose.rotation.matrix();
    let angles = euler_angles_degrees(rotation);
    let axes = projected_axes(camera, target, &fit);
    let translation = fit.pose.translation;
    let xyz = [translation.x, translation.y, translation.z];
    let residuals = ResidualStats::of(&fit);

    for value in [&mut result.event, &mut result.debug] {
        value["ok"] = json!(true);
        value["x"] = json!(xyz[0]);
        value["y"] = json!(xyz[1]);
        value["z"] = json!(xyz[2]);
        value["alpha"] = json!(angles[0]);
        value["beta"] = json!(angles[1]);
        value["gamma"] = json!(angles[2]);
        value["rms"] = json!(fit.rms);
        value["points"] = json!(fit.used);
        value["rejected"] = json!(fit.rejected);
        value["median"] = json!(residuals.median);
        value["p95"] = json!(residuals.p95);
        value["max"] = json!(residuals.max);
        value["bias"] = json!(residuals.bias);
        value["axes"] = json!(axes);
        value["distance"] = json!(translation.norm());
    }
    // `points` is the count in the event and the list in the debug data.
    result.debug["points"] = points_json(&view, result.point_quality(), Some(&fit));
    result.debug["grid"] = json!(board_grid(camera, target, &view, &fit));
    result.debug["pose"] = json!({
        "rotation": (0..3).map(|i| (0..3).map(|j| rotation[(i, j)]).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "translation": xyz,
    });
    result.pose = Some(PoseSample { xyz, angles, fit });
    result.view = Some(view);
    result
}

/// Adds to the event and the debug data what the attempt shown went through:
/// where its carriers came from, the coarse walk and the fine points lost.
fn describe_attempt(attempt: &Attempt, event: &mut Value, debug: &mut Value) {
    event["from_previous"] = json!(attempt.from_previous);
    event["period"] = json!(attempt.period);
    event["coarse"] = json!(attempt.coarse.len());
    event["funnel"] = json!(attempt.funnel.iter().map(|f| f.1).collect::<Vec<_>>());
    if let Some(Ok(code)) = &attempt.code {
        let (x, y) = code.bit_errors();
        event["bit_errors"] = json!([x, y]);
        if let Code::Checkerboard(code) = code {
            event["false_accept"] = json!(code.false_accept);
        }
    }
    debug["coarse"] = json!(
        attempt
            .coarse
            .iter()
            .map(|(pixel, quality)| [pixel[0], pixel[1], *quality])
            .collect::<Vec<_>>()
    );
    debug["refused"] = json!(attempt.coarse_refused);
    debug["dropped"] = json!(
        attempt
            .dropped
            .iter()
            .map(|(pixel, stage)| json!([pixel[0], pixel[1], stage]))
            .collect::<Vec<_>>()
    );
    debug["seed"] = json!(
        attempt
            .seed
            .map(|(pixel, quality)| [pixel[0], pixel[1], quality])
    );
    debug["carriers"] = json!(attempt.carriers);
}

/// `α, β, γ` in degrees such that `rotation = rotz(α)·roty(β)·rotx(γ)`.
fn euler_angles_degrees(rotation: &Matrix3<Real>) -> [Real; 3] {
    let r = rotation;
    [
        r[(1, 0)].atan2(r[(0, 0)]).to_degrees(),
        (-r[(2, 0)]).clamp(-1.0, 1.0).asin().to_degrees(),
        r[(2, 1)].atan2(r[(2, 2)]).to_degrees(),
    ]
}

/// The board's origin and the ends of its axes, ten squares long with z
/// towards the camera, in pixels; empty unless all four project.
fn projected_axes(camera: &Camera, target: &Target, fit: &PnpSolution) -> Vec<[Real; 2]> {
    let project =
        |p: Vector3<Real>| camera.project(&(fit.pose.rotation * p + fit.pose.translation));
    let length = 10.0 * target.square;
    let ends = [
        Vector3::zeros(),
        Vector3::new(length, 0.0, 0.0),
        Vector3::new(0.0, length, 0.0),
        Vector3::new(0.0, 0.0, -length),
    ];
    let axes = ends.iter().filter_map(|&p| project(p)).collect::<Vec<_>>();
    if axes.len() == 4 { axes } else { Vec::new() }
}

/// How far the points the fit kept land from where the pose puts them.
struct ResidualStats {
    /// Lengths in pixels; `None` when no point was kept.
    median: Option<Real>,
    p95: Option<Real>,
    max: Option<Real>,
    /// The mean residual vector. Points pulled one way on average show a
    /// pose or calibration bias rather than noise.
    bias: [Real; 2],
}

impl ResidualStats {
    fn of(fit: &PnpSolution) -> Self {
        let kept_residuals = || {
            fit.residuals
                .iter()
                .zip(&fit.inliers)
                .filter(|&(_, &inlier)| inlier)
                .map(|(residual, _)| residual)
        };
        let mut lengths: Vec<Real> = kept_residuals().map(|r| r[0].hypot(r[1])).collect();
        lengths.sort_by(|a, b| a.total_cmp(b));
        let quantile = |q: Real| {
            lengths
                .get(((lengths.len() as Real - 1.0) * q).round() as usize)
                .copied()
        };
        let n = fit.used.max(1) as Real;
        let bias =
            kept_residuals().fold([0.0, 0.0], |sum, r| [sum[0] + r[0] / n, sum[1] + r[1] / n]);
        Self {
            median: quantile(0.5),
            p95: quantile(0.95),
            max: lengths.last().copied(),
            bias,
        }
    }
}

// ---------- debug data ----------

fn camera_json(camera: &Camera) -> Value {
    json!({
        "model": camera.model.name(),
        "width": camera.width, "height": camera.height,
        "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy,
        "distortion": camera.distortion,
    })
}

/// Whether the spectrum was searched this frame, how long it took and the
/// peaks found.
fn search_json(trace: &Trace) -> Value {
    json!({
        "searched": trace.searched,
        "ms": trace.search_ms,
        "peaks": trace.peaks.iter().map(|p| json!({
            "k": p.k, "score": p.score, "lifted": p.lifted,
        })).collect::<Vec<_>>(),
    })
}

/// A carrier pair followed: its carriers, how far the walk and the fine grid
/// got, the code read, and the time per stage.
fn attempt_json(attempt: &Attempt) -> Value {
    let carrier = |k: [Real; 2]| {
        json!({
            "k": k,
            "period": std::f64::consts::TAU / k[0].hypot(k[1]),
            "angle": k[1].atan2(k[0]).to_degrees(),
        })
    };
    let code = match &attempt.code {
        None => Value::Null,
        Some(Ok(Code::Megarena(c))) => json!({
            "ok": true,
            "transform": c.transform,
            "delta": [c.delta.0, c.delta.1],
            "bits": [c.bits.0, c.bits.1],
            "bit_errors": [c.bit_errors.0, c.bit_errors.1],
            "agreement": c.agreement,
        }),
        Some(Ok(Code::Checkerboard(c))) => json!({
            "ok": true,
            "transform": c.transform,
            "delta": [c.delta.0, c.delta.1],
            "centre_square": [c.centre_square.0, c.centre_square.1],
            "k": [c.k_x, c.k_y],
            "x_window": c.x_window,
            "y_window": c.y_window,
            "check_bits": c.check_bits,
            "bit_errors": [c.bit_errors.0, c.bit_errors.1],
            "false_accept": c.false_accept,
            "runner_up_false_accept": c.runner_up_false_accept,
        }),
        Some(Err(e)) => json!({ "ok": false, "error": e.to_string() }),
    };
    json!({
        "carriers": [carrier(attempt.carriers[0]), carrier(attempt.carriers[1])],
        "from_previous": attempt.from_previous,
        "period": attempt.period,
        "seed": attempt.seed.map(|(pixel, quality)| [pixel[0], pixel[1], quality]),
        "coarse_step": attempt.coarse_step,
        "coarse": attempt.coarse.len(),
        "refused": attempt.coarse_refused.len(),
        "fine_step": attempt.fine_step,
        "defocus": attempt.defocus,
        "funnel": named_values_json(&attempt.funnel),
        "timings": named_values_json(&attempt.timings),
        "code": code,
        "error": attempt.error.map(|e| e.to_string()),
    })
}

/// `[name, value]` pairs as a list of two-element arrays.
fn named_values_json<T: serde::Serialize>(pairs: &[(&str, T)]) -> Vec<Value> {
    pairs
        .iter()
        .map(|(name, value)| json!([name, value]))
        .collect()
}

/// Per point: pixel, board, carrier quality, window offset, and with a fit
/// the residual and whether it was kept, as flat arrays to keep it small.
fn points_json(view: &View, quality: &[(Real, Real)], fit: Option<&PnpSolution>) -> Value {
    let round = |v: Real, scale: Real| (v * scale).round() / scale;
    json!(
        view.points
            .iter()
            .enumerate()
            .map(|(i, point)| {
                let (carrier_quality, window_offset) =
                    quality.get(i).copied().unwrap_or((Real::NAN, Real::NAN));
                let (residual, inlier) =
                    fit.map_or(([Real::NAN; 2], true), |f| (f.residuals[i], f.inliers[i]));
                json!([
                    point.pixel[0],
                    point.pixel[1],
                    round(point.board[0], 1e4),
                    round(point.board[1], 1e4),
                    round(carrier_quality, 1e3),
                    round(window_offset, 1e3),
                    round(residual[0], 1e4),
                    round(residual[1], 1e4),
                    inlier
                ])
            })
            .collect::<Vec<_>>()
    )
}

/// Lines between the squares the view covers, projected with the fitted pose:
/// where they part from the image's own edges, the pose or the calibration is
/// off. Polylines in pixels, broken where a point does not project.
fn board_grid(
    camera: &Camera,
    target: &Target,
    view: &View,
    fit: &PnpSolution,
) -> Vec<Vec<[Real; 2]>> {
    // A board point in squares, shifted half a square to match the `+ 0.5`
    // the lines are drawn at below.
    let in_squares = |board: [Real; 2]| {
        let (i, j) = target.layout.to_lattice(board[0], board[1]);
        (i / target.square - 0.5, j / target.square - 0.5)
    };
    let (mut first, mut last) = ([Real::INFINITY; 2], [Real::NEG_INFINITY; 2]);
    for point in &view.points {
        let (i, j) = in_squares(point.board);
        first = [first[0].min(i), first[1].min(j)];
        last = [last[0].max(i), last[1].max(j)];
    }
    if !(first[0].is_finite() && first[1].is_finite()) {
        return Vec::new();
    }
    // One square of margin around the points.
    let first = [first[0].floor() - 1.0, first[1].floor() - 1.0];
    let last = [last[0].ceil() + 1.0, last[1].ceil() + 1.0];
    // At most about 60 lines each way, so the overlay stays light.
    let span = (last[0] - first[0]).max(last[1] - first[1]);
    let spacing = (span / 60.0).ceil().max(1.0);

    let project = |i: Real, j: Real| {
        let board = target.board_point(i, j);
        let on_board = Vector3::new(board[0], board[1], 0.0);
        camera.project(&(fit.pose.rotation * on_board + fit.pose.translation))
    };
    let mut lines = Vec::new();
    // Splits a run of projected points into polylines at the points that do
    // not project, rounded to a tenth of a pixel to keep the JSON small.
    let mut add_polylines = |points: &mut dyn Iterator<Item = Option<[Real; 2]>>| {
        let mut line = Vec::new();
        for point in points {
            match point {
                Some(p) => line.push([(p[0] * 10.0).round() / 10.0, (p[1] * 10.0).round() / 10.0]),
                None if line.len() > 1 => lines.push(std::mem::take(&mut line)),
                None => line.clear(),
            }
        }
        if line.len() > 1 {
            lines.push(line);
        }
    };
    // Every half square from `a` past `b`, close enough for a distorted line
    // to bend.
    let half_squares = |a: Real, b: Real| {
        let n = ((b - a) * 2.0).ceil() as usize;
        (0..=n).map(move |s| a + s as Real * 0.5)
    };
    let mut i = first[0];
    while i <= last[0] {
        add_polylines(
            &mut half_squares(first[1] - 0.5, last[1] + 0.5).map(|j| project(i + 0.5, j)),
        );
        i += spacing;
    }
    let mut j = first[1];
    while j <= last[1] {
        add_polylines(
            &mut half_squares(first[0] - 0.5, last[0] + 0.5).map(|i| project(i, j + 0.5)),
        );
        j += spacing;
    }
    lines
}

/// Everything known at one pixel of a frame.
fn probe(snapshot: &Snapshot, x: usize, y: usize) -> Value {
    let (width, height) = (snapshot.width, snapshot.height);
    if x >= width || y >= height {
        return json!({ "error": "outside the frame" });
    }
    let i = y * width + x;
    let attempt = snapshot.attempt.as_ref();
    let phase = attempt.and_then(|a| a.phase.as_ref()).map(|p| p[i]);
    // The spectrum image has zero frequency at the centre.
    let frequency = |at: usize, size: usize| {
        (at as f64 - (size / 2) as f64) * std::f64::consts::TAU / size as f64
    };
    json!({
        "x": x, "y": y,
        "intensity": snapshot.pixels[i],
        "restored": attempt.and_then(|a| a.restored.as_ref()).map(|r| r[i]),
        "phase": phase.filter(|p| p[0].is_finite()),
        "square": attempt.and_then(|a| a.pattern_square(x, y, width)).map(|(i, j)| [i, j]),
        "spectrum": snapshot.spectrum.as_ref().map(|s| s.magnitude[i]),
        // Only once a field view has worked it out: it is too slow to compute
        // for a hover.
        "field": snapshot.field.get().and_then(|f| f.as_ref()).map(|f| json!({
            "phase": [f.phase[0][i], f.phase[1][i]],
            "amplitude": [f.amplitude[0][i], f.amplitude[1][i]],
        })),
        "frequency": [frequency(x, width), frequency(y, height)],
    })
}

// ---------- images ----------

/// Intensities from 0 to 1 as 8-bit grey levels.
fn gray_bytes(data: &[f32]) -> Vec<u8> {
    data.iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Grey or, with `rgb`, colour bytes as a JPEG.
fn encode_jpeg(bytes: &[u8], width: usize, height: usize, rgb: bool) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let kind = if rgb {
        image::ExtendedColorType::Rgb8
    } else {
        image::ExtendedColorType::L8
    };
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, 85)
        .encode(bytes, width as u32, height as u32, kind)
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// Grey bytes as a PNG.
fn encode_png(bytes: &[u8], width: usize, height: usize) -> Result<Vec<u8>, String> {
    use image::ImageEncoder;
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(
            bytes,
            width as u32,
            height as u32,
            image::ExtendedColorType::L8,
        )
        .map_err(|e| e.to_string())?;
    Ok(out)
}

/// The colour of hue `h`, in turns, at full saturation and value.
fn hue(h: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0) * 6.0;
    let channel = |n: f32| {
        let k = (n + h) % 6.0;
        1.0 - (k.min(4.0 - k).clamp(0.0, 1.0))
    };
    [channel(5.0), channel(3.0), channel(1.0)]
}

/// RGB bytes of an image coloured pixel by pixel; where `colour` gives
/// nothing, the frame shows through, dimmed.
fn colour_image(snapshot: &Snapshot, colour: impl Fn(usize) -> Option<[f32; 3]>) -> Vec<u8> {
    let dimmed = |i: usize| 0.3 * snapshot.pixels[i].clamp(0.0, 1.0);
    (0..snapshot.width * snapshot.height)
        .flat_map(|i| {
            let rgb = colour(i).unwrap_or([dimmed(i); 3]);
            rgb.map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        })
        .collect()
}

/// The error for a debug image this frame has no data for.
fn missing(what: &str) -> Result<Vec<u8>, String> {
    Err(format!("no {what} for this frame"))
}

/// One of the debug images of a frame, as a JPEG; `kind` is the page's image
/// mode.
fn render(snapshot: &Snapshot, kind: &str, target: &Target) -> Result<Vec<u8>, String> {
    let (width, height) = (snapshot.width, snapshot.height);
    let attempt = snapshot.attempt.as_ref();
    let colour_jpeg = |bytes: Vec<u8>| encode_jpeg(&bytes, width, height, true);
    match kind {
        "frame" => Ok(snapshot.jpeg.clone()),
        "restored" => match attempt.and_then(|a| a.restored.as_ref()) {
            Some(restored) => encode_jpeg(&gray_bytes(restored), width, height, false),
            None => missing("restored frame (the code was not read)"),
        },
        "spectrum" => match &snapshot.spectrum {
            Some(spectrum) => {
                encode_jpeg(&spectrum_bytes(&spectrum.magnitude), width, height, false)
            }
            None => missing("spectrum"),
        },
        "phase1" | "phase2" => match attempt.and_then(|a| a.phase.as_ref()) {
            Some(phase) => {
                let carrier = usize::from(kind == "phase2");
                colour_jpeg(phase_colours(snapshot, phase, carrier))
            }
            None => missing("phase map"),
        },
        "squares" => match attempt.filter(|a| a.phase.is_some()) {
            Some(attempt) => colour_jpeg(square_colours(snapshot, attempt, target)),
            None => missing("phase map"),
        },
        "field1" | "field2" | "slow1" | "slow2" => match snapshot.field() {
            Some(field) => {
                let carrier = usize::from(kind.ends_with('2'));
                let slow = kind.starts_with("slow");
                colour_jpeg(field_colours(snapshot, field, carrier, slow))
            }
            None => missing("demodulated field (no carrier was followed)"),
        },
        "lines" => match (snapshot.field(), attempt) {
            (Some(field), Some(attempt)) => {
                colour_jpeg(square_edge_colours(snapshot, field, attempt))
            }
            _ => missing("demodulated field (no carrier was followed)"),
        },
        _ => Err(format!("unknown image {kind}")),
    }
}

/// The spectrum's log magnitude as grey levels: black up to the median, so
/// the broad background drops out, white at the strongest peak, and a gamma
/// that brings out the weaker peaks.
fn spectrum_bytes(magnitude: &[f32]) -> Vec<u8> {
    // The median of every seventh value is close enough, and much cheaper.
    let mut sample: Vec<f32> = magnitude.iter().step_by(7).copied().collect();
    sample.sort_by(|a, b| a.total_cmp(b));
    let black = sample[sample.len() / 2];
    let white = magnitude.iter().copied().fold(black, f32::max);
    magnitude
        .iter()
        .map(|&v| {
            (((v - black) / (white - black).max(1e-6))
                .clamp(0.0, 1.0)
                .powf(0.7)
                * 255.0) as u8
        })
        .collect()
}

/// One carrier's phase as hue, a turn of the colour wheel per period.
fn phase_colours(snapshot: &Snapshot, phase: &[[f32; 2]], carrier: usize) -> Vec<u8> {
    colour_image(snapshot, |i| {
        let p = phase[i][carrier];
        p.is_finite().then(|| hue(p / std::f32::consts::TAU))
    })
}

/// The pattern square under each pixel: hue along i, brightness along j and
/// a darker shade on alternate squares, so a slip in either shows.
fn square_colours(snapshot: &Snapshot, attempt: &Attempt, target: &Target) -> Vec<u8> {
    let width = snapshot.width;
    let period = target.period_squares() as Real;
    colour_image(snapshot, |i| {
        let (si, sj) = attempt.pattern_square(i % width, i / width, width)?;
        let (si, sj) = (si.round().rem_euclid(period), sj.round().rem_euclid(period));
        let shade = 0.45 + 0.55 * ((sj as i64 % 4) as f32 / 3.0);
        let parity = if (si as i64 + sj as i64) % 2 == 0 {
            1.0
        } else {
            0.8
        };
        Some(hue((si % 12.0) as f32 / 12.0).map(|v| v * shade * parity))
    })
}

/// One carrier of the demodulated field: phase as hue, amplitude as
/// brightness. `slow` takes off the carrier's global frequency, the phase a
/// square-on view would have, leaving the geometry of the view.
fn field_colours(snapshot: &Snapshot, field: &Field, carrier: usize, slow: bool) -> Vec<u8> {
    let width = snapshot.width;
    let k = field.carriers[carrier];
    colour_image(snapshot, |i| {
        let mut phase = field.phase[carrier][i] as Real;
        if slow {
            phase -= k[0] * (i % width) as Real + k[1] * (i / width) as Real;
        }
        let brightness = (field.amplitude[carrier][i] / FULL_AMPLITUDE).clamp(0.0, 1.0);
        Some(hue((phase / std::f64::consts::TAU) as f32).map(|v| v * brightness))
    })
}

/// Lines of constant square coordinate from the field's phases, placed by the
/// code, over the frame: they should fall on the square edges.
fn square_edge_colours(snapshot: &Snapshot, field: &Field, attempt: &Attempt) -> Vec<u8> {
    let (width, height) = (snapshot.width, snapshot.height);
    let phase_maps = attempt
        .phase
        .as_ref()
        .expect("a field comes from phase maps");
    // The nearest square at each pixel where the board was measured and the
    // carrier is there.
    let squares: Vec<Option<(Real, Real)>> = (0..width * height)
        .map(|i| {
            let strong = field.amplitude[0][i].min(field.amplitude[1][i]) > 0.1;
            (strong && phase_maps[i][0].is_finite()).then(|| {
                let phase = [field.phase[0][i] as Real, field.phase[1][i] as Real];
                let square = attempt.to_pattern(phase);
                (square.0.round(), square.1.round())
            })
        })
        .collect();
    // Whether the nearest square's coordinate along `axis` changes to the
    // right or below, that is whether the pixel is on a square's edge.
    let on_edge = |i: usize, axis: usize| {
        let here = squares[i]?;
        let coordinate = |s: (Real, Real)| if axis == 0 { s.0 } else { s.1 };
        let neighbours = [
            (i % width + 1 < width).then(|| i + 1),
            (i / width + 1 < height).then(|| i + width),
        ];
        Some(
            neighbours
                .into_iter()
                .flatten()
                .filter_map(|j| squares[j])
                .any(|s| coordinate(s) != coordinate(here)),
        )
    };
    colour_image(snapshot, |i| {
        let grey = 0.55 * snapshot.pixels[i].clamp(0.0, 1.0) + 0.2;
        match (on_edge(i, 0), on_edge(i, 1)) {
            (Some(true), _) => Some(LINE_I),
            (_, Some(true)) => Some(LINE_J),
            _ => Some([grey; 3]),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use vernier_camera::{Model, RigidPose, Scene};

    fn camera() -> Camera {
        let mut camera = Camera::ideal(Model::Pinhole, 480, 360, 450.0, 452.0, 243.0, 177.0);
        camera.distortion = vec![-0.22, 0.07, 0.0006, -0.0004, 0.0];
        camera
    }

    /// A board swaying in front of the camera, `t` in seconds.
    fn frame(camera: &Camera, target: &Target, index: u64, t: f64) -> Frame {
        let pose = RigidPose::from_vectors(
            Vector3::new(0.3 * (0.7 * t).sin(), 0.25 * (0.5 * t).cos(), 0.3 + 0.2 * t),
            Vector3::new(10.0 * (0.3 * t).sin(), 0.0, 200.0 + 20.0 * (0.4 * t).sin()),
        );
        let data = Scene {
            camera,
            target,
            half_size: [200.0, 160.0],
            background: 0.45,
            supersample: 1,
        }
        .render(&pose);
        Frame {
            index,
            width: camera.width,
            height: camera.height,
            data,
        }
    }

    #[test]
    fn a_frame_leaves_every_debug_image_and_its_points() {
        let (camera, target) = (camera(), Target::new(5.0, 6));
        let f = frame(&camera, &target, 0, 0.0);
        let mut done = process(Some(&camera), &target, &f.data, f.width, f.height, None);
        assert_eq!(done.event["ok"], json!(true), "{}", done.event);
        let points = done.debug["points"].as_array().expect("points");
        assert!(points.len() > 100);
        assert!(done.debug["grid"].as_array().is_some_and(|g| !g.is_empty()));
        assert!(done.debug["search"]["searched"].as_bool().unwrap());
        assert!(done.spectrum.is_some());
        let snapshot = Snapshot {
            index: 0,
            width: f.width,
            height: f.height,
            jpeg: encode_jpeg(&gray_bytes(&f.data), f.width, f.height, false).unwrap(),
            pixels: f.data,
            debug_json: done.debug.to_string(),
            spectrum: done.spectrum.map(|magnitude| {
                Arc::new(Spectrum {
                    frame: 0,
                    magnitude,
                })
            }),
            attempt: done.attempt,
            field: OnceLock::new(),
        };
        let previous = done.view.take();
        for kind in [
            "frame", "spectrum", "phase1", "phase2", "squares", "restored", "field1", "field2",
            "slow1", "slow2", "lines",
        ] {
            render(&snapshot, kind, &target).unwrap_or_else(|e| panic!("{kind}: {e}"));
        }
        let probed = probe(&snapshot, 240, 180);
        assert!(probed["square"].is_array(), "{probed}");

        // The next frame reuses the carriers and skips the spectrum.
        let g = frame(&camera, &target, 1, 0.05);
        let done = process(
            Some(&camera),
            &target,
            &g.data,
            g.width,
            g.height,
            previous.as_ref(),
        );
        assert_eq!(done.event["from_previous"], json!(true), "{}", done.event);
        assert_eq!(done.event["searched"], json!(false));
    }

    /// Serves the page on port 8099 with a synthetic board, for looking at it
    /// in a browser: `cargo test -p vernier-cli --release -- --ignored demo`.
    #[test]
    #[ignore]
    fn demo() {
        let (camera, target) = (camera(), Target::new(5.0, 6));
        let shared = listen(("127.0.0.1", 8099), target, None, None).unwrap();
        eprintln!("open http://localhost:8099/");
        let mut tracker = Tracker::new(shared, target, None).unwrap();
        let start = Instant::now();
        let seconds: f64 = std::env::var("DEMO_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(120.0);
        for i in 0.. {
            let t = start.elapsed().as_secs_f64();
            if t > seconds {
                break;
            }
            let mut f = frame(&camera, &target, i, t);
            // Every ninth second the board leaves the frame.
            if (t as u64) % 9 == 8 {
                f.data.fill(0.45);
            }
            tracker.step(Some(&camera), f, |_, _, _| {}).unwrap();
        }
    }
}
