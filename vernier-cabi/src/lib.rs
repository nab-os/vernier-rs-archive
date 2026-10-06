//! C ABI for vernier-rs.
//!
//! Create a handle with `vernier_detector_new` (CPU) or
//! `vernier_detector_new_cuda` (GPU), call the detection functions, then
//! release the handle with `vernier_detector_free`. Camera calibration and
//! PnP from the coded checkerboard go through `vernier_measure_view`,
//! `vernier_calibrate` and `vernier_solve_pnp`. Each function is
//! thread-safe in the sense that separate handles may be used concurrently;
//! a single handle must not be used from multiple threads simultaneously.

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};

use vernier_camera as camera;
use vernier_patterns::checkerboard::{CodeLayout, CodePacking};

use vernier_core::buffer::BufferLayout;
use vernier_core::image::GrayImage;
use vernier_core::{Complex32, Real};
use vernier_cpu::CpuBackend;
use vernier_spectral::spectrum::{Detection, analyze_two};
use vernier_pose::{Calibration, absolute, periodic};

// ─── Thread-local error storage ──────────────────────────────────────────────

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = RefCell::new(None);
}

fn set_last_error(msg: impl std::fmt::Display) {
    let s = msg.to_string();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(s).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Returns the last error message on this thread, or NULL if the last call
/// succeeded.
///
/// The pointer is valid until the next vernier call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_last_error() -> *const c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

// ─── Backend enum ─────────────────────────────────────────────────────────────

enum BackendInner {
    Cpu(CpuBackend),
    #[cfg(feature = "cuda")]
    Cuda(vernier_cuda::CudaBackend),
}

impl BackendInner {
    fn analyze_two(
        &self,
        data: &[Complex32],
        layout: BufferLayout,
        sigma: Real,
        min_frequency: usize,
        max_frequency: usize,
        smoothing_sigma: Real,
    ) -> vernier_core::Result<Detection> {
        match self {
            BackendInner::Cpu(b) => {
                analyze_two(b, data, layout, sigma, min_frequency, max_frequency, smoothing_sigma)
            }
            #[cfg(feature = "cuda")]
            BackendInner::Cuda(b) => {
                analyze_two(b, data, layout, sigma, min_frequency, max_frequency, smoothing_sigma)
            }
        }
    }
}

// ─── Detector handle ─────────────────────────────────────────────────────────

/// Opaque handle to a Vernier detector. Create with `vernier_detector_new`
/// (CPU) or `vernier_detector_new_cuda` (GPU); free with
/// `vernier_detector_free`.
pub struct VernierDetector {
    backend: BackendInner,
}

/// Creates a CPU-backed detector.
///
/// Must be freed with `vernier_detector_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_new() -> *mut VernierDetector {
    Box::into_raw(Box::new(VernierDetector {
        backend: BackendInner::Cpu(CpuBackend::new()),
    }))
}

/// Creates a CUDA-backed detector. Returns NULL if no CUDA device is
/// available or if the library was not compiled with CUDA support (check
/// `vernier_last_error` for details).
///
/// Must be freed with `vernier_detector_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_new_cuda() -> *mut VernierDetector {
    #[cfg(feature = "cuda")]
    {
        match vernier_cuda::CudaBackend::new() {
            Ok(b) => Box::into_raw(Box::new(VernierDetector {
                backend: BackendInner::Cuda(b),
            })),
            Err(e) => {
                set_last_error(e);
                std::ptr::null_mut()
            }
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        set_last_error("CUDA support not compiled in (rebuild with --features cuda)");
        std::ptr::null_mut()
    }
}

/// Frees a detector. Passing NULL is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detector_free(det: *mut VernierDetector) {
    if !det.is_null() {
        unsafe { drop(Box::from_raw(det)) };
    }
}

// ─── Result type ─────────────────────────────────────────────────────────────

/// Pose returned by all detection functions.
///
/// Check `found` before reading `x`, `y`, `theta`. On failure (`found == 0`)
/// call `vernier_last_error()` for a description.
#[repr(C)]
pub struct VernierPose {
    pub x: f64,
    pub y: f64,
    pub theta: f64,
    /// 1 on success, 0 on failure.
    pub found: i32,
}

impl VernierPose {
    fn not_found() -> Self {
        Self { x: 0.0, y: 0.0, theta: 0.0, found: 0 }
    }

    fn found(pose: &vernier_core::Pose) -> Self {
        Self { x: pose.x, y: pose.y, theta: pose.theta, found: 1 }
    }
}

// ─── Periodic (relative / fine) detection ────────────────────────────────────

/// Periodic (relative) detection: recovers `x`, `y` modulo the pattern period
/// and the in-image orientation `theta`.
///
/// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
/// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
/// - `period`           — pattern spatial period in physical units.
/// - `sigma`            — bandpass filter half-width in frequency bins.
/// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
/// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
/// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
///                        search; 0 disables blurring.
///
/// Returns a pose with `found == 0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detect_periodic(
    det: *mut VernierDetector,
    pixels: *const f32,
    width: usize,
    height: usize,
    period: f64,
    sigma: f64,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: f64,
) -> VernierPose {
    guarded(VernierPose::not_found(), || {
        let det = unsafe { det.as_ref() }.ok_or("null detector pointer")?;
        let image = to_image(pixels, width, height)?;
        let detection = det
            .backend
            .analyze_two(
                &image.to_complex(),
                image.layout(),
                sigma as Real,
                min_frequency,
                max_frequency,
                smoothing_sigma as Real,
            )
            .map_err(|e| e.to_string())?;
        let calib = Calibration::new(period as Real, width, height);
        let pose = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, &calib);
        Ok(VernierPose::found(&pose))
    })
}

// ─── Megarena absolute detection ─────────────────────────────────────────────

/// Megarena absolute detection: recovers an unambiguous `(x, y, theta)` using
/// the LFSR binary code embedded in the pattern.
///
/// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
/// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
/// - `physical_period`  — pattern spatial period in micrometres (9 µm for the
///                        reference pattern).
/// - `code_size`        — LFSR order in bits (12 for the reference pattern).
/// - `sigma`            — bandpass filter half-width in frequency bins.
/// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
/// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
/// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
///                        search; 0 disables blurring.
///
/// Returns a pose with `found == 0` on failure.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_detect_megarena(
    det: *mut VernierDetector,
    pixels: *const f32,
    width: usize,
    height: usize,
    physical_period: f64,
    code_size: u32,
    sigma: f64,
    min_frequency: usize,
    max_frequency: usize,
    smoothing_sigma: f64,
) -> VernierPose {
    guarded(VernierPose::not_found(), || {
        let det = unsafe { det.as_ref() }.ok_or("null detector pointer")?;
        let image = to_image(pixels, width, height)?;
        let detection = det
            .backend
            .analyze_two(
                &image.to_complex(),
                image.layout(),
                sigma as Real,
                min_frequency,
                max_frequency,
                smoothing_sigma as Real,
            )
            .map_err(|e| e.to_string())?;
        let calib = Calibration::new(physical_period as Real, width, height);
        let pose = absolute::solve_megarena(&detection, image.as_slice(), &calib, code_size)
            .map_err(|e| e.to_string())?;
        Ok(VernierPose::found(&pose))
    })
}

// ─── Camera calibration and PnP ──────────────────────────────────────────────
//
// Measure each frame of the coded checkerboard into a `VernierView`, then
// either calibrate a camera from several views (`vernier_calibrate`) or, the
// camera being known, find the board's pose in one (`vernier_solve_pnp`).
// Poses and distortion coefficients follow OpenCV's conventions, so they can
// be handed to `cv::projectPoints`, `cv::undistort` and the like as they are.

/// `VernierCamera::model`: pinhole with Brown-Conrady distortion
/// `k1, k2, p1, p2, k3`.
pub const VERNIER_MODEL_PINHOLE: u32 = 0;
/// `VernierCamera::model`: Kannala-Brandt fisheye (OpenCV's `cv::fisheye`),
/// distortion `k1, k2, k3, k4`.
pub const VERNIER_MODEL_FISHEYE: u32 = 1;

/// `VernierTarget::layout`: upright squares, the default.
pub const VERNIER_LAYOUT_SQUARES: u32 = 0;
/// `VernierTarget::layout`: squares turned 45°.
pub const VERNIER_LAYOUT_DIAMONDS: u32 = 1;

/// `VernierTarget::packing`: one code bit per axis in each 3×3 supercell,
/// the default.
pub const VERNIER_PACKING_ONE_BIT: u32 = 0;
/// `VernierTarget::packing`: two code bits per axis in each 5×5 supercell.
pub const VERNIER_PACKING_TWO_BITS: u32 = 1;

/// `VernierTarget::kind`: the coded checkerboard of `vernier render-checkerboard`.
pub const VERNIER_TARGET_CHECKERBOARD: u32 = 0;
/// `VernierTarget::kind`: the megarena dot grid of `vernier render-megarena`.
pub const VERNIER_TARGET_MEGARENA: u32 = 1;

/// The printed board: which pattern, its size and how its code is laid out.
/// What `vernier render-checkerboard` prints by default is
/// `{ square, order, VERNIER_LAYOUT_SQUARES, VERNIER_PACKING_ONE_BIT,
/// VERNIER_TARGET_CHECKERBOARD }`, which `vernier_target_default` returns;
/// `vernier_target_megarena` gives a megarena.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VernierTarget {
    /// Side of one square, or for a megarena the dot pitch, in the unit poses
    /// should come out in (e.g. mm).
    pub square: f64,
    /// LFSR order (code size) the board was rendered with, 4 to 12.
    pub order: u32,
    /// `VERNIER_LAYOUT_*`. Checkerboard only.
    pub layout: u32,
    /// `VERNIER_PACKING_*`. Checkerboard only.
    pub packing: u32,
    /// `VERNIER_TARGET_*`.
    pub kind: u32,
}

/// One measured point: where it is in the frame and where it is on the board.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VernierPointMatch {
    /// In pixels, OpenCV's convention (pixel centres on integers).
    pub pixel: [f64; 2],
    /// On the board plane (z = 0), in the target's unit.
    pub board: [f64; 2],
}

/// Camera intrinsics, laid out as OpenCV's camera matrix and distortion
/// vector.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VernierCamera {
    /// `VERNIER_MODEL_*`.
    pub model: u32,
    /// Image size, pixels.
    pub width: usize,
    pub height: usize,
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    /// Pinhole: `k1, k2, p1, p2, k3`. Fisheye: `k1, k2, k3, k4`, the fifth
    /// entry unused and zero.
    pub distortion: [f64; 5],
}

/// Board to camera, `p_camera = R(rvec) · p_board + tvec`: the pair OpenCV's
/// `solvePnP` returns.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VernierRigidPose {
    /// Rodrigues rotation vector, radians.
    pub rvec: [f64; 3],
    /// Translation, in the target's unit.
    pub tvec: [f64; 3],
}

/// The fit of one view: the board's pose and how well the points agree.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VernierViewFit {
    pub pose: VernierRigidPose,
    /// Reprojection error over the points kept, pixels.
    pub rms: f64,
    /// Points kept.
    pub used: usize,
    /// Points dropped as outliers.
    pub rejected: usize,
}

/// Opaque handle to the points measured in one frame. Create with
/// `vernier_measure_view`; free with `vernier_view_free`.
pub struct VernierView {
    view: camera::View,
}

/// Runs `f`, turning an error into the thread's last error and a panic into
/// a generic one, so that no panic unwinds into C.
fn guarded<T>(fallback: T, f: impl FnOnce() -> Result<T, String>) -> T {
    clear_last_error();
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            set_last_error(message);
            fallback
        }
        Err(_) => {
            set_last_error("internal error (panic) in vernier");
            fallback
        }
    }
}

fn to_target(t: *const VernierTarget) -> Result<camera::Target, String> {
    let t = unsafe { t.as_ref() }.ok_or("null target pointer")?;
    if !(t.square > 0.0) {
        return Err(format!("square size {} must be positive", t.square));
    }
    let layout = match t.layout {
        VERNIER_LAYOUT_SQUARES => CodeLayout::Squares,
        VERNIER_LAYOUT_DIAMONDS => CodeLayout::Diamonds,
        other => return Err(format!("unknown code layout {other}")),
    };
    let packing = match t.packing {
        VERNIER_PACKING_ONE_BIT => CodePacking::OneBit,
        VERNIER_PACKING_TWO_BITS => CodePacking::TwoBits,
        other => return Err(format!("unknown code packing {other}")),
    };
    let target = match t.kind {
        VERNIER_TARGET_CHECKERBOARD => camera::Target::new(t.square, t.order)
            .with_layout(layout)
            .with_packing(packing),
        VERNIER_TARGET_MEGARENA => camera::Target::megarena(t.square, t.order),
        other => return Err(format!("unknown target kind {other}")),
    };
    // Building the pattern is what checks the order.
    if target.checkerboard().is_none() && target.megarena_pattern().is_none() {
        return Err(format!("unsupported code size {}; must be 4..=12", t.order));
    }
    Ok(target)
}

fn to_model(model: u32) -> Result<camera::Model, String> {
    match model {
        VERNIER_MODEL_PINHOLE => Ok(camera::Model::Pinhole),
        VERNIER_MODEL_FISHEYE => Ok(camera::Model::Fisheye),
        other => Err(format!("unknown camera model {other}")),
    }
}

fn to_camera(c: *const VernierCamera) -> Result<camera::Camera, String> {
    let c = unsafe { c.as_ref() }.ok_or("null camera pointer")?;
    let model = to_model(c.model)?;
    let mut camera = camera::Camera::ideal(model, c.width, c.height, c.fx, c.fy, c.cx, c.cy);
    camera.distortion = c.distortion[..model.distortion_len()].to_vec();
    Ok(camera)
}

fn from_camera(c: &camera::Camera) -> VernierCamera {
    let mut distortion = [0.0; 5];
    distortion[..c.distortion.len()].copy_from_slice(&c.distortion);
    VernierCamera {
        model: match c.model {
            camera::Model::Pinhole => VERNIER_MODEL_PINHOLE,
            camera::Model::Fisheye => VERNIER_MODEL_FISHEYE,
        },
        width: c.width,
        height: c.height,
        fx: c.fx,
        fy: c.fy,
        cx: c.cx,
        cy: c.cy,
        distortion,
    }
}

fn from_fit(fit: &camera::ViewFit) -> VernierViewFit {
    let (r, t) = (fit.pose.rvec(), fit.pose.translation);
    VernierViewFit {
        pose: VernierRigidPose {
            rvec: [r.x, r.y, r.z],
            tvec: [t.x, t.y, t.z],
        },
        rms: fit.rms,
        used: fit.used,
        rejected: fit.rejected,
    }
}

fn pixel_slice<'a>(pixels: *const f32, width: usize, height: usize) -> Result<&'a [f32], String> {
    if pixels.is_null() {
        return Err("null pixels pointer".into());
    }
    if width == 0 || height == 0 {
        return Err(format!("empty image {width}×{height}"));
    }
    let len = width
        .checked_mul(height)
        .filter(|&len| len <= isize::MAX as usize / size_of::<f32>())
        .ok_or_else(|| format!("image {width}×{height} is too large"))?;
    Ok(unsafe { std::slice::from_raw_parts(pixels, len) })
}

fn to_image(pixels: *const f32, width: usize, height: usize) -> Result<GrayImage, String> {
    let slice = pixel_slice(pixels, width, height)?;
    GrayImage::from_vec(width, height, slice.to_vec()).ok_or_else(|| "mismatched image size".into())
}

/// A target as `vernier render-checkerboard` prints it by default: upright
/// squares, one code bit per supercell.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_target_default(square: f64, order: u32) -> VernierTarget {
    VernierTarget {
        square,
        order,
        layout: VERNIER_LAYOUT_SQUARES,
        packing: VERNIER_PACKING_ONE_BIT,
        kind: VERNIER_TARGET_CHECKERBOARD,
    }
}

/// A megarena as `vernier render-megarena` draws it, of dot pitch `pitch`
/// (in the unit poses should come out in) and LFSR order `order`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_target_megarena(pitch: f64, order: u32) -> VernierTarget {
    VernierTarget {
        kind: VERNIER_TARGET_MEGARENA,
        ..vernier_target_default(pitch, order)
    }
}

/// Measures one frame of the target (coded checkerboard or megarena) into
/// pixel ↔ board correspondences.
///
/// - `pixels` — row-major f32 grayscale image, `width × height` elements in [0, 1].
/// - `target` — the printed board.
///
/// Returns NULL when the board is not found (see `vernier_last_error`). Free
/// the view with `vernier_view_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_measure_view(
    pixels: *const f32,
    width: usize,
    height: usize,
    target: *const VernierTarget,
) -> *mut VernierView {
    vernier_measure_view_after(pixels, width, height, target, std::ptr::null())
}

/// `vernier_measure_view` for a frame of a video: the carriers of `previous`
/// (the view of the frame before; may be NULL) are tried first, which is
/// faster while the board barely moves.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_measure_view_after(
    pixels: *const f32,
    width: usize,
    height: usize,
    target: *const VernierTarget,
    previous: *const VernierView,
) -> *mut VernierView {
    guarded(std::ptr::null_mut(), || {
        let data = pixel_slice(pixels, width, height)?;
        let target = to_target(target)?;
        let view = match unsafe { previous.as_ref() } {
            Some(p) => camera::measure_view_after(data, width, height, &target, &p.view),
            None => camera::measure_view(data, width, height, &target),
        }
        .map_err(|e| e.to_string())?;
        Ok(Box::into_raw(Box::new(VernierView { view })))
    })
}

/// Frees a view. Passing NULL is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_view_free(view: *mut VernierView) {
    if !view.is_null() {
        unsafe { drop(Box::from_raw(view)) };
    }
}

/// Copies up to `capacity` of the view's points into `out` and returns how
/// many the view holds, so a first call with `out == NULL` gives the size to
/// allocate. Returns 0 for a NULL view.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_view_copy_points(
    view: *const VernierView,
    out: *mut VernierPointMatch,
    capacity: usize,
) -> usize {
    let Some(view) = (unsafe { view.as_ref() }) else {
        return 0;
    };
    let points = &view.view.points;
    if !out.is_null() {
        let out = unsafe { std::slice::from_raw_parts_mut(out, capacity.min(points.len())) };
        for (dst, src) in out.iter_mut().zip(points) {
            *dst = VernierPointMatch {
                pixel: src.pixel,
                board: src.board,
            };
        }
    }
    points.len()
}

/// 1 when the board's code was read, so the points (and a pose solved from
/// them) are in the board's own frame; 0 otherwise. A view whose code was not
/// read still serves for calibration, but its pose is only known up to a
/// quarter-turn and a whole number of code periods.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_view_is_absolute(view: *const VernierView) -> i32 {
    unsafe { view.as_ref() }.is_some_and(|v| v.view.is_absolute()) as i32
}

/// Calibrates a camera from views of the board taken at varied angles (at
/// least 2; 10 or more is better), all of the same image size.
///
/// - `views`      — array of `count` view pointers.
/// - `model`      — `VERNIER_MODEL_*`.
/// - `out_camera` — receives the intrinsics (must not be NULL).
/// - `out_rms`    — receives the reprojection rms over all points kept, pixels (may be NULL).
/// - `out_fits`   — array of `count` entries receiving each view's pose and fit (may be NULL).
///
/// Returns 1 on success, 0 on failure (see `vernier_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vernier_calibrate(
    views: *const *const VernierView,
    count: usize,
    model: u32,
    out_camera: *mut VernierCamera,
    out_rms: *mut f64,
    out_fits: *mut VernierViewFit,
) -> i32 {
    guarded(0, || {
        if views.is_null() && count > 0 {
            return Err("null views pointer".into());
        }
        let out_camera = unsafe { out_camera.as_mut() }.ok_or("null out_camera pointer")?;
        let model = to_model(model)?;
        let pointers: &[*const VernierView] = if count == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(views, count) }
        };
        let owned: Vec<camera::View> = pointers
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                unsafe { p.as_ref() }
                    .map(|v| v.view.clone())
                    .ok_or_else(|| format!("view {i} is NULL"))
            })
            .collect::<Result<_, _>>()?;

        let result = camera::calibrate(&owned, model).map_err(|e| e.to_string())?;
        *out_camera = from_camera(&result.camera);
        if let Some(rms) = unsafe { out_rms.as_mut() } {
            *rms = result.rms;
        }
        if !out_fits.is_null() {
            let fits = unsafe { std::slice::from_raw_parts_mut(out_fits, count) };
            for (dst, src) in fits.iter_mut().zip(&result.views) {
                *dst = from_fit(src);
            }
        }
        Ok(1)
    })
}

/// Pose of the board in one view, the camera being known. The pose is in the
/// board's own frame only when `vernier_view_is_absolute(view)`.
///
/// Returns 1 on success, 0 on failure (see `vernier_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vernier_solve_pnp(
    camera: *const VernierCamera,
    view: *const VernierView,
    out: *mut VernierViewFit,
) -> i32 {
    guarded(0, || {
        let camera = to_camera(camera)?;
        let view = unsafe { view.as_ref() }.ok_or("null view pointer")?;
        let out = unsafe { out.as_mut() }.ok_or("null out pointer")?;
        let fit = camera::solve_pnp(&camera, &view.view).map_err(|e| e.to_string())?;
        *out = from_fit(&fit);
        Ok(1)
    })
}

// ─── Files: camera.json and images ───────────────────────────────────────────

/// The camera file `vernier calibrate` writes and `vernier solve-pnp` reads:
/// the camera's fields at the top level, then how the calibration went.
#[derive(serde::Serialize, serde::Deserialize)]
struct CameraFile {
    #[serde(flatten)]
    camera: camera::Camera,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    rms: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    views: Option<usize>,
}

fn path_of<'a>(path: *const c_char) -> Result<&'a str, String> {
    if path.is_null() {
        return Err("null path".into());
    }
    unsafe { std::ffi::CStr::from_ptr(path) }
        .to_str()
        .map_err(|_| "path is not UTF-8".into())
}

/// Reads a camera file as `vernier calibrate` writes it (JSON: `model`,
/// `width`, `height`, `fx`, `fy`, `cx`, `cy`, `distortion`).
///
/// Returns 1 on success, 0 on failure (see `vernier_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vernier_camera_load(path: *const c_char, out: *mut VernierCamera) -> i32 {
    guarded(0, || {
        let path = path_of(path)?;
        let out = unsafe { out.as_mut() }.ok_or("null out pointer")?;
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read {path}: {e}"))?;
        let file: CameraFile = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        let c = file.camera;
        let expected = c.model.distortion_len();
        if c.distortion.len() != expected {
            return Err(format!(
                "{path}: a {} camera has {expected} distortion coefficients",
                c.model.name()
            ));
        }
        *out = from_camera(&c);
        Ok(1)
    })
}

/// Writes a camera file in the format `vernier calibrate` writes, which
/// `vernier_camera_load` and `vernier solve-pnp` read back. `rms` (pixels) and
/// `views` record how the calibration went; a negative `rms` or zero `views`
/// leaves them out.
///
/// Returns 1 on success, 0 on failure (see `vernier_last_error`).
#[unsafe(no_mangle)]
pub extern "C" fn vernier_camera_save(
    path: *const c_char,
    camera: *const VernierCamera,
    rms: f64,
    views: usize,
) -> i32 {
    guarded(0, || {
        let path = path_of(path)?;
        let file = CameraFile {
            camera: to_camera(camera)?,
            rms: (rms >= 0.0).then_some(rms),
            views: (views > 0).then_some(views),
        };
        let text = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        std::fs::write(path, text + "\n").map_err(|e| format!("failed to write {path}: {e}"))?;
        Ok(1)
    })
}

/// Loads an image file (PNG, JPEG, BMP, TIFF, PGM/PPM) as grayscale,
/// row-major, `width × height` floats in [0, 1]: what the measurement and
/// detection functions take.
///
/// Returns NULL on failure (see `vernier_last_error`). Free the pixels with
/// `vernier_image_free`.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_image_load(
    path: *const c_char,
    width: *mut usize,
    height: *mut usize,
) -> *mut f32 {
    guarded(std::ptr::null_mut(), || {
        let path = path_of(path)?;
        let (width, height) = unsafe { (width.as_mut(), height.as_mut()) };
        let (Some(width), Some(height)) = (width, height) else {
            return Err("null width or height pointer".into());
        };
        let image = image::open(path).map_err(|e| format!("failed to open {path}: {e}"))?;
        // Through 16-bit luma, to keep the depth of 16-bit files.
        let luma = image.to_luma16();
        *width = luma.width() as usize;
        *height = luma.height() as usize;
        let pixels: Box<[f32]> = luma
            .pixels()
            .map(|p| p.0[0] as f32 / u16::MAX as f32)
            .collect();
        Ok(Box::into_raw(pixels) as *mut f32)
    })
}

/// Frees pixels from `vernier_image_load`, given the size it returned.
/// Passing NULL is a no-op.
#[unsafe(no_mangle)]
pub extern "C" fn vernier_image_free(pixels: *mut f32, width: usize, height: usize) {
    if !pixels.is_null() {
        let slice = std::ptr::slice_from_raw_parts_mut(pixels, width.saturating_mul(height));
        unsafe { drop(Box::from_raw(slice)) };
    }
}
