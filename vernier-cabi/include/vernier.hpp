#pragma once
#include "vernier.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// C++ RAII wrapper around the vernier C ABI.
namespace vernier {

/// Detected in-plane pose.
/// `x` and `y` are in the same physical units as the `period` argument.
/// `theta` is in radians.
struct Pose {
    double x;
    double y;
    double theta;
};

/// CPU- or CUDA-backed pose detector.
///
/// Reuse across frames — the internal FFT planner caches its plan, so
/// repeated calls on same-size images are cheaper than constructing a new
/// Detector each time.
class Detector {
public:
    /// Creates a CPU-backed detector.
    /// Throws `std::runtime_error` on allocation failure.
    Detector() {
        handle_ = vernier_detector_new();
        if (!handle_)
            throw std::runtime_error("vernier: failed to create CPU detector");
    }

    /// Creates a CUDA-backed detector.
    /// Throws `std::runtime_error` if no CUDA device is available or if the
    /// library was not built with CUDA support (rebuild with `--features cuda`).
    static Detector cuda() {
        VernierDetector* h = vernier_detector_new_cuda();
        if (!h) {
            const char* err = vernier_last_error();
            throw std::runtime_error(err ? std::string(err)
                                        : "vernier: CUDA detector creation failed");
        }
        return Detector(h);
    }

    Detector(Detector&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Detector& operator=(Detector&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    Detector(const Detector&)            = delete;
    Detector& operator=(const Detector&) = delete;

    ~Detector() { reset(); }

    /// Periodic (relative) detection.
    ///
    /// Recovers `x` and `y` modulo `period` and the in-image orientation
    /// `theta`. Throws `std::runtime_error` if no carrier peaks are found.
    ///
    /// @param pixels          Row-major float array, `width * height` elements in [0, 1].
    /// @param width           Image width in pixels.
    /// @param height          Image height in pixels.
    /// @param period          Pattern spatial period in physical units.
    /// @param sigma           Bandpass filter half-width in frequency bins.
    /// @param min_frequency   Inner spectral annulus radius; 0 = no limit.
    /// @param max_frequency   Outer spectral annulus radius; 0 = no limit.
    /// @param smoothing_sigma Gaussian blur sigma on magnitude spectrum; 0 disables.
    Pose detect_periodic(
        const float* pixels,
        std::size_t  width,
        std::size_t  height,
        float        period,
        float        sigma           = 3.0f,
        std::size_t  min_frequency   = 0,
        std::size_t  max_frequency   = 0,
        float        smoothing_sigma = 0.5f)
    {
        VernierPose p = vernier_detect_periodic(
            handle_, pixels, width, height,
            period, sigma, min_frequency, max_frequency, smoothing_sigma);
        if (!p.found)
            throw_last_error("periodic detection failed");
        return {p.x, p.y, p.theta};
    }

    /// Megarena absolute detection.
    ///
    /// Recovers an unambiguous `(x, y, theta)` by combining the fine phase
    /// measurement with the LFSR binary code embedded in the Megarena pattern.
    /// Throws `std::runtime_error` if detection or LFSR decode fails.
    ///
    /// @param pixels          Row-major float array, `width * height` elements in [0, 1].
    /// @param width           Image width in pixels.
    /// @param height          Image height in pixels.
    /// @param physical_period Pattern spatial period in micrometres (9.0 for the reference pattern).
    /// @param code_size       LFSR order in bits (12 for the reference pattern).
    /// @param sigma           Bandpass filter half-width in frequency bins.
    /// @param min_frequency   Inner spectral annulus radius; 0 = no limit.
    /// @param max_frequency   Outer spectral annulus radius; 0 = no limit.
    /// @param smoothing_sigma Gaussian blur sigma on magnitude spectrum.
    Pose detect_megarena(
        const float*  pixels,
        std::size_t   width,
        std::size_t   height,
        float         physical_period,
        std::uint32_t code_size,
        float         sigma           = 3.0f,
        std::size_t   min_frequency   = 20,
        std::size_t   max_frequency   = 500,
        float         smoothing_sigma = 0.5f)
    {
        VernierPose p = vernier_detect_megarena(
            handle_, pixels, width, height,
            physical_period, code_size,
            sigma, min_frequency, max_frequency, smoothing_sigma);
        if (!p.found)
            throw_last_error("megarena detection failed");
        return {p.x, p.y, p.theta};
    }

private:
    VernierDetector* handle_;

    explicit Detector(VernierDetector* h) : handle_(h) {}

    void reset() noexcept {
        if (handle_) {
            vernier_detector_free(handle_);
            handle_ = nullptr;
        }
    }

    [[noreturn]] static void throw_last_error(const char* fallback) {
        const char* err = vernier_last_error();
        throw std::runtime_error(std::string("vernier: ") +
                                 (err ? err : fallback));
    }
};

// ─── Camera calibration and PnP ──────────────────────────────────────────────
//
// Measure each frame of the coded checkerboard into a `View`, then either
// `calibrate` a camera from several views or, the camera being known,
// `solve_pnp` for the board's pose in one. Poses and distortion follow
// OpenCV's conventions, so they can be handed to `cv::projectPoints`,
// `cv::undistort` and the like as they are.

namespace detail {
[[noreturn]] inline void throw_last_error(const char* fallback) {
    const char* err = vernier_last_error();
    throw std::runtime_error(std::string("vernier: ") + (err ? err : fallback));
}
}  // namespace detail

/// Lens model of a `Camera`.
enum class Model : std::uint32_t {
    /// Brown-Conrady distortion `k1, k2, p1, p2, k3`.
    Pinhole = VERNIER_MODEL_PINHOLE,
    /// Kannala-Brandt (OpenCV's `cv::fisheye`), distortion `k1, k2, k3, k4`.
    Fisheye = VERNIER_MODEL_FISHEYE,
};

/// Squares upright or turned 45°.
enum class CodeLayout : std::uint32_t {
    Squares  = VERNIER_LAYOUT_SQUARES,
    Diamonds = VERNIER_LAYOUT_DIAMONDS,
};

/// Code bits per axis in each supercell.
enum class CodePacking : std::uint32_t {
    OneBit  = VERNIER_PACKING_ONE_BIT,
    TwoBits = VERNIER_PACKING_TWO_BITS,
};

/// Which pattern a `Target` is.
enum class PatternKind : std::uint32_t {
    /// The coded checkerboard of `vernier render-checkerboard`.
    Checkerboard = VERNIER_TARGET_CHECKERBOARD,
    /// The megarena dot grid of `vernier render-megarena`.
    Megarena     = VERNIER_TARGET_MEGARENA,
};

/// The printed board. The defaults match what `vernier render-checkerboard`
/// prints, so usually only `square` and `order` need setting; for a megarena
/// use `Target::megarena(pitch, order)`.
struct Target {
    /// Side of one square, or for a megarena the dot pitch, in the unit poses
    /// should come out in (e.g. mm).
    double        square;
    /// LFSR order (code size) the board was rendered with, 4 to 12.
    std::uint32_t order;
    /// Checkerboard only.
    CodeLayout    layout  = CodeLayout::Squares;
    /// Checkerboard only.
    CodePacking   packing = CodePacking::OneBit;
    PatternKind   kind    = PatternKind::Checkerboard;

    /// A megarena as `vernier render-megarena` draws it: dot pitch and LFSR
    /// order.
    static Target megarena(double pitch, std::uint32_t order) {
        return {pitch, order, CodeLayout::Squares, CodePacking::OneBit, PatternKind::Megarena};
    }

    VernierTarget to_c() const {
        return {square, order, static_cast<std::uint32_t>(layout),
                static_cast<std::uint32_t>(packing), static_cast<std::uint32_t>(kind)};
    }
};

/// One measured point: where it is in the frame and where it is on the board.
struct PointMatch {
    /// Pixels, OpenCV's convention (pixel centres on integers).
    std::array<double, 2> pixel;
    /// On the board plane (z = 0), in the target's unit.
    std::array<double, 2> board;
};

/// Camera intrinsics: OpenCV's camera matrix and distortion vector.
struct Camera {
    Model       model  = Model::Pinhole;
    std::size_t width  = 0;
    std::size_t height = 0;
    double      fx = 0, fy = 0, cx = 0, cy = 0;
    /// Pinhole: `k1, k2, p1, p2, k3` (5). Fisheye: `k1, k2, k3, k4` (4).
    std::vector<double> distortion;

    static Camera from_c(const VernierCamera& c) {
        Camera cam;
        cam.model  = static_cast<Model>(c.model);
        cam.width  = c.width;
        cam.height = c.height;
        cam.fx = c.fx; cam.fy = c.fy; cam.cx = c.cx; cam.cy = c.cy;
        const std::size_t n = cam.model == Model::Fisheye ? 4 : 5;
        cam.distortion.assign(c.distortion, c.distortion + n);
        return cam;
    }

    /// Throws `std::invalid_argument` when `distortion` does not hold as many
    /// coefficients as the model takes.
    VernierCamera to_c() const {
        const std::size_t n = model == Model::Fisheye ? 4 : 5;
        if (distortion.size() != n)
            throw std::invalid_argument(
                "vernier: a " + std::string(model == Model::Fisheye ? "fisheye" : "pinhole") +
                " camera has " + std::to_string(n) + " distortion coefficients");
        VernierCamera c{};
        c.model  = static_cast<std::uint32_t>(model);
        c.width  = width;
        c.height = height;
        c.fx = fx; c.fy = fy; c.cx = cx; c.cy = cy;
        for (std::size_t i = 0; i < n; ++i) c.distortion[i] = distortion[i];
        return c;
    }
};

/// Board to camera, `p_camera = R(rvec) · p_board + tvec`: OpenCV's
/// `solvePnP` output.
struct RigidPose {
    /// Rodrigues rotation vector, radians.
    std::array<double, 3> rvec;
    /// Translation, in the target's unit.
    std::array<double, 3> tvec;
};

/// The fit of one view: the board's pose and how well the points agree.
struct ViewFit {
    RigidPose   pose;
    /// Reprojection error over the points kept, pixels.
    double      rms;
    /// Points kept.
    std::size_t used;
    /// Points dropped as outliers.
    std::size_t rejected;

    static ViewFit from_c(const VernierViewFit& f) {
        return {{{f.pose.rvec[0], f.pose.rvec[1], f.pose.rvec[2]},
                 {f.pose.tvec[0], f.pose.tvec[1], f.pose.tvec[2]}},
                f.rms, f.used, f.rejected};
    }
};

/// A calibrated camera, with the fit of every view that went into it.
struct Calibration {
    Camera               camera;
    /// Reprojection rms over every point kept, pixels.
    double               rms;
    /// One per view passed to `calibrate`, in the same order.
    std::vector<ViewFit> views;
};

/// The points measured in one frame. Move-only.
class View {
public:
    /// Measures one frame of the board.
    ///
    /// @param pixels Row-major float array, `width * height` elements in [0, 1].
    /// Throws `std::runtime_error` when the board is not found.
    static View measure(const float* pixels, std::size_t width, std::size_t height,
                        const Target& target) {
        VernierTarget t = target.to_c();
        VernierView*  h = vernier_measure_view(pixels, width, height, &t);
        if (!h) detail::throw_last_error("board not found");
        return View(h);
    }

    /// `measure` for a frame of a video: the carriers of `previous` (the view
    /// of the frame before) are tried first, which is faster while the board
    /// barely moves.
    static View measure_after(const float* pixels, std::size_t width, std::size_t height,
                              const Target& target, const View& previous) {
        VernierTarget t = target.to_c();
        VernierView*  h =
            vernier_measure_view_after(pixels, width, height, &t, previous.handle_);
        if (!h) detail::throw_last_error("board not found");
        return View(h);
    }

    View(View&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }

    View& operator=(View&& other) noexcept {
        if (this != &other) {
            vernier_view_free(handle_);
            handle_       = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    View(const View&)            = delete;
    View& operator=(const View&) = delete;

    ~View() { vernier_view_free(handle_); }

    /// Whether the board's code was read, so the points (and a pose solved
    /// from them) are in the board's own frame. A view whose code was not
    /// read still serves for calibration, but its pose is only known up to a
    /// quarter-turn and a whole number of code periods.
    bool is_absolute() const { return vernier_view_is_absolute(handle_) != 0; }

    /// Number of points measured.
    std::size_t size() const { return vernier_view_copy_points(handle_, nullptr, 0); }

    /// The pixel ↔ board correspondences, e.g. for `cv::solvePnP` with
    /// `board` as object points at z = 0.
    std::vector<PointMatch> points() const {
        std::vector<VernierPointMatch> raw(size());
        vernier_view_copy_points(handle_, raw.data(), raw.size());
        std::vector<PointMatch> out;
        out.reserve(raw.size());
        for (const auto& p : raw)
            out.push_back({{p.pixel[0], p.pixel[1]}, {p.board[0], p.board[1]}});
        return out;
    }

    const VernierView* handle() const { return handle_; }

private:
    VernierView* handle_;

    explicit View(VernierView* h) : handle_(h) {}
};

/// Calibrates a camera from views of the board taken at varied angles (at
/// least 2; 10 or more is better), all of the same image size.
/// Throws `std::runtime_error` on failure.
inline Calibration calibrate(const std::vector<View>& views, Model model = Model::Pinhole) {
    std::vector<const VernierView*> handles;
    handles.reserve(views.size());
    for (const auto& v : views) handles.push_back(v.handle());

    VernierCamera               camera{};
    double                      rms = 0;
    std::vector<VernierViewFit> fits(views.size());
    if (!vernier_calibrate(handles.data(), handles.size(), static_cast<std::uint32_t>(model),
                           &camera, &rms, fits.data()))
        detail::throw_last_error("calibration failed");

    Calibration result{Camera::from_c(camera), rms, {}};
    result.views.reserve(fits.size());
    for (const auto& f : fits) result.views.push_back(ViewFit::from_c(f));
    return result;
}

/// Pose of the board in one view, the camera being known. The pose is in the
/// board's own frame only when `view.is_absolute()`.
/// Throws `std::runtime_error` on failure.
inline ViewFit solve_pnp(const Camera& camera, const View& view) {
    VernierCamera  c = camera.to_c();
    VernierViewFit fit{};
    if (!vernier_solve_pnp(&c, view.handle(), &fit))
        detail::throw_last_error("solve_pnp failed");
    return ViewFit::from_c(fit);
}

}  // namespace vernier
