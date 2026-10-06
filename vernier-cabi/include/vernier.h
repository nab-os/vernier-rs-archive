#ifndef VERNIER_H
#define VERNIER_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

// `VernierCamera::model`: pinhole with Brown-Conrady distortion
// `k1, k2, p1, p2, k3`.
#define VERNIER_MODEL_PINHOLE 0

// `VernierCamera::model`: Kannala-Brandt fisheye (OpenCV's `cv::fisheye`),
// distortion `k1, k2, k3, k4`.
#define VERNIER_MODEL_FISHEYE 1

// `VernierTarget::layout`: upright squares, the default.
#define VERNIER_LAYOUT_SQUARES 0

// `VernierTarget::layout`: squares turned 45°.
#define VERNIER_LAYOUT_DIAMONDS 1

// `VernierTarget::packing`: one code bit per axis in each 3×3 supercell,
// the default.
#define VERNIER_PACKING_ONE_BIT 0

// `VernierTarget::packing`: two code bits per axis in each 5×5 supercell.
#define VERNIER_PACKING_TWO_BITS 1

// `VernierTarget::kind`: the coded checkerboard of `vernier render-checkerboard`.
#define VERNIER_TARGET_CHECKERBOARD 0

// `VernierTarget::kind`: the megarena dot grid of `vernier render-megarena`.
#define VERNIER_TARGET_MEGARENA 1

// Opaque handle to a Vernier detector. Create with `vernier_detector_new`
// (CPU) or `vernier_detector_new_cuda` (GPU); free with
// `vernier_detector_free`.
typedef struct VernierDetector VernierDetector;

// Opaque handle to the points measured in one frame. Create with
// `vernier_measure_view`; free with `vernier_view_free`.
typedef struct VernierView VernierView;

// Pose returned by all detection functions.
//
// Check `found` before reading `x`, `y`, `theta`. On failure (`found == 0`)
// call `vernier_last_error()` for a description.
typedef struct VernierPose {
    double x;
    double y;
    double theta;
    // 1 on success, 0 on failure.
    int32_t found;
} VernierPose;

// The printed board: which pattern, its size and how its code is laid out.
// What `vernier render-checkerboard` prints by default is
// `{ square, order, VERNIER_LAYOUT_SQUARES, VERNIER_PACKING_ONE_BIT,
// VERNIER_TARGET_CHECKERBOARD }`, which `vernier_target_default` returns;
// `vernier_target_megarena` gives a megarena.
typedef struct VernierTarget {
    // Side of one square, or for a megarena the dot pitch, in the unit poses
    // should come out in (e.g. mm).
    double square;
    // LFSR order (code size) the board was rendered with, 4 to 12.
    uint32_t order;
    // `VERNIER_LAYOUT_*`. Checkerboard only.
    uint32_t layout;
    // `VERNIER_PACKING_*`. Checkerboard only.
    uint32_t packing;
    // `VERNIER_TARGET_*`.
    uint32_t kind;
} VernierTarget;

// One measured point: where it is in the frame and where it is on the board.
typedef struct VernierPointMatch {
    // In pixels, OpenCV's convention (pixel centres on integers).
    double pixel[2];
    // On the board plane (z = 0), in the target's unit.
    double board[2];
} VernierPointMatch;

// Camera intrinsics, laid out as OpenCV's camera matrix and distortion
// vector.
typedef struct VernierCamera {
    // `VERNIER_MODEL_*`.
    uint32_t model;
    // Image size, pixels.
    size_t width;
    size_t height;
    double fx;
    double fy;
    double cx;
    double cy;
    // Pinhole: `k1, k2, p1, p2, k3`. Fisheye: `k1, k2, k3, k4`, the fifth
    // entry unused and zero.
    double distortion[5];
} VernierCamera;

// Board to camera, `p_camera = R(rvec) · p_board + tvec`: the pair OpenCV's
// `solvePnP` returns.
typedef struct VernierRigidPose {
    // Rodrigues rotation vector, radians.
    double rvec[3];
    // Translation, in the target's unit.
    double tvec[3];
} VernierRigidPose;

// The fit of one view: the board's pose and how well the points agree.
typedef struct VernierViewFit {
    struct VernierRigidPose pose;
    // Reprojection error over the points kept, pixels.
    double rms;
    // Points kept.
    size_t used;
    // Points dropped as outliers.
    size_t rejected;
} VernierViewFit;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Returns the last error message on this thread, or NULL if the last call
// succeeded.
//
// The pointer is valid until the next vernier call on this thread.
const char *vernier_last_error(void);

// Creates a CPU-backed detector. Returns NULL on allocation failure.
//
// Must be freed with `vernier_detector_free`.
struct VernierDetector *vernier_detector_new(void);

// Creates a CUDA-backed detector. Returns NULL if no CUDA device is
// available or if the library was not compiled with CUDA support (check
// `vernier_last_error` for details).
//
// Must be freed with `vernier_detector_free`.
struct VernierDetector *vernier_detector_new_cuda(void);

// Frees a detector. Passing NULL is a no-op.
void vernier_detector_free(struct VernierDetector *det);

// Periodic (relative) detection: recovers `x`, `y` modulo the pattern period
// and the in-image orientation `theta`.
//
// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
// - `period`           — pattern spatial period in physical units.
// - `sigma`            — bandpass filter half-width in frequency bins.
// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
//                        search; 0 disables blurring.
//
// Returns a pose with `found == 0` on failure.
struct VernierPose vernier_detect_periodic(struct VernierDetector *det,
                                           const float *pixels,
                                           size_t width,
                                           size_t height,
                                           double period,
                                           double sigma,
                                           size_t min_frequency,
                                           size_t max_frequency,
                                           double smoothing_sigma);

// Megarena absolute detection: recovers an unambiguous `(x, y, theta)` using
// the LFSR binary code embedded in the pattern.
//
// - `det`              — handle from `vernier_detector_new[_cuda]` (must not be NULL).
// - `pixels`           — row-major f32 image, `width × height` elements in [0, 1].
// - `physical_period`  — pattern spatial period in micrometres (9 µm for the
//                        reference pattern).
// - `code_size`        — LFSR order in bits (12 for the reference pattern).
// - `sigma`            — bandpass filter half-width in frequency bins.
// - `min_frequency`    — inner annulus radius for peak search (0 = no limit).
// - `max_frequency`    — outer annulus radius for peak search (0 = no limit).
// - `smoothing_sigma`  — Gaussian blur on the magnitude spectrum before peak
//                        search; 0 disables blurring.
//
// Returns a pose with `found == 0` on failure.
struct VernierPose vernier_detect_megarena(struct VernierDetector *det,
                                           const float *pixels,
                                           size_t width,
                                           size_t height,
                                           double physical_period,
                                           uint32_t code_size,
                                           double sigma,
                                           size_t min_frequency,
                                           size_t max_frequency,
                                           double smoothing_sigma);

// A target as `vernier render-checkerboard` prints it by default: upright
// squares, one code bit per supercell.
struct VernierTarget vernier_target_default(double square, uint32_t order);

// A megarena as `vernier render-megarena` draws it, of dot pitch `pitch`
// (in the unit poses should come out in) and LFSR order `order`.
struct VernierTarget vernier_target_megarena(double pitch, uint32_t order);

// Measures one frame of the target (coded checkerboard or megarena) into
// pixel ↔ board correspondences.
//
// - `pixels` — row-major f32 grayscale image, `width × height` elements in [0, 1].
// - `target` — the printed board.
//
// Returns NULL when the board is not found (see `vernier_last_error`). Free
// the view with `vernier_view_free`.
struct VernierView *vernier_measure_view(const float *pixels,
                                         size_t width,
                                         size_t height,
                                         const struct VernierTarget *target);

// `vernier_measure_view` for a frame of a video: the carriers of `previous`
// (the view of the frame before; may be NULL) are tried first, which is
// faster while the board barely moves.
struct VernierView *vernier_measure_view_after(const float *pixels,
                                               size_t width,
                                               size_t height,
                                               const struct VernierTarget *target,
                                               const struct VernierView *previous);

// Frees a view. Passing NULL is a no-op.
void vernier_view_free(struct VernierView *view);

// Copies up to `capacity` of the view's points into `out` and returns how
// many the view holds, so a first call with `out == NULL` gives the size to
// allocate. Returns 0 for a NULL view.
size_t vernier_view_copy_points(const struct VernierView *view,
                                struct VernierPointMatch *out,
                                size_t capacity);

// 1 when the board's code was read, so the points (and a pose solved from
// them) are in the board's own frame; 0 otherwise. A view whose code was not
// read still serves for calibration, but its pose is only known up to a
// quarter-turn and a whole number of code periods.
int32_t vernier_view_is_absolute(const struct VernierView *view);

// Calibrates a camera from views of the board taken at varied angles (at
// least 2; 10 or more is better), all of the same image size.
//
// - `views`      — array of `count` view pointers.
// - `model`      — `VERNIER_MODEL_*`.
// - `out_camera` — receives the intrinsics (must not be NULL).
// - `out_rms`    — receives the reprojection rms over all points kept, pixels (may be NULL).
// - `out_fits`   — array of `count` entries receiving each view's pose and fit (may be NULL).
//
// Returns 1 on success, 0 on failure (see `vernier_last_error`).
int32_t vernier_calibrate(const struct VernierView *const *views,
                          size_t count,
                          uint32_t model,
                          struct VernierCamera *out_camera,
                          double *out_rms,
                          struct VernierViewFit *out_fits);

// Pose of the board in one view, the camera being known. The pose is in the
// board's own frame only when `vernier_view_is_absolute(view)`.
//
// Returns 1 on success, 0 on failure (see `vernier_last_error`).
int32_t vernier_solve_pnp(const struct VernierCamera *camera,
                          const struct VernierView *view,
                          struct VernierViewFit *out);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  /* VERNIER_H */
