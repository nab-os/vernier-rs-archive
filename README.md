# vernier-rs

A Rust port of the Vernier pose-measurement library.  Given a camera image of a calibrated sinusoidal grid pattern, it recovers the in-plane position and rotation of the sensor with nanometre-level resolution.

Two detection modes are available:

- **Periodic** — recovers position modulo the pattern period (relative, fast).
- **Megarena** — recovers an unambiguous absolute position by decoding the LFSR binary code embedded in the pattern.

The library runs on CPU by default and can dispatch to a Vulkan GPU (any vendor) or a CUDA GPU.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — how the crates fit together.
- [`docs/vernier-course.md`](docs/vernier-course.md) — a step-by-step course, from a raw image to a final pose.

---

## Building

You need a recent Rust toolchain (1.85+).

```bash
# Everything, the CLI with its CPU and Vulkan backends (`gpu`, a default feature)
cargo build --release

# The CLI without the Vulkan backend
cargo build --release --no-default-features -p vernier-cli

# Add the CUDA backend (requires CUDA 12.6 with cuFFT)
cargo build --release --features cuda -p vernier-cabi
cargo build --release --features cuda -p vernier-py
```

After a release build the CLI binary is at `target/release/vernier`.

---

## CLI quick start

Every command reads its pattern from a pattern file, a JSON file `make-pattern` writes (see [Pattern files](#pattern-files)).

```bash
# Run periodic + absolute detection on the reference image, a megarena of
# 9 µm dots and code size 12
./target/release/vernier make-pattern megarena --pitch 9 --code-size 12
./target/release/vernier detect-megarena \
    --image resources/images/megarenaPatternImage_12bits_9um.jpg \
    --pattern megarena-pitch9-code12-512x512-20px.json

# Benchmark the detection pipeline (CPU)
./target/release/vernier bench --size 1024 --iters 50

# Same benchmark on the GPU (Vulkan)
./target/release/vernier bench --size 1024 --iters 50 --backend gpu
```

Expected output for the reference image:

```
Estimated pose: x=-34827.2630 µm, y=-26759.6244 µm, θ=2.998949 rad (quadrant k3=1)
```

(C++ reference on the same image: x=-34827.2626 µm, y=-26759.6172 µm,
α=2.99895 rad — agreement is a few nanometres in position and ~1 µrad in
orientation.)

---

## Camera calibration

`vernier-camera` calibrates a camera, pinhole or fisheye, from views of the coded checkerboard, and finds the pose of the board in a picture (PnP). A view is read through the phase of the board's carriers rather than its corners, so every visible square counts and one view gives hundreds to thousands of sub-pixel correspondences. The code tells which square is which, so the board does not have to be fully in view.

Print a board. Describe it in a pattern file, with the image to render it to: here an A4 sheet at 300 dpi, 2480×3508 pixels, with 5 mm squares, 59 pixels each. Then render it:

```bash
./target/release/vernier make-pattern checkerboard --square 5 --code-size 6 \
    --width 2480 --height 3508 --square-px 59 --output board.json
./target/release/vernier render-pattern --pattern board.json --output board.png
```

The same works for a screen: a megarena for a Pixel 8a, whose pixels are about 0.0591 mm, is `make-pattern megarena --pitch 1.772 --code-size 6 --width 1080 --height 2400 --square-px 30`.

Measure a printed square with a ruler and give that as `--square`: it is the unit poses come out in. A view needs about `3 × (code size + 3)` squares across to read the code, which is why a small code suits a small sheet. Views where the code is not read still count towards the calibration.

Calibrate a webcam (frames come through `ffmpeg`, which must be installed). Hold the board up and change its angle and place between captures; the command keeps 15 distinct views, calibrates and writes `camera.json`:

```bash
./target/release/vernier calibrate-webcam --pattern board.json \
    --video-size 1280x720 --save-frames frames
```

`--input-format`, `--video-size` and `--framerate` pick the camera's mode, on `calibrate-webcam` and `track` alike. Many webcams send their larger sizes only as Motion-JPEG, so ask for `--input-format mjpeg`; `ffmpeg -f v4l2 -list_formats all -i /dev/video0` lists what the camera offers.

Or work from photos:

```bash
./target/release/vernier calibrate --pattern board.json --model fisheye frames/*.png
./target/release/vernier solve-pnp --camera camera.json --pattern board.json photo.png
./target/release/vernier undistort --camera camera.json --output straight.png photo.png
```

Follow the board live: every frame is solved and the pose is traced on a page at `http://localhost:8080/`, next to the camera picture with the board's axes drawn on it (`--csv poses.csv` also logs every pose):

```bash
./target/release/vernier track --camera camera.json --pattern board.json
```

A calibration only holds at the resolution it was made at. `--device` also takes a video file, which is a way to rehearse without a camera.

`track`, `phone` and `calibrate-webcam` take `--backend gpu` to demodulate the frames on a Vulkan GPU; the spectral search, the code and the board's restoration stay on the CPU. The points come out the same to within 1e-6 of a square. `cargo run --release -p vernier-gpu --example local_demod` compares both backends on a 1280×720 view.

A megarena works the same way as the checkerboard: give its pattern file to `calibrate`, `track` and the rest.

### Pattern files

Commands take their pattern only from a pattern file, `--pattern`. `make-pattern checkerboard` or `make-pattern megarena` writes one, with the options of that pattern. Without `--output` it is named after the pattern and its parameters, e.g. `megarena-pitch2-code8-512x512-20px.json`:

```bash
./target/release/vernier make-pattern checkerboard --square 5 --code-size 6 --diamonds
./target/release/vernier make-pattern megarena --pitch 1.772 --code-size 6 \
    --width 1080 --height 2400 --square-px 30
```

```json
{ "pattern": "checkerboard", "square": 5.0, "code_size": 6, "layout": "diamonds", "packing": "one-bit",
  "corner_radius": 0.0, "plain": false, "image": { "width": 512, "height": 512, "square": 12.0 } }
{ "pattern": "megarena", "pitch": 1.772, "code_size": 6,
  "image": { "width": 1080, "height": 2400, "square": 30.0 } }
```

`square` (`pitch`, the dot spacing, for a megarena) is the printed size, in the unit poses come out in. `image` is how `render-pattern` draws it: the image size and the square side (dot pitch) in pixels. `roundtrip-megarena` and `checkerboard-figures` render from it too, and `detect-megarena` and `roundtrip-megarena` read a megarena's `pitch` in micrometres.

`code_size` defaults to 8, a megarena's `pitch` to 2.0, `layout` to `squares` (or `diamonds`), `packing` to `one-bit` (or `two-bits`), `corner_radius` to 0 (up to 0.5 for round corners), `plain` (the uncoded checkerboard) to false, and `image` to 512×512 with 12 px squares, or a 20 px dot pitch.

---

## Language bindings

All bindings except Python share the same native library (`libvernier_cabi`).  Build the library once, then build whichever language wrapper you need.

Periodic and megarena detection are available from every language. Camera calibration, PnP and the camera/image helpers are only in the C and C++ APIs so far.

### C

The C ABI is a plain `cdylib` + header generated by cbindgen.

```bash
# Build the native library
cargo build --release -p vernier-cabi

# Build and run the example
make -C vernier-cabi/examples
./vernier-cabi/examples/detect
```

The public header is at `vernier-cabi/include/vernier.h`.  Link your project with `-lvernier_cabi -L<path/to/target/release>`.

### C++

`vernier-cabi/include/vernier.hpp` wraps the C ABI in RAII classes that throw `std::runtime_error` on failure.  Besides the `Detector`, it covers camera calibration and PnP from the coded checkerboard:

```cpp
#include "vernier.hpp"

vernier::Target target{5.0, 6};                  // 5 mm squares, code size 6
std::vector<vernier::View> views;
for (const auto& path : paths) {                 // PNG, JPEG, BMP, TIFF or PGM
    vernier::Image img = vernier::load_image(path);
    views.push_back(vernier::View::measure(img.pixels.data(), img.width, img.height, target));
}

vernier::Calibration cal = vernier::calibrate(views, vernier::Model::Pinhole);
// cal.camera: fx, fy, cx, cy and OpenCV-ordered distortion; cal.rms in px
cal.camera.save("camera.json", cal.rms, cal.views.size());   // as `vernier calibrate` writes it

vernier::Camera camera = vernier::Camera::load("camera.json");
vernier::ViewFit fit = vernier::solve_pnp(camera, views[0]);
// fit.pose.rvec / fit.pose.tvec: board → camera, as cv::solvePnP returns
```

A printed megarena works the same way; only the target changes, and poses come out in the megarena's own frame (origin at its dot (0, 0)):

```cpp
vernier::Target target = vernier::Target::megarena(2.0, 8);   // 2 mm dot pitch, code size 8
```

`View::points()` gives the raw pixel ↔ board correspondences if you would rather hand them to OpenCV.  Compile with `-std=c++17 -I vernier-cabi/include -L target/release -lvernier_cabi`.

Two complete examples in `vernier-cabi/examples` take either board (`--megarena` for a megarena):

- `calibrate_camera [--megarena] [--fisheye] <size> <code-size> <camera.json> <image>...` calibrates from a list of photos and writes `camera.json`;
- `solve_pnp [--megarena] <camera.json> <size> <code-size> <image>...` reads a known `camera.json` and prints each photo's pose.

`camera.json` is the same file `vernier calibrate` writes and `vernier solve-pnp` reads, so the CLI and the examples can be mixed.  To try them on the fmac checkerboard set and on synthetic megarena views, each against its truth:

```bash
make -C vernier-cabi/examples run-pnp            # compare with resources/fmac-calibration/poses.csv
make -C vernier-cabi/examples run-pnp-megarena   # compare with megarena-views/truth.txt
```

### Python

The Python extension is built with [maturin](https://github.com/PyO3/maturin).

```bash
pip install maturin

# Build a wheel, then install it
cd vernier-py
maturin build --release
pip install ../target/wheels/vernier_py-*.whl
cd ..

# Inside an active virtualenv you can also use the faster dev-install:
# maturin develop --release
```

```python
import numpy as np
import vernier_py

det  = vernier_py.Detector()           # CPU
# det = vernier_py.Detector.cuda()    # CUDA (if built with --features cuda)

img  = np.asarray(..., dtype=np.float32)   # [H, W] in [0, 1]
pose = det.detect_megarena(img, physical_period=9.0, code_size=12,
                           min_frequency=20, max_frequency=500)
print(pose.x, pose.y, pose.theta)
```

See `vernier-py/examples/detect.py` for a full example.

### .NET / C#

The .NET binding uses P/Invoke.  Build the native library first, then run with `dotnet`.

```bash
cargo build --release -p vernier-cabi

LD_LIBRARY_PATH=$PWD/target/release \
    dotnet run --project vernier-dotnet/examples/Detect/Detect.csproj
```

To use the `Vernier` package in your own project, add a `<ProjectReference>` to `vernier-dotnet/Vernier.csproj` and make sure `libvernier_cabi.so` is on `LD_LIBRARY_PATH` at runtime.

```csharp
using Vernier;

using var det = new Detector();
// using var det = Detector.CreateCuda();   // CUDA

Pose p = det.DetectMegarena(pixels, width, height,
    physicalPeriod: 9.0f, codeSize: 12,
    minFrequency: 20, maxFrequency: 500);
```

See `vernier-dotnet/examples/Detect/Program.cs` for a full example.

### MATLAB

The MATLAB binding wraps `calllib` calls in a `+vernier` package.  Point MATLAB at the package directory and set the library path if needed.

```matlab
addpath('path/to/vernier-rs/vernier-matlab')

% Optional: override the default library search path
vernier.Detector.set_lib_path('/path/to/libvernier_cabi.so')

det  = vernier.Detector();
img  = single(imread('pattern.jpg')) / 255;   % [H x W] float32
pose = det.detect_megarena(img, 9.0, 12, 'min_frequency', 20, 'max_frequency', 500);
fprintf('x=%.2f  y=%.2f  theta=%.6f\n', pose.x, pose.y, pose.theta);
```

The library looks for `libvernier_cabi.so` in `vernier-matlab/lib/` by default.  Copy or symlink it there from `target/release/`.  The header is read from `vernier-matlab/include/vernier.h` if present, otherwise from `vernier-cabi/include/vernier.h`.

See `vernier-matlab/examples/detect.m` for a full example.

---

## Workspace layout

| Crate | Purpose |
|---|---|
| `vernier-core` | Shared types: `Pose`, `ComputeBackend`, `BufferLayout` |
| `vernier-cpu` | CPU backend (rustfft + ndarray) |
| `vernier-gpu` | Vulkan GPU backend (vulkano) |
| `vernier-cuda` | CUDA backend (cudarc + cuFFT) |
| `vernier-spectral` | Spectral detection pipeline |
| `vernier-patterns` | Pattern rendering (periodic, megarena) |
| `vernier-pose` | Pose estimation and LFSR absolute decode |
| `vernier-detector` | C++-style detector and layout objects, created by class name or from JSON |
| `vernier-render` | Vulkan rasteriser for patterns (behind `vernier-patterns`' `vulkan` feature) |
| `vernier-camera` | Camera calibration and PnP from the coded checkerboard |
| `vernier-cli` | Command-line tool |
| `vernier-webapp` | Dioxus/WebAssembly pattern generator and spectrum explorer (own workspace) |
| `vernier-cabi` | C ABI shared/static library |
| `vernier-py` | Python bindings (PyO3/maturin) |
| `vernier-dotnet` | .NET bindings (P/Invoke) |
| `vernier-matlab` | MATLAB bindings (calllib) |
