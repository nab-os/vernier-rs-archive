# fmac calibration set

Fifteen views of the coded checkerboard rendered by fmac (the Vernier
project's marker renderer) through a known camera, to calibrate with and
compare against the truth.

- `view_00.png` … `view_14.png`: 512×512, 8-bit grayscale. fmac's thin-lens
  camera: a 6 mm lens on 15 µm pixels, f/2, focused at 300 mm, so the blur
  grows with the distance from focus. The images also include diffraction at
  550 nm and gamma.
- `truth.json`: the true camera, in the format `vernier calibrate` writes:
  pinhole, fx = fy = 400, cx 256.3, cy 255.6, distortion
  (k1, k2, p1, p2, k3) = (−0.12, 0.08, 0.0005, −0.0003, 0).
- `poses.csv`: the true pose of each view, `name,rx,ry,rz,tx,ty,tz`: an OpenCV
  rotation vector and a translation in mm, from board to camera. The board
  origin is its centre.
- `board.png`, `fmac_camera.json`: what fmac rendered from: the board bitmap
  (56×40 squares, 64 px each) and fmac's camera file.

The board has 5 mm squares and an LFSR code of order 6:

```text
vernier calibrate --square 5 --code-size 6 resources/fmac-calibration/view_*.png
```

The views: one nearly square on; four tilted 35° about either board axis;
four tilted 25° about the diagonals, pushed into each corner of the frame;
four tilted 45°, further off; two close views at the left and right edges.
The board is turned about the optical axis differently in each and sits 270
to 380 mm away. Not every view shows the whole board, but the code is read
in all of them.

Calibrating with the command above gave fx 399.989, fy 399.988,
cx 256.305, cy 255.602, distortion (−0.11999, 0.07999, 0.00050, −0.00029,
−0.00016), at 0.013 px reprojection rms over 8430 points.

To make the set again, build the renderer as `vernier-camera/examples/fmac/`
explains, then run:

```text
cargo run --release -p vernier-camera --example fmac_precision -- dataset <dir>
<render-build>/render <dir>/fmac_camera.json <dir>/board.png <dir>/poses.csv <dir>
```
