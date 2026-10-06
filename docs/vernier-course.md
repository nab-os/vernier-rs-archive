# Vernier, from image to pose — a step-by-step course

This is a guided tour of what happens inside `vernier-rs` when you hand it a
photo of a pattern and ask "where is the camera?". It goes in order, from the
moment the image file is opened to the moment a final `(x, y, angle)` is
returned. It is written to be read top to bottom.

There is almost no maths notation here. Where an idea is usually written as an
equation, it is explained in words and then shown as the actual Rust code that
does it. If you can read code, you can follow every step.

---

## Table of contents

- **Chapter 0** — The big idea (why it's called "Vernier")
- **Chapter 1** — The two kinds of pattern
- **Chapter 2** — The map: which crate does what
- **Chapter 3** — Importing the image
- **Chapter 4** — Turning pixels into complex numbers
- **Chapter 5** — The Fourier transform: seeing the image as waves
- **Chapter 6** — Finding the two "carrier" peaks
- **Chapter 7** — Isolating one direction (the band-pass filter)
- **Chapter 8** — Going back to space, and reading the phase
- **Chapter 9** — Unwrapping the phase
- **Chapter 10** — Fitting a plane: where the resolution comes from
- **Chapter 11** — From plane to pose (the "periodic" answer)
- **Chapter 12** — The catch: ambiguity
- **Chapter 13** — The Megarena code: making position absolute
- **Chapter 14** — The LFSR: a ruler where every window is unique
- **Chapter 15** — Reading the code out of the image
- **Chapter 16** — Assembling the final absolute pose
- **Chapter 17** — The whole call chain in one place
- **Chapter 18** — Bonus: measuring tilt (3D)
- **Chapter 19** — Bonus: running on the GPU with CUDA

---

## Chapter 0 — The big idea (why it's called "Vernier")

A **vernier caliper** is that measuring tool with two scales that slide past each
other. The trick is this: the main scale gives you a rough reading, and a second
finer scale — offset just slightly — lets your eye catch exactly which line
"lines up". The combination reads far more precisely than either scale alone.

This library does the same thing, but with light and a printed pattern:

- The **coarse scale** is "which repeat of the pattern am I looking at?"
- The **fine scale** is "how far into that repeat am I, measured as a fraction?"

A printed pattern of tiny repeating dots is photographed by a camera. The
pattern repeats every so many micrometres — that spacing is called the
**period**. By measuring the *phase* of that repeating pattern (think: "how far
between one dot and the next am I, expressed as a fraction of the way across"),
the library can pin down position to roughly **one part in a thousand of a
pixel**. That is the whole point: a cheap camera photo, turned into a
nanometre-scale position measurement.

The output of the whole system is a **Pose**: where the camera is relative to
the pattern, and how it is rotated. Here is that type — the thing every chapter
is building toward (`vernier-core/src/pose.rs:13`):

```rust
pub struct Pose {
    pub x: Real,          // position across the pattern
    pub y: Real,          // position along the pattern
    pub z: Real,          // height (only used in 3D mode)
    pub theta: Real,      // rotation (in radians) — C++ calls this "alpha"
    pub beta: Real,       // tilt (3D only)
    pub gamma: Real,      // tilt (3D only)
    pub pixel_size: Real, // how many micrometres one pixel is worth
    pub is_3d: bool,
}
```

For most of this course we only care about three of those fields: `x`, `y`, and
`theta`. That is a flat 2D measurement: where am I on the pattern, and which way
am I turned.

---

## Chapter 1 — The two kinds of pattern

There are two flavours of pattern, and they correspond to two different
questions:

**1. Periodic pattern** — a plain grid of repeating dots. Measuring it tells you
your position *within one cell* extremely precisely, and your rotation. But it
**cannot** tell you which cell you are in, because every cell looks identical.
Like a ruler with no numbers printed on it: you can see you're exactly 0.3 of the
way between two marks, but you don't know if it's the 5th mark or the 500th.

**2. Megarena pattern** — the same repeating grid, but with a hidden code baked
into it. Some dots are deliberately present or absent in a special sequence.
That sequence acts like the printed numbers on the ruler: it tells you *exactly*
which cell you are in. Combined with the fine within-cell measurement, you get an
unambiguous absolute position.

The library is honest about this split. The periodic detector's own doc comment
says so (`vernier-detector/src/periodic.rs:30`):

```rust
/// Estimates the pose of a periodic pattern with subpixel resolution.
///
/// Recovers `x`, `y` modulo the pattern period and the in-image orientation
/// `theta`. Absolute position is ambiguous modulo the period; use
/// [`MegarenaPatternDetector`] for an unambiguous absolute pose.
```

"Modulo the pattern period" is the formal way of saying "within one cell, but we
don't know which cell." Chapters 3–11 build the periodic answer. Chapters 12–16
add the code that removes the ambiguity.

---

## Chapter 2 — The map: which crate does what

`vernier-rs` is split into small crates (Rust's word for a library module). You
don't need to memorise them, but it helps to know the pipeline's shape. Here is
the journey an image takes, and which crate owns each leg:

| Stage | Crate | What it does |
|---|---|---|
| Load the photo | `vernier-cli` (`imageio.rs`) | Read JPEG/PNG → grey numbers |
| Store pixels | `vernier-core` | Shared types: image buffer, `Pose`, complex numbers |
| Fourier + filtering | `vernier-cpu` | The heavy number-crunching "backend" |
| Orchestrate the spectral steps | `vernier-spectral` | FFT → peak → band-pass → phase → unwrap → plane |
| Turn planes into a pose | `vernier-pose` | The final geometry, both periodic and absolute |
| The friendly object API | `vernier-detector` | `PeriodicPatternDetector`, `MegarenaPatternDetector` |

Two words you'll see a lot:

- **Backend** — the thing that actually runs the maths. There are three: a CPU
  backend (`vernier-cpu`), a Vulkan/GLSL compute-shader backend (`vernier-gpu`),
  and an NVIDIA CUDA + cuFFT backend (`vernier-cuda`). They're interchangeable
  because they all implement the same `ComputeBackend` trait (Rust's version of
  an interface). Every function that does real computation is written as "give
  me *any* backend and I'll run on it." Chapter 19 walks through the CUDA one.
- **Detector** — the high-level object you actually call. It holds the
  configuration and hands work to a backend.

The rest of this course follows the pipeline in the order above.

---

## Chapter 3 — Importing the image

Everything starts with a picture file. The one place in the whole system that
knows how to read `.jpg`/`.png`/`.tiff` is `imageio.rs` — kept deliberately in a
corner so the maths crates never depend on image codecs.

Here is the entire loader (`vernier-cli/src/imageio.rs:26`):

```rust
pub fn load_grayscale(path: &Path) -> Result<LoadedImage, String> {
    let img = image::open(path).map_err(|e| format!("failed to open {}: {e}", path.display()))?;
    let (img_width, img_height) = img.dimensions();
    let (width, height) = (img_width as usize, img_height as usize);

    // Convert to 16-bit luma to preserve depth, then normalize.
    let luma16 = img.to_luma16();
    let data: Vec<f32> = luma16
        .pixels()
        .map(|p| p.0[0] as f32 / u16::MAX as f32)
        .collect();

    Ok(LoadedImage { width, height, data })
}
```

Three things happen here, and each matters:

1. **Open the file.** The `image` crate figures out the format for us.
2. **Convert to grey.** Colour is irrelevant — the pattern is black-and-white
   dots — so a colour photo is collapsed to a single brightness (luminance)
   value per pixel.
3. **Normalise to `0.0 .. 1.0`.** Whatever the original bit depth (8-bit,
   12-bit, 16-bit), every pixel becomes a floating-point number between 0 (black)
   and 1 (white). The `/ u16::MAX` is just dividing by the biggest possible
   value, which squashes everything into that range.

Why normalise? Because the Vernier targets are often photographed at 12-bit
depth. If the pipeline cared about raw pixel values it would behave differently
for 8-bit vs 12-bit images. By always mapping to `0.0 .. 1.0`, the rest of the
pipeline works identically no matter the camera. The comment in the code says
exactly this.

The result is a `LoadedImage`: a width, a height, and one long flat list of
`f32` brightness values in **row-major** order (row 0 left-to-right, then row 1,
and so on).

---

## Chapter 4 — Turning pixels into complex numbers

The next stage — the Fourier transform — works on **complex numbers**, not plain
brightness values. So before anything else, each grey pixel is wrapped into a
complex number.

Don't let "complex number" scare you. A complex number is just a pair: a "real"
part and an "imaginary" part. For an input image we only have brightness, so the
brightness goes in the real part and the imaginary part is set to zero.

The image buffer lives in `vernier-core`, and it has a method that does exactly
this conversion (`vernier-core/src/image.rs:54`):

```rust
pub fn to_complex(&self) -> Vec<Complex32> {
    // each pixel value v becomes Complex32 { re: v, im: 0.0 }
```

So a `GrayImage` of brightnesses becomes a `Vec<Complex32>` — a flat list of
`{ real = brightness, imaginary = 0 }`. Later stages *will* fill in that
imaginary part; that's where the phase information ends up living. But at import
time it's just zero.

Why bother with complex numbers at all? Because the Fourier transform's whole
job is to describe the image as a sum of waves, and a wave has two things: how
big it is, and where its crests fall (its phase). A complex number is the natural
way to carry both of those at once — the "size" and the "shift" bundled into one
value. The next chapter makes this concrete.

---

## Chapter 5 — The Fourier transform: seeing the image as waves

This is the conceptual heart of the whole method, so let's go slowly and without
equations.

**The idea.** Take a photo of a repeating dot grid. Along any horizontal line
through it, the brightness goes light-dark-light-dark-light-dark... That's a
wave. The Fourier transform is a machine that takes the image and answers the
question: *"which waves, at which spacings and directions, do I need to add
together to rebuild this image?"*

For a clean repeating pattern, the answer is beautifully simple: **just a couple
of waves.** One wave running across the pattern, one running along it. Everything
else is nearly zero. So a messy grid of thousands of pixels collapses into a tiny
number of strong "peaks", each peak describing one of those waves.

**The vocabulary.** After the Fourier transform, the image is in the *frequency
domain* (also called the *spectrum*). Instead of "pixel at row 12, column 40" you
now have "bin at frequency 3-across, 0-down". Each bin holds a complex number
whose:

- **size** (magnitude) says *how strong* that wave is in the image, and
- **angle** (phase) says *where the crests of that wave fall*.

That phase is the treasure. It's what will eventually give sub-pixel position.

**The code.** Running the transform is a one-liner in the spectral crate
(`vernier-spectral/src/spectrum.rs:29`):

```rust
pub fn forward<B: ComputeBackend>(backend: &B, buffer: &mut B::Buffer2D) -> Result<()> {
    let mut job = backend.begin()?;
    job.fft2d(buffer)?;   // 2D Fast Fourier Transform, in place
    job.submit()
}
```

`fft2d` is the 2D **F**ast **F**ourier **T**ransform. "Fast" is just the name of
the efficient algorithm; the result is the same wave decomposition described
above. It runs *in place*: the buffer that held the image now holds its spectrum.

On the CPU backend, `fft2d` simply forwards to a well-tested FFT planner — the
library doesn't reinvent the transform itself (`vernier-cpu/src/backend.rs:183`):

```rust
fn fft2d(&mut self, buf: &mut CpuBuffer) -> Result<()> {
    self.backend.planner.borrow_mut().forward(buf);
    Ok(())
}
```

One useful sanity fact, straight from the tests: FFT followed by inverse-FFT
gives you back exactly what you started with (`fft_then_ifft_is_identity`,
`vernier-cpu/src/backend.rs:524`). The transform loses nothing — it's just a
different way of looking at the same data. We'll use that "inverse" direction in
Chapter 8.

After this chapter, hold this picture in your head: **the image is now a mostly-
empty field with two bright spots in it** — one per pattern direction. Finding
those two spots is the next job.

---

## Chapter 6 — Finding the two "carrier" peaks

The two bright spots in the spectrum are called the **carrier peaks** (the
pattern "carries" the position information on those two waves). We need their
locations, because a peak's location tells us the pattern's spacing and
orientation, and a peak's phase tells us position.

The peak search lives in `peak_search` (`vernier-cpu/src/backend.rs:321`). It's
worth reading in pieces.

**Step 1 — work with magnitudes.** We only care *where* the energy is, not its
phase yet, so first convert every complex bin to its size:

```rust
for v in spectrum.as_mut_slice().iter_mut() {
    v.re = (v.re * v.re + v.im * v.im).sqrt();  // magnitude
    v.im = 0.0;
}
```

That `sqrt(re² + im²)` is just the length of the complex number — how strong that
wave is. (This is the one place a formula sneaks in, and it's only Pythagoras:
the length of a little arrow with sides `re` and `im`.)

**Step 2 — ignore the useless bins.** The very centre of the spectrum (called
*DC*) is always huge — it just represents the average brightness of the whole
image, which tells us nothing about position. Very high frequencies are usually
noise. So an "annulus" (a ring/donut shape) mask blanks out everything too close
to the centre or too far out, keeping only the believable middle band:

```rust
Self::annulus_mask(&mut spectrum, min_frequency, max_frequency);
```

The `min_frequency` and `max_frequency` are configuration values (defaults are 20
and 500 — see `SpectralConfig` in `vernier-detector/src/lib.rs:139`). Optionally
a gentle Gaussian blur is applied first so noise doesn't create a fake sharp
spike:

```rust
if smoothing_sigma > 0.0 {
    self.gaussian_blur_2d(&mut spectrum, smoothing_sigma).unwrap();
}
```

**Step 3 — find the strongest bin (first peak).** Now just pick the brightest
remaining bin. There's a subtlety: because the spectrum is symmetric (every wave
shows up twice, mirror-imaged through the centre), we only search *half* of it —
the "half-plane" — so we don't count the same wave twice:

```rust
let (cx1, cy1) = {
    if let Some(a) = Self::halfplane_argmax(&spectrum, width, height) {
        a
    } else {
        return Ok(None);   // no peak at all → nothing detected
    }
};
```

`argmax` just means "the location of the maximum". If nothing is found, the whole
detection returns "no pattern here" rather than crashing.

**Step 4 — find the second peak.** The second pattern direction is roughly
perpendicular to the first and at a similar distance from the centre. So the
code carves out a ring at that same radius and, within it, avoids the angle where
the first peak already sits:

```rust
let distance = (sfx1 * sfx1 + sfy1 * sfy1).sqrt();   // how far peak 1 is from centre
let center_angle = sfy1.atan2(sfx1);                 // the angle of peak 1
let half_width = (3.0 * sigma).atan2(distance);      // a small angular guard around it
// ...keep only bins between half and double that radius...
let (cx2, cy2) = Self::halfplane_argmax_angular_excl(
    &spectrum, width, height, center_angle, half_width
)?;
```

**Step 5 — return the two locations in a stable order.** Finally it decides which
peak to call "direction 1" and which "direction 2" (sorting by their x
coordinate, so the answer is deterministic) and returns the four numbers — two
`(x, y)` peak coordinates:

```rust
let (d1x, d1y, d2x, d2y) = if sfx1 >= sfx2 {
    (cx1, cy1, cx2, cy2)
} else {
    (cx2, cy2, cx1, cy1)
};
```

At the end of this chapter we know **where** the two carrier waves are in the
spectrum. Next we isolate them one at a time.

---

## Chapter 7 — Isolating one direction (the band-pass filter)

We now process each direction separately. The idea: keep only the neighbourhood
around one peak, and erase everything else in the spectrum. That leaves a
spectrum that describes *only* the wave running in that one direction — a single
clean sinusoid.

This is a **band-pass filter**: it passes a band of frequencies around the peak
and blocks the rest. It's done by multiplying the whole spectrum by a soft bump
(a Gaussian) centred on the peak — 1.0 right at the peak, fading to 0.0 as you
move away.

Here's the core of it (`vernier-cpu/src/backend.rs:275`):

```rust
fn bandpass_filter(&mut self, buf: &mut CpuBuffer, cx: usize, cy: usize, sigma: Real) -> Result<()> {
    // Precompute a soft falloff along x and along y, centred on the peak (cx, cy):
    let gain_x: Vec<f32> = (0..width).map(|fx| {
        let dx = circular_delta(fx, cx, width);
        (dx * dx * inv_denom).exp() as f32   // 1.0 at the peak, fading outward
    }).collect();
    // ...same for gain_y...

    // Multiply every bin by its falloff. Bins far from the peak get ~0.
    for fy in 0..height {
        for (fx, v) in row.iter_mut().enumerate() {
            let gain = gain_y[fy] * gain_x[fx];
            v.re *= gain;
            v.im *= gain;
        }
    }
}
```

`sigma` controls how wide the bump is — how much of the neighbourhood you keep.
The `circular_delta` helper handles the fact that the spectrum wraps around at
the edges (frequency 0 and the maximum frequency are neighbours).

In the orchestration code, this is done twice — once per direction — starting
from two copies of the same spectrum (`vernier-spectral/src/spectrum.rs:109`):

```rust
let mut spec1 = job.copy_buffer(&buffer)?;
let mut spec2 = job.copy_buffer(&buffer)?;
job.bandpass_from_peaks(&mut spec1, &peaks_buffer, 0, sigma)?;  // keep only peak 0
job.bandpass_from_peaks(&mut spec2, &peaks_buffer, 1, sigma)?;  // keep only peak 1
```

Now `spec1` contains only the first direction's wave, and `spec2` only the
second. Each is a pure single sinusoid dressed up as a spectrum.

---

## Chapter 8 — Going back to space, and reading the phase

We have a spectrum containing just one wave. Now we do the Fourier transform
*backwards* (the "inverse FFT") to turn it back into an image — but this time
the image is a single clean sine wave sweeping across the frame, with no dots and
no noise, just that one direction's ripple.

```rust
job.ifft2d(&mut spec1)?;
job.ifft2d(&mut spec2)?;
```

Here's the payoff. Remember every pixel is a complex number. After this inverse
transform, at each pixel the complex number's **angle** (its phase) tells you
*exactly where in the wave's cycle that pixel sits* — the fraction of the way
between one crest and the next.

That phase is the fine "vernier" reading. If you're a quarter of the way through
a cycle, the phase is a quarter turn; halfway, a half turn; and so on. And phase
can be measured to a tiny fraction of a cycle, which is why the method reaches
sub-pixel precision.

Getting the phase out of a pixel is just the angle of its complex number
(`arg`). You can see it done directly in the `unwrap_and_fit` helper, where the
pipeline needs the raw phase map (`vernier-spectral/src/spectrum.rs:80`):

```rust
// Per-pixel arg of the band-passed IFFT — the raw, still-wrapped phase.
let mut phase: Vec<Real> = backend.download(spec)?.iter().map(|c| c.arg()).collect();
```

`c.arg()` is "the angle of this complex number." One per pixel gives a whole map
of phases: at every point in the image, how far through the wave cycle we are.

That map is still *wrapped* — the very next line of `unwrap_and_fit` calls
`quarters_unwrap_phase` on it. That's the subject of the next chapter.

---

## Chapter 9 — Unwrapping the phase

Phase is an angle, so it naturally lives in a range of one full turn — the code
uses `-π to +π` (think of it as `-180°` to `+180°`). The moment the true phase
climbs past `+180°`, the measured value snaps back to `-180°` and keeps climbing.
It's exactly like a clock's second hand: after 59 seconds it jumps back to 0, not
on to 60.

But real position is *continuous*. If the camera slides across many periods, the
true phase keeps increasing smoothly, even though our measurement keeps snapping
back. Those snap-backs are false cliffs in the data, and they would wreck the
next step. **Unwrapping** removes them: whenever the measurement suddenly jumps by
almost a full turn, we add or subtract a full turn to cancel the jump and restore
the smooth ramp.

The 1D version shows the whole idea in a few lines
(`vernier-spectral/src/unwrap.rs:11`):

```rust
pub fn unwrap_1d(phases: &mut [Real]) {
    let mut offset: Real = 0.0;
    let mut prev = phases[0];
    for p in phases.iter_mut().skip(1) {
        let raw = *p;
        let mut delta = raw - prev;
        // Reduce the step into (-π, π]:
        while delta > PI  { delta -= TAU; offset -= TAU; }   // jumped up a turn → subtract one
        while delta <= -PI { delta += TAU; offset += TAU; }  // jumped down a turn → add one
        prev = raw;
        *p = raw + offset;   // apply the running correction
    }
}
```

`TAU` is one full turn (2π). The logic: walk along the values; whenever two
neighbours differ by more than half a turn, that's a false cliff, so nudge a
running `offset` by a whole turn to smooth it out. `PI` is half a turn — the
natural threshold, because a real signal never genuinely jumps by more than half
a turn between adjacent pixels, so any bigger jump must be a wrap.

An image is 2D, so this is done in both directions. The straightforward version
unwraps each row, then reconciles the rows using the first column
(`vernier-spectral/src/planefit.rs:72`):

```rust
pub fn unwrap_2d(wrapped: &[Real], width: usize, height: usize) -> Vec<Real> {
    let mut phase = wrapped.to_vec();
    // Unwrap each row in place.
    for r in 0..height {
        let start = r * width;
        unwrap_1d(&mut phase[start..start + width]);
    }
    // Then unwrap down the first column and shift each row to match.
    // ...
    phase
}
```

(There's also a fancier version, `quarters_unwrap_phase`, that unwraps outward
from the centre in four quadrants — more robust on curved phase maps, and it
matches the original C++ library exactly. Same idea, more careful bookkeeping.)

After unwrapping, we have a **continuous phase surface**: a smooth ramp across
the whole image with no cliffs. This surface is what the next chapter turns into
a precise measurement.

---

## Chapter 10 — Fitting a plane: where the resolution comes from

Here is the single most important idea for *precision*, so the file even says so
in its opening line (`vernier-spectral/src/planefit.rs:1`):

> *Least-squares phase-plane fitting — where the resolution comes from.*

We now have a phase value at every one of, say, a million pixels. For a flat
pattern, that phase surface is essentially a tilted flat sheet — a **plane**. It
rises steadily as you move across the pattern (because you're marching through
the wave), and its steepness and direction encode the pattern's spacing and
angle.

Naively, you could read the phase at the centre pixel and call it a day. But one
pixel is noisy. Instead, the code fits a plane through **all** the phase values at
once, letting a million measurements vote on the answer. Averaging over that many
points is exactly what beats the noise down to the ~1/1000-pixel level. The doc
comment spells it out (`vernier-spectral/src/planefit.rs:6`):

> *Rather than read one value, the method fits a plane `φ(i,j) = a·i + b·j + c`
> across the whole (unwrapped) phase map by least squares. That averages the
> redundant phase over every pixel, which is what reaches ~1/1000-pixel
> resolution.*

A plane is described by just three numbers, and this is the whole vocabulary you
need going forward (`vernier-spectral/src/planefit.rs:22`):

```rust
pub struct PhasePlane {
    /// Phase gradient along the column (x) axis, radians per pixel.
    pub a: Real,
    /// Phase gradient along the row (y) axis, radians per pixel.
    pub b: Real,
    /// Phase at the image center, radians. This is the high-resolution phase.
    pub c: Real,
}
```

Read those three in plain words:

- **`a`** — how fast the phase rises as you move sideways (across columns).
- **`b`** — how fast it rises as you move down (across rows).
- **`c`** — the phase right at the centre of the image.

Together, `a` and `b` give the **tilt** of the sheet, which tells you the
pattern's **orientation** and **spacing**. And `c` — the centre phase — is the
**fine position reading**, the vernier fraction. Those three numbers are the
entire output of all the spectral work so far.

**How the fit is computed.** "Least squares" means: find the `a, b, c` that make
the plane pass as close as possible to all the measured phases, with the total
squared error as small as possible. It boils down to accumulating some running
sums over every pixel and solving a tiny 3×3 system of equations
(`vernier-spectral/src/planefit.rs:127`):

```rust
for r in row_off..(height - row_off) {
    let j = (r - row_off) as Real - center_y;
    for col in col_off..(width - col_off) {
        let i = (col - col_off) as Real - center_x;
        let p = phase[r * width + col];
        sii += i * i;  sjj += j * j;  sij += i * j;   // sums for the tilt
        si += i;  sj += j;  sn += 1.0;
        spi += p * i;  spj += p * j;  sp += p;         // sums linking phase to position
    }
}
// Solve the 3x3 system for [a, b, c]:
let (a, b, c) = solve_3x3(
    [[sii, sij, si], [sij, sjj, sj], [si, sj, sn]],
    [spi, spj, sp],
);
PhasePlane { a, b, c }
```

You don't need to trace the arithmetic. The important part is conceptual: *every
pixel contributes one vote to those running sums, and the final three numbers are
the best compromise across all of them.*

Notice `row_off`/`col_off`: the fit ignores a border around the edge of the
image (the `crop_factor`, default 0.5, keeps the central half). Edges get
distorted by the band-pass filtering, so trusting only the clean middle gives a
better answer.

> **Note on the "two ways" in the codebase.** There are two routes to the plane.
> The one the detection path actually uses (`analyze_two` → `unwrap_and_fit` →
> `fit_plane_to_unwrapped`, `vernier-spectral/src/planefit.rs:104`) is exactly
> the unwrap-then-least-squares fit just described. Its running sums are
> accumulated in `f64` (see the comment at `planefit.rs:121`): `sii` reaches
> ~1e9 on a 512² crop, well past `f32`'s 24-bit precision, so the fit would drift
> if it stayed in `f32`. The alternative (`spectral_plane_fit_two`,
> `vernier-cpu/src/backend.rs:405`, used by the single-direction helper
> `analyze_direction`) computes `a` and `b` directly from the peak's position in
> the spectrum and `c` from a weighted average, skipping the explicit unwrap.
> Both produce the same `PhasePlane`. Wherever this course says "the plane",
> that's the struct above.

Each direction gets its own plane. So after this chapter we hold **two
`PhasePlane`s**, `plane1` and `plane2`, one per pattern direction. That's
everything the geometry needs.

---

## Chapter 11 — From plane to pose (the "periodic" answer)

Now we convert those two planes into an actual position and rotation. This is
short, because all the hard work is already done. It lives in `vernier-pose`
(`vernier-pose/src/periodic.rs:25`):

```rust
pub fn estimate(plane1: &PhasePlane, plane2: &PhasePlane, calib: &Calibration) -> Pose {
    // Sub-period displacements from the center phases (mod one period).
    let x = (plane1.c / TAU) * calib.period;
    let y = (plane2.c / TAU) * calib.period;

    // Orientation from the first plane's gradients (high-resolution angle).
    let theta = plane1.orientation();

    Pose::new(x, y, theta)
}
```

Let's read it line by line, because this is where abstract phase becomes real
micrometres.

**Position.** `plane1.c` is the centre phase for direction 1 — a fraction of a
full turn. Dividing by `TAU` (one full turn) turns it into a plain fraction
between 0 and 1: "how far through one cell am I?". Multiply by `calib.period`
(the real-world size of one cell, e.g. 9 micrometres) and you get a real
distance. Same for `y` using the second plane. This is literally "fraction of the
way through a cell × size of a cell = distance."

The `Calibration` is just the physical scale that makes this conversion possible
(`vernier-pose/src/lib.rs:30`):

```rust
pub struct Calibration {
    pub period: Real,        // physical size of one pattern cell (e.g. micrometres)
    pub image_width: usize,
    pub image_height: usize,
}
```

**Orientation.** The plane's tilt direction *is* the pattern's rotation. It's
recovered from the two gradients `a` and `b` (`vernier-spectral/src/planefit.rs:36`):

```rust
pub fn orientation(&self) -> Real {
    self.b.atan2(self.a)
}
```

`atan2(b, a)` is the standard "angle of the arrow pointing in direction
`(a, b)`". If the phase rises purely sideways, the angle is 0; purely downward, a
quarter turn; and so on. That's the pattern's in-image angle.

And that's the **periodic pose**. We now know, very precisely, where the camera
sits *inside one cell*, and how it's rotated.

But — and it's a big but — we do **not** know which cell. That's the ambiguity,
and it's the whole reason for the second half of this course.

---

## Chapter 12 — The catch: ambiguity

The periodic answer has two ambiguities, both baked into the physics:

1. **Position repeats every period.** Because every cell is identical, `x` is
   only known "modulo the period." If cells are 9 µm apart, the answer could be
   the true position, or that plus 9 µm, or plus 18 µm, or minus 90 µm. Our
   fine measurement can't tell them apart. (Ruler with no numbers.)

2. **Rotation repeats every quarter turn.** A square grid looks identical when
   you rotate it 90°. So the measured angle is only known within a quarter turn —
   we can't tell "0°" from "90°" from "180°".

`vernier-pose` names these missing pieces explicitly at the top of the periodic
module (`vernier-pose/src/periodic.rs:1`):

```text
- x = (φ1 / 2π)·λ + k1·λ     (sub-period part + integer period order)
- y = (φ2 / 2π)·λ + k2·λ
- α = atan2(b1, a1) + k3·(π/2)
```

In words: the position is our fine fraction (`φ/2π · λ`) **plus** some unknown
whole number of periods (`k1·λ`). The angle is our fine angle **plus** some
unknown number of quarter-turns (`k3·π/2`). The fine part we already have. The
whole-number parts — `k1`, `k2`, `k3` — are what's missing.

To find those integers, we need extra information printed into the pattern
itself. That's the Megarena code.

---

## Chapter 13 — The Megarena code: making position absolute

A **Megarena** pattern is the same repeating dot grid, but with a secret: whether
each dot is *present or absent* spells out a code. Read that code and you learn
exactly which cell you're in — the missing `k1`, `k2`, `k3` from the last chapter.

The strategy is the caliper again, stated cleanly in the absolute module's
opening (`vernier-pose/src/absolute.rs:1`):

> *Megarena patterns embed an LFSR position code, so a coarse decode tells you
> which period you're in (`k1, k2`) and which quadrant (`k3`); the fine phase
> then refines within it. Coarse + fine gives an unambiguous `(x, y, θ)`.*

- **Coarse:** read the printed code → which cell (the whole-number part).
- **Fine:** the phase-plane measurement from Chapters 3–11 → where inside that
  cell (the fractional part).

Add them and the ambiguity is gone. The remaining chapters explain the coarse
half: what the code *is* (Chapter 14), how it's read off the image (Chapter 15),
and how it's combined with the fine part (Chapter 16).

---

## Chapter 14 — The LFSR: a ruler where every window is unique

How do you print numbers onto a grid using only "dot present / dot absent"? You
use a very clever sequence of bits called a **maximal-length LFSR sequence**. Its
one magic property is stated at the top of the file
(`vernier-patterns/src/lfsr.rs:1`):

> *A maximal LFSR of order `n` produces `2ⁿ − 1` bits where every window of `n`
> consecutive bits is unique. That's what makes the position absolute: any local
> view of `n` bits tells you where you are in the sequence.*

Read that twice, because it's the whole trick. Imagine a very long ribbon of 0s
and 1s. It has a special property: **any `n` bits in a row you look at appear
nowhere else in the ribbon.** So if you can see any `n` consecutive bits, you can
look them up and instantly know your exact position along the ribbon — even
though you can only see a tiny local stretch of it.

That's the "numbers on the ruler." The pattern prints this ribbon along each
axis using present/absent dots. Order `n = 12` (the default) gives a ribbon of
`2¹² − 1 = 4095` bits — enough unique positions to cover a big pattern.

**Generating the ribbon.** An LFSR (Linear-Feedback Shift Register) is a tiny
machine that produces the sequence one bit at a time by shifting and XOR-ing
(`vernier-patterns/src/lfsr.rs:36`):

```rust
let mut code = code_count; // start from the all-ones state
bits.push(1u8);
for _ in 1..code_count {
    let nb = cpp_next_bit(order, code);       // next bit from a fixed XOR rule
    code = (code * 2) % code_max + nb as u32; // shift left, drop in the new bit
    bits.push(nb);
}
```

You don't need the bit-twiddling. What matters: this deterministically produces
*the* ribbon, and the decoder later regenerates the same ribbon to look positions
up in it.

**Looking up a position.** Given `n` bits we read from the image, finding their
position is a dictionary lookup. The code pre-builds a map from "n-bit window" →
"position", so each lookup is instant (`vernier-patterns/src/lfsr.rs:115`):

```rust
pub fn locate(&self, window: &[u8]) -> Option<usize> {
    // pack the window's bits into a number, then look it up
    let mut word = 0u32;
    for &b in window {
        word = (word << 1) | (b & 1) as u32;
    }
    self.map.get(&word).copied()   // Some(position) or None if the bits are garbage
}
```

If the bits we read are corrupt (a smudge on the pattern, say), the window won't
exist in the map and `locate` returns `None` — the decode fails cleanly rather
than lying.

So: **read `n` bits along each axis → look each up → get the cell index for that
axis.** Those indices are the `k1` and `k2` we were missing. All that's left is
actually reading the bits out of a photo, which is the fiddliest part.

---

## Chapter 15 — Reading the code out of the image

This is the messiest stage in the whole system, because reading "is this dot
present or absent?" from a real, slightly-blurry, slightly-tilted photo is hard.
The code for it is `vernier-pose/src/absolute.rs`, and we'll stay at the level of
*what* each step does rather than every line.

The reason this even works is that Chapters 3–11 already gave us the *phase* at
every pixel. Phase tells us where each pixel sits relative to the dot grid — so
we know, for any pixel, which cell it belongs to and whether it's near a dot
centre or in the gap between dots. That's what lets us pool pixels correctly.

**Step 1 — sort pixels into cells (`accumulate_cell_pools`,
`vernier-pose/src/absolute.rs:160`).** Using the two phase maps, every pixel is
assigned to a grid cell `(cell_x, cell_y)` and labelled as either "near a dot
centre" (white pool) or "out in the background" (background pool):

```rust
let fx = phase_x[flat_index] / TAU;   // how many whole cells across, as a decimal
let fy = phase_y[flat_index] / TAU;
let cell_x = fx.round();              // nearest whole cell
let cell_y = fy.round();
let rx = (fx - cell_x).abs();         // how far from the dot centre (0 = dead centre)
let ry = (fy - cell_y).abs();
// ...
if rx < white_r && ry < white_r && period_consistent {
    // this pixel is inside a dot → add its brightness to that cell's WHITE pool
} else if rx > bg_r || ry > bg_r {
    // this pixel is out in the gap → add it to the BACKGROUND pool
}
```

Now each cell has an average "dot brightness" and an average "background
brightness". A present dot makes the white pool bright; an absent dot leaves it
as dark as the background.

**Step 2 — find the orientation (`detect_coding_orientation`,
`vernier-pose/src/absolute.rs:277`).** Remember the 90° rotation ambiguity? The
Megarena pattern solves it with a deliberate "missing corner" in a repeating 3×3
super-cell. By folding all cells into a 3×3 average and template-matching against
the 36 possible arrangements (3 rows × 3 columns × 4 rotations), the code finds
which way is up. This yields the **quadrant** `k3` — the missing rotation integer.

**Step 3 — read the bits (`decode_axis_bits`,
`vernier-pose/src/absolute.rs:364`).** For each cell along a coding line, compare
its dot brightness against two references — the background, and known-white
neighbours. Whichever it's closer to decides the bit:

```rust
let bit = if (mean_coding - mean_back).abs() < (mean_white - mean_coding).abs() {
    0u8   // dot looks like background → absent → bit 0
} else {
    1u8   // dot looks white → present → bit 1
};
```

Notice there's no fixed brightness threshold — it's always "closer to white, or
closer to background?" That's robust to lighting changes across the image, which
a fixed threshold wouldn't be.

**Step 4 — turn bits into positions (`extract_code`,
`vernier-pose/src/absolute.rs:520`).** Collect a run of bits along each axis and
look it up in the LFSR ribbon from Chapter 14:

```rust
let window: Vec<u8> = (0..window_size).map(|j| bits[&(first_triple + j as i64)]).collect();
if widx.locate(&window).is_some() {
    return Some((window, first_triple));   // found our place in the ribbon
}
```

The result of all this is an `ExtractedCode`: the decoded bit windows for both
axes, the quadrant `k3`, and some bookkeeping about direction (the `msb`/period-
shift fields) needed to line the code up with the fine phase. If the pattern is
occluded or too small, extraction returns `None` and the detector reports "no
pattern found" rather than a wrong answer.

The honest takeaway: this chapter is real-world engineering — handling missing
dots, edge cells with too few pixels, and the exact conventions of the original
C++ library (which the comments constantly reference for parity). But the *shape*
is simple: **pool pixels into cells → decide present/absent → look up the code.**

---

## Chapter 16 — Assembling the final absolute pose

Now we combine the two halves — coarse (which cell) and fine (where in the cell)
— into one unambiguous answer. The clean statement of the combination is
`assemble` (`vernier-pose/src/absolute.rs:651`):

```rust
pub fn assemble(fine: &Pose, orders: CoarseOrders, calib: &Calibration) -> Pose {
    let x = orders.k1 as Real * calib.period + fine.x;
    let y = orders.k2 as Real * calib.period + fine.y;
    let theta = fine.theta + orders.k3 as Real * (PI / 2.0);
    Pose::new(x, y, theta)
}
```

This is the caliper reading, made literal:

- `orders.k1 * period` — **coarse:** which cell (from the code) times the size of
  a cell = roughly where you are.
- `+ fine.x` — **fine:** the exact fraction within that cell (from the phase).
- `theta = fine.theta + k3 * quarter-turn` — the fine angle plus the whole
  quarter-turns the code told us about.

Coarse gets you to the right cell; fine places you precisely inside it. The
ambiguity from Chapter 12 is gone.

The real end-to-end function that a detector calls is `solve_megarena`
(`vernier-pose/src/absolute.rs:702`). It ties the whole absolute path together:
fine pose from the two planes, code extracted from the image, checked, and
combined:

```rust
pub fn solve_megarena(
    detection: &Detection, intensity: &[Real], calib: &Calibration, code_size: u32,
) -> core::result::Result<Pose, MegarenaError> {
    let fine = periodic::estimate(&detection.dir1.plane, &detection.dir2.plane, calib);

    let code = extract_code(detection, intensity, code_size).ok_or(MegarenaError::CodeExtraction)?;

    let decoder = MegarenaDecoder::new(code_size, code.x_window.clone(), code.y_window.clone(), code.k3)
        .ok_or(MegarenaError::UnsupportedCodeSize(code_size))?;

    if decoder.decode().is_none() {
        return Err(MegarenaError::DecodeFailed);
    }
    // ...combine fine phase with the decoded period-shifts into absolute x, y...
    Ok(Pose::new(x, y, fine.theta))
}
```

The three ways it can fail (`MegarenaError`, `vernier-pose/src/absolute.rs:673`)
are all "couldn't read the code," never "produced a wrong number": the code
region was unreadable, the code size isn't supported, or the decoded bits didn't
localise in the ribbon.

That `Pose` is the final result — the answer the whole library exists to
produce.

---

## Chapter 17 — The whole call chain in one place

Let's zoom all the way out and watch a single image travel through the system,
top to bottom, naming the real functions.

**The friendly entry point.** You create a detector and call `compute` on it. For
the absolute case that's `MegarenaPatternDetector::compute`
(`vernier-detector/src/megarena.rs:65`):

```rust
fn compute(&mut self, image: &GrayImage) -> Result<()> {
    let detection = run_detection(&self.backend, image, &self.config)?;   // spectral half
    let calib = Calibration::new(self.config.physical_period as Real, image.width(), image.height());
    self.pose = absolute::solve_megarena(&detection, image.as_slice(), &calib, self.code_size).ok();  // decode half
    Ok(())
}
```

Two calls: `run_detection` (Chapters 5–10) and `solve_megarena` (Chapters 13–16).

**The spectral half.** `run_detection` forwards to `analyze_two`, whose job
block is the clearest single view of the spectral pipeline
(`vernier-spectral/src/spectrum.rs:91`):

```rust
let mut job = backend.begin()?;
let mut buffer = job.upload(data, layout)?;          // Ch.4  pixels → complex buffer
job.fft2d(&mut buffer)?;                              // Ch.5  image → spectrum
let peaks_buffer = job.peak_search(&mut buffer, ...)? // Ch.6  find the two carriers
    .ok_or_else(|| VernierError::Backend("no carrier peaks found in spectrum".into()))?;
let mut spec1 = job.copy_buffer(&buffer)?;
let mut spec2 = job.copy_buffer(&buffer)?;
job.bandpass_from_peaks(&mut spec1, &peaks_buffer, 0, sigma)?;   // Ch.7  isolate dir 1
job.bandpass_from_peaks(&mut spec2, &peaks_buffer, 1, sigma)?;   // Ch.7  isolate dir 2
job.ifft2d(&mut spec1)?;                             // Ch.8  back to space (phase)
job.ifft2d(&mut spec2)?;
job.submit()?;
// ...then, host-side, unwrap each phase map and fit its plane in f64:
let (plane1, phase1) = unwrap_and_fit(backend, &spec1, width, height)?;  // Ch.9-10
let (plane2, phase2) = unwrap_and_fit(backend, &spec2, width, height)?;
```

Out of this come the two `PhasePlane`s wrapped in a `Detection` struct.

**The pose.** For a plain periodic pattern the story ends at
`periodic::estimate` (Chapter 11): two planes in, a `Pose` out, ambiguous modulo
the period. For a Megarena pattern, `solve_megarena` additionally reads the code
and calls `assemble` (Chapter 16) to fix the ambiguity, giving an absolute
`Pose`.

**The one-paragraph summary of the entire library:**

> Load the photo and normalise it (Ch.3). Wrap pixels as complex numbers (Ch.4).
> Fourier-transform to see the image as a few waves (Ch.5). Find the two carrier
> peaks (Ch.6) and isolate each with a band-pass filter (Ch.7). Inverse-transform
> to recover each direction's phase (Ch.8), unwrap it (Ch.9), and fit a plane
> through all of it to get a super-precise fine reading (Ch.10). Convert the two
> planes into a within-cell position and angle (Ch.11). If the pattern carries a
> Megarena code, read the present/absent dots (Ch.15), look them up in the LFSR
> ribbon to learn which cell you're in (Ch.14), and add that to the fine reading
> (Ch.16). Return the `Pose`.

---

## Chapter 18 — Bonus: measuring tilt (3D)

Everything above assumed the pattern faces the camera flat. If the surface is
*tilted*, the pattern looks squished in the photo, and the phase surface from
Chapter 10 is no longer a perfectly flat sheet — it curves slightly. That extra
information lets the library estimate two tilt angles, `beta` and `gamma` (the
extra fields in `Pose`).

The catch: a tilt and its mirror-image tilt produce almost the same squish, so
there are **four** candidate 3D poses that all fit. `all_3d_poses` returns all
four (`vernier-pose/src/periodic.rs:41`), and then `compute_phase_gradients`
(`vernier-pose/src/periodic.rs:89`) inspects the faint *curvature* of the phase
surface to pick the correct signs and choose among them. You can see the pick
happen in the periodic detector (`vernier-detector/src/periodic.rs:113`):

```rust
fn get_3d_pose(&self, id: i32) -> Pose {
    let mut pose = /* the first of the four candidates */;
    if let (Some((p1, p2)), Some((m1, m2, w, h))) = (&self.planes, &self.measured) {
        let (beta_sign, gamma_sign) =
            periodic::compute_phase_gradients(m1, m2, *w, *h, p1, p2, 0.5);
        pose.beta  = pose.beta.abs()  * beta_sign as Real;   // fix the tilt signs
        pose.gamma = pose.gamma.abs() * gamma_sign as Real;
    }
    pose
}
```

This is genuinely advanced and optional — for flat measurements you never touch
it. But it shows the same philosophy one more time: the plane fit throws away the
curvature to get a clean fine reading, and when 3D is needed, that discarded
curvature is measured separately to resolve the remaining ambiguity. Coarse and
fine, again and again — the vernier idea, all the way down.

---

## Chapter 19 — Bonus: running on the GPU with CUDA

Everything so far quietly assumed the maths ran on the CPU (`vernier-cpu`). But
none of Chapters 5–10 actually *named* the CPU. They were written against the
`ComputeBackend`/`ComputeJob` traits (`vernier-core/src/backend.rs:111`), so the
whole spectral pipeline is generic: `analyze_two<B: ComputeBackend>` runs on
*any* backend you hand it. `vernier-cuda` is one such backend — it runs the exact
same pipeline on an NVIDIA GPU. Nothing in the algorithm changes; only *where the
arithmetic happens* does.

**The contract.** `CudaBackend` implements `ComputeBackend`
(`vernier-cuda/src/backend.rs:123`) and `CudaJob` implements `ComputeJob`. Every
primitive the pipeline calls — `fft2d`, `peak_search`, `bandpass_from_peaks`,
`ifft2d`, `extract_phase` — has a CUDA version with an identical signature. Swap
the backend, and Chapters 5–10 replay on the GPU unchanged.

**Setup: compile kernels once, at startup.** The GPU can't run Rust; it runs
kernels written in CUDA C. Those kernels live as one big string, `KERNEL_SRC`
(`vernier-cuda/src/kernels.rs:6`), and are compiled *at runtime* by NVRTC (NVIDIA's
just-in-time compiler) into PTX, the GPU's assembly (`vernier-cuda/src/backend.rs:62`):

```rust
let ptx = compile_ptx(KERNEL_SRC).map_err(|e| VernierError::Backend(e.to_string()))?;
// ...then load every named kernel and cache its callable handle:
dev.load_ptx(ptx, "vernier", kernel_names)?;
```

All 14 kernels (`scale`, `magnitude_inplace`, `filter_annulus`,
`gaussian_blur_h/v`, `argmax_local/global`, `band_angular_filter`, `peak_order`,
`bandpass`, `extract_phase`, the two plane-fit reductions, …) are compiled once
into a `CudaContext` (`vernier-cuda/src/backend.rs:58`) and reused for every frame.

**Memory: host and device are separate worlds.** The GPU has its own memory. A
`CudaBuffer` is just a slab of device memory (`CudaSlice<f32>`) plus a width and
height. Getting the image onto the GPU is an explicit copy — and here's the neat
part: a `Complex32` is already `{ re, im }` as two `f32`s, so the whole slice is
*reinterpreted* as a flat `float` array (no conversion) and shipped across
(`vernier-cuda/src/backend.rs:131`):

```rust
let floats: &[f32] = bytemuck::cast_slice(data);       // Complex32[] viewed as f32[]
let dev_slice = self.ctx.dev.htod_sync_copy(floats)?;  // host -> device copy
```

`download` does the reverse (`dtoh_sync_copy`, then reinterpret back to
`Complex32`). These two copies are the only bridge between CPU and GPU; everything
between them stays on the device.

**The FFT: borrow NVIDIA's, don't rewrite it.** Chapter 5's `fft2d` on CPU
forwarded to an FFT planner; on CUDA it forwards to **cuFFT**, NVIDIA's FFT
library, called through a tiny raw-FFI shim (`vernier-cuda/src/backend.rs:22`).
`cufft_2d` (`vernier-cuda/src/backend.rs:171`) looks up a cached complex-to-complex
2D plan for this image size (building one the first time), then executes it *in
place* on the device pointer:

```rust
let mut plans = self.ctx.fft_plans.lock().unwrap();   // plan cache, keyed by (height, width)
// ...cufftMakePlan2d(...) once per size, then reused every frame...
cufftExecC2C(handle, ptr, ptr, direction)             // FORWARD or INVERSE, in place
```

One wrinkle: cuFFT's inverse transform comes out unnormalised (scaled up by
`width·height`), so `ifft2d` chases it with a one-line `scale` kernel that
multiplies every value by `1/(w·h)` (`vernier-cuda/src/backend.rs:298`). And unlike
the CPU's radix FFT, cuFFT handles non-power-of-two sizes natively (there's a test
for exactly that).

**Each pipeline step becomes a kernel launch.** Where the CPU ran a `for` loop
over pixels, the GPU launches thousands of threads that each handle a few pixels
at once. A launch just says "run this kernel over this grid of threads":

```rust
self.ctx.bandpass_fn.clone().launch(cfg, (&mut buf.data, width, height, cx, cy, sigma))?;
```

The interesting cases are the ones that *reduce* many values to a few — like
Chapter 6's "find the brightest bin". On the CPU that's a serial scan; on the GPU
it's a **two-stage parallel reduction** (`peak_search`,
`vernier-cuda/src/backend.rs:399`): `argmax_local` has each block of 256 threads
reduce its chunk to a single best `(magnitude, index)`, then `argmax_global`
reduces those partial winners to the one global maximum. The band-pass, the
annulus mask, the blur, the phase extraction, and the direct-from-peak plane fit
(`spectral_plane_fit_two`, the Chapter 10 alternative, done as a
`partial`→`global` reduction at `vernier-cuda/src/backend.rs:519`) all follow the
same shape.

**The job model, and why `submit` matters.** Just like the CPU, you `begin()` a
job, issue operations, and `submit()`. But GPU launches are *asynchronous* — they
queue up and return immediately. `submit` is the barrier that finally waits for
the GPU to finish (`vernier-cuda/src/backend.rs:559`):

```rust
fn submit(self) -> Result<()> {
    self.ctx.dev.synchronize()   // block until every queued kernel has completed
}
```

That's why the pipeline batches all its FFT/filter/IFFT launches inside one job
block and only synchronises once — issuing work is cheap, waiting is what costs.

**The one-paragraph summary.** Copy the image to the GPU once; run the identical
FFT → peak-search → band-pass → IFFT → plane-fit pipeline as CUDA kernels (with
cuFFT doing the transforms and parallel reductions doing the argmax/plane-fit);
copy the handful of result numbers back. The `Detection` that comes out is the
same struct Chapter 17 produced — the physics is backend-independent, and CUDA is
just a very fast place to run it. (The `vernier-gpu` crate does the same thing
with Vulkan/GLSL compute shaders instead of CUDA, for non-NVIDIA hardware.)

---

## Where to go next in the code

If you want to keep exploring, here's a reading order that follows this course:

1. `vernier-cli/src/imageio.rs` — Chapter 3, image loading.
2. `vernier-spectral/src/spectrum.rs` — Chapters 5–8, the orchestration.
3. `vernier-cpu/src/backend.rs` — Chapters 6–7, the actual FFT/peak/band-pass.
4. `vernier-spectral/src/unwrap.rs` and `planefit.rs` — Chapters 9–10.
5. `vernier-pose/src/periodic.rs` — Chapter 11.
6. `vernier-patterns/src/lfsr.rs` — Chapter 14, the code sequence.
7. `vernier-pose/src/absolute.rs` — Chapters 15–16, the decode and assembly.
8. `vernier-detector/src/megarena.rs` — Chapter 17, the top-level entry point.
9. `vernier-cuda/src/backend.rs` and `kernels.rs` — Chapter 19, the same
   pipeline on the GPU.

Every chapter's code snippet is real and lightly trimmed; open the referenced
file and line to see the full, exact version.
```