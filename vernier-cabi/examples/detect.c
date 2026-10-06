/*
 * detect.c — vernier-cabi quick-start example.
 *
 * Generates a 512×512 synthetic sinusoidal periodic image (period 40 px) and
 * runs vernier_detect_periodic.  No external image I/O libraries are needed.
 * The pattern's crests sit on multiples of 40 px and the image centre,
 * 256 = 6·40 + 16, is 16 px past one, so x and y come out as -16 (modulo the
 * period, with the sign convention of the C++ library).
 *
 * Build
 * -----
 *   cc -std=c11 -I../include -L../../target/release \
 *       detect.c -lvernier_cabi -Wl,-rpath,../../target/release \
 *       -lm -o detect
 *   ./detect
 *
 * Or use the provided Makefile:
 *   make -C vernier-cabi/examples
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "vernier.h"

#define W       512
#define H       512
#define PERIOD  40.0f
#define TWO_PI  6.28318530717958f

int main(void)
{
    /* Synthetic image: cos(2π·col/T) × cos(2π·row/T) mapped to [0, 1]. */
    float *pixels = malloc((size_t)W * H * sizeof(float));
    if (!pixels) { fputs("out of memory\n", stderr); return 1; }

    for (int row = 0; row < H; ++row)
        for (int col = 0; col < W; ++col)
            pixels[row * W + col] =
                0.5f * (1.0f + cosf(TWO_PI * (float)col / PERIOD))
              * 0.5f * (1.0f + cosf(TWO_PI * (float)row / PERIOD));

    /* ── CPU detector ─────────────────────────────────────────────────────── */
    VernierDetector *det = vernier_detector_new();
    if (!det) {
        fprintf(stderr, "detector creation failed: %s\n", vernier_last_error());
        free(pixels);
        return 1;
    }

    VernierPose p = vernier_detect_periodic(
        det, pixels, W, H,
        PERIOD,   /* period in the same units as pixel coordinates */
        3.0f,     /* sigma: bandpass half-width in frequency bins  */
        5,        /* min_frequency: skip DC neighbourhood           */
        0,        /* max_frequency: 0 = no upper limit             */
        0.5f      /* smoothing_sigma                               */
    );

    if (p.found)
        printf("periodic  x=%.4f  y=%.4f  theta=%.6f rad\n",
               p.x, p.y, p.theta);
    else
        fprintf(stderr, "detection failed: %s\n", vernier_last_error());

    vernier_detector_free(det);

    /* ── CUDA (always in ABI; returns NULL + message when unavailable) ─────── */
    VernierDetector *gpu = vernier_detector_new_cuda();
    if (gpu) {
        puts("cuda backend available");
        vernier_detector_free(gpu);
    } else {
        printf("cuda: %s\n", vernier_last_error());
    }

    free(pixels);
    return p.found ? 0 : 1;
}
