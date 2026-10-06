#version 450

// Window sums of the local demodulation, one workgroup per window.
//
// Each window is a Gaussian of `sigma` pixels around (x, y), cut at `radius`
// pixels either way and at the frame edge. Against each carrier's reference
// ψ(q) = k·q + ½·qᵀHq, with q the offset from the window centre, the
// workgroup sums, with w the window weight, I the intensity and e = e^{−iψ}:
//
//   [0..3)   Σw, Σw·I, Σw·I²
//   then per carrier (12 floats each, re/im pairs):
//            Σw·I·e, Σw·e, Σw·I·e·qx, Σw·e·qx, Σw·I·e·qy, Σw·e·qy
//
// I is taken less the intensity at the window centre: z = Σw·(I − mean)·e
// does not change, and the variance Σw·I²/Σw − mean² no longer cancels
// where the window is flat. The host turns the sums into a phase, a quality
// and an offset in double precision.
//
// Window layout, 16 floats: x, y, sigma, radius, then per carrier
// kx, ky, hxx, hxy, hyy, then two of padding.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Frame   { float frame[];   };
layout(set = 0, binding = 1) readonly buffer Windows { float windows[]; };
layout(set = 0, binding = 2) writeonly buffer Sums   { float sums[];    };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint count;
    uint groups_x;  // workgroups along x; windows run past 65535 along y
} pc;

const uint THREADS = 64u;
const uint TERMS = 27u;
const uint STRIDE = 28u;  // floats per window in `sums`
const float TAU = 6.283185307179586;

shared float partial[TERMS * THREADS];

void main() {
    uint index = gl_WorkGroupID.y * pc.groups_x + gl_WorkGroupID.x;
    // Uniform across the workgroup, so no barrier is skipped by part of it.
    if (index >= pc.count) return;
    uint thread = gl_LocalInvocationID.x;

    uint base = index * 16u;
    int x = int(windows[base + 0u]);
    int y = int(windows[base + 1u]);
    float sigma = windows[base + 2u];
    int radius = int(windows[base + 3u]);
    vec2 k1 = vec2(windows[base + 4u], windows[base + 5u]);
    vec3 h1 = vec3(windows[base + 6u], windows[base + 7u], windows[base + 8u]);
    vec2 k2 = vec2(windows[base + 9u], windows[base + 10u]);
    vec3 h2 = vec3(windows[base + 11u], windows[base + 12u], windows[base + 13u]);

    int x0 = max(x - radius, 0);
    int x1 = min(x + radius, int(pc.width) - 1);
    int y0 = max(y - radius, 0);
    int y1 = min(y + radius, int(pc.height) - 1);
    uint columns = uint(x1 - x0 + 1);
    uint pixels = columns * uint(y1 - y0 + 1);
    float exponent = -0.5 / (sigma * sigma);
    float level = frame[uint(y) * pc.width + uint(x)];

    float acc[TERMS];
    for (uint j = 0u; j < TERMS; j++) acc[j] = 0.0;

    for (uint p = thread; p < pixels; p += THREADS) {
        int px = x0 + int(p % columns);
        int py = y0 + int(p / columns);
        float qx = float(px - x);
        float qy = float(py - y);
        float w = exp((qx * qx + qy * qy) * exponent);
        float v = frame[uint(py) * pc.width + uint(px)] - level;
        acc[0] += w;
        acc[1] += w * v;
        acc[2] += w * v * v;

        for (uint c = 0u; c < 2u; c++) {
            vec2 k = c == 0u ? k1 : k2;
            vec3 h = c == 0u ? h1 : h2;
            float psi = k.x * qx + k.y * qy
                + 0.5 * (h.x * qx * qx + 2.0 * h.y * qx * qy + h.z * qy * qy);
            // Into −π..π, where sin and cos are held to their precision.
            psi -= TAU * round(psi / TAU);
            vec2 unit = w * vec2(cos(psi), -sin(psi));
            vec2 signal = unit * v;
            uint o = 3u + 12u * c;
            acc[o + 0u] += signal.x;
            acc[o + 1u] += signal.y;
            acc[o + 2u] += unit.x;
            acc[o + 3u] += unit.y;
            acc[o + 4u] += signal.x * qx;
            acc[o + 5u] += signal.y * qx;
            acc[o + 6u] += unit.x * qx;
            acc[o + 7u] += unit.y * qx;
            acc[o + 8u] += signal.x * qy;
            acc[o + 9u] += signal.y * qy;
            acc[o + 10u] += unit.x * qy;
            acc[o + 11u] += unit.y * qy;
        }
    }

    for (uint j = 0u; j < TERMS; j++) partial[j * THREADS + thread] = acc[j];
    barrier();
    for (uint stride = THREADS / 2u; stride > 0u; stride >>= 1u) {
        if (thread < stride) {
            for (uint j = 0u; j < TERMS; j++) {
                partial[j * THREADS + thread] += partial[j * THREADS + thread + stride];
            }
        }
        barrier();
    }
    if (thread < TERMS) sums[index * STRIDE + thread] = partial[thread * THREADS];
}
