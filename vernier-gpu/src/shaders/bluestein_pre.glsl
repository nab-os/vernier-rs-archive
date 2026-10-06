#version 450

// Bluestein chirp-Z pre-weight pass.
//
// Fills work_a[] with the chirp-pre-weighted input (zero-padded to M) and
// work_b[] with the wrapped chirp sequence b_circ[] of length M.
//
// Forward (is_inverse=0):
//   a[n] = data[n] * exp(-jπn²/N)   n ∈ [0,N), else 0
//   b_circ[m] = exp(+jπm²/N)        m ∈ [0,N) ∪ (M-N, M)
//
// Inverse (is_inverse=1):
//   a[k] = data[k] * exp(+jπk²/N)
//   b_circ[m] = exp(-jπm²/N)
//
// Memory layouts:
//   pass=0 (row):    work_a is (height × M),  indexed work_a[r*M + n]
//   pass=1 (col):    work_a is (M × width),   indexed work_a[n*width + c]
//   work_b is always M elements.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };
layout(set = 0, binding = 1) buffer WorkA   { vec2 work_a[]; };
layout(set = 0, binding = 2) buffer WorkB   { vec2 work_b[]; };

layout(push_constant) uniform PushConstantData {
    uint N;
    uint M;
    uint width;
    uint height;
    uint pass;
    uint is_inverse;
} pc;

const float PI = 3.14159265358979323846;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// πn²/N reduced to (-π, π]. n²/N reaches thousands of half-turns, where an
// f32 angle keeps few fractional bits and GPU sin/cos lose accuracy, but the
// chirp only depends on n² mod 2N, which integers give exactly (n < N ≤ 4096).
float chirp_angle(uint n) {
    uint q = (n * n) % (2u * pc.N);
    float centred = q > pc.N ? float(q) - 2.0 * float(pc.N) : float(q);
    return PI * centred / float(pc.N);
}

// exp(-jπn²/N) for forward, exp(+jπn²/N) for inverse.
vec2 pre_chirp(uint n) {
    float angle = chirp_angle(n);
    if (pc.is_inverse == 0u) angle = -angle;
    return vec2(cos(angle), sin(angle));
}

// exp(+jπm²/N) for forward, exp(-jπm²/N) for inverse.
vec2 b_chirp(uint m) {
    float angle = chirp_angle(m);
    if (pc.is_inverse != 0u) angle = -angle;
    return vec2(cos(angle), sin(angle));
}

void main() {
    uint gid = gl_GlobalInvocationID.x;

    // --- Fill work_a ----------------------------------------------------------
    if (pc.pass == 0u) {
        // Row pass: work_a[r * M + n], total = height * M
        uint total = pc.height * pc.M;
        if (gid < total) {
            uint r = gid / pc.M;
            uint n = gid % pc.M;
            if (n < pc.N) {
                work_a[gid] = cmul(data[r * pc.width + n], pre_chirp(n));
            } else {
                work_a[gid] = vec2(0.0);
            }
        }
    } else {
        // Column pass: work_a[n * width + c], total = M * width
        uint total = pc.M * pc.width;
        if (gid < total) {
            uint n = gid / pc.width;
            uint c = gid % pc.width;
            if (n < pc.N) {
                work_a[gid] = cmul(data[n * pc.width + c], pre_chirp(n));
            } else {
                work_a[gid] = vec2(0.0);
            }
        }
    }

    // --- Fill work_b (only first M threads) -----------------------------------
    // Each thread writes exactly one element; no races.
    if (gid < pc.M) {
        uint m = gid;
        if (m < pc.N) {
            work_b[m] = b_chirp(m);
        } else if (m > pc.M - pc.N) {
            // Wrap: b_circ[M - k] = b_chirp(k) for k = M - m ∈ [1, N-1]
            work_b[m] = b_chirp(pc.M - m);
        } else {
            work_b[m] = vec2(0.0);
        }
    }
}
