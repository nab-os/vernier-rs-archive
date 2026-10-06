#version 450

// Bluestein chirp-Z post-weight pass.
//
// Extracts N output elements from work_a[], applies the post-weight chirp,
// accounts for the 1/M normalization applied by the IFFT shader, and writes
// back to data[].
//
// The internal IFFT shader already normalizes by 1/M, so work_a_ifft[n] is
// the true circular convolution (a ⊛ b_circ)[n].  No extra M factor needed.
//
// Forward (is_inverse=0):
//   data[n] = exp(-jπn²/N) * work_a_ifft[n]
//
// Inverse (is_inverse=1):
//   data[n] = (1/N) * exp(+jπn²/N) * work_a_ifft[n]
//   (the 1/N factor mirrors the 1/height normalization of the PoT IFFT pass)
//
// Memory layouts:
//   pass=0 (row):  work_a[r*M + n],  data[r*width + n],  n ∈ [0, N=width)
//   pass=1 (col):  work_a[n*width + c], data[n*width + c], n ∈ [0, N=height)

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf { vec2 data[]; };
layout(set = 0, binding = 1) buffer WorkA   { vec2 work_a[]; };

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

// πn²/N reduced to (-π, π], exactly in integers; see bluestein_pre.glsl.
float chirp_angle(uint n) {
    uint q = (n * n) % (2u * pc.N);
    float centred = q > pc.N ? float(q) - 2.0 * float(pc.N) : float(q);
    return PI * centred / float(pc.N);
}

// exp(-jπn²/N) for forward, exp(+jπn²/N) for inverse.
vec2 post_chirp(uint n) {
    float angle = chirp_angle(n);
    if (pc.is_inverse == 0u) angle = -angle;
    return vec2(cos(angle), sin(angle));
}

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint total = pc.height * pc.width;
    if (gid >= total) return;

    float scale = 1.0;
    if (pc.is_inverse != 0u) scale = 1.0 / float(pc.N);

    uint n;
    uint work_idx;
    if (pc.pass == 0u) {
        // Row pass: gid = r * width + col, n = col
        uint r = gid / pc.width;
        n = gid % pc.width;          // n ∈ [0, N=width)
        work_idx = r * pc.M + n;
    } else {
        // Column pass: gid = row * width + c, n = row
        n = gid / pc.width;          // n ∈ [0, N=height)
        // work_a layout is M × width: index = n * width + c = gid
        work_idx = gid;
    }

    data[gid] = cmul(work_a[work_idx], post_chirp(n)) * scale;
}
