#version 450

// 2D forward FFT — in-place, row-major Complex32 buffer.
//
// Dispatched twice per 2D transform:
//   pass=0  →  row pass,    dispatch [1,      height, 1]
//   pass=1  →  column pass, dispatch [width,  1,      1]
//
// For power-of-two N ≤ 4096: Cooley-Tukey DIT radix-2 (in-place, shared mem).
// The host sends every other N through Bluestein (bluestein_*.glsl), whose
// padded length is a power of two, so the direct-transform branch below is
// never dispatched.
//
// Max supported N: 4096 (shared memory = 4096 × 8 B = 32 KB).

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) buffer DataBuf {
    vec2 data[];
};

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    uint pass;
} pc;

shared vec2 s[4096];

const float TAU = 6.28318530717958647692;

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

uint bit_reverse(uint v, uint bits) {
    uint r = 0u;
    for (uint i = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v >>= 1u;
    }
    return r;
}

bool is_pow2(uint n) {
    return n > 0u && (n & (n - 1u)) == 0u;
}

void main() {
    uint thread_id = gl_LocalInvocationID.x;
    uint transform_size;

    // ---- Load one row or column into shared memory -------------------------
    if (pc.pass == 0u) {
        transform_size = pc.width;
        uint base = gl_WorkGroupID.y * pc.width;
        if (thread_id          < transform_size) s[thread_id]          = data[base + thread_id];
        if (thread_id + 1024u  < transform_size) s[thread_id + 1024u]  = data[base + thread_id + 1024u];
        if (thread_id + 2048u  < transform_size) s[thread_id + 2048u]  = data[base + thread_id + 2048u];
        if (thread_id + 3072u  < transform_size) s[thread_id + 3072u]  = data[base + thread_id + 3072u];
    } else {
        transform_size = pc.height;
        uint col = gl_WorkGroupID.x;
        if (thread_id          < transform_size) s[thread_id]          = data[thread_id          * pc.width + col];
        if (thread_id + 1024u  < transform_size) s[thread_id + 1024u]  = data[(thread_id + 1024u) * pc.width + col];
        if (thread_id + 2048u  < transform_size) s[thread_id + 2048u]  = data[(thread_id + 2048u) * pc.width + col];
        if (thread_id + 3072u  < transform_size) s[thread_id + 3072u]  = data[(thread_id + 3072u) * pc.width + col];
    }
    barrier();

    // ---- Choose algorithm based on whether N is a power of two ------------
    if (is_pow2(transform_size) && transform_size <= 4096u) {
        // ---- Cooley-Tukey DIT (in-place on s[]) ---------------------------
        uint bits = 0u;
        for (uint t = transform_size; t > 1u; t >>= 1u) bits++;

        if (thread_id          < transform_size) { uint rev = bit_reverse(thread_id,          bits); if (rev > thread_id)          { vec2 tmp = s[thread_id];         s[thread_id]         = s[rev]; s[rev] = tmp; } }
        if (thread_id + 1024u  < transform_size) { uint rev = bit_reverse(thread_id + 1024u,  bits); if (rev > thread_id + 1024u)  { vec2 tmp = s[thread_id + 1024u]; s[thread_id + 1024u] = s[rev]; s[rev] = tmp; } }
        if (thread_id + 2048u  < transform_size) { uint rev = bit_reverse(thread_id + 2048u,  bits); if (rev > thread_id + 2048u)  { vec2 tmp = s[thread_id + 2048u]; s[thread_id + 2048u] = s[rev]; s[rev] = tmp; } }
        if (thread_id + 3072u  < transform_size) { uint rev = bit_reverse(thread_id + 3072u,  bits); if (rev > thread_id + 3072u)  { vec2 tmp = s[thread_id + 3072u]; s[thread_id + 3072u] = s[rev]; s[rev] = tmp; } }
        barrier();

        for (uint size = 2u; size <= transform_size; size <<= 1u) {
            uint hstep = size >> 1u;
            for (uint k = thread_id; k < transform_size / 2u; k += 1024u) {
                uint grp = k / hstep, pos = k % hstep;
                uint i = grp * size + pos, j = i + hstep;
                float angle = -TAU * float(pos) / float(size);
                vec2 twiddle = vec2(cos(angle), sin(angle));
                vec2 upper = s[i], lower = cmul(twiddle, s[j]);
                s[i] = upper + lower;
                s[j] = upper - lower;
            }
            barrier();
        }

        // Write back
        if (pc.pass == 0u) {
            uint base = gl_WorkGroupID.y * pc.width;
            if (thread_id          < transform_size) data[base + thread_id]          = s[thread_id];
            if (thread_id + 1024u  < transform_size) data[base + thread_id + 1024u]  = s[thread_id + 1024u];
            if (thread_id + 2048u  < transform_size) data[base + thread_id + 2048u]  = s[thread_id + 2048u];
            if (thread_id + 3072u  < transform_size) data[base + thread_id + 3072u]  = s[thread_id + 3072u];
        } else {
            uint col = gl_WorkGroupID.x;
            if (thread_id          < transform_size) data[thread_id          * pc.width + col] = s[thread_id];
            if (thread_id + 1024u  < transform_size) data[(thread_id + 1024u) * pc.width + col] = s[thread_id + 1024u];
            if (thread_id + 2048u  < transform_size) data[(thread_id + 2048u) * pc.width + col] = s[thread_id + 2048u];
            if (thread_id + 3072u  < transform_size) data[(thread_id + 3072u) * pc.width + col] = s[thread_id + 3072u];
        }
    } else {
        // ---- Direct DFT: read from s[], write straight to global memory ---
        // (safe because each thread owns distinct output indices)
        if (pc.pass == 0u) {
            uint base = gl_WorkGroupID.y * pc.width;
            for (uint k = thread_id; k < transform_size; k += 1024u) {
                vec2 dft_sum = vec2(0.0);
                for (uint m = 0u; m < transform_size; m++) {
                    float angle = -TAU * float(k) * float(m) / float(transform_size);
                    dft_sum += cmul(s[m], vec2(cos(angle), sin(angle)));
                }
                data[base + k] = dft_sum;
            }
        } else {
            uint col = gl_WorkGroupID.x;
            for (uint k = thread_id; k < transform_size; k += 1024u) {
                vec2 dft_sum = vec2(0.0);
                for (uint m = 0u; m < transform_size; m++) {
                    float angle = -TAU * float(k) * float(m) / float(transform_size);
                    dft_sum += cmul(s[m], vec2(cos(angle), sin(angle)));
                }
                data[k * pc.width + col] = dft_sum;
            }
        }
    }
}
