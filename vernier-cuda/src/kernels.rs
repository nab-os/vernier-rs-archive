/// All CUDA C kernels compiled at runtime via NVRTC.
///
/// Complex buffers are `float*` with interleaved layout: element i has
/// real part at `data[2*i]` and imaginary part at `data[2*i+1]`.
/// No `float2` typedef is used to avoid header dependencies.
pub const KERNEL_SRC: &str = r#"
typedef unsigned int uint;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

__device__ int signed_freq(uint f, uint n) {
    int fi = (int)f;
    int ni = (int)n;
    return (fi > ni / 2) ? (fi - ni) : fi;
}

__device__ float circular_delta(uint a, uint c, uint n) {
    int d = (int)a - (int)c;
    int ni = (int)n;
    d = ((d % ni) + ni) % ni;
    if (d > ni / 2) d -= ni;
    return (float)d;
}

// ---------------------------------------------------------------------------
// 1. scale: multiply all floats by scalar s
// ---------------------------------------------------------------------------
extern "C" __global__ void scale(float* data, uint n, float s) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] *= s;
}

// ---------------------------------------------------------------------------
// 2. bandpass: Gaussian band-pass centred at (cx, cy)
// ---------------------------------------------------------------------------
extern "C" __global__ void bandpass(float* data, uint width, uint height,
                                     uint cx, uint cy, float sigma) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float dx = circular_delta(x, cx, width);
    float dy = circular_delta(y, cy, height);
    float r2 = dx * dx + dy * dy;
    float two_sigma_sq = 2.0f * sigma * sigma;
    float gain = expf(-r2 / two_sigma_sq);

    uint idx = y * width + x;
    data[2*idx]   *= gain;
    data[2*idx+1] *= gain;
}

// ---------------------------------------------------------------------------
// 3. bandpass_from_peaks: Gaussian band-pass reading coords from peaks buffer.
//    peaks layout (8 floats): [cx1, 0, cy1, 0, cx2, 0, cy2, 0]
//    direction=0 -> peaks[0]=cx1, peaks[2]=cy1
//    direction=1 -> peaks[4]=cx2, peaks[6]=cy2
// ---------------------------------------------------------------------------
extern "C" __global__ void bandpass_from_peaks(float* data, const float* peaks,
                                                uint width, uint height,
                                                uint direction, float sigma) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint base = direction * 4u;
    uint cx = (uint)peaks[base];
    uint cy = (uint)peaks[base + 2u];

    float dx = circular_delta(x, cx, width);
    float dy = circular_delta(y, cy, height);
    float r2 = dx * dx + dy * dy;
    float two_sigma_sq = 2.0f * sigma * sigma;
    float gain = expf(-r2 / two_sigma_sq);

    uint idx = y * width + x;
    data[2*idx]   *= gain;
    data[2*idx+1] *= gain;
}

// ---------------------------------------------------------------------------
// 4. filter_annulus: zero bins outside [min_freq, max_freq]
// ---------------------------------------------------------------------------
extern "C" __global__ void filter_annulus(float* data, uint width, uint height,
                                           uint min_freq, uint max_freq) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int sfx = signed_freq(x, width);
    int sfy = signed_freq(y, height);
    float r2 = (float)(sfx * sfx + sfy * sfy);

    float min_r2 = (float)(min_freq * min_freq);
    float max_r2 = (max_freq > 0u) ? (float)(max_freq * max_freq) : 1.0e30f;

    if (r2 < min_r2 || r2 > max_r2) {
        uint idx = y * width + x;
        data[2*idx]   = 0.0f;
        data[2*idx+1] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// 5. extract_phase: per-pixel atan2(im, re) into dst real lane
// ---------------------------------------------------------------------------
extern "C" __global__ void extract_phase(const float* src, float* dst, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float re = src[2*i];
    float im = src[2*i+1];
    dst[2*i]   = atan2f(im, re);
    dst[2*i+1] = 0.0f;
}

// ---------------------------------------------------------------------------
// 6. magnitude_inplace: re = sqrt(re^2 + im^2), im = 0
// ---------------------------------------------------------------------------
extern "C" __global__ void magnitude_inplace(float* data, uint n) {
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float re = data[2*i];
    float im = data[2*i+1];
    data[2*i]   = sqrtf(re * re + im * im);
    data[2*i+1] = 0.0f;
}

// ---------------------------------------------------------------------------
// 7. gaussian_blur_h: horizontal separable Gaussian blur on real lane
//    src -> dst, clamps at borders
// ---------------------------------------------------------------------------
extern "C" __global__ void gaussian_blur_h(const float* src, float* dst,
                                            uint width, uint height, float sigma) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = (int)ceilf(3.0f * sigma);
    float weighted_value = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -radius; k <= radius; k++) {
        int sx = (int)x + k;
        if (sx < 0) sx = 0;
        if (sx >= (int)width) sx = (int)width - 1;
        float kw = expf(-(float)(k * k) / (2.0f * sigma * sigma));
        weighted_value += src[2*(y * width + (uint)sx)] * kw;
        weight_sum += kw;
    }

    uint idx = y * width + x;
    dst[2*idx]   = (weight_sum > 0.0f) ? weighted_value / weight_sum : 0.0f;
    dst[2*idx+1] = 0.0f;
}

// ---------------------------------------------------------------------------
// 8. gaussian_blur_v: vertical separable Gaussian blur on real lane
//    src -> dst, clamps at borders
// ---------------------------------------------------------------------------
extern "C" __global__ void gaussian_blur_v(const float* src, float* dst,
                                            uint width, uint height, float sigma) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int radius = (int)ceilf(3.0f * sigma);
    float weighted_value = 0.0f;
    float weight_sum = 0.0f;

    for (int k = -radius; k <= radius; k++) {
        int sy = (int)y + k;
        if (sy < 0) sy = 0;
        if (sy >= (int)height) sy = (int)height - 1;
        float kw = expf(-(float)(k * k) / (2.0f * sigma * sigma));
        weighted_value += src[2*((uint)sy * width + x)] * kw;
        weight_sum += kw;
    }

    uint idx = y * width + x;
    dst[2*idx]   = (weight_sum > 0.0f) ? weighted_value / weight_sum : 0.0f;
    dst[2*idx+1] = 0.0f;
}

// ---------------------------------------------------------------------------
// 9. argmax_local: stage-1 reduction, finds per-group max in upper half-plane.
//    Each workgroup of 256 threads writes one entry to intermediate:
//    intermediate[2*gid]   = best_mag
//    intermediate[2*gid+1] = best_linear_index, its bits stored in a float
//                            (a float value is only exact up to 2^24)
// ---------------------------------------------------------------------------
extern "C" __global__ void argmax_local(const float* src, float* intermediate,
                                         uint width, uint height, uint n) {
    __shared__ float s_mag[256];
    __shared__ float s_idx[256];

    uint thread_id = threadIdx.x;
    uint global_index = blockIdx.x * 256u + thread_id;

    float best_mag = -1.0f;
    float best_idx = __uint_as_float(0u);

    if (global_index < n) {
        uint x = global_index % width;
        uint y = global_index / width;
        int sfx = signed_freq(x, width);
        int sfy = signed_freq(y, height);

        // Upper half-plane: sfy > 0, or (sfy == 0 and sfx > 0).
        bool valid = (sfy > 0) || (sfy == 0 && sfx > 0);
        if (valid) {
            best_mag = src[2*global_index];
            best_idx = __uint_as_float(global_index);
        }
    }

    s_mag[thread_id] = best_mag;
    s_idx[thread_id] = best_idx;
    __syncthreads();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride && s_mag[thread_id + stride] > s_mag[thread_id]) {
            s_mag[thread_id] = s_mag[thread_id + stride];
            s_idx[thread_id] = s_idx[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0u) {
        uint gid = blockIdx.x;
        intermediate[2*gid]   = s_mag[0];
        intermediate[2*gid+1] = s_idx[0];
    }
}

// ---------------------------------------------------------------------------
// 10. argmax_global: stage-2 reduction, single block.
//     Reads intermediate[2*i], intermediate[2*i+1] per group.
//     Writes peak[0]=cx, peak[1]=cy (2 floats).
// ---------------------------------------------------------------------------
extern "C" __global__ void argmax_global(const float* intermediate, float* peak,
                                          uint n_groups, uint width) {
    __shared__ float s_mag[256];
    __shared__ float s_idx[256];

    uint thread_id = threadIdx.x;

    float best_mag = -1.0f;
    float best_idx = __uint_as_float(0u);

    for (uint i = thread_id; i < n_groups; i += 256u) {
        float mag = intermediate[2*i];
        if (mag > best_mag) {
            best_mag = mag;
            best_idx = intermediate[2*i+1];
        }
    }

    s_mag[thread_id] = best_mag;
    s_idx[thread_id] = best_idx;
    __syncthreads();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride && s_mag[thread_id + stride] > s_mag[thread_id]) {
            s_mag[thread_id] = s_mag[thread_id + stride];
            s_idx[thread_id] = s_idx[thread_id + stride];
        }
        __syncthreads();
    }

    if (thread_id == 0u) {
        uint linear_index = __float_as_uint(s_idx[0]);
        peak[0] = (float)(linear_index % width);
        peak[1] = (float)(linear_index / width);
    }
}

// ---------------------------------------------------------------------------
// 11. band_angular_filter: zeros bins outside band [0.5*r1, 2*r1] or inside
//     angular cone around peak1.
//     peak1 is a 2-float buffer: [cx, cy]
// ---------------------------------------------------------------------------
extern "C" __global__ void band_angular_filter(float* mag, const float* peak1,
                                                uint width, uint height, float sigma) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float cx1 = peak1[0];
    float cy1 = peak1[1];

    int sfx1i = signed_freq((uint)cx1, width);
    int sfy1i = signed_freq((uint)cy1, height);
    float sfx1 = (float)sfx1i;
    float sfy1 = (float)sfy1i;
    float r1   = sqrtf(sfx1 * sfx1 + sfy1 * sfy1);

    float band_min_r2 = (r1 * 0.5f) * (r1 * 0.5f);
    float band_max_r2 = (r1 * 2.0f) * (r1 * 2.0f);
    float center_angle = atan2f(sfy1, sfx1);
    float half_width   = atan2f(3.0f * sigma, (r1 > 1.0f ? r1 : 1.0f));

    int sfx = signed_freq(x, width);
    int sfy = signed_freq(y, height);
    float r2 = (float)(sfx * sfx + sfy * sfy);

    uint flat_index = y * width + x;

    if (r2 < band_min_r2 || r2 > band_max_r2) {
        mag[2*flat_index] = 0.0f;
        return;
    }

    const float PI  = 3.14159265358979323846f;
    const float TAU = 6.28318530717958647692f;
    float angle = atan2f((float)sfy, (float)sfx);
    float diff  = fmodf(angle - center_angle + PI, TAU) - PI;
    if (fabsf(diff) < half_width) {
        mag[2*flat_index] = 0.0f;
    }
}

// ---------------------------------------------------------------------------
// 12. peak_order: single-thread ordering of two peaks.
//     pa[0]=cx_a, pa[1]=cy_a; pb[0]=cx_b, pb[1]=cy_b
//     out is 8 floats: [cx1, 0, cy1, 0, cx2, 0, cy2, 0]
//     direction 1 = larger signed column frequency
// ---------------------------------------------------------------------------
extern "C" __global__ void peak_order(const float* pa, const float* pb,
                                       float* out, uint width, uint height) {
    float cx_a = pa[0], cy_a = pa[1];
    float cx_b = pb[0], cy_b = pb[1];

    int sfx_a = signed_freq((uint)cx_a, width);
    int sfx_b = signed_freq((uint)cx_b, width);

    // direction 1 = larger signed column frequency
    if (sfx_a < sfx_b) {
        float tmp;
        tmp = cx_a; cx_a = cx_b; cx_b = tmp;
        tmp = cy_a; cy_a = cy_b; cy_b = tmp;
    }

    out[0] = cx_a; out[1] = 0.0f;
    out[2] = cy_a; out[3] = 0.0f;
    out[4] = cx_b; out[5] = 0.0f;
    out[6] = cy_b; out[7] = 0.0f;
}

// ---------------------------------------------------------------------------
// 15. spectral_plane_fit_partial: 64-thread block, one workgroup per 64-pixel
//     chunk. peaks is 8 floats: [cx1, 0, cy1, 0, cx2, 0, cy2, 0]
//     Writes 10 floats per group into partial[].
// ---------------------------------------------------------------------------
extern "C" __global__ void spectral_plane_fit_partial(const float* S,
                                                        const float* peaks,
                                                        float* partial,
                                                        uint W, uint H, uint N,
                                                        float sigma) {
    __shared__ float sh_c_re1[64];  __shared__ float sh_c_im1[64];
    __shared__ float sh_sfx_num1[64]; __shared__ float sh_sfy_num1[64]; __shared__ float sh_den1[64];
    __shared__ float sh_c_re2[64];  __shared__ float sh_c_im2[64];
    __shared__ float sh_sfx_num2[64]; __shared__ float sh_sfy_num2[64]; __shared__ float sh_den2[64];

    uint thread_id   = threadIdx.x;
    uint global_index = blockIdx.x * 64u + thread_id;

    float neg_inv_two_sigma_sq = -1.0f / (2.0f * sigma * sigma);

    // peaks layout: [cx1, 0, cy1, 0, cx2, 0, cy2, 0]
    float sfx1 = (float)signed_freq((uint)peaks[0], W);
    float sfy1 = (float)signed_freq((uint)peaks[2], H);
    float sfx2 = (float)signed_freq((uint)peaks[4], W);
    float sfy2 = (float)signed_freq((uint)peaks[6], H);

    float c_re1 = 0.0f, c_im1 = 0.0f, sfx_num1 = 0.0f, sfy_num1 = 0.0f, den1 = 0.0f;
    float c_re2 = 0.0f, c_im2 = 0.0f, sfx_num2 = 0.0f, sfy_num2 = 0.0f, den2 = 0.0f;

    if (global_index < N) {
        uint  fx  = global_index % W;
        uint  fy  = global_index / W;
        float sfx = (float)signed_freq(fx, W);
        float sfy = (float)signed_freq(fy, H);
        float sign = ((fx + fy) % 2u == 0u) ? 1.0f : -1.0f;

        float s_re = S[2*global_index]   * sign;
        float s_im = S[2*global_index+1] * sign;
        float magnitude_sq = S[2*global_index]*S[2*global_index]
                           + S[2*global_index+1]*S[2*global_index+1];

        float dx1 = sfx - sfx1, dy1 = sfy - sfy1;
        float weight1 = expf(neg_inv_two_sigma_sq * (dx1*dx1 + dy1*dy1));
        float wm1 = weight1 * magnitude_sq;
        c_re1    = weight1 * s_re;  c_im1    = weight1 * s_im;
        sfx_num1 = wm1 * sfx;       sfy_num1 = wm1 * sfy;  den1 = wm1;

        float dx2 = sfx - sfx2, dy2 = sfy - sfy2;
        float weight2 = expf(neg_inv_two_sigma_sq * (dx2*dx2 + dy2*dy2));
        float wm2 = weight2 * magnitude_sq;
        c_re2    = weight2 * s_re;  c_im2    = weight2 * s_im;
        sfx_num2 = wm2 * sfx;       sfy_num2 = wm2 * sfy;  den2 = wm2;
    }

    sh_c_re1[thread_id]=c_re1; sh_c_im1[thread_id]=c_im1;
    sh_sfx_num1[thread_id]=sfx_num1; sh_sfy_num1[thread_id]=sfy_num1; sh_den1[thread_id]=den1;
    sh_c_re2[thread_id]=c_re2; sh_c_im2[thread_id]=c_im2;
    sh_sfx_num2[thread_id]=sfx_num2; sh_sfy_num2[thread_id]=sfy_num2; sh_den2[thread_id]=den2;
    __syncthreads();

    for (uint stride = 32u; stride > 0u; stride >>= 1u) {
        if (thread_id < stride) {
            sh_c_re1[thread_id]+=sh_c_re1[thread_id+stride]; sh_c_im1[thread_id]+=sh_c_im1[thread_id+stride];
            sh_sfx_num1[thread_id]+=sh_sfx_num1[thread_id+stride]; sh_sfy_num1[thread_id]+=sh_sfy_num1[thread_id+stride]; sh_den1[thread_id]+=sh_den1[thread_id+stride];
            sh_c_re2[thread_id]+=sh_c_re2[thread_id+stride]; sh_c_im2[thread_id]+=sh_c_im2[thread_id+stride];
            sh_sfx_num2[thread_id]+=sh_sfx_num2[thread_id+stride]; sh_sfy_num2[thread_id]+=sh_sfy_num2[thread_id+stride]; sh_den2[thread_id]+=sh_den2[thread_id+stride];
        }
        __syncthreads();
    }

    if (thread_id == 0u) {
        uint base = blockIdx.x * 10u;
        partial[base+0u]=sh_c_re1[0]; partial[base+1u]=sh_c_im1[0];
        partial[base+2u]=sh_sfx_num1[0]; partial[base+3u]=sh_sfy_num1[0]; partial[base+4u]=sh_den1[0];
        partial[base+5u]=sh_c_re2[0]; partial[base+6u]=sh_c_im2[0];
        partial[base+7u]=sh_sfx_num2[0]; partial[base+8u]=sh_sfy_num2[0]; partial[base+9u]=sh_den2[0];
    }
}

// ---------------------------------------------------------------------------
// 16. spectral_plane_fit_global: single 256-thread block.
//     Writes 12 floats: [a1,0, b1,0, c1,0, a2,0, b2,0, c2,0]
// ---------------------------------------------------------------------------
extern "C" __global__ void spectral_plane_fit_global(const float* partial,
                                                       float* result,
                                                       uint n_groups,
                                                       float W, float H) {
    __shared__ float sh_c_re1[256];  __shared__ float sh_c_im1[256];
    __shared__ float sh_sfx_num1[256]; __shared__ float sh_sfy_num1[256]; __shared__ float sh_den1[256];
    __shared__ float sh_c_re2[256];  __shared__ float sh_c_im2[256];
    __shared__ float sh_sfx_num2[256]; __shared__ float sh_sfy_num2[256]; __shared__ float sh_den2[256];

    uint local_id = threadIdx.x;

    float c_re1=0.0f, c_im1=0.0f, sfx_num1=0.0f, sfy_num1=0.0f, den1=0.0f;
    float c_re2=0.0f, c_im2=0.0f, sfx_num2=0.0f, sfy_num2=0.0f, den2=0.0f;

    for (uint g = local_id; g < n_groups; g += 256u) {
        uint base = g * 10u;
        c_re1+=partial[base+0u]; c_im1+=partial[base+1u];
        sfx_num1+=partial[base+2u]; sfy_num1+=partial[base+3u]; den1+=partial[base+4u];
        c_re2+=partial[base+5u]; c_im2+=partial[base+6u];
        sfx_num2+=partial[base+7u]; sfy_num2+=partial[base+8u]; den2+=partial[base+9u];
    }

    sh_c_re1[local_id]=c_re1; sh_c_im1[local_id]=c_im1;
    sh_sfx_num1[local_id]=sfx_num1; sh_sfy_num1[local_id]=sfy_num1; sh_den1[local_id]=den1;
    sh_c_re2[local_id]=c_re2; sh_c_im2[local_id]=c_im2;
    sh_sfx_num2[local_id]=sfx_num2; sh_sfy_num2[local_id]=sfy_num2; sh_den2[local_id]=den2;
    __syncthreads();

    for (uint stride = 128u; stride > 0u; stride >>= 1u) {
        if (local_id < stride) {
            sh_c_re1[local_id]+=sh_c_re1[local_id+stride]; sh_c_im1[local_id]+=sh_c_im1[local_id+stride];
            sh_sfx_num1[local_id]+=sh_sfx_num1[local_id+stride]; sh_sfy_num1[local_id]+=sh_sfy_num1[local_id+stride]; sh_den1[local_id]+=sh_den1[local_id+stride];
            sh_c_re2[local_id]+=sh_c_re2[local_id+stride]; sh_c_im2[local_id]+=sh_c_im2[local_id+stride];
            sh_sfx_num2[local_id]+=sh_sfx_num2[local_id+stride]; sh_sfy_num2[local_id]+=sh_sfy_num2[local_id+stride]; sh_den2[local_id]+=sh_den2[local_id+stride];
        }
        __syncthreads();
    }

    if (local_id == 0u) {
        const float two_pi = 6.283185307179586f;
        float a1 = (sh_den1[0] > 0.0f) ? (two_pi * sh_sfx_num1[0] / sh_den1[0] / W) : 0.0f;
        float b1 = (sh_den1[0] > 0.0f) ? (two_pi * sh_sfy_num1[0] / sh_den1[0] / H) : 0.0f;
        float c1 = atan2f(sh_c_im1[0], sh_c_re1[0]);
        float a2 = (sh_den2[0] > 0.0f) ? (two_pi * sh_sfx_num2[0] / sh_den2[0] / W) : 0.0f;
        float b2 = (sh_den2[0] > 0.0f) ? (two_pi * sh_sfy_num2[0] / sh_den2[0] / H) : 0.0f;
        float c2 = atan2f(sh_c_im2[0], sh_c_re2[0]);
        result[0]=a1; result[1]=0.0f;
        result[2]=b1; result[3]=0.0f;
        result[4]=c1; result[5]=0.0f;
        result[6]=a2; result[7]=0.0f;
        result[8]=b2; result[9]=0.0f;
        result[10]=c2; result[11]=0.0f;
    }
}
"#;
