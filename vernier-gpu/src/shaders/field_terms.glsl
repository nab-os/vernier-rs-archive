#version 450

// First pass of the demodulated field: the ten per-pixel terms the Gaussian
// then blurs, one plane of `width × height` each:
//
//   I, I², then per carrier  I·cos ψ, −I·sin ψ, cos ψ, −sin ψ
//
// with ψ the reference phase of the carrier at the pixel, already wrapped
// into −π..π by the host, and I the intensity less the frame's mean `level`:
// z = Σw·(I − mean)·e^{−iψ} does not change, and the variance
// Σw·I²/Σw − mean² cancels less in f32.

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Frame      { float frame[];      };
layout(set = 0, binding = 1) readonly buffer References { float references[]; };
layout(set = 0, binding = 2) writeonly buffer Terms     { float terms[];      };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    float level;
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;
    uint n = pc.width * pc.height;
    uint i = y * pc.width + x;

    float v = frame[i] - pc.level;
    terms[i] = v;
    terms[n + i] = v * v;
    for (uint c = 0u; c < 2u; c++) {
        float psi = references[c * n + i];
        vec2 e = vec2(cos(psi), -sin(psi));
        uint o = (2u + 4u * c) * n + i;
        terms[o] = v * e.x;
        terms[o + n] = v * e.y;
        terms[o + 2u * n] = e.x;
        terms[o + 3u * n] = e.y;
    }
}
