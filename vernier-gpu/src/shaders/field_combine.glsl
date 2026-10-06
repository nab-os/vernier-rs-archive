#version 450

// Last pass of the demodulated field: the ten blurred term planes combined
// per pixel. Writes five planes: per carrier the real and imaginary parts of
// z = Σw·(I − mean)·e^{−iψ} over Σw, then the local deviation of I. The host
// takes the phase and the amplitude in double precision.

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Blurred { float blurred[]; };
layout(set = 0, binding = 1) writeonly buffer Out    { float outs[];    };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
} pc;

void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= pc.width || y >= pc.height) return;
    uint n = pc.width * pc.height;
    uint i = y * pc.width + x;
    float mean = blurred[i];
    for (uint c = 0u; c < 2u; c++) {
        uint p = (2u + 4u * c) * n + i;
        outs[(2u * c) * n + i] = blurred[p] - mean * blurred[p + 2u * n];
        outs[(2u * c + 1u) * n + i] = blurred[p + n] - mean * blurred[p + 3u * n];
    }
    outs[4u * n + i] = sqrt(max(blurred[n + i] - mean * mean, 0.0));
}
