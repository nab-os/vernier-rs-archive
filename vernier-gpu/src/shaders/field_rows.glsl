#version 450

// Second pass of the demodulated field: a term plane (the workgroup's z)
// blurred along its rows by a Gaussian of `sigma` pixels cut at `radius`,
// normalized by the weight that falls inside the frame.
//
// A workgroup covers TX pixels of one row. The kernel is walked in segments
// of SEG taps: the TX + SEG source pixels a segment needs are staged in
// shared memory once and read by every thread.

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Source  { float source[];  };
layout(set = 0, binding = 1) writeonly buffer Target { float target[];  };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    float sigma;
    int radius;
} pc;

const uint TX = 256u;
const uint SEG = 256u;

shared float tile[TX + SEG];
shared float kernel_weights[SEG];

void main() {
    uint lx = gl_LocalInvocationID.x;
    int x0 = int(gl_WorkGroupID.x * TX);
    int x = x0 + int(lx);
    uint y = gl_WorkGroupID.y;
    int w = int(pc.width);
    uint base = gl_WorkGroupID.z * pc.width * pc.height + y * pc.width;
    float exponent = -0.5 / (pc.sigma * pc.sigma);

    float sum = 0.0;
    float weight = 0.0;
    for (int seg = -pc.radius; seg <= pc.radius; seg += int(SEG)) {
        int taps = min(int(SEG), pc.radius - seg + 1);
        barrier();
        // Tile entry t holds source pixel x0 + seg + t.
        for (uint t = lx; t < TX + SEG; t += TX) {
            int sx = x0 + seg + int(t);
            tile[t] = (sx >= 0 && sx < w) ? source[base + uint(sx)] : 0.0;
        }
        float tap = float(seg + int(lx));
        kernel_weights[lx] = exp(tap * tap * exponent);
        barrier();
        // Taps that land inside the frame.
        int first = max(0, -(x + seg));
        int last = min(taps, w - (x + seg));
        for (int i = first; i < last; i++) {
            float k = kernel_weights[i];
            sum += k * tile[uint(int(lx) + i)];
            weight += k;
        }
    }
    if (x < w) target[base + uint(x)] = sum / weight;
}
