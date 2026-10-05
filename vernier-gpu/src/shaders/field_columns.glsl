#version 450

// Third pass of the demodulated field: a row-blurred term plane (the
// workgroup's z) blurred down its columns by a Gaussian of `sigma` pixels
// cut at `radius`, normalized by the weight that falls inside the frame.
//
// A workgroup covers a block of TX columns by TY rows, each thread four rows
// of one column, 8 apart, in the lanes of a vec4. The kernel is walked in
// segments of SEG taps: the TY + SEG source rows a segment needs are staged
// in shared memory once and read by every row of the block.

layout(local_size_x = 32, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0) readonly buffer Source  { float source[]; };
layout(set = 0, binding = 1) writeonly buffer Target { float target[]; };

layout(push_constant) uniform PushConstantData {
    uint width;
    uint height;
    float sigma;
    int radius;
} pc;

const uint TX = 32u;
const uint TY = 32u;
const uint SEG = 32u;

shared float tile[(TY + SEG) * TX];
shared float kernel_weights[SEG];

void main() {
    uint lx = gl_LocalInvocationID.x;
    uint ly = gl_LocalInvocationID.y;
    // Workgroups run down a strip of columns one block after the other, so
    // the rows a block reads are still in cache for the next.
    uint x = gl_WorkGroupID.y * TX + lx;
    int y0 = int(gl_WorkGroupID.x * TY);
    bool column = x < pc.width;
    int h = int(pc.height);
    uint base = gl_WorkGroupID.z * pc.width * pc.height;
    float exponent = -0.5 / (pc.sigma * pc.sigma);

    // The rows this thread writes.
    ivec4 rows = y0 + int(ly) + ivec4(0, 8, 16, 24);
    vec4 sum = vec4(0.0);
    vec4 weight = vec4(0.0);
    for (int seg = -pc.radius; seg <= pc.radius; seg += int(SEG)) {
        int taps = min(int(SEG), pc.radius - seg + 1);
        barrier();
        // Tile row t holds source row y0 + seg + t.
        for (uint t = ly; t < TY + SEG; t += 8u) {
            int sy = y0 + seg + int(t);
            tile[t * TX + lx] = (column && sy >= 0 && sy < h)
                ? source[base + uint(sy) * pc.width + x]
                : 0.0;
        }
        uint thread = ly * TX + lx;
        if (thread < SEG) {
            float tap = float(seg + int(thread));
            kernel_weights[thread] = exp(tap * tap * exponent);
        }
        barrier();
        for (int i = 0; i < taps; i++) {
            float k = kernel_weights[i];
            uint t = (ly + uint(i)) * TX + lx;
            // Rows past the frame read 0 from the tile, and weigh nothing.
            vec4 v = vec4(tile[t], tile[t + 8u * TX], tile[t + 16u * TX], tile[t + 24u * TX]);
            ivec4 sy = rows + seg + i;
            vec4 inside = vec4(greaterThanEqual(sy, ivec4(0))) * vec4(lessThan(sy, ivec4(h)));
            sum += k * v;
            weight += k * inside;
        }
    }

    if (!column) return;
    vec4 blurred = sum / weight;
    for (uint o = 0u; o < 4u; o++) {
        if (rows[o] < h) target[base + uint(rows[o]) * pc.width + x] = blurred[o];
    }
}
