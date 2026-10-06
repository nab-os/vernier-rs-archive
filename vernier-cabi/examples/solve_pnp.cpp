// solve_pnp.cpp — the pose of the coded checkerboard or of a megarena in each
// photo, from a known camera.
//
// camera.json is the file `vernier calibrate` or calibrate_camera.cpp writes.
// The pose is board → camera, as OpenCV's solvePnP gives it: a Rodrigues
// rotation vector and a translation in the unit of <size>, the board's origin
// at its centre (a megarena's at its dot (0, 0)).
//
// Usage:
//   solve_pnp [--megarena] <camera.json> <size> <code-size> <image>...
//
//   size        side of one square, or for a megarena the dot pitch
//   code-size   LFSR order the board was rendered with (4..12)
//   --megarena  the board is a megarena (default: coded checkerboard)
//
// Images may be PNG, JPEG, BMP, TIFF or PGM, taken with the calibrated camera
// at the size it was calibrated at.

#include "vernier.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    bool megarena = false;
    int  arg      = 1;
    for (; arg < argc && std::string(argv[arg]).rfind("--", 0) == 0; ++arg) {
        const std::string option = argv[arg];
        if (option == "--megarena") megarena = true;
        else { std::cerr << "unknown option " << option << "\n"; return 2; }
    }
    if (argc - arg < 4) {
        std::cerr << "usage: " << argv[0]
                  << " [--megarena] <camera.json> <size> <code-size> <image>...\n";
        return 2;
    }

    vernier::Camera camera;
    try {
        camera = vernier::Camera::load(argv[arg]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    const double size  = std::atof(argv[arg + 1]);
    const auto   order = static_cast<std::uint32_t>(std::atoi(argv[arg + 2]));
    const vernier::Target target =
        megarena ? vernier::Target::megarena(size, order) : vernier::Target{size, order};

    int failures = 0;
    for (int i = arg + 3; i < argc; ++i) {
        const char* name = argv[i];
        try {
            const vernier::Image img = vernier::load_image(name);
            if (img.width != camera.width || img.height != camera.height) {
                std::printf("%s: %zux%zu, but the camera was calibrated at %zux%zu\n", name,
                            img.width, img.height, camera.width, camera.height);
                ++failures;
                continue;
            }
            const vernier::View view =
                vernier::View::measure(img.pixels.data(), img.width, img.height, target);

            // Without the code, the points sit on the right lattice but in an
            // unknown place on the board, so the pose would mean nothing.
            if (!view.is_absolute()) {
                std::printf("%s: code not read, so the board frame is unknown\n", name);
                ++failures;
                continue;
            }

            const vernier::ViewFit fit = vernier::solve_pnp(camera, view);
            const auto&            r   = fit.pose.rvec;
            const auto&            t   = fit.pose.tvec;
            std::printf("%s: rvec (%.5f %.5f %.5f)  tvec (%.3f %.3f %.3f)  rms %.4f px, "
                        "%zu points\n",
                        name, r[0], r[1], r[2], t[0], t[1], t[2], fit.rms, fit.used);
        } catch (const std::exception& e) {
            std::printf("%s: %s\n", name, e.what());
            ++failures;
        }
    }
    return failures ? 1 : 0;
}
