// calibrate_camera.cpp — calibrates a camera from photos of the coded
// checkerboard or of a megarena, and writes camera.json.
//
// The photos should show the board at varied tilts and places in the frame
// (10 or more is best), all at the same image size. camera.json is the file
// `vernier calibrate` writes, which solve_pnp.cpp and `vernier solve-pnp` read.
//
// Usage:
//   calibrate_camera [--megarena] [--fisheye] <size> <code-size> <camera.json> <image>...
//
//   size        side of one square, or for a megarena the dot pitch, in the
//               unit poses should come out in (e.g. mm)
//   code-size   LFSR order the board was rendered with (4..12)
//   --megarena  the board is a megarena (default: coded checkerboard)
//   --fisheye   fit the fisheye model (default: pinhole)
//
// Images may be PNG, JPEG, BMP, TIFF or PGM. For example, on the fmac set:
//   calibrate_camera 5 6 camera.json ../../resources/fmac-calibration/view_*.png

#include "vernier.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    bool megarena = false, fisheye = false;
    int  arg      = 1;
    for (; arg < argc && std::string(argv[arg]).rfind("--", 0) == 0; ++arg) {
        const std::string option = argv[arg];
        if (option == "--megarena") megarena = true;
        else if (option == "--fisheye") fisheye = true;
        else { std::cerr << "unknown option " << option << "\n"; return 2; }
    }
    if (argc - arg < 4) {
        std::cerr << "usage: " << argv[0]
                  << " [--megarena] [--fisheye] <size> <code-size> <camera.json> <image>...\n";
        return 2;
    }
    const double        size   = std::atof(argv[arg]);
    const auto          order  = static_cast<std::uint32_t>(std::atoi(argv[arg + 1]));
    const std::string   output = argv[arg + 2];
    const vernier::Target target =
        megarena ? vernier::Target::megarena(size, order) : vernier::Target{size, order};

    // Measure every image, keeping those where the board is found.
    std::vector<vernier::View> views;
    std::vector<std::string>   names;
    for (int i = arg + 3; i < argc; ++i) {
        try {
            const vernier::Image img = vernier::load_image(argv[i]);
            vernier::View view =
                vernier::View::measure(img.pixels.data(), img.width, img.height, target);
            std::printf("%s: %zu points, code %s\n", argv[i], view.size(),
                        view.is_absolute() ? "read" : "not read");
            views.push_back(std::move(view));
            names.push_back(argv[i]);
        } catch (const std::exception& e) {
            std::printf("%s: skipped, %s\n", argv[i], e.what());
        }
    }

    try {
        // Intrinsics and distortion from all of them, and each view's fit.
        const vernier::Calibration cal = vernier::calibrate(
            views, fisheye ? vernier::Model::Fisheye : vernier::Model::Pinhole);
        const vernier::Camera& c = cal.camera;
        std::printf("\n%s camera %zux%zu  fx %.3f  fy %.3f  cx %.3f  cy %.3f\ndistortion",
                    fisheye ? "fisheye" : "pinhole", c.width, c.height, c.fx, c.fy, c.cx, c.cy);
        for (double k : c.distortion) std::printf(" %.5f", k);
        std::printf("\nrms %.4f px over %zu views\n", cal.rms, cal.views.size());
        for (std::size_t i = 0; i < cal.views.size(); ++i)
            std::printf("  %s: rms %.4f px, %zu points kept, %zu rejected\n", names[i].c_str(),
                        cal.views[i].rms, cal.views[i].used, cal.views[i].rejected);

        c.save(output, cal.rms, cal.views.size());
        std::printf("wrote %s\n", output.c_str());
    } catch (const std::exception& e) {
        std::fflush(stdout);
        std::cerr << "calibration failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
