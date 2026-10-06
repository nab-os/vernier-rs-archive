// Calibrates a camera from photos of the coded checkerboard, then solves the
// board's pose in each photo with the camera found.
//
// Images are binary 8-bit PGM, to keep the example free of an image library:
//
//     magick view_00.png view_00.pgm
//
// Usage: calibrate <square> <code-size> <image.pgm>...
//   e.g. calibrate 5 6 ../../resources/fmac-calibration/*.pgm

#include "vernier.hpp"
#include "pgm.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <square> <code-size> <image.pgm>...\n";
        return 2;
    }
    const vernier::Target target{std::atof(argv[1]),
                                 static_cast<std::uint32_t>(std::atoi(argv[2]))};

    // Measure every photo, skipping those where the board is not found.
    std::vector<vernier::View>  views;
    std::vector<std::string>    names;
    for (int i = 3; i < argc; ++i) {
        try {
            Image         img  = load_pgm(argv[i]);
            vernier::View view = vernier::View::measure(img.pixels.data(), img.width,
                                                        img.height, target);
            std::printf("%s: %zu points, code %s\n", argv[i], view.size(),
                        view.is_absolute() ? "read" : "not read");
            views.push_back(std::move(view));
            names.push_back(argv[i]);
        } catch (const std::exception& e) {
            std::printf("%s: skipped, %s\n", argv[i], e.what());
        }
    }

    try {
        const vernier::Calibration cal = vernier::calibrate(views, vernier::Model::Pinhole);
        const vernier::Camera&     c   = cal.camera;
        std::printf("\ncamera %zux%zu  fx %.3f  fy %.3f  cx %.3f  cy %.3f\n", c.width,
                    c.height, c.fx, c.fy, c.cx, c.cy);
        std::printf("distortion");
        for (double k : c.distortion) std::printf(" %.5f", k);
        std::printf("\nrms %.4f px over %zu views\n\n", cal.rms, cal.views.size());

        // The camera known, the board's pose in each view on its own.
        for (std::size_t i = 0; i < views.size(); ++i) {
            const vernier::ViewFit fit = vernier::solve_pnp(c, views[i]);
            const auto&            r = fit.pose.rvec;
            const auto&            t = fit.pose.tvec;
            std::printf("%s: rvec (%.4f %.4f %.4f)  tvec (%.2f %.2f %.2f)  rms %.4f px, "
                        "%zu points\n",
                        names[i].c_str(), r[0], r[1], r[2], t[0], t[1], t[2], fit.rms,
                        fit.used);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
