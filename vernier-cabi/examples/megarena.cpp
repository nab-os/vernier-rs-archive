// Calibrates a camera from photos of a printed megarena, then solves the
// megarena's pose in each photo with the camera found.
//
// The photos should show the megarena at varied tilts (10 or more is best),
// with at least ~5 pixels per dot. Images are binary 8-bit PGM (see pgm.hpp).
//
// Usage: megarena <pitch> <code-size> <image.pgm>...
//   pitch      distance between dots, in the unit poses should come out in
//   code-size  LFSR order the megarena was rendered with (4..12)
//
// A synthetic set with its truth, to try it on:
//   cargo run --release -p vernier-camera --example megarena_views -- megarena-views
//   ./megarena 2 8 megarena-views/view_*.pgm
//   cat megarena-views/truth.txt

#include "vernier.hpp"
#include "pgm.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <pitch> <code-size> <image.pgm>...\n";
        return 2;
    }
    // The only difference from a checkerboard: the target.
    const vernier::Target target = vernier::Target::megarena(
        std::atof(argv[1]), static_cast<std::uint32_t>(std::atoi(argv[2])));

    // Measure every photo into dot ↔ pixel correspondences, skipping those
    // where the megarena is not found.
    std::vector<vernier::View> views;
    std::vector<std::string>   names;
    for (int i = 3; i < argc; ++i) {
        try {
            const Image   img  = load_pgm(argv[i]);
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
        // Intrinsics and distortion from all the views.
        const vernier::Calibration cal = vernier::calibrate(views, vernier::Model::Pinhole);
        const vernier::Camera&     c   = cal.camera;
        std::printf("\ncamera %zux%zu  fx %.3f  fy %.3f  cx %.3f  cy %.3f\n", c.width,
                    c.height, c.fx, c.fy, c.cx, c.cy);
        std::printf("distortion");
        for (double k : c.distortion) std::printf(" %.5f", k);
        std::printf("\nrms %.4f px over %zu views\n\n", cal.rms, cal.views.size());

        // The camera known, the megarena's pose in each view on its own:
        // board → camera, in the pitch's unit, origin at the megarena's
        // dot (0, 0). Only views whose code was read are in the megarena's
        // own frame.
        for (std::size_t i = 0; i < views.size(); ++i) {
            if (!views[i].is_absolute()) continue;
            const vernier::ViewFit fit = vernier::solve_pnp(c, views[i]);
            const auto&            r   = fit.pose.rvec;
            const auto&            t   = fit.pose.tvec;
            std::printf("%s: rvec (%.4f %.4f %.4f)  tvec (%.3f %.3f %.3f)  rms %.4f px\n",
                        names[i].c_str(), r[0], r[1], r[2], t[0], t[1], t[2], fit.rms);
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    return 0;
}
