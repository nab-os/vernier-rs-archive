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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct Image {
    std::size_t        width = 0, height = 0;
    std::vector<float> pixels;  // row-major, [0, 1]
};

// Reads a binary (P5) 8-bit PGM.
static Image load_pgm(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::string   magic;
    int           max = 0;
    Image         img;
    in >> magic;
    // Skip comments between header fields.
    auto field = [&](auto& value) {
        in >> std::ws;
        while (in.peek() == '#') {
            std::string line;
            std::getline(in, line);
            in >> std::ws;
        }
        in >> value;
    };
    field(img.width);
    field(img.height);
    field(max);
    in.get();
    if (!in || magic != "P5" || max <= 0 || max > 255)
        throw std::runtime_error(path + ": not an 8-bit binary PGM");
    std::vector<unsigned char> raw(img.width * img.height);
    in.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(raw.size()));
    if (!in) throw std::runtime_error(path + ": truncated");
    img.pixels.resize(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) img.pixels[i] = raw[i] / float(max);
    return img;
}

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
