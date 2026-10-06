// A binary 8-bit PGM reader, to keep the examples free of an image library.
// Convert other formats with e.g. `magick view_00.png view_00.pgm`.
#pragma once

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Image {
    std::size_t        width = 0, height = 0;
    std::vector<float> pixels;  // row-major, [0, 1]
};

// Reads a binary (P5) 8-bit PGM.
inline Image load_pgm(const std::string& path) {
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
