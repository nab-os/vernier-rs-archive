// Renders the poses listed in a CSV file with fmac's thin-lens camera, for
// the `fmac_precision` example of vernier-camera.
//
//   render <camera.json> <board.png> <poses.csv> <output-dir>
//
// Each line of poses.csv is `name,rx,ry,rz,tx,ty,tz`: an OpenCV rotation
// vector and translation, board to camera, in the camera file's unit.
// Writes <output-dir>/<name>.png.

#include <fstream>
#include <iostream>
#include <sstream>

#include "ThinLensCamera.hpp"

int main(int argc, char **argv) {
    if (argc != 5) {
        std::cerr << "usage: render <camera.json> <board.png> <poses.csv> <output-dir>" << std::endl;
        return 2;
    }
    ThinLensCamera camera(argv[1], argv[2]);
    std::cerr << camera;
    std::ifstream poses(argv[3]);
    std::string line;
    while (std::getline(poses, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream fields(line);
        std::string name, value;
        std::getline(fields, name, ',');
        double v[6];
        for (double &x : v) {
            std::getline(fields, value, ',');
            x = std::stod(value);
        }
        cv::Mat image;
        camera.render(cv::Vec3d(v[0], v[1], v[2]), cv::Vec3d(v[3], v[4], v[5]), image);
        cv::imwrite(std::string(argv[4]) + "/" + name + ".png", image);
        std::cerr << name << std::endl;
    }
    return 0;
}
