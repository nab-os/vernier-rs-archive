LIB_DIR  := ../../target/release
INC_DIR  := ../include
CFLAGS   := -std=c11 -O2 -Wall -Wextra -I$(INC_DIR)
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(INC_DIR)
LDFLAGS  := -L$(LIB_DIR) -lvernier_cabi -Wl,-rpath,$(abspath $(LIB_DIR)) -lm

FMAC     := ../../resources/fmac-calibration

.PHONY: all clean run bench run-pnp run-pnp-megarena

all: detect bench calibrate_camera solve_pnp

detect: detect.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

bench: bench.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

calibrate_camera: calibrate_camera.cpp $(INC_DIR)/vernier.h $(INC_DIR)/vernier.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

solve_pnp: solve_pnp.cpp $(INC_DIR)/vernier.h $(INC_DIR)/vernier.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: detect
	./detect

# Calibrates from the fmac checkerboard set into camera.json, then solves
# every view's pose with it; compare with $(FMAC)/poses.csv.
run-pnp: calibrate_camera solve_pnp
	./calibrate_camera 5 6 camera.json $(FMAC)/view_*.png
	./solve_pnp camera.json 5 6 $(FMAC)/view_*.png

# The same from synthetic megarena views; compare with megarena-views/truth.txt.
run-pnp-megarena: calibrate_camera solve_pnp
	cargo run --release -q -p vernier-camera --example megarena_views -- $(abspath megarena-views)
	./calibrate_camera --megarena 2 8 camera-megarena.json megarena-views/view_*.pgm
	./solve_pnp --megarena camera-megarena.json 2 8 megarena-views/view_*.pgm

clean:
	rm -rf bench calibrate_camera solve_pnp megarena-views camera.json camera-megarena.json
