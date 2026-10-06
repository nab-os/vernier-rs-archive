LIB_DIR  := ../../target/release
INC_DIR  := ../include
CFLAGS   := -std=c11 -O2 -Wall -Wextra -I$(INC_DIR)
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(INC_DIR)
LDFLAGS  := -L$(LIB_DIR) -lvernier_cabi -Wl,-rpath,$(abspath $(LIB_DIR)) -lm

FMAC     := ../../resources/fmac-calibration

.PHONY: all clean run bench run-calibrate

all: detect bench calibrate

detect: detect.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

bench: bench.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

calibrate: calibrate.cpp $(INC_DIR)/vernier.h $(INC_DIR)/vernier.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: detect
	./detect

# Calibrates from the fmac set (needs ImageMagick to turn the PNGs into PGM).
run-calibrate: calibrate
	mkdir -p fmac-pgm
	for f in $(FMAC)/view_*.png; do magick "$$f" "fmac-pgm/$$(basename "$${f%.png}").pgm"; done
	./calibrate 5 6 fmac-pgm/view_*.pgm

clean:
	rm -rf detect bench calibrate fmac-pgm
