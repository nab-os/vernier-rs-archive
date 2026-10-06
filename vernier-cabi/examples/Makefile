LIB_DIR  := ../../target/release
INC_DIR  := ../include
CFLAGS   := -std=c11 -O2 -Wall -Wextra -I$(INC_DIR)
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(INC_DIR)
LDFLAGS  := -L$(LIB_DIR) -lvernier_cabi -Wl,-rpath,$(abspath $(LIB_DIR)) -lm

FMAC     := ../../resources/fmac-calibration

.PHONY: all clean run bench run-calibrate run-megarena

all: detect bench calibrate megarena

detect: detect.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

bench: bench.c $(INC_DIR)/vernier.h
	$(CC) $(CFLAGS) $< $(LDFLAGS) -o $@

calibrate: calibrate.cpp pgm.hpp $(INC_DIR)/vernier.h $(INC_DIR)/vernier.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

megarena: megarena.cpp pgm.hpp $(INC_DIR)/vernier.h $(INC_DIR)/vernier.hpp
	$(CXX) $(CXXFLAGS) $< $(LDFLAGS) -o $@

run: detect
	./detect

# Calibrates from the fmac set (needs ImageMagick to turn the PNGs into PGM).
run-calibrate: calibrate
	mkdir -p fmac-pgm
	for f in $(FMAC)/view_*.png; do magick "$$f" "fmac-pgm/$$(basename "$${f%.png}").pgm"; done
	./calibrate 5 6 fmac-pgm/view_*.pgm

# Renders synthetic views of a megarena, then calibrates from them; compare
# with megarena-views/truth.txt.
run-megarena: megarena
	cargo run --release -q -p vernier-camera --example megarena_views -- $(abspath megarena-views)
	./megarena 2 8 megarena-views/view_*.pgm
	cat megarena-views/truth.txt

clean:
	rm -rf detect bench calibrate megarena fmac-pgm megarena-views
