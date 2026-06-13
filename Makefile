# Thin convenience wrapper around CMake. Plain `make` configures + builds.
# CMake remains the source of truth; these are just shortcuts.

BUILD_DIR ?= build

.PHONY: all configure build test run format clean

all: build

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release

build: configure
	cmake --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

run: build
	./$(BUILD_DIR)/inference-engine --help

format:
	clang-format -i src/*.cpp src/*.hpp tests/*.cpp

clean:
	rm -rf $(BUILD_DIR)
