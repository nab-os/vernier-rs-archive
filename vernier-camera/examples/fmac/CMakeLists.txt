# Builds the fmac renderer the `fmac_precision` example drives, against a
# build of fmac, the Vernier project's marker renderer:
#
#   cmake -S <fmac> -B <fmac-build> && cmake --build <fmac-build> --target fmac
#   cmake -S vernier-camera/examples/fmac -B <render-build> \
#         -DFMAC_DIR=<fmac> -DFMAC_BUILD=<fmac-build>
#   cmake --build <render-build>
cmake_minimum_required(VERSION 3.10...3.31)
project(fmac_render LANGUAGES C CXX)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(FMAC_DIR "" CACHE PATH "fmac source directory")
set(FMAC_BUILD "" CACHE PATH "fmac build directory")
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP)
add_executable(render render.cpp)
target_include_directories(render PRIVATE ${FMAC_DIR}/include ${FMAC_DIR}/3rdparty/bessel
  ${FMAC_DIR}/3rdparty/gruenschloss/kuo-2d-proj ${FMAC_DIR}/3rdparty/gruenschloss/halton)
target_link_libraries(render ${FMAC_BUILD}/src/libfmac.a Eigen3::Eigen ${OpenCV_LIBS})
if(OpenMP_CXX_FOUND)
  target_link_libraries(render OpenMP::OpenMP_CXX)
endif()
