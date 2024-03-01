add_library(quickhull OBJECT quickhull/QuickHull.cpp)
target_include_directories(quickhull PUBLIC
    $<INSTALL_INTERFACE:quickhull>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/quickhull>)

target_compile_features(quickhull PUBLIC cxx_std_17)
install(TARGETS quickhull EXPORT manifoldTargets)

set(VORO_BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(VORO_BUILD_EXAMPLES    OFF CACHE BOOL "")
set(VORO_BUILD_CMD_LINE    OFF CACHE BOOL "")
set(VORO_ENABLE_DOXYGEN    OFF CACHE BOOL "")
add_subdirectory(voro)
