include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CMAKE_SOURCE_DIR}/example/common)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)
add_executable(AMRPoisson AMRPoisson.cpp)
target_link_libraries(AMRPoisson ${MAIN_LIB} ${COMMON_LIB} mpi)