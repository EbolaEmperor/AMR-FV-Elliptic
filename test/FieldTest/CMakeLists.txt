add_executable(testPoissonCell testPoissonCell.cpp)
target_link_libraries(testPoissonCell ${MAIN_LIB} mpi)

add_executable(testBR testBR.cpp)
target_link_libraries(testBR ${MAIN_LIB} mpi)