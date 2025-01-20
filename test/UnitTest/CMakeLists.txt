add_library(${UNITTEST_LIB} 
  catch_amalgamated.cpp
  catch_amalgamated.hpp
  TestBRMeshRefine.h
  TestCellOperators.h
  TestComputeNorm.h
  TestLevelData.h
  TestWrapperSilo.h
)

add_executable(PerformUnitTests PerformUnitTests.cpp)
target_link_libraries(PerformUnitTests ${MAIN_LIB} ${COMMON_LIB} ${UNITTEST_LIB} mpi)