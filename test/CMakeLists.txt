include_directories(${CMAKE_SOURCE_DIR}/src)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_SOURCE_DIR}/bin)



add_subdirectory(UnitTest)
add_subdirectory(FieldTest)
