add_library(${COMMON_LIB}
  DomainFactory.cpp
  DomainFactory.h
  FunctionFactory.cpp
  FunctionFactory.h
)
target_link_libraries(${COMMON_LIB} ${MAIN_LIB})
