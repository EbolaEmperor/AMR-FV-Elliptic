cmake_minimum_required(VERSION 3.5)
project(NS4)

set(CMAKE_CXX_STANDARD 20)
set (CMAKE_C_COMPILER mpicc)
set (CMAKE_CXX_COMPILER mpicxx)
if(NOT DIM)
  set(DIM "2")
endif()

add_compile_options("-Wno-deprecated-enum-enum-conversion")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unknown-pragmas -Wno-unused-result -Wno-non-template-friend -Wno-unused-variable -DDIM=${DIM}")

# ifsanitize=address")

# Format
include (etc/clang_format.cmake)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wno-unused-result -DDIM=${DIM}")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DBGLEVEL "The verbosity. " -1)

if(DBGLEVEL GREATER -1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDBGLEVEL=${DBGLEVEL}")
endif()

option(USE_OPENMP "Whether to use the OpenMP parallelism." ON)

option(PROFILE "Whether to enable profiling. " OFF)

if(PROFILE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -pg")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(HDF5 REQUIRED hdf5)
include_directories(${HDF5_INCLUDE_DIRS})
link_directories(${HDF5_LIBRARY_DIRS})

if(USE_OPENMP)
  include_directories(/opt/homebrew/Cellar/libomp/19.1.2/include)
  link_directories(/opt/homebrew/Cellar/libomp/19.1.2/lib)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENMP -fopenmp")
endif()

# pkg_check_modules(SILO REQUIRED silo)
# include_directories(${SILO_INCLUDE_DIRS})
# link_directories(${SILO_LIBRARY_DIRS})
include_directories(/opt/silo/include)
link_directories(/opt/silo/lib)
pkg_check_modules(PETSC REQUIRED petsc)
include_directories(${PETSC_INCLUDE_DIRS})
link_directories(${PETSC_LIBRARY_DIRS})
MESSAGE(STATUS "get ${HDF5_LIBRARIES} from ${HDF5_INCLUDE_DIRS} and ${HDF5_LIBRARY_DIRS}")
MESSAGE(STATUS "get ${SILO_LIBRARIES} from ${SILO_INCLUDE_DIRS} and ${SILO_LIBRARY_DIRS}")
# MESSAGE(STATUS "get ${PETSC_LIBRARIES} from ${PETSC_INCLUDE_DIRS} and ${PETSC_LIBRARY_DIRS}")

# find_package(Boost 1.71.0 REQUIRED timer)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package(CGAL)

option(USE_MKL "Whether to use the Intel(R) MKL. " OFF)
option(USE_OPENBLAS "Whether to use the OpenBLAS. " ON)

if(USE_MKL)
  if(NOT DEFINED MKLROOT)
    if(NOT DEFINED ENV{MKLROOT})
      message(SEND_ERROR "Please set the MKLROOT variable. ")
    endif()

    set(MKLROOT $ENV{MKLROOT})
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_MKL -m64 -I${MKLROOT}/include")
elseif(USE_AOCL)
  if(NOT DEFINED AOCL_ROOT)
    if(NOT DEFINED ENV{AOCL_ROOT})
      message(SEND_ERROR "Please set the AOCL_ROOT variable. ")
    endif()

    set(AOCL_ROOT $ENV{AOCL_ROOT})
  endif()
elseif(USE_OPENBLAS)
  find_package(openblas CONFIG REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_OPENBLAS")
else()
  # find_package(PkgConfig REQUIRED)
  pkg_check_modules(BLAS REQUIRED blas)
  pkg_check_modules(CBLAS cblas)

  if(NOT ${BLAS_FOUND} AND NOT ${CBLAS_FOUND})
    message(SEND_ERROR "No BLAS library found. ")
  endif()

  pkg_check_modules(LAPACK REQUIRED lapack)
  pkg_check_modules(LAPACKE REQUIRED lapacke)
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

message("System : ${CMAKE_SYSTEM_NAME}")
message("Build type : ${CMAKE_BUILD_TYPE}")
message("CMAKE_CXX_FLAGS : ${CMAKE_CXX_FLAGS}")

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")
set(MAIN_LIB "NS4")
set(COMMON_LIB "${MAIN_LIB}_common")
set(UNITTEST_LIB "${MAIN_LIB}_UnitTest")

add_subdirectory(src)
add_subdirectory(example)

option(ENABLE_TEST "Whether to enable the tests. " ON)

if(ENABLE_TEST)
  add_subdirectory(test)
endif()

configure_file(${CMAKE_SOURCE_DIR}/src/Core/configure.h.in ${CMAKE_SOURCE_DIR}/src/Core/configure.h)
