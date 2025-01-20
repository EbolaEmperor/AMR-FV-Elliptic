#!/bin/bash
if [ -e output ]; then
    echo "directory output/ exists."
else
    mkdir output
fi
# cmake -S . -B  build/ -DCMAKE_BUILD_TYPE=Debug -DDBGLEVEL=1
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=RELEASE -DPETSC_DIR=/opt/petsc/ -Dopenblas_DIR=/opt/homebrew/Cellar/openblas/0.3.28/lib/cmake/openblas
