#include "TestBRMeshRefine.h"
#include "TestCellOperators.h"
#include "TestComputeNorm.h"
#include "TestLevelData.h"
#include "TestWrapperSilo.h"

#include <Core/MPI.h>

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int result = Catch::Session().run(argc, argv);
  MPI_Finalize();
  return result;
}