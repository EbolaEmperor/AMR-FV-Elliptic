
#include "Core/Profile.h"

#include "petsclog.h"

PetscLogStage COMPUTE_RESIDUAL;
PetscLogStage COMPUTE_NORM;
PetscLogStage DIRECTSOLVER;
PetscLogStage BLOCKRELAXATION;
PetscLogStage APPLY_SMOOTHER;
PetscLogStage APPLY_RESTRICTION;
PetscLogStage APPLY_PROLONGATION;
PetscLogStage FILL_GHOSTS;
PetscLogStage EXCHANGE_ALL;
