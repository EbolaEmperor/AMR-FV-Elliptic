
#pragma once

#include "SpatialOp/Petsc.h"
#include "petsclog.h"

#define registerLogStage(stage) PetscLogStageRegister(#stage, &stage)
#define pushLogStage(stage) PetscLogStagePush(stage)
#define popLogStage() PetscLogStagePop()

extern PetscLogStage COMPUTE_RESIDUAL;
extern PetscLogStage COMPUTE_NORM;
extern PetscLogStage DIRECTSOLVER;
extern PetscLogStage BLOCKRELAXATION;
extern PetscLogStage APPLY_SMOOTHER;
extern PetscLogStage APPLY_RESTRICTION;
extern PetscLogStage APPLY_PROLONGATION;
extern PetscLogStage FILL_GHOSTS;
extern PetscLogStage EXCHANGE_ALL;
