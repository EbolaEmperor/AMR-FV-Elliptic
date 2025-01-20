# A fourth-order FV solver for elliptic equations with AMR and PC techniques 

My undergraduate graduation project.

## Dependent

gcc13, g++13, cmake, mpich, hdf5, silo

## How to compile

Run the following commands.

```bash
./configure.sh
./compile.sh
```

## How to run

### Poisson solver on AMR

Run the following command.

```bash
mpirun -np 4 ./bin/AMRPoisson example/AMRPoisson/Poisson-MultiGaussian-rect.json
```

The number after `-np` is the number of processes. The last argument is the json input file. You can try other json files or use your own json file.

Also, if your equation has no analytic solution, run the following command.

```bash
mpirun -np 4 ./bin/AMRPoissonNoAnalyticSol example/AMRPoissonNoAnalyticSol/Poisson-Constant-L.json
```

### Unit Test

Run the following command.

```bash
mpirun -np 4 ./bin/PerformUnitTests
```

The number after `-np` is the number of processes. The unit test whose default number of processes is not the `np` you set will be skipped.