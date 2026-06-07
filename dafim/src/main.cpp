#include <mpi.h>
#include "dafim.hpp"

int RANK = 0;
int SIZE = 1;

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &RANK);
    MPI_Comm_size(MPI_COMM_WORLD, &SIZE);

    int rc = dafim_run(argc, argv);

    MPI_Finalize();
    return rc;
}
