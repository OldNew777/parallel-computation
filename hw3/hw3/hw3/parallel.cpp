#include <vector>
#include <mpi.h>

#include "timer.h"
#include "func.h"
#include "log.h"

using namespace std;

int id, np;
vector<RangeP> rangePVec;

namespace ParallelVersion {

    void simulate(int nBodies) {
        const float dt = 0.01f; // time step
        const int nIters = 10;  // simulation iterations

        // init
        int bytes = nBodies * sizeof(Body);
        float* buf = (float*)malloc(bytes);
        Body* p = (Body*)buf;

        rangePVec.resize(np);
        double invR = (double)nBodies / np;
        rangePVec[0].l = 0;
        rangePVec[np - 1].r = nBodies;
        for (int i = 1; i < np; ++i) {
            int index = i * invR;
            rangePVec[i - 1].r = index;
            rangePVec[i].l = index;
        }
        for (int i = 0; i < np; ++i)
            rangePVec[i].size = rangePVec[i].r - rangePVec[i].l;
        RangeP &rangeP = rangePVec[id];
        LOG_INFO("nBodies = %d, P%d [%d, %d)", nBodies, id, rangeP.l, rangeP.r);

        initialize((float*)(p + rangeP.l), rangeP.size); // Init pos / vel data

        auto mgr = CustomManager::GetInstance();
        for (int i = 0; i < np; ++i)
            MPI_Bcast(p + rangePVec[i].l, rangePVec[i].size, *mgr.DataType(), i, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        double totalTime = 0.0;

        for (int iter = 1; iter <= nIters; iter++) {
            Timer::Tik();

            forceP(p, dt, nBodies, rangeP); // compute interbody forces

            for (int i = rangeP.l; i < rangeP.r; i++) { // integrate position
                p[i].x += p[i].vx * dt;
                p[i].y += p[i].vy * dt;
                p[i].z += p[i].vz * dt;
            }
            for (int i = 0; i < np; ++i)
                MPI_Bcast(p + rangePVec[i].l, rangePVec[i].size, *mgr.DataType(), i, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            const double tElapsed = Timer::Tok();
            if (iter > 1) { // First iter is warm up
                totalTime += tElapsed;
            }
#ifndef SHMOO
            if (id == 0) {
                LOG_INFO("Iteration %d: %.3f seconds", iter, tElapsed);
            }
#endif
        }
        double avgTime = totalTime / (double)(nIters - 1);

        MPI_Barrier(MPI_COMM_WORLD);

#ifdef SHMOO
        if (id == 0) {
            LOG_INFO("%d, %0.3f", nBodies, 1e-9 * nBodies * nBodies / avgTime);
        }
#else
        if (id == 0) {
            LOG_INFO("Average rate for iterations 2 through %d: %.3f steps per second.",
                nIters, 1.f / avgTime);
            LOG_INFO("%d Bodies: average %0.3f Billion Interactions / second", nBodies, 1e-9 * nBodies * nBodies / avgTime);
        }
#endif
        free(buf);
    }

}

int main(int argc, char** argv) {
    int nBodies = 30000;
    if (argc > 1) nBodies = atoi(argv[1]);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    ParallelVersion::simulate(nBodies);

    MPI_Finalize();

    return 0;
}