#include "timer.h"
#include "func.h"

namespace SequenceVersion {

    void simulate(int nBodies) {
        const float dt = 0.01f; // time step
        const int nIters = 10;  // simulation iterations

        int bytes = nBodies * sizeof(Body);
        float* buf = (float*)malloc(bytes);
        Body* p = (Body*)buf;

        initialize(buf, 6 * nBodies); // Init pos / vel data

        double totalTime = 0.0;

        for (int iter = 1; iter <= nIters; iter++) {
            Timer::Tik();

            force(p, dt, nBodies); // compute interbody forces

            for (int i = 0; i < nBodies; i++) { // integrate position
                p[i].x += p[i].vx * dt;
                p[i].y += p[i].vy * dt;
                p[i].z += p[i].vz * dt;
            }

            const double tElapsed = Timer::Tok();
            if (iter > 1) { // First iter is warm up
                totalTime += tElapsed;
            }
#ifndef SHMOO
            printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
#endif
        }
        double avgTime = totalTime / (double)(nIters - 1);

#ifdef SHMOO
        printf("%d, %0.3f\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#else
        printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
            nIters, 1.f / avgTime);
        printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);
#endif
        free(buf);
    }

}

//int main(const int argc, const char** argv) {
//    int nBodies = 30000;
//    if (argc > 1) nBodies = atoi(argv[1]);
//
//    SequenceVersion::simulate(nBodies);
//}
