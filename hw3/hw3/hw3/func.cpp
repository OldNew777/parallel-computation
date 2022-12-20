#include "func.h"

void initialize(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

void force(Body* p, float dt, int n) {
    RangeP rangeP{0, n, n};
    forceP(p, dt, n, rangeP);
}

void forceP(Body* p, float dt, int n, RangeP rangeP) {
    for (int i = rangeP.l; i < rangeP.r; i++) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = 1.0f / sqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx;
        p[i].vy += dt * Fy;
        p[i].vz += dt * Fz;
    }
}

//void CustomManagerFunc(Body* in, Body* inout, int* len, MPI_Datatype* dptr) {
//    for (int i = 0; i < *len; i++) {
//        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;
//
//        for (int j = 0; j < *len; j++) {
//            float dx = in[j].x - in[i].x;
//            float dy = in[j].y - in[i].y;
//            float dz = in[j].z - in[i].z;
//            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
//            float invDist = 1.0f / sqrtf(distSqr);
//            float invDist3 = invDist * invDist * invDist;
//
//            Fx += dx * invDist3;
//            Fy += dy * invDist3;
//            Fz += dz * invDist3;
//        }
//
//        inout[i].vx = in[i].vx + dt * Fx;
//        inout[i].vy = in[i].vy + dt * Fy;
//        inout[i].vz = in[i].vz + dt * Fz;
//    }
//}
//
//
//void forceR(Body* p, float dt, int n, RangeP rangeP) {
//    auto f = CustomManager::GetInstance();
//    MPI_Reduce(p, p, n, *f.DataType(), *f.Op(), id, MPI_COMM_WORLD);
//}