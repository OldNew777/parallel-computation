#pragma once

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <mpi.h>

#define SOFTENING 1e-9f

typedef struct { float x, y, z, vx, vy, vz; } Body;

struct RangeP {
    int l, r;
    int size;
};

void initialize(float* data, int n);

void force(Body* p, float dt, int n);
void forceP(Body* p, float dt, int n, RangeP rangeP);
void forceR(Body* p, float dt, int n, RangeP rangeP);

class CustomManager {
private:
    MPI_Datatype dataType;

    CustomManager() {
        MPI_Type_contiguous(6, MPI_FLOAT, &dataType);
        MPI_Type_commit(&dataType);
    }

public:
    static CustomManager& GetInstance() {
        static CustomManager instance;
        return instance;
    }

    auto RegisterOp(MPI_User_function* f) const {
        MPI_Op op;
        MPI_Op_create(f, 1, &op);
        return &op;
    }
    auto DataType() const {
        return &dataType;
    }
};
