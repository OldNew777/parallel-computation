//
// Created by ChenXin on 2022/12/20.
//

#include <omp.h>

#include "log.h"
#include "timer.h"
#include "SparceMatrix.h"

int main(int argc, char **argv) {

    CrossList matrix(100, 100, 5);
    LOG_DEBUG("%s", matrix.to_string().c_str());

    return 0;
}