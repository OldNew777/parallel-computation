#pragma warning(disable:4996)

#include <cstdio>
#include <mpi.h>
#include <chrono>
#include <cassert>
#include <string>
#include <fstream>
#include <functional>

#include "cxxopts.hpp"

#include "log.h"

int buff_size;
int n_repeat;
double write_ratio;
int id, n;
int comm = MPI_COMM_WORLD;
string filename = "__IO__.bin";
string file_input;
vector<bool> is_valid;

std::pair<double, double> test_rw() {
    MPI_File f;
    srand((unsigned) time(nullptr));
    MPI_File_open(comm, filename.c_str(), MPI_MODE_RDWR | MPI_MODE_CREATE, MPI_INFO_NULL, &f);
    auto buff = new char[buff_size];
    uint64_t n_read = 0, n_write = 0;
    auto time_start = std::chrono::high_resolution_clock::now();
    if (is_valid[id]) {
        for (int i = 0; i < n_repeat; ++i) {
            if (double(rand()) / RAND_MAX < write_ratio) { // write
                ++n_write;
                MPI_File_write(f, buff, buff_size, MPI_CHAR, MPI_STATUS_IGNORE);
            }
            else { // read
                ++n_read;
                MPI_File_read(f, buff, buff_size, MPI_CHAR, MPI_STATUS_IGNORE);
            }
        }
    }
    auto time_end = std::chrono::high_resolution_clock::now();
    double time_consumption = (double)std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / std::chrono::microseconds::period::den;
    MPI_File_close(&f);
    delete[] buff;
    return { buff_size * n_read / time_consumption, buff_size * n_write / time_consumption};
}

int parse_args(int argc, char **argv) {
    //if (id == 0) {
    //    LOG_INFO("===== args =====");
    //    for (int i = 0; i < argc; ++i)
    //        LOG_INFO("%s", argv[i]);
    //    LOG_INFO("================");
    //}

    string valid_str;

    cxxopts::Options opts("MPI file IO test");
    opts.add_options()
        ("s,size", "Size of send/recv buffer", cxxopts::value<int>(buff_size)->default_value("4096"))
        ("r,repeat", "Repeating times", cxxopts::value<int>(n_repeat)->default_value("1000"))
        ("w,write", "Ratio of write operations [0.0, 1.0]", cxxopts::value<double>(write_ratio)->default_value("0.3"))
        ("valid", "Valid nodes divided by ',' (empty = all)", cxxopts::value<std::string>(valid_str)->default_value(""))
        ("orders_file", "Orders file (will override other settings)", cxxopts::value<std::string>(file_input)->default_value(""))
        ("h,help", "Print help");

    auto result = opts.parse(argc, argv);
    //if (id == 0) {
    //    LOG_INFO("size = %d", buff_size);
    //    LOG_INFO("repeat = %d", n_repeat);
    //    LOG_INFO("write = %f", write_ratio);
    //    LOG_INFO("valid = %s", valid_str.c_str());
    //    LOG_INFO("orders_file = %s", file_input.c_str());
    //}

    if (result.count("help")) {
        if (id == 0) {
            printf("%s", opts.help().c_str());
        }
        fflush(stdout);
        return 0;
    }

    if (file_input != "") {
        return 2;
    }

    if (valid_str != "") {
        is_valid.resize(n, false);
        for (int i = 0; i < n; ++i) is_valid[i] = false;
        stringstream ss(valid_str);
        string item;
        while (getline(ss, item, ',')) {
            int index = stoi(item);
            is_valid[index] = true;
        }
    }
    else {
        is_valid.resize(n, true);
        for (int i = 0; i < n; ++i) is_valid[i] = true;
    }

    return 1;
}

void run() {
    if (is_valid[id]) {
        MPI_Comm_split(MPI_COMM_WORLD, is_valid[id], id, &comm);
    }
    else {
        MPI_Comm_split(MPI_COMM_WORLD, is_valid[id], id, &comm);
    }

    auto time_start = std::chrono::high_resolution_clock::now();
    auto rw_bandwidth = test_rw();
    auto time_end = std::chrono::high_resolution_clock::now();
    double time_consumption = (double)std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start).count() / std::chrono::microseconds::period::den;

    MPI_Comm_free(&comm);

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        LOG_INFO("********** bandwidth test **********");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (is_valid[id]) {
        LOG_INFO("P%d, Read: %.6ef, Write: %.6ef", id, rw_bandwidth.first, rw_bandwidth.second);
    }
    else {
        LOG_INFO("P%d, inactivate", id);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        LOG_INFO("***** all finished in %.6fs ****\n", time_consumption);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n);

    auto result = parse_args(argc, argv);
    MPI_Barrier(MPI_COMM_WORLD);

    std::function<void(int)> f = [&](int result) {
        if (result < 0) {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        else if (result == 0) {
            MPI_Finalize();
        }
        else if (result == 1) {
            run();
        }
        else if (result == 2) {
            ifstream fin(file_input, ios::in);
            if (!fin) {
                if (id == 0)
                    LOG_ERROR("File not existing");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            char buf[4096];
            char** orders = new char* [20];
            for (int i = 0; i < 20; ++i) orders[i] = new char[256];
            while (fin.getline(buf, 4096)) {
                auto order = string(buf);
                if (order == "") continue;
                if (id == 0) {
                    LOG_INFO("order = %s", order.c_str());
                }

                int n_param = 0;
                strcpy(orders[n_param++], argv[0]);

                stringstream ss(order);
                string item;
                while (getline(ss, item, ' ')) {
                    if (item[0] == '\"') {
                        assert(item[item.size() - 1] == '\"');
                        memcpy(orders[n_param], item.c_str() + 1, item.size() - 2);
                        orders[n_param][item.size() - 2] = '\0';
                    }
                    else {
                        strcpy(orders[n_param], item.c_str());
                    }
                    ++n_param;
                }

                result = parse_args(n_param, orders);
                MPI_Barrier(MPI_COMM_WORLD);

                f(result);
            }
            for (int i = 0; i < 20; ++i) delete[] orders[i];
            delete[] orders;
        }
    };

    f(result);

    MPI_Finalize();

    return 0;
}
