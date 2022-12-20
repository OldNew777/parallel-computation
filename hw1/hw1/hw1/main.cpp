#pragma warning(disable:4996)

#include <cstdio>
#include <mpi.h>
#include <ctime>
#include <cassert>
#include <fstream>

#include "cxxopts.hpp"
#include "log.h"
#include "func.h"

#define MAX_PROCESSES 256

string mode;
int buff_size;
int n_repeat;
int id, id_new;
int comm = MPI_COMM_WORLD;
string file_str;
int id_old2new[MAX_PROCESSES];
int id_new2old[MAX_PROCESSES];

int n, n_valid, n_root = 0;
vector<bool> is_root;
vector<bool> is_valid;

void check0() {
    if (n_root == 0) {
        if (id == 0) {
            LOG_ERROR("Root nodes undefined");
            LOG_FLUSH();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (n_valid - n_root < 0) {
        if (id == 0) {
            LOG_ERROR("Valid nodes error");
            LOG_FLUSH();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double test_bcast() {
    check0();
    vector<char> buff(n * buff_size);
    reset_2d_buffer(buff, n, buff_size);
    if (is_root[id]) {
        fill_buffer(&buff[id_new * buff_size], buff_size);
    }
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        for (int i = 0; i < n; ++i) {
            if (is_root[id_new2old[i]]) {
                MPI_Bcast(&buff[i * buff_size], buff_size, MPI_CHAR, i, comm);
            }
        }
        time_consumption += clock() - time_start;
        for (int i = 0; i < n; ++i) {
            assert_buffer(&buff[i * buff_size], buff_size);
        }
        reset_2d_buffer(buff, n, buff_size);
        if (is_root[id]) {
            fill_buffer(&buff[id_new * buff_size], buff_size);
        }
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC / n_root;
}

double test_gather() {
    check0();
    vector<char> send_buff(buff_size), recv_buff(n * n * buff_size);
    reset_3d_buffer(recv_buff, n, n, buff_size);
    if (is_valid[id]) {
        fill_buffer(send_buff, buff_size);
    } else {
        reset_buffer(send_buff, buff_size);
    }
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        for (int i = 0; i < n; ++i) {
            if (is_root[id_new2old[i]]) {
                MPI_Gather(send_buff.data(), buff_size, MPI_CHAR, &recv_buff[i * n * buff_size], buff_size, MPI_CHAR, i, comm);
            }
        }
        time_consumption += clock() - time_start;
        if (is_root[id]) {
            for (int i = 0; i < n; ++i) {
                if (!is_valid[id_new2old[i]]) {
                    continue;
                }
                assert_buffer(&recv_buff[i* n * buff_size], buff_size);
            }
        }
        reset_3d_buffer(recv_buff, n, n, buff_size);
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC / n_root;
}

double test_reduce() {
    check0();
    vector<char> send_buff(buff_size), recv_buff(n * buff_size);
    for (int i = 0; i < buff_size; ++i) send_buff[i] = 1;
    reset_2d_buffer(recv_buff, n, buff_size);
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        for (int i = 0; i < n; ++i) {
            if (is_root[id_new2old[i]]) {
                MPI_Reduce(send_buff.data(), &recv_buff[i * buff_size], buff_size, MPI_CHAR, MPI_SUM, i, comm);
            }
        }
        time_consumption += clock() - time_start;
        if (is_root[id]) {
            for (int i = 0; i < n; ++i) {
                if (!is_valid[id_new2old[i]]) continue;
                for (int j = 0; j < buff_size; ++j) {
                    assert(recv_buff[i * buff_size + j] == n_valid);
                }
            }
        }
        reset_2d_buffer(recv_buff, n, buff_size);
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC / n_root;
}

double test_allreduce() {
    vector<char> send_buff(buff_size), recv_buff(buff_size);
    for (int i = 0; i < buff_size; ++i) send_buff[i] = 1;
    reset_buffer(recv_buff, buff_size);
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        MPI_Allreduce(send_buff.data(), recv_buff.data(), buff_size, MPI_CHAR, MPI_SUM, comm);
        time_consumption += clock() - time_start;
        for (int i = 0; i < buff_size; ++i) {
            assert(recv_buff[i] == n_valid);
        }
        reset_buffer(recv_buff, buff_size);
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC;
}

double test_scan() {
    vector<char> send_buff(buff_size), recv_buff(buff_size);
    for (int i = 0; i < buff_size; ++i) send_buff[i] = 1;
    reset_buffer(recv_buff, buff_size);
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        MPI_Scan(send_buff.data(), recv_buff.data(), buff_size, MPI_CHAR, MPI_SUM, comm);
        time_consumption += clock() - time_start;
        for (int i = 0; i < buff_size; ++i) {
            assert(recv_buff[i] == id_new + 1);
        }
        reset_buffer(recv_buff, buff_size);
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC;
}

double test_alltoall() {
    vector<char> send_buff(n * buff_size), recv_buff(n * buff_size);
    for (int i = 0; i < n * buff_size; ++i) send_buff[i] = (char) id_new;
    reset_2d_buffer(recv_buff, n, buff_size);
    MPI_Barrier(comm);
    clock_t time_consumption = 0;
    for (int r = 0; r < n_repeat; ++r) {
        auto time_start = clock();
        MPI_Alltoall(send_buff.data(), buff_size, MPI_CHAR, recv_buff.data(), buff_size, MPI_CHAR, comm);
        time_consumption += clock() - time_start;
        for (int i = 0; i < n; ++i) {
            if (!is_valid[id_new2old[i]]) {
                continue;
            }
            for (int j = i * buff_size; j < (i + 1) * buff_size; ++j) {
                assert(recv_buff[j] == i);
            }
        }
        reset_2d_buffer(recv_buff, n, buff_size);
    }
    return double(time_consumption) / n_repeat / CLOCKS_PER_SEC;
}

int parse_args(int argc, char **argv) {
    //if (id == 0) {
    //    for (int i = 0; i < argc; ++i) {
    //        LOG_INFO("%s", argv[i]);
    //        LOG_FLUSH();
    //    }
    //}

    string root_str, valid_str;

    cxxopts::Options opts("Collective communication bandwidth test");
    opts.add_options()
            ("m,mode", "Testing mode (bcast/gather/reduce/allreduce/scan/alltoall)",
             cxxopts::value<std::string>(mode)->default_value("bcast"))
            ("s,size", "Number of elements in send buffer", cxxopts::value<int>(buff_size)->default_value("4096"))
            ("r,repeat", "Repeating times", cxxopts::value<int>(n_repeat)->default_value("1000"))
            ("root", "Root processes id divided by ','", cxxopts::value<std::string>(root_str)->default_value("0"))
            ("v,valid", "Valid nodes divided by ',' (empty = all)", cxxopts::value<std::string>(valid_str))
            ("orders_file", "Orders file (will override other settings)", cxxopts::value<std::string>(file_str))
            ("h,help", "Print help");

    auto result = opts.parse(argc, argv);

    // help
    if (result.count("help")) {
        if (id == 0) {
            printf("%s\n", opts.help().c_str());
            fflush(stdout);
        }
        return 0;
    }

    // file
    if (result.count("orders_file")) {
        return 2;
    }

    // valid
    if (result.count("valid")) {
        n_valid = 0;
        is_valid.resize(n, false);
        for (int i = 0; i < n; ++i) is_valid[i] = false;
        stringstream ss(valid_str);
        string item;
        while (getline(ss, item, ',')) {
            is_valid[stoi(item)] = true;
            ++n_valid;
        }
    } else {
        n_valid = n;
        is_valid.resize(n, true);
        for (int i = 0; i < n; ++i) is_valid[i] = true;
    }

    // root
    {
        n_root = 0;
        is_root.resize(n, false);
        for (int i = 0; i < n; ++i) is_root[i] = false;
        stringstream ss(root_str);
        string item;
        while (getline(ss, item, ',')) {
            int index = stoi(item);
            if (!is_valid[index]) {
                continue;
            }
            is_root[index] = true;
            ++n_root;
        }
    }

    // send/recv count
    if (buff_size < 0) {
        if (id == 0) {
            printf("Invalid send count!\n");
        }
        return -1;
    }

    return 1;
}

void function() {
    for (int i = 0; i < n; ++i) {
        id_old2new[i] = -1;
        id_new2old[i] = -1;
    }
    int id_valid = 0;
    for (int i = 0; i < n; ++i) {
        if (is_valid[i] && is_root[i]) {
            id_old2new[i] = id_valid;
            id_new2old[id_valid] = i;
            ++id_valid;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (is_valid[i] && !is_root[i]) {
            id_old2new[i] = id_valid;
            id_new2old[id_valid] = i;
            ++id_valid;
        }
    }
    id_new = id_old2new[id];
    MPI_Comm_split(MPI_COMM_WORLD, is_valid[id], id_new, &comm);
    LOG_INFO("id = %d, id_new = %d", id, id_new);

    double (*test)() = nullptr;
    if (mode == "bcast") test = test_bcast;
    else if (mode == "gather") test = test_gather;
    else if (mode == "reduce") test = test_reduce;
    else if (mode == "allreduce") test = test_allreduce;
    else if (mode == "scan") test = test_scan;
    else if (mode == "alltoall") test = test_alltoall;
    else {
        if (id == 0) {
            LOG_ERROR("Unknown mode: %s", mode.c_str());
            LOG_FLUSH();
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    double time_consumption = test();

    MPI_Comm_free(&comm);

    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        LOG_INFO("********** %s bandwidth test **********", mode.c_str());
        LOG_FLUSH();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (is_valid[id]) {
        LOG_INFO("Node %d: %.6ef", id, buff_size / time_consumption);
        LOG_FLUSH();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (id == 0) {
        LOG_INFO("");
        LOG_FLUSH();
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char **argv) {
    // init MPI and get process info
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    if (n <= 1) {
        LOG_ERROR("Number of processes must be greater than 1");
        LOG_FLUSH();
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto result = parse_args(argc, argv);
    if (result == 0) {
        MPI_Finalize();
        return 0;
    }
    else if (result == -1) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    else if (result == 1) {
        function();
    }
    else if (result == 2) {
        ifstream fin(file_str, ios::in);
        if (!fin) {
            LOG_ERROR("File not existing");
            LOG_FLUSH();
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        char buf[4096];
        char **orders = new char*[20];
        for (int i = 0; i < 20; ++i) orders[i] = new char[128];
        while (fin.getline(buf, 4096)) {
            auto order = string(buf);
            if (order == "") continue;
            int n_param = 0;

            LOG_INFO("order = %s", order.c_str());
            LOG_FLUSH();

            stringstream ss(order);
            string item;
            while (getline(ss, item, ' ')) {
                if (item.size() > 0 && item[0] == '\"') {
                    memcpy(orders[n_param], item.c_str() + 1, item.size() - 2);
                    orders[n_param][item.size() - 2] = '\0';
                }
                else {
                    memcpy(orders[n_param], item.c_str(), item.size());
                    orders[n_param][item.size()] = '\0';
                }
                n_param++;
            }

            parse_args(n_param, orders);
            function();
        }
        for (int i = 0; i < 20; ++i) delete[] orders[i];
        delete[] orders;
    }

    MPI_Finalize();
    return 0;
}
