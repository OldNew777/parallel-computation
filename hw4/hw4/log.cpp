#include "log.h"
#include <unordered_map>

void log_by_level(int level, const char *msg, ...) {
    static const std::unordered_map<int, string> level_map = {
            {LOG_COLOR::INFO,  "INFO"},
            {LOG_COLOR::WARN,  "WARN"},
            {LOG_COLOR::ERROR, "ERROR"}
    };
    
    string header = "\033[" + to_string(level) + ";40m[" + level_map.find(level)->second + "]\033[0m ";
    printf("%s", header.c_str());

    va_list args;
    va_start(args, msg);
    vprintf(msg, args);
    va_end(args);

    printf("\n");

    //va_list args;
    //va_start(args, msg);
    //vprintf(msg, args);
    //va_end(args);

    fflush(stdout);
}