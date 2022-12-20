#include <string>
#include <stdarg.h>

using namespace std;

enum LOG_COLOR {
    DEBUG = 90,
    INFO = 32,
    WARN = 33,
    ERROR = 31
};

void log_by_level(int level, const char *msg, ...);

#define LOG_DEBUG(msg, ...) log_by_level(LOG_COLOR::DEBUG, msg, ##__VA_ARGS__)
#define LOG_INFO(msg, ...) log_by_level(LOG_COLOR::INFO, msg, ##__VA_ARGS__)
#define LOG_WARN(msg, ...) log_by_level(LOG_COLOR::WARN, msg, ##__VA_ARGS__)
#define LOG_ERROR(msg, ...) log_by_level(LOG_COLOR::ERROR, msg, ##__VA_ARGS__)
#define LOG_FLUSH() fflush(stdout)