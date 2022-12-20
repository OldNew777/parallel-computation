#include "log.h"

int main() {
    LOG_INFO("Hello world");
    LOG_WARN("Hello world");
    LOG_ERROR("Hello world");
    LOG_FLUSH();
    return 0;
}