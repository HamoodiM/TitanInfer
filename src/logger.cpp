#include "titaninfer/logger.hpp"
#include <chrono>

namespace titaninfer {

Logger& Logger::instance() {
    static Logger logger;
    return logger;
}

Logger::Logger()
    : level_(LogLevel::INFO)
    , stream_(&std::cerr)
{}

void Logger::set_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(mutex_);
    level_ = level;
}

LogLevel Logger::level() const noexcept {
    return level_;
}

void Logger::set_stream(std::ostream& os) {
    std::lock_guard<std::mutex> lock(mutex_);
    stream_ = &os;
}

void Logger::log(LogLevel level, const std::string& message) {
    if (level < level_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    // Re-check under lock (level_ may have changed)
    if (level < level_) {
        return;
    }
    *stream_ << "[" << level_string(level) << "] ["
             << format_timestamp() << "] " << message << '\n';
    stream_->flush();
}

void Logger::debug(const std::string& msg) { log(LogLevel::DEBUG, msg); }
void Logger::info(const std::string& msg) { log(LogLevel::INFO, msg); }
void Logger::warning(const std::string& msg) { log(LogLevel::WARNING, msg); }
void Logger::error(const std::string& msg) { log(LogLevel::ERROR, msg); }

const char* Logger::level_string(LogLevel level) noexcept {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO";
        case LogLevel::WARNING: return "WARNING";
        case LogLevel::ERROR:   return "ERROR";
        case LogLevel::SILENT:  return "SILENT";
    }
    return "UNKNOWN";
}

std::string Logger::format_timestamp() {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto since_epoch = now.time_since_epoch();
    auto total_ms = duration_cast<milliseconds>(since_epoch).count();

    auto ms_part  = total_ms % 1000;
    auto total_s  = total_ms / 1000;
    auto seconds  = total_s % 60;
    auto total_m  = total_s / 60;
    auto minutes  = total_m % 60;
    auto hours    = (total_m / 60) % 24;

    char buf[64];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d",
                  static_cast<int>(hours),
                  static_cast<int>(minutes),
                  static_cast<int>(seconds),
                  static_cast<int>(ms_part));
    return buf;
}

} // namespace titaninfer
