#pragma once

#include <iostream>
#include <mutex>
#include <string>

namespace titaninfer {

/**
 * @brief Log severity levels
 */
enum class LogLevel { DEBUG = 0, INFO = 1, WARNING = 2, ERROR = 3, SILENT = 4 };

/**
 * @brief Thread-safe singleton logger with configurable level and output stream
 *
 * Default level is INFO, default output is std::cerr.
 * Use set_stream() in tests to redirect to std::ostringstream.
 *
 * Thread safety: all public methods are guarded by a mutex.
 */
class Logger {
public:
    static Logger& instance();

    void set_level(LogLevel level);
    LogLevel level() const noexcept;

    void set_stream(std::ostream& os);

    void log(LogLevel level, const std::string& message);

    void debug(const std::string& msg);
    void info(const std::string& msg);
    void warning(const std::string& msg);
    void error(const std::string& msg);

    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;

private:
    Logger();

    mutable std::mutex mutex_;
    LogLevel level_;
    std::ostream* stream_;

    static const char* level_string(LogLevel level) noexcept;
    static std::string format_timestamp();
};

} // namespace titaninfer

// Convenience macros — skip string construction when level is filtered
#define TITANINFER_LOG_DEBUG(msg)   \
    do { if (::titaninfer::Logger::instance().level() <= ::titaninfer::LogLevel::DEBUG) \
             ::titaninfer::Logger::instance().debug(msg); } while(false)

#define TITANINFER_LOG_INFO(msg)    \
    do { if (::titaninfer::Logger::instance().level() <= ::titaninfer::LogLevel::INFO) \
             ::titaninfer::Logger::instance().info(msg); } while(false)

#define TITANINFER_LOG_WARNING(msg) \
    do { if (::titaninfer::Logger::instance().level() <= ::titaninfer::LogLevel::WARNING) \
             ::titaninfer::Logger::instance().warning(msg); } while(false)

#define TITANINFER_LOG_ERROR(msg)   \
    do { if (::titaninfer::Logger::instance().level() <= ::titaninfer::LogLevel::ERROR) \
             ::titaninfer::Logger::instance().error(msg); } while(false)
