// bbport BB_STALE_DESC_DIAG=1 (diagnostics, never default): a texture or color descriptor whose
// formats no surface format exists for (a "stale descriptor": read from game memory while the
// game rewrites it) is reported with its context: the thread, the caller, the descriptor's dwords,
// its image address and whether the draw was prepared by a draw-preparation worker; the first
// ones with a backtrace. Then the read-as-Undefined path continues instead of the assertion, so a
// whole run records every stale read, and every 5 s the counts are printed.
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <execinfo.h>
#include <pthread.h>
#include <unistd.h>

namespace BbStaleDesc {

inline bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("BB_STALE_DESC_DIAG");
        return env && env[0] == '1';
    }();
    return enabled;
}

/// What the converting code is working on (set by the callers of SurfaceFormat).
struct Context {
    const char* caller = nullptr;
    const std::uint32_t* dwords = nullptr;
    std::uint32_t num_dwords = 0;
    std::uint64_t address = 0;
};
inline thread_local Context context{};
/// Set by the rasterizer per draw: 1 prepared by a worker, 0 by this thread, -1 unknown.
inline thread_local int prepared_by_worker = -1;

/// Sets the context for one conversion and clears it again.
struct Scope {
    Scope(const char* caller, const void* descriptor, std::size_t bytes, std::uint64_t address) {
        if (Enabled()) {
            context = {caller, static_cast<const std::uint32_t*>(descriptor),
                       static_cast<std::uint32_t>(bytes / 4), address};
        }
    }
    ~Scope() {
        if (Enabled()) {
            context = {};
        }
    }
};

/// One stale read: `recovered` when a fallback number format gave a usable format.
inline void Report(unsigned data_format, unsigned num_format, bool recovered) {
    static std::mutex mutex;
    static std::uint64_t total = 0, unrecovered = 0, window_total = 0, window_unrecovered = 0;
    static auto last = std::chrono::steady_clock::now();
    std::scoped_lock lock{mutex};
    ++total;
    ++window_total;
    if (!recovered) {
        ++unrecovered;
        ++window_unrecovered;
    }
    char name[64] = "?";
    pthread_getname_np(pthread_self(), name, sizeof(name));
    if (total <= 200) {
        std::printf("Stale descriptor #%llu: data_format=%u num_format=%u %s; thread %s; caller %s; "
                    "image %#llx; prepared by worker %d; dwords",
                    (unsigned long long)total, data_format, num_format,
                    recovered ? "recovered by a fallback" : "UNKNOWN (would assert)", name,
                    context.caller ? context.caller : "?", (unsigned long long)context.address,
                    prepared_by_worker);
        for (std::uint32_t i = 0; i < context.num_dwords && i < 16; ++i) {
            std::printf(" %08x", context.dwords[i]);
        }
        std::printf("\n");
        if (total <= 20) {
            void* frames[12];
            const int n = backtrace(frames, 12);
            std::fflush(stdout);
            backtrace_symbols_fd(frames, n, STDOUT_FILENO);
        }
    }
    if (std::chrono::steady_clock::now() - last > std::chrono::seconds(5)) {
        last = std::chrono::steady_clock::now();
        std::printf("Stale descriptor counts (5 s): %llu reads, %llu unknown; in all %llu, %llu "
                    "unknown\n",
                    (unsigned long long)window_total, (unsigned long long)window_unrecovered,
                    (unsigned long long)total, (unsigned long long)unrecovered);
        window_total = window_unrecovered = 0;
    }
}

} // namespace BbStaleDesc
