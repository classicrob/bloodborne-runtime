// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_STAGE_A_LOG=<file> (diagnostics): the GPU command thread's (stage A's) time per part of
// its per-draw work, one CSV line per frame: what grows in slow frames. Inclusive (nested parts
// count in their parent too). Ticks of the CPU counter; each line has the frame's ticks and
// nanoseconds of wall time to convert.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#if defined(__x86_64__)
#include <x86intrin.h> // arm64: bbport_arm64_compat.h
#endif

namespace BbStageA {
enum Id : std::uint32_t {
    Draw,      ///< Rasterizer::Draw
    Pipeline,  ///< PipelineCache::GetGraphicsPipeline
    Prepared,  ///< PipelineCache::TryPreparedPipeline
    Select,    ///< GetGraphicsPipeline's own selection (RefreshGraphicsKey and lookup)
    PostDraw,  ///< Rasterizer::PostDraw (the packet for stage B)
    Rings,     ///< CollectRingBindings (constants into the ring, pending reads)
    Dispatch,  ///< Rasterizer::DispatchDirect
    Count,
};
inline constexpr const char* Names[Count] = {"draw", "pipeline", "prepared", "select",
                                             "postdraw", "rings", "dispatch"};
enum Counter : std::uint32_t {
    Draws,
    PreparedUsed, ///< draws whose pipeline came from a worker's preparation
    Dispatches,
    NumCounters,
};
inline constexpr const char* CounterNames[NumCounters] = {"draws", "prepared_used", "dispatches"};

inline std::FILE* File() {
    static std::FILE* file = [] {
        const char* path = std::getenv("BB_STAGE_A_LOG");
        return path && *path ? std::fopen(path, "w") : nullptr;
    }();
    return file;
}
inline bool Enabled() {
    static const bool on = File() != nullptr;
    return on;
}

/// Stage A only (single thread): set by the GPU command thread on its first draw.
inline thread_local bool stage_a = false;
inline std::array<std::uint64_t, Count> ticks{};
inline std::array<std::uint64_t, NumCounters> counts{};

inline void Add(Counter c, std::uint64_t n = 1) {
    if (stage_a) {
        counts[c] += n;
    }
}

/// Stage A, once per draw: at a new frame number, writes the last frame's line.
/// `wait_ticks`: the draw pipe's stage A wait counter (drain_cycles), cumulative.
inline void FrameCheck(std::uint32_t frame, std::uint64_t wait_ticks) {
    static std::uint32_t last_frame = ~0u;
    static std::uint64_t last_tick = 0, last_wait = 0;
    static std::chrono::steady_clock::time_point last_time{};
    if (frame == last_frame) {
        return;
    }
    const std::uint64_t now_tick = __rdtsc();
    const auto now = std::chrono::steady_clock::now();
    auto* file = File();
    if (last_frame == ~0u) {
        std::fprintf(file, "frame,wall_ns,wall_ticks,wait_ticks");
        for (const char* name : Names) {
            std::fprintf(file, ",%s", name);
        }
        for (const char* name : CounterNames) {
            std::fprintf(file, ",%s", name);
        }
        std::fprintf(file, "\n");
    } else {
        std::fprintf(file, "%u,%lld,%llu,%llu", last_frame,
                     static_cast<long long>(
                         std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_time)
                             .count()),
                     static_cast<unsigned long long>(now_tick - last_tick),
                     static_cast<unsigned long long>(wait_ticks - last_wait));
        for (const auto t : ticks) {
            std::fprintf(file, ",%llu", static_cast<unsigned long long>(t));
        }
        for (const auto c : counts) {
            std::fprintf(file, ",%llu", static_cast<unsigned long long>(c));
        }
        std::fprintf(file, "\n");
        if ((last_frame & 63) == 0) {
            std::fflush(file);
        }
    }
    ticks.fill(0);
    counts.fill(0);
    last_frame = frame;
    last_tick = now_tick;
    last_wait = wait_ticks;
    last_time = now;
}

struct Scope {
    explicit Scope(Id id_) : id{id_}, start{stage_a ? __rdtsc() : 0} {}
    ~Scope() {
        if (start) {
            ticks[id] += __rdtsc() - start;
        }
    }
    Id id;
    std::uint64_t start;
};
} // namespace BbStageA

#define BB_STAGE_A_JOIN2(a, b) a##b
#define BB_STAGE_A_JOIN(a, b) BB_STAGE_A_JOIN2(a, b)
#define BB_STAGE_A(id) BbStageA::Scope BB_STAGE_A_JOIN(bb_stage_a_scope_, __LINE__){BbStageA::id}
