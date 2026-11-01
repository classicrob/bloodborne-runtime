// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_METAL_GPU_TIMING=1: see mtl_gpu_timing.h.

#include "video_core/renderer_metal/mtl_gpu_timing.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "bbport_toggles.h"

namespace MetalGpuTiming {

namespace {

constexpr unsigned SamplesPerBuffer = 1024; // 8 KiB: ~250 render encoders of one command buffer
constexpr unsigned MaxBuffers = 512;

struct Encoder {
    char kind; ///< R(ender), C(ompute), B(lit)
    unsigned first; ///< first sample index
    unsigned work{}; ///< draws or dispatches
    unsigned long long frame;
    // Render passes: what the pass draws into.
    std::string signature;
    const char* origin{};
    unsigned colors{}, width{}, height{};
    bool depth{};
};

/// One encoder's GPU interval, resolved.
struct Record {
    Encoder encoder;
    double start, end; ///< ns from the calibration point
    double vertex, fragment; ///< ns, render passes
};

bool enabled = false;
id<MTLDevice> device;
id<MTLCounterSet> timestamps;
std::mutex pool_mutex;
std::vector<id<MTLCounterSampleBuffer>> pool;
unsigned created = 0;
std::atomic<unsigned long long> sampled{0}, unsampled{0};

// Calibration: GPU ticks to ns, refined at each report.
MTLTimestamp gpu0{};
std::chrono::steady_clock::time_point steady0;
double ns_per_tick = 1.0;

std::mutex records_mutex;
std::vector<Record> records;
std::chrono::steady_clock::time_point last_report;

const char* FormatName(MTLPixelFormat f) {
    switch (f) {
    case MTLPixelFormatRGBA8Unorm: return "RGBA8";
    case MTLPixelFormatRGBA8Unorm_sRGB: return "RGBA8s";
    case MTLPixelFormatBGRA8Unorm: return "BGRA8";
    case MTLPixelFormatBGRA8Unorm_sRGB: return "BGRA8s";
    case MTLPixelFormatRG11B10Float: return "R11G11B10F";
    case MTLPixelFormatRGB10A2Unorm: return "RGB10A2";
    case MTLPixelFormatRGBA16Float: return "RGBA16F";
    case MTLPixelFormatRGBA16Unorm: return "RGBA16";
    case MTLPixelFormatRG16Float: return "RG16F";
    case MTLPixelFormatRG16Unorm: return "RG16";
    case MTLPixelFormatR16Float: return "R16F";
    case MTLPixelFormatR16Unorm: return "R16";
    case MTLPixelFormatR32Float: return "R32F";
    case MTLPixelFormatRG32Float: return "RG32F";
    case MTLPixelFormatRGBA32Float: return "RGBA32F";
    case MTLPixelFormatR8Unorm: return "R8";
    case MTLPixelFormatRG8Unorm: return "RG8";
    case MTLPixelFormatR32Uint: return "R32U";
    case MTLPixelFormatDepth16Unorm: return "D16";
    case MTLPixelFormatDepth32Float: return "D32";
    case MTLPixelFormatDepth32Float_Stencil8: return "D32S8";
    case MTLPixelFormatStencil8: return "S8";
    default: return nullptr;
    }
}

std::string Format(MTLPixelFormat f) {
    if (const char* name = FormatName(f)) {
        return name;
    }
    return "fmt" + std::to_string(unsigned(f));
}

Command* Ensure(Command*& state);

} // namespace

struct Command {
    id<MTLCounterSampleBuffer> buffer;
    unsigned used{};
    std::vector<Encoder> encoders;
};

namespace {

id<MTLCounterSampleBuffer> TakeBuffer() {
    std::scoped_lock lock{pool_mutex};
    if (!pool.empty()) {
        id<MTLCounterSampleBuffer> b = pool.back();
        pool.pop_back();
        return b;
    }
    if (created >= MaxBuffers) {
        return nil;
    }
    MTLCounterSampleBufferDescriptor* d = [MTLCounterSampleBufferDescriptor new];
    d.counterSet = timestamps;
    d.storageMode = MTLStorageModeShared;
    d.sampleCount = SamplesPerBuffer;
    d.label = @"bbport GPU timing";
    NSError* error = nil;
    id<MTLCounterSampleBuffer> b = [device newCounterSampleBufferWithDescriptor:d error:&error];
    if (!b) {
        static bool told = false;
        if (!told) {
            told = true;
            std::printf("GPU timing: no more sample buffers after %u (%s)\n", created,
                        error ? error.localizedDescription.UTF8String : "unknown");
        }
        created = MaxBuffers;
        return nil;
    }
    ++created;
    return b;
}

void GiveBack(id<MTLCounterSampleBuffer> b) {
    if (!b) {
        return;
    }
    std::scoped_lock lock{pool_mutex};
    pool.push_back(b);
}

Command* Ensure(Command*& state) {
    if (!state) {
        state = new Command;
        state->buffer = TakeBuffer();
    }
    return state;
}

/// Room for `count` samples in this command buffer's sample buffer.
bool Room(Command* c, unsigned count) {
    if (c->buffer && c->used + count <= SamplesPerBuffer) {
        sampled.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    unsampled.fetch_add(1, std::memory_order_relaxed);
    return false;
}

void Calibrate(MTLTimestamp& gpu, std::chrono::steady_clock::time_point& steady) {
    MTLTimestamp cpu = 0;
    steady = std::chrono::steady_clock::now();
    [device sampleTimestamps:&cpu gpuTimestamp:&gpu];
}

void Report();

} // namespace

bool Enabled() {
    return enabled;
}

void Init(id<MTLDevice> dev) {
    const char* env = std::getenv("BB_METAL_GPU_TIMING");
    if (!env || env[0] != '1' || enabled) {
        return;
    }
    device = dev;
    if (![device supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
        std::printf("GPU timing: off (this GPU does not sample counters at stage boundaries)\n");
        return;
    }
    for (id<MTLCounterSet> set in device.counterSets) {
        if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            timestamps = set;
        }
    }
    if (!timestamps) {
        std::printf("GPU timing: off (no timestamp counter set)\n");
        return;
    }
    Calibrate(gpu0, steady0);
    last_report = steady0;
    enabled = true;
    std::printf("GPU timing: on (timestamps at stage boundaries; draw boundaries %s)\n",
                [device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary]
                    ? "also supported" : "not supported");
}

void AttachRender(Command*& state, MTLRenderPassDescriptor* pass, const char* origin) {
    if (!enabled || !pass) {
        return;
    }
    Command* c = Ensure(state);
    if (!Room(c, 4)) {
        DetachRender(pass);
        return;
    }
    MTLRenderPassSampleBufferAttachmentDescriptor* a = pass.sampleBufferAttachments[0];
    a.sampleBuffer = c->buffer;
    a.startOfVertexSampleIndex = c->used;
    a.endOfVertexSampleIndex = c->used + 1;
    a.startOfFragmentSampleIndex = c->used + 2;
    a.endOfFragmentSampleIndex = c->used + 3;
    Encoder e{'R', c->used};
    e.frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    e.origin = origin;
    std::string colors;
    for (unsigned i = 0; i < 8; ++i) {
        id<MTLTexture> t = pass.colorAttachments[i].texture;
        if (!t) {
            continue;
        }
        if (!e.width) {
            e.width = unsigned(t.width >> pass.colorAttachments[i].level);
            e.height = unsigned(t.height >> pass.colorAttachments[i].level);
        }
        colors += (colors.empty() ? "" : " ") + Format(t.pixelFormat);
        ++e.colors;
    }
    std::string depth;
    if (id<MTLTexture> t = pass.depthAttachment.texture) {
        e.depth = true;
        depth = Format(t.pixelFormat);
        if (!e.width) {
            e.width = unsigned(t.width >> pass.depthAttachment.level);
            e.height = unsigned(t.height >> pass.depthAttachment.level);
        }
    }
    bool clears = pass.depthAttachment.texture && pass.depthAttachment.loadAction == MTLLoadActionClear;
    for (unsigned i = 0; i < 8; ++i) {
        clears |= pass.colorAttachments[i].texture &&
                  pass.colorAttachments[i].loadAction == MTLLoadActionClear;
    }
    e.signature = std::to_string(e.width) + "x" + std::to_string(e.height) + " [" + colors + "]" +
                  (depth.empty() ? "" : " depth " + depth) + (clears ? " clears" : "");
    c->used += 4;
    c->encoders.push_back(std::move(e));
}

void DetachRender(MTLRenderPassDescriptor* pass) {
    if (enabled && pass) {
        pass.sampleBufferAttachments[0].sampleBuffer = nil;
    }
}

MTLComputePassDescriptor* ComputePass(Command*& state) {
    if (!enabled) {
        return nil;
    }
    Command* c = Ensure(state);
    if (!Room(c, 2)) {
        return nil;
    }
    MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
    d.dispatchType = MTLDispatchTypeSerial;
    d.sampleBufferAttachments[0].sampleBuffer = c->buffer;
    d.sampleBufferAttachments[0].startOfEncoderSampleIndex = c->used;
    d.sampleBufferAttachments[0].endOfEncoderSampleIndex = c->used + 1;
    Encoder e{'C', c->used};
    e.frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    c->used += 2;
    c->encoders.push_back(std::move(e));
    return d;
}

MTLBlitPassDescriptor* BlitPass(Command*& state) {
    if (!enabled) {
        return nil;
    }
    Command* c = Ensure(state);
    if (!Room(c, 2)) {
        return nil;
    }
    MTLBlitPassDescriptor* d = [MTLBlitPassDescriptor blitPassDescriptor];
    d.sampleBufferAttachments[0].sampleBuffer = c->buffer;
    d.sampleBufferAttachments[0].startOfEncoderSampleIndex = c->used;
    d.sampleBufferAttachments[0].endOfEncoderSampleIndex = c->used + 1;
    Encoder e{'B', c->used};
    e.frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    c->used += 2;
    c->encoders.push_back(std::move(e));
    return d;
}

void NoteWork(Command* state) {
    if (state && !state->encoders.empty()) {
        ++state->encoders.back().work;
    }
}

void Drop(Command*& state) {
    if (!state) {
        return;
    }
    GiveBack(state->buffer);
    delete state;
    state = nullptr;
}

void Commit(Command*& state, id<MTLCommandBuffer> cmd) {
    if (!state) {
        return;
    }
    if (state->encoders.empty()) {
        Drop(state);
        return;
    }
    Command* c = state;
    state = nullptr;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      NSData* data = [c->buffer resolveCounterRange:NSMakeRange(0, c->used)];
      if (data && data.length >= c->used * sizeof(MTLCounterResultTimestamp)) {
          const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
          const auto valid = [](u_int64_t t) { return t != MTLCounterErrorValue && t != 0; };
          const auto ns = [&](u_int64_t t) { return (double(t) - double(gpu0)) * ns_per_tick; };
          std::vector<Record> done;
          done.reserve(c->encoders.size());
          for (auto& e : c->encoders) {
              const unsigned n = e.kind == 'R' ? 4 : 2;
              double lo = 0, hi = 0;
              bool any = false;
              for (unsigned i = 0; i < n; ++i) {
                  const u_int64_t t = ts[e.first + i].timestamp;
                  if (!valid(t)) {
                      continue;
                  }
                  const double v = ns(t);
                  lo = any ? std::min(lo, v) : v;
                  hi = any ? std::max(hi, v) : v;
                  any = true;
              }
              if (!any || hi < lo) {
                  continue;
              }
              Record r{std::move(e), lo, hi, 0, 0};
              if (r.encoder.kind == 'R') {
                  const auto span = [&](unsigned a, unsigned b) {
                      const u_int64_t x = ts[r.encoder.first + a].timestamp;
                      const u_int64_t y = ts[r.encoder.first + b].timestamp;
                      return valid(x) && valid(y) && y >= x ? (ns(y) - ns(x)) : 0.0;
                  };
                  r.vertex = span(0, 1);
                  r.fragment = span(2, 3);
              }
              done.push_back(std::move(r));
          }
          std::scoped_lock lock{records_mutex};
          records.insert(records.end(), std::make_move_iterator(done.begin()),
                         std::make_move_iterator(done.end()));
      }
      GiveBack(c->buffer);
      delete c;
      Report();
    }];
}

namespace {

struct Sum {
    double ms{}, vertex{}, fragment{};
    double encoders{}, work{};
};

/// The kind of a render pass by what it draws into, `full` the largest pass area seen.
std::string KindOf(const Encoder& e, unsigned long long full) {
    if (e.kind == 'C') return "compute";
    if (e.kind == 'B') return "blit";
    if (std::string(e.origin) != "pass") return std::string("render: ") + e.origin;
    if (e.colors == 0) return e.depth ? "render: depth only" : "render: no targets";
    const unsigned long long area = (unsigned long long)e.width * e.height;
    const char* size = area * 10 >= full * 9 ? "full size" : "smaller";
    if (e.colors >= 4) return std::string("render: 4+ targets, ") + size;
    return std::string("render: ") + std::to_string(e.colors) + (e.colors == 1 ? " target, " : " targets, ") + size;
}

void Report() {
    static std::mutex report_mutex;
    std::unique_lock report{report_mutex, std::try_to_lock};
    if (!report.owns_lock()) {
        return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - last_report < std::chrono::seconds(5)) {
        return;
    }
    last_report = now;
    std::vector<Record> taken;
    {
        std::scoped_lock lock{records_mutex};
        if (records.empty()) {
            return;
        }
        unsigned long long newest = 0;
        for (const auto& r : records) newest = std::max(newest, r.encoder.frame);
        // The newest frames may still have command buffers in flight: kept for the next report.
        std::vector<Record> keep;
        for (auto& r : records) {
            (r.encoder.frame + 3 > newest ? keep : taken).push_back(std::move(r));
        }
        records = std::move(keep);
    }
    if (taken.empty()) {
        return;
    }
    // Refine ticks to ns over the whole time since Init (applies to later reports' records).
    MTLTimestamp gpu1{};
    std::chrono::steady_clock::time_point steady1;
    Calibrate(gpu1, steady1);
    if (gpu1 > gpu0) {
        ns_per_tick = double(std::chrono::duration_cast<std::chrono::nanoseconds>(steady1 - steady0).count()) /
                      double(gpu1 - gpu0);
    }
    std::map<unsigned long long, std::vector<const Record*>> frames;
    unsigned long long full = 0;
    for (const auto& r : taken) {
        frames[r.encoder.frame].push_back(&r);
        if (r.encoder.kind == 'R' && std::string(r.encoder.origin) == "pass") {
            full = std::max(full, (unsigned long long)r.encoder.width * r.encoder.height);
        }
    }
    const double n = double(frames.size());
    std::map<std::string, Sum> by_kind, by_pass;
    std::map<std::pair<unsigned, std::string>, Sum> by_position; // GPU order in the frame
    std::map<std::string, Sum> gaps; // idle between encoders, by the kinds on either side
    double span = 0, busy = 0, idle = 0, between = 0, gap_count = 0, sum_all = 0;
    double prev_end = -1;
    for (auto& [frame, list] : frames) {
        std::sort(list.begin(), list.end(), [](const Record* a, const Record* b) { return a->start < b->start; });
        const double first = list.front()->start;
        double reach = first, last = first;
        const Record* reaching = list.front();
        unsigned position = 0;
        for (const Record* r : list) {
            const auto& e = r->encoder;
            const double ms = (r->end - r->start) / 1e6;
            const std::string kind = KindOf(e, full);
            for (auto* s : {&by_kind[kind], &by_pass[e.kind == 'R' ? std::string(1, 'R') + " " + e.signature + (std::string(e.origin) != "pass" ? std::string(" (") + e.origin + ")" : "") : kind],
                            &by_position[{position, e.kind == 'R' ? e.signature : kind}]}) {
                s->ms += ms;
                s->vertex += r->vertex / 1e6;
                s->fragment += r->fragment / 1e6;
                s->encoders += 1;
                s->work += e.work;
            }
            ++position;
            sum_all += ms;
            if (r->start > reach) {
                const double g = (r->start - reach) / 1e6;
                idle += g;
                gap_count += 1;
                auto& s = gaps[KindOf(reaching->encoder, full) + " -> " + kind];
                s.ms += g;
                s.encoders += 1;
            }
            if (r->end > reach) {
                busy += (r->end - std::max(reach, r->start)) / 1e6;
                reach = r->end;
                reaching = r;
            }
            last = std::max(last, r->end);
        }
        span += (last - first) / 1e6;
        if (prev_end >= 0 && first > prev_end) {
            between += (first - prev_end) / 1e6;
        }
        prev_end = last;
    }
    const auto s = sampled.exchange(0), u = unsampled.exchange(0);
    std::printf("GPU timing: %.0f frames; per frame: GPU span %.2f ms, busy %.2f (encoder sum %.2f: "
                "overlap %.2f), idle between encoders %.2f in %.1f gaps, between frames %.2f "
                "(present and unsampled work included); %.0f%% of encoders sampled\n",
                n, span / n, busy / n, sum_all / n, (sum_all - busy) / n, idle / n, gap_count / n,
                between / n, s + u ? 100.0 * double(s) / double(s + u) : 0.0);
    const auto print = [&](const char* title, const auto& table, size_t top) {
        std::vector<std::pair<std::string, Sum>> rows;
        for (const auto& [k, v] : table) rows.emplace_back(k, v);
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
        std::printf("  %s (ms/frame, vertex, fragment, encoders/frame, draws or dispatches/frame):\n", title);
        for (size_t i = 0; i < std::min(top, rows.size()); ++i) {
            const Sum& v = rows[i].second;
            std::printf("    %7.3f %6.3f %6.3f %6.1f %7.1f  %s\n", v.ms / n, v.vertex / n,
                        v.fragment / n, v.encoders / n, v.work / n, rows[i].first.c_str());
        }
    };
    print("by kind", by_kind, 20);
    print("by pass", by_pass, 20);
    std::map<std::string, Sum> positions;
    for (const auto& [k, v] : by_position) {
        positions["#" + std::to_string(k.first) + " " + k.second] = v;
    }
    print("costliest encoders by position in the frame (GPU order)", positions, 20);
    std::printf("  idle between encoders by the kinds on either side (ms/frame, gaps/frame):\n");
    std::vector<std::pair<std::string, Sum>> g(gaps.begin(), gaps.end());
    std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    for (size_t i = 0; i < std::min<size_t>(12, g.size()); ++i) {
        std::printf("    %7.3f %6.1f  %s\n", g[i].second.ms / n, g[i].second.encoders / n, g[i].first.c_str());
    }
}

} // namespace

} // namespace MetalGpuTiming
