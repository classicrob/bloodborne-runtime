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
    std::string shader; ///< the first draw's vertex function or the first dispatch's kernel
    unsigned long long command{}; ///< the command buffer's serial (boundaries between them)
};

/// One encoder's GPU interval, resolved.
struct Record {
    Encoder encoder;
    double start, end; ///< ns from the calibration point
    double vertex, fragment; ///< ns, render passes
};

bool enabled = false;
bool split = false;    ///< BB_METAL_GPU_TIMING=2: vertex and fragment stage ends sampled too
unsigned every = 1;    ///< BB_METAL_GPU_TIMING_EVERY: one frame in this many sampled
unsigned seq_from = 0, seq_to = 0; ///< BB_METAL_GPU_TIMING_SEQ=from-to: these positions in order
std::string seq_anchor; ///< BB_METAL_GPU_TIMING_SEQ=<text>:<count>[:<before>]: around the first encoder named so
long seq_before = 0;
id<MTLDevice> device;
id<MTLCounterSet> timestamps;
std::mutex pool_mutex;
std::vector<id<MTLCounterSampleBuffer>> pool;
unsigned created = 0;
std::atomic<unsigned long long> sampled{0}, unsampled{0}, command_serial{0};

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
    unsigned long long serial{};
    double commit_ns{}; ///< since the calibration point: no sample of this buffer is older
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
        state->serial = ++command_serial;
    }
    return state;
}

/// Room for `count` samples in this command buffer's sample buffer (and this frame is sampled).
bool Room(Command* c, unsigned count) {
    if (every > 1 && BbStats::gpu_frames.load(std::memory_order_relaxed) % every != 0) {
        return false;
    }
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

/// GPU ticks to ns over the time since Init.
void Refine() {
    MTLTimestamp gpu1{};
    std::chrono::steady_clock::time_point steady1;
    Calibrate(gpu1, steady1);
    if (gpu1 > gpu0) {
        ns_per_tick = double(std::chrono::duration_cast<std::chrono::nanoseconds>(steady1 - steady0).count()) /
                      double(gpu1 - gpu0);
    }
}

} // namespace

bool Enabled() {
    return enabled;
}

void Init(id<MTLDevice> dev) {
    const char* env = std::getenv("BB_METAL_GPU_TIMING");
    if (!env || (env[0] != '1' && env[0] != '2') || enabled) {
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
    split = env[0] == '2';
    if (const char* q = std::getenv("BB_METAL_GPU_TIMING_SEQ")) {
        const std::string v{q};
        if (const auto colon = v.find(':'); colon != std::string::npos) {
            // Positions vary from frame to frame: count from the first encoder whose label holds
            // the text (a pass signature or shader name), starting <before> encoders ahead of it.
            seq_anchor = v.substr(0, colon);
            int count = 1, before = 0;
            std::sscanf(v.c_str() + colon + 1, "%d:%d", &count, &before);
            seq_before = std::max(0, before);
            seq_from = 0;
            seq_to = unsigned(std::max(1, count + before)) - 1;
        } else {
            std::sscanf(q, "%u-%u", &seq_from, &seq_to);
        }
    }
    if (const char* n = std::getenv("BB_METAL_GPU_TIMING_EVERY")) {
        every = std::max(1, std::atoi(n));
    }
    enabled = true;
    std::printf("GPU timing: on (timestamps at stage boundaries%s; one frame in %u; draw boundaries "
                "%s)\n",
                split ? ", vertex and fragment ends too" : ", encoder start and end only", every,
                [device supportsCounterSampling:MTLCounterSamplingPointAtDrawBoundary]
                    ? "also supported" : "not supported");
}

void AttachRender(Command*& state, MTLRenderPassDescriptor* pass, const char* origin) {
    if (!enabled || !pass) {
        return;
    }
    Command* c = Ensure(state);
    const unsigned samples = split ? 4 : 2;
    if (!Room(c, samples)) {
        DetachRender(pass);
        return;
    }
    // Samples: start of vertex, end of fragment (the encoder's span), then with split the end of
    // vertex and the start of fragment. Each sample inside a pass costs it overlap with its
    // neighbours, so by default only the ends are sampled.
    MTLRenderPassSampleBufferAttachmentDescriptor* a = pass.sampleBufferAttachments[0];
    a.sampleBuffer = c->buffer;
    a.startOfVertexSampleIndex = c->used;
    a.endOfFragmentSampleIndex = c->used + 1;
    a.endOfVertexSampleIndex = split ? c->used + 2 : MTLCounterDontSample;
    a.startOfFragmentSampleIndex = split ? c->used + 3 : MTLCounterDontSample;
    Encoder e{'R', c->used};
    e.frame = BbStats::gpu_frames.load(std::memory_order_relaxed);
    e.command = c->serial;
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
    c->used += samples;
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
    e.command = c->serial;
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
    e.command = c->serial;
    c->used += 2;
    c->encoders.push_back(std::move(e));
    return d;
}

void NoteWork(Command* state, const char* function) {
    if (!state || state->encoders.empty()) {
        return;
    }
    Encoder& e = state->encoders.back();
    if (e.work++ == 0 && function) {
        e.shader = function;
    }
}

void NoteWork(Command* state, id<MTLFunction> function) {
    if (!state || state->encoders.empty()) {
        return;
    }
    Encoder& e = state->encoders.back();
    if (e.work++ == 0 && function) {
        e.shader = function.name.UTF8String ?: "";
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
    c->commit_ns = double(std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now() - steady0).count());
    [cmd addCompletedHandler:^(id<MTLCommandBuffer>) {
      // The first resolves a second after Init get a measured tick length (reports refine it).
      static std::atomic<bool> refined{false};
      if (!refined.load(std::memory_order_relaxed) &&
          std::chrono::steady_clock::now() - steady0 > std::chrono::seconds(1) &&
          !refined.exchange(true)) {
          Refine();
      }
      NSData* data = [c->buffer resolveCounterRange:NSMakeRange(0, c->used)];
      if (data && data.length >= c->used * sizeof(MTLCounterResultTimestamp)) {
          const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
          const auto ns = [&](u_int64_t t) { return (double(t) - double(gpu0)) * ns_per_tick; };
          // A stage that did not run leaves its slot as it was: a value from the buffer's last
          // use, older than this commit (2 ms allowed for the calibration's error).
          const auto valid = [&](u_int64_t t) {
              return t != MTLCounterErrorValue && t != 0 && ns(t) >= c->commit_ns - 2e6;
          };
          std::vector<Record> done;
          done.reserve(c->encoders.size());
          for (auto& e : c->encoders) {
              const unsigned n = e.kind == 'R' && split ? 4 : 2;
              // A render pass without draws runs no vertex stage: its start of vertex stays stale
              // (rejected above); with split its start of fragment stands in.
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
                  if (split) {
                      r.vertex = span(0, 2);
                      r.fragment = span(3, 1);
                  }
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
    double exclusive{}; ///< busy time shared out among the encoders running at once (sums to busy)
    double ms{}, vertex{}, fragment{};
    double encoders{}, work{};
};

/// The kind of a render pass by what it draws into, `full` the largest color pass area seen.
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
    Refine(); // over the whole time since Init (applies to later reports' records)
    std::map<unsigned long long, std::vector<const Record*>> frames;
    // Full size: the size of the color pass with the most draws (the scene's G-buffer).
    std::map<unsigned long long, unsigned long long> draws_by_area;
    for (const auto& r : taken) {
        frames[r.encoder.frame].push_back(&r);
        if (r.encoder.kind == 'R' && std::string(r.encoder.origin) == "pass" && r.encoder.colors) {
            draws_by_area[(unsigned long long)r.encoder.width * r.encoder.height] += r.encoder.work;
        }
    }
    unsigned long long full = 0, most = 0;
    for (const auto& [area, d] : draws_by_area) {
        if (d > most) {
            most = d;
            full = area;
        }
    }
    const double n = double(frames.size());
    std::map<std::string, Sum> by_kind, by_pass;
    std::map<std::pair<unsigned, std::string>, Sum> by_position; // GPU order in the frame
    std::map<std::string, Sum> gaps; // idle between encoders, by the kinds on either side
    std::map<std::string, Sum> by_shader, by_kernel; // render passes by first vertex function, compute by first kernel
    std::map<std::string, Sum> gap_pairs; // idle by the two encoders on either side and what separates them
    struct Step {
        double exclusive{}, span{}, gap{}, count{}, work{};
        std::string label;
    };
    std::map<unsigned, Step> sequence; // BB_METAL_GPU_TIMING_SEQ
    double span = 0, busy = 0, idle = 0, between = 0, gap_count = 0, sum_all = 0;
    double prev_end = -1;
    for (auto& [frame, list] : frames) {
        std::sort(list.begin(), list.end(), [](const Record* a, const Record* b) { return a->start < b->start; });
        // Exclusive time: each stretch of busy time is shared among the encoders running in it.
        std::unordered_map<const Record*, double> exclusive;
        {
            std::vector<std::pair<double, std::pair<int, const Record*>>> events;
            for (const Record* r : list) {
                events.push_back({r->start, {1, r}});
                events.push_back({r->end, {-1, r}});
            }
            std::sort(events.begin(), events.end(), [](const auto& a, const auto& b) {
                // Starts before ends at one instant: a zero-length encoder is added before it goes.
                return a.first != b.first ? a.first < b.first : a.second.first > b.second.first;
            });
            std::vector<const Record*> active;
            double at = events.empty() ? 0 : events.front().first;
            for (const auto& [t, ev] : events) {
                if (!active.empty() && t > at) {
                    const double share = (t - at) / 1e6 / double(active.size());
                    for (const Record* a : active) exclusive[a] += share;
                }
                at = t;
                if (ev.first > 0) {
                    active.push_back(ev.second);
                } else {
                    if (const auto it = std::find(active.begin(), active.end(), ev.second);
                        it != active.end()) {
                        active.erase(it);
                    }
                }
            }
        }
        const auto label_of = [](const Encoder& x) {
            return (x.kind == 'R' ? "R " + x.signature : x.kind == 'C' ? "C " + x.shader : std::string("blit")) +
                   (x.kind == 'R' && std::string(x.origin) != "pass" ? std::string(" (") + x.origin + ")" : std::string()) +
                   (x.kind == 'R' && !x.shader.empty() ? " vs " + x.shader : std::string());
        };
        long anchor = 0;
        if (!seq_anchor.empty()) {
            anchor = -1;
            for (size_t i = 0; i < list.size(); ++i) {
                if (label_of(list[i]->encoder).find(seq_anchor) != std::string::npos) {
                    anchor = long(i);
                    break;
                }
            }
        }
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
                s->exclusive += exclusive[r];
                s->vertex += r->vertex / 1e6;
                s->fragment += r->fragment / 1e6;
                s->encoders += 1;
                s->work += e.work;
            }
            if (e.kind != 'B') {
                Sum& t = e.kind == 'R' ? by_shader[e.shader.empty() ? "(no draws) " + e.signature : e.shader]
                                       : by_kernel[e.shader.empty() ? "(no dispatches)" : e.shader];
                t.ms += ms;
                t.exclusive += exclusive[r];
                t.encoders += 1;
                t.work += e.work;
            }
            const long rel = anchor < 0 ? -1 : long(position) - anchor + seq_before;
            if ((seq_to || !seq_anchor.empty()) && rel >= long(seq_from) && rel <= long(seq_to)) {
                Step& st = sequence[unsigned(rel)];
                st.exclusive += exclusive[r];
                st.span += ms;
                st.gap += r->start > reach ? (r->start - reach) / 1e6 : 0.0;
                st.count += 1;
                st.work += e.work;
                st.label = label_of(e);
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
                const auto name = [&](const Encoder& x) {
                    return x.kind == 'R' ? "R " + x.signature
                           : x.kind == 'C' ? "C " + (x.shader.empty() ? std::string("?") : x.shader)
                                           : std::string("blit");
                };
                const Encoder& p = reaching->encoder;
                const char* between = p.command != e.command ? "command buffer boundary"
                                      : p.kind == 'R' && e.kind == 'R' && p.signature == e.signature
                                          ? "same pass begun again"
                                      : p.kind == e.kind && p.kind != 'R' ? "same kind (barrier)"
                                                                          : "encoder change";
                auto& q = gap_pairs[name(p) + "  ->  " + name(e) + "  [" + between + "]"];
                q.ms += g;
                q.encoders += 1;
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
    std::printf("GPU timing: %.0f frames; per frame: GPU span %.2f ms, busy %.2f (encoder spans sum to "
                "%.2f: overlap %.2f), idle between encoders %.2f in %.1f gaps, between frames %.2f "
                "(present and unsampled work included); %.0f%% of encoders sampled\n",
                n, span / n, busy / n, sum_all / n, (sum_all - busy) / n, idle / n, gap_count / n,
                between / n, s + u ? 100.0 * double(s) / double(s + u) : 0.0);
    // BB_METAL_GPU_TIMING_TOP=N: N rows per table instead of each table's default.
    static const size_t top_rows = [] {
        const char* v = std::getenv("BB_METAL_GPU_TIMING_TOP");
        return v ? size_t(std::max(1, std::atoi(v))) : size_t(0);
    }();
    const auto print = [&](const char* title, const auto& table, size_t top) {
        top = top_rows ? top_rows : top;
        std::vector<std::pair<std::string, Sum>> rows;
        for (const auto& [k, v] : table) rows.emplace_back(k, v);
        std::sort(rows.begin(), rows.end(),
                  [](const auto& a, const auto& b) { return a.second.exclusive > b.second.exclusive; });
        std::printf("  %s (ms/frame: exclusive, span, vertex, fragment; encoders/frame, draws or "
                    "dispatches/frame):\n", title);
        for (size_t i = 0; i < std::min(top, rows.size()); ++i) {
            const Sum& v = rows[i].second;
            std::printf("    %7.3f %7.3f %6.3f %6.3f %6.1f %7.1f  %s\n", v.exclusive / n, v.ms / n,
                        v.vertex / n, v.fragment / n, v.encoders / n, v.work / n,
                        rows[i].first.c_str());
        }
    };
    print("by kind", by_kind, 20);
    print("by pass", by_pass, 20);
    std::map<std::string, Sum> positions;
    for (const auto& [k, v] : by_position) {
        positions["#" + std::to_string(k.first) + " " + k.second] = v;
    }
    print("costliest encoders by position in the frame (GPU order)", positions, 20);
    if (!sequence.empty()) {
        std::printf("  positions %u-%u in GPU order%s%s (ms per occurrence: exclusive, span, idle before; "
                    "share of frames, draws):\n", seq_from, seq_to,
                    seq_anchor.empty() ? "" : " from the first encoder named ", seq_anchor.c_str());
        for (const auto& [pos, st] : sequence) {
            std::printf("    #%-4u %6.3f %6.3f %6.3f %5.2f %5.1f  %s\n", pos, st.exclusive / st.count,
                        st.span / st.count, st.gap / st.count, st.count / n, st.work / st.count,
                        st.label.c_str());
        }
    }
    print("render passes by their first draw's vertex function", by_shader, 20);
    print("compute encoders by their first kernel", by_kernel, 15);
    std::printf("  idle between encoders by the kinds on either side (ms/frame, gaps/frame):\n");
    std::vector<std::pair<std::string, Sum>> g(gaps.begin(), gaps.end());
    std::sort(g.begin(), g.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    for (size_t i = 0; i < std::min<size_t>(12, g.size()); ++i) {
        std::printf("    %7.3f %6.1f  %s\n", g[i].second.ms / n, g[i].second.encoders / n, g[i].first.c_str());
    }
    std::printf("  largest idle by the encoders on either side and what separates them (ms/frame, "
                "gaps/frame, us per gap):\n");
    std::vector<std::pair<std::string, Sum>> pg(gap_pairs.begin(), gap_pairs.end());
    std::sort(pg.begin(), pg.end(), [](const auto& a, const auto& b) { return a.second.ms > b.second.ms; });
    for (size_t i = 0; i < std::min<size_t>(25, pg.size()); ++i) {
        const Sum& v = pg[i].second;
        std::printf("    %7.3f %6.1f %6.0f  %s\n", v.ms / n, v.encoders / n,
                    v.encoders ? 1000.0 * v.ms / v.encoders : 0.0, pg[i].first.c_str());
    }
}

} // namespace

} // namespace MetalGpuTiming
