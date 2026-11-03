// bbport: BarrierTracker with touching ranges merged (BB_TRACKER_MERGE_TOUCHING) must answer every
// FindRange exactly as without it: merging changes the nodes, not the covered bytes.
#include <cassert>
#include <cstdio>
#include <random>
#include "video_core/renderer_vulkan/vk_barrier_tracker.h"

using Vulkan::Access;
using Vulkan::AddressRange;
using Vulkan::BarrierTracker;

int main() {
    std::mt19937_64 rng(1234);
    for (int round = 0; round < 200; ++round) {
        BarrierTracker plain, merged;
        plain.SetMergeTouching(false);
        merged.SetMergeTouching(true);
        // Ranges like ring allocations: many back to back, some gaps, some overlaps, a few resources.
        u64 cursor[4] = {0, 0, 0, 0};
        const int inserts = 1 + int(rng() % 300);
        for (int i = 0; i < inserts; ++i) {
            AddressRange r;
            r.resource = 1 + rng() % 4;
            u64& at = cursor[r.resource - 1];
            const u64 kind = rng() % 10;
            if (kind < 6) {
                r.range_start = at;                        // adjoining the previous one
            } else if (kind < 8) {
                r.range_start = at + 1 + rng() % 512;      // a gap
            } else {
                r.range_start = at > 300 ? at - rng() % 300 : 0; // overlapping
            }
            r.range_end = r.range_start + rng() % 256;
            at = r.range_end + 1;
            const Access access = (rng() & 1) ? Access::Read : Access::Write;
            plain.InsertRange(r, access);
            merged.InsertRange(r, access);
        }
        for (int q = 0; q < 2000; ++q) {
            AddressRange r;
            r.resource = 1 + rng() % 5;
            r.range_start = rng() % 80000;
            r.range_end = r.range_start + rng() % 400;
            for (const Access access : {Access::Read, Access::Write}) {
                if (plain.FindRange(r, access) != merged.FindRange(r, access)) {
                    std::printf("round %d query %d: answers differ\n", round, q);
                    return 1;
                }
            }
        }
        plain.Clear();
        merged.Clear();
        assert(plain.Empty() && merged.Empty());
    }
    std::puts("barrier tracker: touching-range merge gives the same answers");
    return 0;
}
