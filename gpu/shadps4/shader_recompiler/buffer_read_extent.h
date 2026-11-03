#pragma once
#include <algorithm>
#include <cstdint>

namespace Shader {
// end is the largest instruction + constant scalar offset + access width.
// For a binding starting on a page boundary, dynamic offsets retain the tracker's legacy page coverage (not a shader bound).
inline uint64_t BufferReadExtent(uint32_t stride, uint32_t records, bool swizzle,
                                 uint32_t index_stride, uint32_t element_size,
                                 bool add_tid, uint64_t end, bool dynamic_offset) {
    const uint64_t nominal = stride ? uint64_t(stride) * records : records;
    if (!stride || !records || !end)
        return nominal;
    const uint64_t baseline = dynamic_offset ? (nominal + 4095) / 4096 * 4096 : nominal;
    (void)add_tid; // The range check applies to the final index including lane ID.
    if (!swizzle)
        return std::max(baseline, uint64_t(records - 1) * stride + end);
    const uint64_t group = uint64_t{8} << index_stride;
    const uint64_t element = uint64_t{2} << element_size;
    const uint64_t last = records - 1;
    const auto extent = [&](uint64_t index) {
        return (index / group * stride + (end - 1) / element * element) * group +
               index % group * element + (end - 1) % element + 1;
    };
    // The maximum index is either in the final partial group or at the end
    // of the preceding full group. Offset mapping is monotonic.
    uint64_t result = extent(last);
    if (last >= group)
        result = std::max(result, extent(last / group * group - 1));
    return std::max(baseline, result);
}
}
