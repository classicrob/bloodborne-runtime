#include "shader_recompiler/buffer_read_extent.h"
#include <cassert>
int main() {
    using Shader::BufferReadExtent;
    assert(BufferReadExtent(0, 100, false, 0, 0, false, 4096, true) == 100);
    assert(BufferReadExtent(16, 10, false, 0, 0, false, 12, false) == 160);
    assert(BufferReadExtent(16, 10, false, 0, 0, false, 36, false) == 180);
    assert(BufferReadExtent(16, 10, true, 0, 0, false, 16, false) == 244);
    // A full previous group can reach farther than the last partial group.
    assert(BufferReadExtent(1, 9, true, 0, 3, false, 2, false) == 114);
    assert(BufferReadExtent(16, 10, false, 0, 0, false, 5000, true) == 5144);
    assert(BufferReadExtent(16, 0, true, 0, 0, false, 100, false) == 0);
    assert(BufferReadExtent(16, 10, false, 0, 0, true, 36, false) == 180);
    assert(BufferReadExtent(16, 10, false, 0, 0, false, 4, true) == 4096);
}
