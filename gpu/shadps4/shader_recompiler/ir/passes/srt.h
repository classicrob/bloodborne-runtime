// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <boost/container/set.hpp>
#include <boost/container/small_vector.hpp>
#include "common/arch.h"
#include "common/types.h"

namespace Serialization {
struct Archive;
}

namespace Shader {

#ifdef ARCH_X86_64
using PFN_SrtWalker = void PS4_SYSV_ABI (*)(const u32* /*user_data*/, u32* /*flat_dst*/);
inline const void* SrtWalkerCode(PFN_SrtWalker walker) {
    return reinterpret_cast<const void*>(walker);
}
#else
// ARM_bloodborne_pc 3d56243 wrote the portable walker; Bloodborne-Android bf1e6cf
// moved it to bbport 0.51. Keep the translated walker as an off switch on arm64.
struct SrtWalker {
    const u8* code{};
    bool portable{};
    explicit operator bool() const { return code != nullptr; }
};
using PFN_SrtWalker = SrtWalker;
inline const void* SrtWalkerCode(PFN_SrtWalker walker) { return walker.code; }
void RunSrtProgram(const u32* program, const u32* user_data, u32* flat_dst);
#endif
bool UsePortableSrtWalker();

/// bbport BB_STALE_RANGE_CHECK=1 (diagnostics): guest ranges whose memory a renderer substitution
/// no longer writes (experiment bit 18's layer clears); the walker counts any load from them, to
/// test whether descriptors are ever read from such memory.
bool StaleRangeCheckOn();
void NoteUnwrittenRange(u64 base, u64 size);
void CheckWalkerLoad(const void* address, u32 size);
/// Any other access by the renderer to such a range (buffer bind, image write-back, image
/// creation, CPU write seen by the caches): counted and logged per site.
void NoteStaleRangeAccess(const char* site, u64 address, u64 size);
PFN_SrtWalker RegisterWalkerCode(const u8* ptr, size_t size);

/// Run x86 code natively, translated x86 code on arm64, or a portable SRT program.
void CallSrtWalker(PFN_SrtWalker walker, const u32* user_data, u32* flat_dst);

struct PersistentSrtInfo {
    // Special case when fetch shader uses step rates.
    struct SrtSharpReservation {
        u32 sgpr_base;
        u32 dword_offset;
        u32 num_dwords;
    };

    PFN_SrtWalker walker_func{};
    size_t walker_func_size{};
    u32 flattened_bufsize_dw = 16; // NumUserDataRegs

    void Serialize(Serialization::Archive& ar) const;
    bool Deserialize(Serialization::Archive& ar);
};

} // namespace Shader
