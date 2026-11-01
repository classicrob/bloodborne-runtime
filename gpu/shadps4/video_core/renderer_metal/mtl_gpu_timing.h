// SPDX-License-Identifier: GPL-2.0-or-later
// bbport BB_METAL_GPU_TIMING=1: GPU time of every Metal encoder of the frame path, from GPU
// timestamps sampled at stage boundaries (MTLCounterSampleBuffer; Apple GPUs sample at encoder
// and stage boundaries, not between draws). Every 5 s: GPU ms per frame by encoder kind and by
// render pass signature (attachment formats and size), the costliest encoders of the frame, and
// the GPU's idle time between encoders by the kinds on either side. Off by default; for
// measuring only (Metal System Trace gives no GPU timeline on some macOS versions).
#pragma once

#import <Metal/Metal.h>

namespace MetalGpuTiming {

/// BB_METAL_GPU_TIMING=1 and the device samples timestamps at stage boundaries.
bool Enabled();
/// Once, with the device (prints whether timing is on and why not).
void Init(id<MTLDevice> device);

/// Per command buffer state (Owned::timing): the sample buffer and the encoders sampled into it.
struct Command;

/// Sets `pass`'s sample attachment for the next render encoder of `cmd`, recording what the
/// encoder is (`origin`: "pass", "clear", "copy draw"). Call Detach(pass) once it is created.
void AttachRender(Command*& state, MTLRenderPassDescriptor* pass, const char* origin);
void DetachRender(MTLRenderPassDescriptor* pass);
/// A compute/blit pass descriptor carrying the sample attachment, or nil (then encode as before).
MTLComputePassDescriptor* ComputePass(Command*& state);
MTLBlitPassDescriptor* BlitPass(Command*& state);
/// Counts a draw or dispatch into the current encoder (shown next to its time).
void NoteWork(Command* state);
/// Before `cmd` commits: resolves its samples once the GPU is done. Resets `state`.
void Commit(Command*& state, id<MTLCommandBuffer> cmd);
/// A command buffer dropped without a commit.
void Drop(Command*& state);

} // namespace MetalGpuTiming
