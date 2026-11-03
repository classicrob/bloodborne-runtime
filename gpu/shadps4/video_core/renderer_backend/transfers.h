// The render adapter's transfers and image uses (docs/RENDER_ADAPTER.md, M2): copies, uploads,
// clears and the accesses the backend turns into barriers, on the texture and buffer caches'
// objects.
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <optional>
#include <span>

#include "common/types.h"
#include "video_core/renderer_backend/resource.h"
#include "video_core/texture_cache/types.h"

namespace VideoCore {
struct Buffer;
struct Image;
} // namespace VideoCore

namespace Vulkan {
class StagingBufferPool;
}

namespace VideoCore::Backend {

class Transfers {
public:
    virtual ~Transfers() = default;

    virtual void CopyBuffer(const Buffer* src, const Buffer* dst,
                            std::span<const BufferCopy> copies) = 0;
    /// Inside a pass whose draws so far did not touch the ranges, the copies run before the
    /// pass began, so the pass is not split; false when they cannot be placed there (the caller
    /// copies the usual way).
    [[nodiscard]] virtual bool CopyBufferBeforePass(const Buffer* src, const Buffer* dst,
                                                    std::span<const BufferCopy> copies) = 0;
    /// From the game's memory itself (`src`: a guest memory chunk's buffer, NativeHandle).
    virtual void CopyFromGuestChunk(u64 src, const Buffer* dst,
                                    std::span<const BufferCopy> copies) = 0;
    virtual void UploadImage(Image* dst, const Buffer* src,
                             std::span<const BufferImageCopy> copies) = 0;
    virtual void DownloadImage(Image* src, const Buffer* dst,
                               std::span<const BufferImageCopy> copies) = 0;
    virtual void FillBuffer(const Buffer* dst, u64 offset, u64 size, u32 value) = 0;
    /// `data` written into `dst` by the GPU in stream order.
    virtual void UpdateBuffer(const Buffer* dst, u64 offset, std::span<const u8> data) = 0;

    virtual void CopyImage(Image* src, Image* dst) = 0;
    virtual void CopyMip(Image* src, Image* dst, u32 mip, u32 slice) = 0;
    /// bbport: level 0 of one layer of `src` into one layer of `dst` (same format and size).
    virtual void CopyImageLayer(Image* src, u32 src_layer, Image* dst, u32 dst_layer) = 0;
    virtual void CopyColorAndDepth(Image* src, Image* dst) = 0;
    virtual void CopyDepthStencil(Image* src, Image* dst, const SubresourceRange& range) = 0;
    virtual void ResolveImage(Image* src, Image* dst, const SubresourceRange& src_range,
                              const SubresourceRange& dst_range) = 0;
    /// Depth (and stencil) of `src` drawn into `dst` at `width` x `height`, for formats without
    /// blits (CanResampleDepth): single-level images by NativeHandle, `src` in Sampled use and
    /// `dst` in DepthAttachment use, with their state values (CommandRecorder::StateOf,
    /// BindingLayout).
    virtual void ResampleDepth(u64 src, u32 src_state, u64 dst, u32 dst_state, Format format,
                               bool stencil, u32 width, u32 height) = 0;
    [[nodiscard]] virtual bool CanResampleDepth(Format format, bool stencil) = 0;

    /// BB_DEFER_CLEARS: a clear is noted on the image; a pass that draws into the whole range
    /// takes it as its load action (TakePendingClear), any other use records it first.
    virtual void ClearImage(Image* dst, const SubresourceRange& range,
                            const ClearColor& value) = 0;
    enum class ClearResolveReason : u8 {
        Copy, Sample, OtherAccess, ReducedPass, UpscalerRedirect, ViewRange,
        SmallRenderArea, MultiLevelOrLayer, BackingSwap, Superseded, Other,
        Count,
    };
    virtual void ResolvePendingClear(Image* image,
                                     ClearResolveReason reason = ClearResolveReason::Other) = 0;
    virtual bool TakePendingClear(Image* image, const SubresourceRange& range,
                                  ClearColor& value) = 0;

    virtual void SetBackingSamples(Image* image, u32 num_samples, bool copy_backing = true) = 0;
    /// bbport: lazy texture usages (Image::allowed_storage): a new backing with every usage the
    /// image may need, the current one's contents copied into it (all levels and layers).
    virtual void UpgradeUsage(Image* image) = 0;

    /// Declared uses; the backend places the barriers (FlushBarriers records pending ones).
    /// True when the use needs a barrier.
    virtual bool Transit(Image* image, ImageAccess access,
                         std::optional<SubresourceRange> range = {}) = 0;
    virtual void AccessBuffer(const Buffer* buffer, u64 offset, u64 size,
                              BufferAccess access) = 0;
    /// A shader reached memory through the page table (BB_LAYER_MEMORY).
    virtual void AccessGlobal(BufferAccess access) = 0;
    [[nodiscard]] virtual bool IsBufferAccessed(const Buffer* buffer, u64 offset, u64 size,
                                                bool check_read_access = false) = 0;
    virtual void FlushBarriers() = 0;
    /// The state value image bindings and attachments of `image` carry (ImageBinding::layout,
    /// RenderAttachment::image_layout): the backend's own state of its current backing.
    [[nodiscard]] virtual u32 BindingLayout(const Image& image) const = 0;

    /// Runs on every declared use of an image (Transit), before its barrier: scene resolution
    /// writes back a reduced proxy first. `writes`: the use writes the image.
    virtual void SetNativeAccessHook(void (*hook)(void*, Image&, bool writes), void* context) = 0;

    /// BB_GPU_PROFILE: a profiler segment for a transfer of `image` until EndTransferMark with
    /// the returned value (0: no profiler); the segment it interrupted continues afterwards.
    [[nodiscard]] virtual u64 BeginTransferMark(const char* what, const Image& image) = 0;
    virtual void EndTransferMark(u64 mark) = 0;

    /// Runs before this thread changes image state (the rasterizer's texture binding helper).
    virtual void SetImageAccessHook(void (*hook)(void*), void* context) = 0;
    virtual void BeforeImageAccess() = 0;

    virtual void TickFrame() = 0;
    [[nodiscard]] virtual Vulkan::StagingBufferPool& GetStagingPool() = 0;
};

/// A Transfers::BeginTransferMark segment while it lives.
class ScopedTransferMark {
public:
    ScopedTransferMark(Transfers& transfers_, const char* what, const Image& image)
        : transfers{transfers_}, mark{transfers_.BeginTransferMark(what, image)} {}
    ~ScopedTransferMark() {
        if (mark) {
            transfers.EndTransferMark(mark);
        }
    }
    ScopedTransferMark(const ScopedTransferMark&) = delete;
    ScopedTransferMark& operator=(const ScopedTransferMark&) = delete;

private:
    Transfers& transfers;
    u64 mark;
};

} // namespace VideoCore::Backend
