// SPDX-License-Identifier: GPL-2.0-or-later
// The render adapter's Metal command stream (mtl_recorder.h).

#include <cstdio>
#include <cstring>

#include <boost/container/static_vector.hpp>

#include "video_core/renderer_metal/mtl_frame.h"
#include "video_core/renderer_metal/mtl_recorder.h"
#include "video_core/renderer_metal/mtl_vk_layer.h"
#include "video_core/renderer_vulkan/vk_breadcrumbs.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_resource_backend.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Metal {
namespace {

namespace Backend = VideoCore::Backend;

/// What CommitDrawState hands the encoder: the changed fields of the draw state.
struct DrawStateDelta {
    enum Bit : u32 {
        Viewports = 1u << 0,
        Scissors = 1u << 1,
        DepthTest = 1u << 2,
        DepthWrite = 1u << 3,
        DepthCompare = 1u << 4,
        DepthBiasEnable = 1u << 5,
        DepthBias = 1u << 6,
        StencilTest = 1u << 7,
        FrontOps = 1u << 8,
        BackOps = 1u << 9,
        FrontReference = 1u << 10,
        BackReference = 1u << 11,
        FrontWriteMask = 1u << 12,
        BackWriteMask = 1u << 13,
        FrontCompareMask = 1u << 14,
        BackCompareMask = 1u << 15,
        RasterizerDiscard = 1u << 16,
        CullMode = 1u << 17,
        FrontFace = 1u << 18,
        BlendConstants = 1u << 19,
    };
    u32 bits{};
    Backend::Viewports viewports;
    Backend::Scissors scissors;
    bool depth_test{}, depth_write{}, depth_bias_enable{}, stencil_test{}, rasterizer_discard{};
    Backend::CompareOp depth_compare{};
    float depth_bias[3]{};
    Backend::StencilOps front_ops{}, back_ops{};
    u32 front_reference{}, back_reference{}, front_write_mask{}, back_write_mask{};
    u32 front_compare_mask{}, back_compare_mask{};
    Backend::CullMode cull_mode{};
    Backend::FrontFace front_face{};
    std::array<float, 4> blend_constants{};
};

void SetFace(MetalFrame::Owned::Face& face, const Backend::StencilOps& ops) {
    face.fail = static_cast<VkStencilOp>(ops.fail_op);
    face.pass = static_cast<VkStencilOp>(ops.pass_op);
    face.depth_fail = static_cast<VkStencilOp>(ops.depth_fail_op);
    face.compare = static_cast<VkCompareOp>(ops.compare_op);
}

/// The encoder's side of the draw state (as the layer's vkCmdSet* encodings set it).
void Apply(MetalFrame::Owned& o, const DrawStateDelta& d) {
    using B = DrawStateDelta;
    if (d.bits & B::Viewports) {
        static_assert(sizeof(Backend::Viewport) == sizeof(VkViewport));
        MetalFrame::SetViewports(o, u32(d.viewports.size()),
                                 reinterpret_cast<const VkViewport*>(d.viewports.data()));
    }
    if (d.bits & B::Scissors) {
        static_assert(sizeof(Backend::Rect2D) == sizeof(VkRect2D));
        MetalFrame::SetScissors(o, u32(d.scissors.size()),
                                reinterpret_cast<const VkRect2D*>(d.scissors.data()));
    }
    if (d.bits & B::DepthTest) o.depth_test = d.depth_test;
    if (d.bits & B::DepthWrite) o.depth_write = d.depth_write;
    if (d.bits & B::DepthCompare) o.depth_compare = static_cast<VkCompareOp>(d.depth_compare);
    if (d.bits & B::DepthBiasEnable) o.depth_bias_enable = d.depth_bias_enable;
    if (d.bits & B::DepthBias) std::copy_n(d.depth_bias, 3, o.depth_bias);
    if (d.bits & B::StencilTest) o.stencil_test = d.stencil_test;
    if (d.bits & B::FrontOps) SetFace(o.front, d.front_ops);
    if (d.bits & B::BackOps) SetFace(o.back, d.back_ops);
    if (d.bits & B::FrontReference) o.front.reference = d.front_reference;
    if (d.bits & B::BackReference) o.back.reference = d.back_reference;
    if (d.bits & B::FrontWriteMask) o.front.write_mask = d.front_write_mask;
    if (d.bits & B::BackWriteMask) o.back.write_mask = d.back_write_mask;
    if (d.bits & B::FrontCompareMask) o.front.compare_mask = d.front_compare_mask;
    if (d.bits & B::BackCompareMask) o.back.compare_mask = d.back_compare_mask;
    if (d.bits & B::RasterizerDiscard) o.rasterizer_discard = d.rasterizer_discard;
    if (d.bits & B::CullMode) o.cull_mode = static_cast<VkCullModeFlags>(d.cull_mode);
    if (d.bits & B::FrontFace) o.front_face = static_cast<VkFrontFace>(d.front_face);
    if (d.bits & B::BlendConstants) {
        std::copy_n(d.blend_constants.data(), 4, o.blend_constants);
    }
}

MetalFrame::Owned* Encoder(vk::CommandBuffer cmdbuf) {
    auto* o = MetalLayer::OwnedFor(static_cast<VkCommandBuffer>(cmdbuf));
    if (!o) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            std::printf("Metal recorder: a command buffer the Metal frame does not own (skipped)\n");
        }
    }
    return o;
}

/// Runs `encode(owned)` as the layer runs its encodings: counted, in an autorelease pool (the
/// recording threads never drain one of their own).
template <typename F>
void Encode(vk::CommandBuffer cmdbuf, F&& encode) {
    if (auto* o = Encoder(cmdbuf)) {
        ++o->commands;
        @autoreleasepool {
            encode(*o);
        }
    }
}

VkBuffer AsBuffer(u64 native) {
    return reinterpret_cast<VkBuffer>(native);
}
VkPipeline AsPipeline(u64 native) {
    return reinterpret_cast<VkPipeline>(native);
}

class FrameRecorder final : public Vulkan::CommandRecorder {
public:
    using Vulkan::CommandRecorder::CommandRecorder;

    // Each command captures only what it encodes, as the Vulkan recorder's (larger captures fill
    // the recording chunks sooner).
    void Draw(const Backend::DrawCommand& d, const Backend::Crumb& crumb) override {
        if (d.args_buffer && d.count_buffer) {
            // Not encoded by the Metal frame either (counted as unsupported there).
            Vulkan::CommandRecorder::Draw(d, crumb);
            return;
        }
        const VkPipeline pipeline = AsPipeline(d.pipeline);
        if (!d.args_buffer) {
            scheduler.RecordCrumb(crumb, [pipeline, indexed = d.indexed, count = d.count,
                                          instances = d.instances, first = d.first,
                                          first_instance = d.first_instance](vk::CommandBuffer cmd) {
                Encode(cmd, [&](MetalFrame::Owned& o) {
                    MetalFrame::BindPipeline(o, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                    if (indexed) {
                        MetalFrame::DrawIndexed(o, count, instances, 0, s32(first), first_instance);
                    } else {
                        MetalFrame::Draw(o, count, instances, first, first_instance);
                    }
                });
            });
            return;
        }
        scheduler.RecordCrumb(crumb, [pipeline, indexed = d.indexed, args = AsBuffer(d.args_buffer),
                                      args_offset = d.args_offset, max_draws = d.max_draws,
                                      stride = d.stride](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::BindPipeline(o, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                MetalFrame::DrawIndirect(o, args, args_offset, max_draws, stride, indexed);
            });
        });
    }

    void Dispatch(const Backend::DispatchCommand& d, const Backend::Crumb& crumb) override {
        const VkPipeline pipeline = AsPipeline(d.pipeline);
        if (!d.args_buffer) {
            scheduler.RecordCrumb(crumb, [pipeline, x = d.groups[0], y = d.groups[1],
                                          z = d.groups[2]](vk::CommandBuffer cmd) {
                Encode(cmd, [&](MetalFrame::Owned& o) {
                    MetalFrame::BindPipeline(o, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                    MetalFrame::Dispatch(o, x, y, z);
                });
            });
            return;
        }
        scheduler.RecordCrumb(crumb, [pipeline, args = AsBuffer(d.args_buffer),
                                      args_offset = d.args_offset,
                                      raw = vk::Buffer{AsBuffer(d.crumb_args_buffer)},
                                      raw_offset = d.crumb_args_offset,
                                      slot = crumb.args_slot](vk::CommandBuffer cmd) {
            Vulkan::Breadcrumbs::CopyArgs(cmd, raw, raw_offset, slot); // what the game wrote
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::BindPipeline(o, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                MetalFrame::DispatchIndirect(o, args, args_offset);
            });
        });
    }

    void BindIndexBuffer(u64 buffer, u64 offset, Backend::IndexType type) override {
        scheduler.Record([handle = AsBuffer(buffer), offset, type](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::BindIndexBuffer(o, handle, offset, static_cast<VkIndexType>(type));
            });
        });
    }

    void BindVertexBuffers(std::span<const u64> buffers, std::span<const u64> offsets,
                           std::span<const u64> sizes, std::span<const u64> strides) override {
        const u32 count = static_cast<u32>(buffers.size());
        // Handles and offsets are 64-bit like u64: read in place. The Metal frame takes no sizes.
        const auto handles = scheduler.RecordData(std::span<const VkBuffer>{
            reinterpret_cast<const VkBuffer*>(buffers.data()), count});
        const auto vk_offsets = scheduler.RecordData(std::span<const VkDeviceSize>{offsets});
        const auto vk_strides = strides.empty()
                                    ? std::span<const VkDeviceSize>{}
                                    : scheduler.RecordData(std::span<const VkDeviceSize>{strides});
        scheduler.Record([count, handles, vk_offsets, vk_strides](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::BindVertexBuffers(o, 0, count, handles.data(), vk_offsets.data(),
                                              vk_strides.empty() ? nullptr : vk_strides.data());
            });
        });
    }

    // Barriers: Metal tracks hazards within the encoders; the Metal frame ends the open encoders
    // at each one, as for the layer's vkCmdPipelineBarrier.
    void Barrier(Backend::MemoryUse, Backend::MemoryUse) override {
        RecordBarrier();
    }
    void BufferBarrier(u64, u64, u64, Backend::MemoryUse, Backend::MemoryUse) override {
        RecordBarrier();
    }
    void TransitImage(const Backend::ImageAllocation& image, Backend::ImageAspect aspect,
                      Backend::ImageAccess access, bool discard) override {
        // The image's tracked use, as the Vulkan recorder keeps it (later uses read it).
        auto& state = static_cast<Vulkan::ImageState*>(image.backend_state)->state;
        state = Vulkan::ImageAccessState(access, Backend::Has(aspect, Backend::ImageAspect::Depth),
                                         Backend::Has(aspect, Backend::ImageAspect::Stencil), false);
        (void)discard;
        RecordBarrier();
    }

    void ClearImage(const Backend::ImageAllocation& image,
                    const Backend::ClearColor& value) override {
        VkClearColorValue clear{};
        std::memcpy(clear.uint32, value.data(), sizeof(value));
        scheduler.Record([image = reinterpret_cast<VkImage>(image.native), clear](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                MetalFrame::ClearColorImage(o, image, &clear, 1, &range);
            });
        });
    }

    void BlitImage(u64 src, u64 dst, const Backend::ImageBlit& blit) override {
        const auto aspect = static_cast<VkImageAspectFlags>(blit.aspect);
        const VkImageBlit region{
            .srcSubresource = {aspect, blit.src_level, 0, 1},
            .srcOffsets = {{0, 0, 0}, {s32(blit.src_width), s32(blit.src_height), 1}},
            .dstSubresource = {aspect, blit.dst_level, 0, 1},
            .dstOffsets = {{0, 0, 0}, {s32(blit.dst_width), s32(blit.dst_height), 1}},
        };
        scheduler.Record([src = reinterpret_cast<VkImage>(src), dst = reinterpret_cast<VkImage>(dst),
                          region](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::BlitImage(o, src, dst, 1, &region, VK_FILTER_NEAREST);
            });
        });
    }

    void FillBuffer(u64 buffer, u64 offset, u64 size, u32 value) override {
        scheduler.Record([buffer = AsBuffer(buffer), offset, size, value](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::FillBuffer(o, buffer, offset, size, value);
            });
        });
    }

    /// A label the guest waits on, as the portable label (vk_gpu_labels.cpp) the layer encoded:
    /// after all earlier work, the high word before the low word the guest polls.
    void WriteLabel(u64 buffer, u64 offset, u64 value, u32 num_bytes) override {
        scheduler.Record([buffer = AsBuffer(buffer), offset, value, num_bytes](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::Barrier(o);
                if (num_bytes == 8) {
                    const u32 high = u32(value >> 32);
                    MetalFrame::UpdateBuffer(o, buffer, offset + 4, sizeof(high), &high);
                    MetalFrame::Barrier(o);
                }
                const u32 low = u32(value);
                MetalFrame::UpdateBuffer(o, buffer, offset, sizeof(low), &low);
                MetalFrame::Barrier(o);
            });
        });
    }

protected:
    void RecordBarrier() {
        scheduler.Record([](vk::CommandBuffer cmd) {
            Encode(cmd, [](MetalFrame::Owned& o) { MetalFrame::Barrier(o); });
        });
    }

    void RecordPushConstants(vk::PipelineLayout, vk::ShaderStageFlags,
                             std::span<const u8> data) override {
        scheduler.Record([data](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::PushConstants(o, 0, u32(data.size()), data.data());
            });
        });
    }

    void RecordPushDescriptors(vk::PipelineBindPoint bind_point, vk::PipelineLayout,
                               std::span<const vk::WriteDescriptorSet> writes) override {
        static_assert(sizeof(vk::WriteDescriptorSet) == sizeof(VkWriteDescriptorSet));
        scheduler.Record([point = static_cast<VkPipelineBindPoint>(bind_point),
                          writes](vk::CommandBuffer cmd) {
            Encode(cmd, [&](MetalFrame::Owned& o) {
                MetalFrame::PushDescriptorSet(
                    o, point, 0, u32(writes.size()),
                    reinterpret_cast<const VkWriteDescriptorSet*>(writes.data()));
            });
        });
    }

public:

    /// The changed draw state as one command (the Vulkan recorder records one per field, and
    /// the layer translated each back).
    void CommitDrawState() const override {
        auto& s = scheduler.GetDynamicState();
        auto& dirty = s.dirty_state;
        using B = DrawStateDelta;
        DrawStateDelta d;
        if (dirty.viewports) {
            dirty.viewports = false;
            d.bits |= B::Viewports;
            d.viewports = s.viewports;
        }
        if (dirty.scissors) {
            dirty.scissors = false;
            d.bits |= B::Scissors;
            d.scissors = s.scissors;
        }
        if (dirty.depth_test_enabled) {
            dirty.depth_test_enabled = false;
            d.bits |= B::DepthTest;
            d.depth_test = s.depth_test_enabled;
        }
        if (dirty.depth_write_enabled) {
            dirty.depth_write_enabled = false;
            d.bits |= B::DepthWrite;
            d.depth_write = s.depth_write_enabled;
        }
        if (s.depth_test_enabled && dirty.depth_compare_op) {
            dirty.depth_compare_op = false;
            d.bits |= B::DepthCompare;
            d.depth_compare = s.depth_compare_op;
        }
        // No depth bounds test in Metal.
        dirty.depth_bounds_test_enabled = false;
        if (s.depth_bounds_test_enabled) {
            dirty.depth_bounds = false;
        }
        if (dirty.depth_bias_enabled) {
            dirty.depth_bias_enabled = false;
            d.bits |= B::DepthBiasEnable;
            d.depth_bias_enable = s.depth_bias_enabled;
        }
        if (s.depth_bias_enabled && dirty.depth_bias) {
            dirty.depth_bias = false;
            d.bits |= B::DepthBias;
            d.depth_bias[0] = s.depth_bias_constant;
            d.depth_bias[1] = s.depth_bias_clamp;
            d.depth_bias[2] = s.depth_bias_slope;
        }
        if (dirty.stencil_test_enabled) {
            dirty.stencil_test_enabled = false;
            d.bits |= B::StencilTest;
            d.stencil_test = s.stencil_test_enabled;
        }
        if (s.stencil_test_enabled) {
            // (Bit-fields: no references to them.)
#define BB_TAKE(flag, bit)                                                                         \
    if (dirty.flag) {                                                                              \
        dirty.flag = false;                                                                        \
        d.bits |= bit;                                                                             \
    }
            BB_TAKE(stencil_front_ops, B::FrontOps)
            BB_TAKE(stencil_back_ops, B::BackOps)
            BB_TAKE(stencil_front_reference, B::FrontReference)
            BB_TAKE(stencil_back_reference, B::BackReference)
            BB_TAKE(stencil_front_write_mask, B::FrontWriteMask)
            BB_TAKE(stencil_back_write_mask, B::BackWriteMask)
            BB_TAKE(stencil_front_compare_mask, B::FrontCompareMask)
            BB_TAKE(stencil_back_compare_mask, B::BackCompareMask)
#undef BB_TAKE
            d.front_ops = s.stencil_front_ops;
            d.back_ops = s.stencil_back_ops;
            d.front_reference = s.stencil_front_reference;
            d.back_reference = s.stencil_back_reference;
            d.front_write_mask = s.stencil_front_write_mask;
            d.back_write_mask = s.stencil_back_write_mask;
            d.front_compare_mask = s.stencil_front_compare_mask;
            d.back_compare_mask = s.stencil_back_compare_mask;
        }
        // Primitive restart is always on for strips in Metal; no line width.
        dirty.primitive_restart_enable = false;
        dirty.line_width = false;
        if (dirty.rasterizer_discard_enable) {
            dirty.rasterizer_discard_enable = false;
            d.bits |= B::RasterizerDiscard;
            d.rasterizer_discard = s.rasterizer_discard_enable;
        }
        if (dirty.cull_mode) {
            dirty.cull_mode = false;
            d.bits |= B::CullMode;
            d.cull_mode = s.cull_mode;
        }
        if (dirty.front_face) {
            dirty.front_face = false;
            d.bits |= B::FrontFace;
            d.front_face = s.front_face;
        }
        if (dirty.blend_constants) {
            dirty.blend_constants = false;
            d.bits |= B::BlendConstants;
            d.blend_constants = s.blend_constants;
        }
        // No dynamic write masks or feedback-loop layouts in the Metal frame; the flags clear as
        // the Vulkan recorder clears them.
        dirty.color_write_masks = false;
        if (dirty.feedback_loop_enabled &&
            scheduler.GetInstance().IsAttachmentFeedbackLoopLayoutSupported()) {
            dirty.feedback_loop_enabled = false;
        }
        if (!d.bits) {
            return;
        }
        scheduler.Record([d = std::move(d)](vk::CommandBuffer cmdbuf) {
            if (auto* o = Encoder(cmdbuf)) {
                Apply(*o, d);
            }
        });
    }
};

/// The scheduler's passes and the runtime's transfers, encoded as the layer encodes the Vulkan
/// commands they replace.
class FrameStream final : public Vulkan::StreamEncoder {
public:
    void CopyBuffer(vk::CommandBuffer cmd, vk::Buffer src, vk::Buffer dst,
                    std::span<const vk::BufferCopy> regions) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::CopyBuffer(o, src, dst, u32(regions.size()),
                                   reinterpret_cast<const VkBufferCopy*>(regions.data()));
        });
    }
    void FillBuffer(vk::CommandBuffer cmd, vk::Buffer dst, u64 offset, u64 size,
                    u32 value) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) { MetalFrame::FillBuffer(o, dst, offset, size, value); });
    }
    void UpdateBuffer(vk::CommandBuffer cmd, vk::Buffer dst, u64 offset,
                      std::span<const u8> data) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::UpdateBuffer(o, dst, offset, data.size_bytes(), data.data());
        });
    }
    void CopyBufferToImage(vk::CommandBuffer cmd, vk::Buffer src, vk::Image dst,
                           std::span<const vk::BufferImageCopy> regions) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::CopyBufferToImage(o, src, dst, u32(regions.size()),
                                          reinterpret_cast<const VkBufferImageCopy*>(regions.data()));
        });
    }
    void CopyImageToBuffer(vk::CommandBuffer cmd, vk::Image src, vk::Buffer dst,
                           std::span<const vk::BufferImageCopy> regions,
                           vk::ImageLayout) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::CopyImageToBuffer(o, src, dst, u32(regions.size()),
                                          reinterpret_cast<const VkBufferImageCopy*>(regions.data()));
        });
    }
    void CopyImage(vk::CommandBuffer cmd, vk::Image src, vk::Image dst,
                   std::span<const vk::ImageCopy> regions, vk::ImageLayout,
                   vk::ImageLayout) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::CopyImage(o, src, dst, u32(regions.size()),
                                  reinterpret_cast<const VkImageCopy*>(regions.data()));
        });
    }
    void BlitImage(vk::CommandBuffer cmd, vk::Image src, vk::ImageLayout, vk::Image dst,
                   vk::ImageLayout, std::span<const vk::ImageBlit> regions,
                   vk::Filter filter) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::BlitImage(o, src, dst, u32(regions.size()),
                                  reinterpret_cast<const VkImageBlit*>(regions.data()),
                                  static_cast<VkFilter>(filter));
        });
    }
    void ClearDepthStencilImage(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout,
                                const vk::ClearDepthStencilValue& value,
                                const vk::ImageSubresourceRange& range) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::ClearDepthStencilImage(
                o, image, reinterpret_cast<const VkClearDepthStencilValue*>(&value), 1,
                reinterpret_cast<const VkImageSubresourceRange*>(&range));
        });
    }
    void ResolveImage(vk::CommandBuffer cmd, vk::Image src, vk::Image dst,
                      const vk::ImageResolve& region) const override {
        // Not encoded by the Metal frame (counted as unsupported there): the layer's way.
        Vulkan::VulkanStreamEncoder().ResolveImage(cmd, src, dst, region);
    }
    void ClearColorImage(vk::CommandBuffer cmd, vk::Image image, const vk::ClearColorValue& color,
                         const vk::ImageSubresourceRange& range) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::ClearColorImage(o, image, reinterpret_cast<const VkClearColorValue*>(&color),
                                        1, reinterpret_cast<const VkImageSubresourceRange*>(&range));
        });
    }
    void Barrier(vk::CommandBuffer cmd, const vk::DependencyInfo& info) const override {
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::Barrier(o, reinterpret_cast<const VkDependencyInfo*>(&info));
        });
    }
    void BeginRendering(vk::CommandBuffer cmd, const vk::RenderingInfo& info) const override {
        static_assert(sizeof(vk::RenderingInfo) == sizeof(VkRenderingInfo));
        Encode(cmd, [&](MetalFrame::Owned& o) {
            MetalFrame::BeginRendering(o, reinterpret_cast<const VkRenderingInfo*>(&info));
        });
    }
    void EndRendering(vk::CommandBuffer cmd) const override {
        Encode(cmd, [](MetalFrame::Owned& o) { MetalFrame::EndRendering(o); });
    }
};

} // namespace

const Vulkan::StreamEncoder& FrameStreamEncoder() {
    static const FrameStream encoder;
    return encoder;
}

std::unique_ptr<Vulkan::CommandRecorder> MakeFrameRecorder(Vulkan::Scheduler& scheduler,
                                                           Vulkan::DescriptorHeap& descriptor_heap,
                                                           vk::Device device) {
    std::printf("Metal recorder: draws, dispatches, bindings, the draw state, barriers, clears, "
                "blits, fills, labels, passes and the runtime's transfers are encoded in Metal directly\n");
    return std::make_unique<FrameRecorder>(scheduler, descriptor_heap, device);
}

} // namespace Metal
