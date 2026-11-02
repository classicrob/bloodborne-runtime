// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "bbport_stale_desc.h"
#include "common/logging/log.h"
#include "shader_recompiler/resource.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_state.h"
#include "video_core/texture_cache/image.h"
#include "video_core/texture_cache/image_view.h"

#include <magic_enum/magic_enum.hpp>

namespace VideoCore {

Backend::ImageViewType ConvertImageViewType(AmdGpu::ImageType type) {
    switch (type) {
    case AmdGpu::ImageType::Color1D:
        return Backend::ImageViewType::View1D;
    case AmdGpu::ImageType::Color1DArray:
        return Backend::ImageViewType::View1DArray;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DMsaa:
        return Backend::ImageViewType::View2D;
    case AmdGpu::ImageType::Color2DArray:
        return Backend::ImageViewType::View2DArray;
    case AmdGpu::ImageType::Color3D:
        return Backend::ImageViewType::View3D;
    default:
        UNREACHABLE();
    }
}

bool IsViewTypeCompatible(AmdGpu::ImageType view_type, AmdGpu::ImageType image_type) {
    switch (view_type) {
    case AmdGpu::ImageType::Color1D:
    case AmdGpu::ImageType::Color1DArray:
        return image_type == AmdGpu::ImageType::Color1D;
    case AmdGpu::ImageType::Color2D:
    case AmdGpu::ImageType::Color2DArray:
    case AmdGpu::ImageType::Color2DMsaa:
    case AmdGpu::ImageType::Color2DMsaaArray:
        return image_type == AmdGpu::ImageType::Color2D || image_type == AmdGpu::ImageType::Color3D;
    case AmdGpu::ImageType::Color3D:
        return image_type == AmdGpu::ImageType::Color3D;
    default:
        UNREACHABLE();
    }
}

ImageViewInfo::ImageViewInfo(const AmdGpu::Image& image, const Shader::ImageResource& desc) noexcept
    : is_storage{desc.is_written} {
    const auto dfmt = image.GetDataFmt();
    auto nfmt = image.GetNumberFmt();
    if (is_storage && nfmt == AmdGpu::NumberFormat::Srgb) {
        nfmt = AmdGpu::NumberFormat::Unorm;
    }
    {
        const BbStaleDesc::Scope scope{"texture T# (view)", &image, sizeof(image), image.Address()};
        format = Vulkan::LiverpoolToVK::SurfaceFormat(dfmt, nfmt);
    }
    if (desc.is_depth) {
        format = Vulkan::LiverpoolToVK::PromoteFormatToDepth(format);
    }

    range.base.level = image.base_level;
    range.base.layer = std::min<u32>(image.base_array, image.NumLayers() - 1);
    range.extent.levels = image.NumViewLevels(desc.is_array);
    range.extent.layers = image.NumViewLayers(desc.is_array);
    type = image.GetViewType(desc.is_array);
    min_lod = static_cast<u32>(image.min_lod);

    if (!is_storage) {
        mapping = Vulkan::LiverpoolToVK::ComponentMapping(image.DstSelect());
    }
}

ImageViewInfo::ImageViewInfo(const AmdGpu::ColorBuffer& col_buffer) noexcept {
    range.base.layer = col_buffer.BaseSlice();
    range.extent.layers = col_buffer.NumSlices() - range.base.layer;
    type = range.extent.layers > 1 ? AmdGpu::ImageType::Color2DArray : AmdGpu::ImageType::Color2D;
    const BbStaleDesc::Scope scope{"color buffer (view)", &col_buffer, sizeof(col_buffer),
                                   col_buffer.Address()};
    format =
        Vulkan::LiverpoolToVK::SurfaceFormat(col_buffer.GetDataFmt(), col_buffer.GetNumberFmt());
}

ImageViewInfo::ImageViewInfo(const AmdGpu::DepthBuffer& depth_buffer, AmdGpu::DepthView view,
                             AmdGpu::DepthControl ctl) {
    format = Vulkan::LiverpoolToVK::DepthFormat(depth_buffer.z_info.format,
                                                depth_buffer.stencil_info.format);
    is_storage = ctl.depth_write_enable;
    range.base.layer = view.slice_start;
    range.extent.layers = view.NumSlices() - range.base.layer;
    type = range.extent.layers > 1 ? AmdGpu::ImageType::Color2DArray : AmdGpu::ImageType::Color2D;
}

ImageView::ImageView(Backend::Device& device_, const ImageViewInfo& info_, const Image& image)
    : info{info_}, device{&device_} {
    using Backend::ImageAspect;
    using Backend::ImageUse;
    const ImageUse image_usage = Vulkan::FromVk(image.usage_flags);
    ImageUse usage = image_usage;
    if (!info.is_storage) {
        usage = static_cast<ImageUse>(static_cast<u32>(usage) & ~static_cast<u32>(ImageUse::Storage));
    }
    float min_lod = 0.0f;
    if (info.min_lod != 0) {
        const float last_level =
            static_cast<float>(info.range.base.level + info.range.extent.levels - 1);
        min_lod = std::min(static_cast<float>(info.min_lod) / 256.f, last_level);
    }
    // When sampling D32/D16 texture from shader, the T# specifies R32/R16 format so adjust it.
    Backend::Format format = info.format;
    const ImageAspect image_aspect = Vulkan::FromVk(image.aspect_mask);
    ImageAspect aspect = image_aspect;
    if (Backend::Has(image_aspect, ImageAspect::Depth) &&
        Vulkan::LiverpoolToVK::IsFormatDepthCompatible(format)) {
        format = image.info.pixel_format;
        aspect = ImageAspect::Depth;
    }
    if (Backend::Has(image_aspect, ImageAspect::Stencil) &&
        Vulkan::LiverpoolToVK::IsFormatStencilCompatible(format)) {
        format = image.info.pixel_format;
        aspect = ImageAspect::Stencil;
    }
    if (!IsViewTypeCompatible(info.type, image.info.type)) {
        LOG_ERROR(Render_Vulkan, "image view type {} is incompatible with image type {}",
                  magic_enum::enum_name(info.type), magic_enum::enum_name(image.info.type));
    }

    const Backend::ImageViewDesc desc{
        .type = ConvertImageViewType(info.type),
        .format = device->SupportedFormat(format, image_usage),
        .aspect = aspect,
        .base_level = info.range.base.level,
        .levels = info.range.extent.levels,
        .base_layer = info.range.base.layer,
        .layers = info.range.extent.layers,
        .mapping = info.mapping,
        .usage = usage,
        .min_lod = min_lod,
    };
    const auto view_aspect = Backend::Has(aspect, ImageAspect::Depth)     ? "Depth"
                             : Backend::Has(aspect, ImageAspect::Stencil) ? "Stencil"
                                                                          : "Color";
    const auto allocation = device->CreateImageView(
        image.backing->handle, desc,
        fmt::format("ImageView {}x{}x{} {:#x}:{:#x} {}:{} {}:{} ({})", image.info.size.width,
                    image.info.size.height, image.info.size.depth, image.info.guest_address,
                    image.info.guest_size, info.range.base.level,
                    info.range.base.level + info.range.extent.levels - 1, info.range.base.layer,
                    info.range.base.layer + info.range.extent.layers - 1, view_aspect));
    handle = allocation.handle;
    native = allocation.native;
}

ImageView::~ImageView() {
    if (handle) {
        device->DestroyImageView(handle);
    }
}

} // namespace VideoCore
