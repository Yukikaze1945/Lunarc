#define DLSS_GPU_RESOURCES_EXPORTS
#include "dlss_gpu_resources.h"

#include <d3d12.h>

namespace {

bool same_device(ID3D12Device* expected, ID3D12DeviceChild* child) {
    ID3D12Device* owner = nullptr;
    if (FAILED(child->GetDevice(IID_PPV_ARGS(&owner))) || !owner) return false;
    IUnknown* expected_identity = nullptr;
    IUnknown* owner_identity = nullptr;
    const bool ok = SUCCEEDED(expected->QueryInterface(
        IID_PPV_ARGS(&expected_identity))) &&
        SUCCEEDED(owner->QueryInterface(IID_PPV_ARGS(&owner_identity))) &&
        expected_identity == owner_identity;
    if (expected_identity) expected_identity->Release();
    if (owner_identity) owner_identity->Release();
    owner->Release();
    return ok;
}

int32_t check_texture(ID3D12Device* device, void* raw,
                      DXGI_FORMAT format, uint32_t width, uint32_t height,
                      bool needs_uav) {
    if (!raw) return DLSS_GPU_BAD_ARGUMENT;
    auto* resource = static_cast<ID3D12Resource*>(raw);
    if (!same_device(device, resource)) return DLSS_GPU_DEVICE_MISMATCH;
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
        desc.SampleDesc.Count != 1) return DLSS_GPU_BAD_RESOURCE;
    if (desc.Width != width || desc.Height != height) return DLSS_GPU_BAD_EXTENT;
    if (desc.Format != format ||
        (needs_uav && !(desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS)))
        return DLSS_GPU_BAD_FORMAT;
    return DLSS_GPU_VALID;
}

}  // namespace

extern "C" DLSS_GPU_API uint32_t dlss_gpu_resources_abi_version(void) {
    return DLSS_GPU_RESOURCES_ABI_VERSION;
}

extern "C" DLSS_GPU_API int32_t dlss_gpu_validate_frame(
    const DlssGpuFrameResources* frame) {
    if (!frame || frame->struct_size != sizeof(DlssGpuFrameResources) ||
        !frame->device || !frame->width || !frame->height ||
        !frame->color || !frame->motion || !frame->depth || !frame->output)
        return DLSS_GPU_BAD_ARGUMENT;
    if ((!frame->producer_fence && frame->producer_value) ||
        (frame->producer_fence && !frame->producer_value))
        return DLSS_GPU_BAD_FENCE;

    auto* device = static_cast<ID3D12Device*>(frame->device);
    if (frame->producer_fence && !same_device(
            device, static_cast<ID3D12Fence*>(frame->producer_fence)))
        return DLSS_GPU_DEVICE_MISMATCH;

    const struct {
        void* resource;
        DXGI_FORMAT format;
        bool needs_uav;
    } fields[] = {
        {frame->color, DXGI_FORMAT_R8G8B8A8_UNORM, false},
        {frame->motion, DXGI_FORMAT_R16G16_FLOAT, false},
        {frame->depth, DXGI_FORMAT_R32_FLOAT, false},
        {frame->output, DXGI_FORMAT_R8G8B8A8_UNORM, true},
    };
    for (const auto& field : fields) {
        const int32_t result = check_texture(
            device, field.resource, field.format,
            frame->width, frame->height, field.needs_uav);
        if (result != DLSS_GPU_VALID) return result;
    }
    if (frame->color == frame->output) return DLSS_GPU_BAD_RESOURCE;
    return DLSS_GPU_VALID;
}
