#include "dlss_gpu_resources.h"
#include <d3d12.h>
#include <dxgi1_6.h>
#include <cassert>

static ID3D12Resource* texture(ID3D12Device* device, DXGI_FORMAT format,
                              D3D12_RESOURCE_FLAGS flags) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 64;
    desc.Height = 64;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
    desc.Flags = flags;
    ID3D12Resource* result = nullptr;
    const HRESULT hr = device->CreateCommittedResource(
        &heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
        nullptr, IID_PPV_ARGS(&result));
    assert(SUCCEEDED(hr) && result);
    return result;
}

int main() {
    assert(dlss_gpu_resources_abi_version() == DLSS_GPU_RESOURCES_ABI_VERSION);
    ID3D12Device* device = nullptr;
    assert(SUCCEEDED(D3D12CreateDevice(
        nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))));
    auto* color = texture(device, DXGI_FORMAT_R8G8B8A8_UNORM,
                          D3D12_RESOURCE_FLAG_NONE);
    auto* motion = texture(device, DXGI_FORMAT_R16G16_FLOAT,
                           D3D12_RESOURCE_FLAG_NONE);
    auto* depth = texture(device, DXGI_FORMAT_R32_FLOAT,
                          D3D12_RESOURCE_FLAG_NONE);
    auto* output = texture(device, DXGI_FORMAT_R8G8B8A8_UNORM,
                           D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
    DlssGpuFrameResources frame{};
    frame.struct_size = sizeof(frame);
    frame.device = device;
    frame.color = color;
    frame.motion = motion;
    frame.depth = depth;
    frame.output = output;
    frame.width = 64;
    frame.height = 64;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_VALID);

    frame.width = 32;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_BAD_EXTENT);
    frame.width = 64;
    frame.motion = depth;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_BAD_FORMAT);
    frame.motion = motion;
    frame.producer_value = 1;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_BAD_FENCE);
    frame.producer_value = 0;
    IDXGIFactory4* factory = nullptr;
    assert(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))));
    IDXGIAdapter* warp = nullptr;
    assert(SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp))));
    ID3D12Device* other_device = nullptr;
    assert(SUCCEEDED(D3D12CreateDevice(
        warp, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&other_device))));
    frame.device = other_device;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_DEVICE_MISMATCH);
    frame.device = device;
    other_device->Release();
    warp->Release();
    factory->Release();
    frame.output = color;
    assert(dlss_gpu_validate_frame(&frame) == DLSS_GPU_BAD_FORMAT ||
           dlss_gpu_validate_frame(&frame) == DLSS_GPU_BAD_RESOURCE);

    output->Release();
    depth->Release();
    motion->Release();
    color->Release();
    device->Release();
    return 0;
}
