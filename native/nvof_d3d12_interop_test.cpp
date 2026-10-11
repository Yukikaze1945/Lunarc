#include "nvof_d3d12_interop.h"
#include "nvof_bridge.h"

#include <windows.h>
#include <d3d12.h>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

static float from_half(uint16_t bits) {
    const float sign = (bits & 0x8000) ? -1.0f : 1.0f;
    const int exponent = (bits >> 10) & 31;
    const int mantissa = bits & 1023;
    if (exponent == 0) return sign * std::ldexp(float(mantissa), -24);
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    return sign * std::ldexp(float(1024 + mantissa), exponent - 25);
}

int main() {
    constexpr uint32_t w = 64, h = 64;
    constexpr size_t bytes = size_t(w) * h * 8;
    std::vector<uint8_t> previous(size_t(w) * h * 3);
    std::vector<uint8_t> current(previous.size());
    for (uint32_t y = 0; y < h; ++y) for (uint32_t x = 0; x < w; ++x) {
        const size_t p = (size_t(y) * w + x) * 3;
        previous[p] = static_cast<uint8_t>((x * 7 + y) % 256);
        previous[p + 1] = static_cast<uint8_t>((x + y * 5) % 256);
        previous[p + 2] = static_cast<uint8_t>((x * 3 + y * 2) % 256);
        const uint32_t shifted = x ? x - 1 : 0;
        const size_t q = (size_t(y) * w + shifted) * 3;
        current[p] = previous[q];
        current[p + 1] = previous[q + 1];
        current[p + 2] = previous[q + 2];
    }
    void* device_session = nullptr;
    void* host_session = nullptr;
    if (nvof_bridge_create(w, h, 4, 10, &device_session) ||
        nvof_bridge_create(w, h, 4, 10, &host_session)) {
        std::printf("nvof_create_failed: %s\n", nvof_bridge_get_last_error());
        return 1;
    }
    NvOFBridgeTiming timing{};
    if (nvof_bridge_execute_bgr_sequential_device(device_session,
            current.data(), w * 3, previous.data(), w * 3, 1, &timing)) {
        std::printf("nvof_device_execute_failed: %s\n", nvof_bridge_get_last_error());
        return 2;
    }
    NvOFBridgeCounters counters{};
    nvof_bridge_get_counters(device_session, &counters);
    if (counters.d2h_copy_count || counters.d2h_bytes) return 3;
    uint64_t source = 0;
    size_t pitch = 0;
    uint32_t index = 0, owner = 0;
    void* context = nullptr;
    if (nvof_bridge_get_output_deviceptr(device_session, &source, &pitch) ||
        nvof_bridge_get_session_info(device_session, &index, &context, &owner)) return 4;
    ID3D12Device* d3d = nullptr;
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                   IID_PPV_ARGS(&d3d));
    if (FAILED(hr)) return 5;
    NvofD3D12FlowBridge* bridge = nullptr;
    const int interop_result = nvof_d3d12_create(d3d, context, w, h, &bridge);
    if (interop_result) {
        std::printf("interop_create_failed=%d: %s\n", interop_result,
                    nvof_d3d12_last_error());
        return 6;
    }
    uint64_t cuda_done = 0;
    if (nvof_d3d12_copy_flow(bridge, source, pitch, &cuda_done)) {
        std::printf("interop_copy_failed: %s\n", nvof_d3d12_last_error());
        return 7;
    }
    void* raw_buffer = nullptr;
    void* raw_fence = nullptr;
    nvof_d3d12_get_resources(bridge, &raw_buffer, &raw_fence);
    auto* shared = static_cast<ID3D12Resource*>(raw_buffer);
    auto* cuda_fence = static_cast<ID3D12Fence*>(raw_fence);

    D3D12_HEAP_PROPERTIES readback_props{};
    readback_props.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer_desc{};
    buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer_desc.Width = bytes;
    buffer_desc.Height = 1;
    buffer_desc.DepthOrArraySize = 1;
    buffer_desc.MipLevels = 1;
    buffer_desc.SampleDesc.Count = 1;
    buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ID3D12Resource* readback = nullptr;
    ID3D12Resource* motion_readback = nullptr;
    hr = d3d->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE,
        &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&readback));
    ID3D12CommandQueue* queue = nullptr;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12Fence* done = nullptr;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    if (SUCCEEDED(hr)) hr = d3d->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (SUCCEEDED(hr)) hr = d3d->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (SUCCEEDED(hr)) hr = d3d->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list));
    if (SUCCEEDED(hr)) hr = d3d->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done));
    if (FAILED(hr)) return 8;
    hr = queue->Wait(cuda_fence, cuda_done);
    if (FAILED(hr)) return 9;
    void* raw_motion = nullptr;
    const int convert_result = nvof_d3d12_record_motion(bridge, list, &raw_motion);
    if (convert_result) {
        std::printf("motion_convert_failed=%d: %s\n", convert_result,
                    nvof_d3d12_last_error());
        return 16;
    }
    auto* motion = static_cast<ID3D12Resource*>(raw_motion);
    if (motion->GetDesc().Format != DXGI_FORMAT_R16G16_FLOAT) return 17;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT rows = 0;
    UINT64 row_bytes = 0, motion_bytes = 0;
    const D3D12_RESOURCE_DESC motion_desc = motion->GetDesc();
    d3d->GetCopyableFootprints(&motion_desc, 0, 1, 0, &layout,
                              &rows, &row_bytes, &motion_bytes);
    buffer_desc.Width = motion_bytes;
    hr = d3d->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE,
        &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(&motion_readback));
    if (FAILED(hr)) return 18;
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = shared;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    list->CopyBufferRegion(readback, 0, shared, 0, bytes);
    D3D12_RESOURCE_BARRIER motion_barrier{};
    motion_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    motion_barrier.Transition.pResource = motion;
    motion_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    motion_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    motion_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &motion_barrier);
    D3D12_TEXTURE_COPY_LOCATION source_location{};
    source_location.pResource = motion;
    source_location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION destination_location{};
    destination_location.pResource = motion_readback;
    destination_location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    destination_location.PlacedFootprint = layout;
    list->CopyTextureRegion(&destination_location, 0, 0, 0,
                            &source_location, nullptr);
    hr = list->Close();
    if (FAILED(hr)) return 10;
    ID3D12CommandList* commands[] = {list};
    queue->ExecuteCommandLists(1, commands);
    hr = queue->Signal(done, 1);
    if (FAILED(hr)) return 11;
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event || FAILED(done->SetEventOnCompletion(1, event)) ||
        WaitForSingleObject(event, 15000) != WAIT_OBJECT_0) return 12;
    CloseHandle(event);

    std::vector<float> reference(size_t(w) * h * 2);
    if (nvof_bridge_execute_bgr_sequential(host_session,
            current.data(), w * 3, previous.data(), w * 3,
            reference.data(), w * 8, 1, &timing)) return 13;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped))) return 14;
    const bool identical = std::memcmp(mapped, reference.data(), bytes) == 0;
    readback->Unmap(0, nullptr);
    void* mapped_motion = nullptr;
    if (FAILED(motion_readback->Map(0, nullptr, &mapped_motion))) return 19;
    bool motion_matches = true;
    for (uint32_t y = 0; y < h && motion_matches; ++y) {
        const auto* row = reinterpret_cast<const uint16_t*>(
            static_cast<const uint8_t*>(mapped_motion) + y * layout.Footprint.RowPitch);
        for (uint32_t x = 0; x < w * 2; ++x) {
            const float expected = reference[(size_t(y) * w * 2) + x];
            const float actual = from_half(row[x]);
            if (!std::isfinite(actual) ||
                std::fabs(actual - expected) > 0.02f + std::fabs(expected) * 0.001f) {
                motion_matches = false;
                break;
            }
        }
    }
    motion_readback->Unmap(0, nullptr);
    std::printf("cuda_to_d3d12_bytes=%zu nvof_d2h_copies=%llu float32_parity=%s fp16_motion=%s\n",
        bytes, static_cast<unsigned long long>(counters.d2h_copy_count),
        identical ? "identical" : "different",
        motion_matches ? "matched" : "different");
    done->Release();
    list->Release();
    allocator->Release();
    queue->Release();
    readback->Release();
    motion_readback->Release();
    nvof_d3d12_destroy(bridge);
    d3d->Release();
    nvof_bridge_destroy(device_session);
    nvof_bridge_destroy(host_session);
    return identical && motion_matches ? 0 : 15;
}
