#include "gpu_low_residual_bridge.h"
#include "nvof_d3d12_interop.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdio>
#include <new>

using Microsoft::WRL::ComPtr;
namespace {
thread_local char last_error[512] = "";
void fail(const char* where, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", where,
                  static_cast<unsigned long>(code));
}
uint32_t aligned_pitch(uint32_t width) { return (width * 4u + 255u) & ~255u; }
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}
constexpr char shader[] = R"hlsl(
ByteAddressBuffer WorkingColor : register(t0);
Texture2D<float4> DlssColor : register(t1);
RWTexture2D<float4> SignedResidual : register(u0);
cbuffer Frame : register(b0) { uint Width, Height, Pitch; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    uint packed = WorkingColor.Load(id.y * Pitch + id.x * 4u);
    float3 original = float3(packed & 255u, (packed >> 8) & 255u,
                             (packed >> 16) & 255u);
    float3 denoised = round(DlssColor.Load(int3(id.xy, 0)).rgb * 255.0);
    SignedResidual[id.xy] = float4(denoised - original, 0.0);
}
)hlsl";
}

struct GpuLowResidualBridge {
    uint32_t width = 0, height = 0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    CudaD3D12ColorBridge* color = nullptr;
    void* cuda_context = nullptr;
    HANDLE event = nullptr;
    uint64_t fence_value = 0;
    uint64_t process_count = 0, working_device_copy_bytes = 0;
    bool failed = false;
};

namespace {
bool wait_gpu(GpuLowResidualBridge* bridge) {
    if (bridge->fence->GetCompletedValue() >= bridge->fence_value) return true;
    HRESULT hr = bridge->fence->SetEventOnCompletion(bridge->fence_value,
                                                     bridge->event);
    if (FAILED(hr)) { fail("low residual fence event", hr); return false; }
    if (WaitForSingleObject(bridge->event, 15000) != WAIT_OBJECT_0) {
        fail("low residual fence timeout", ERROR_TIMEOUT);
        return false;
    }
    return true;
}
HRESULT open_resources(GpuLowResidualBridge* bridge) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = bridge->width;
    desc.Height = bridge->height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return bridge->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
        IID_PPV_ARGS(bridge->output.GetAddressOf()));
}
HRESULT open_shader(GpuLowResidualBridge* bridge) {
    ComPtr<ID3DBlob> code, errors;
    HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, "low_residual",
        nullptr, nullptr, "main", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
        0, code.GetAddressOf(), errors.GetAddressOf());
    if (FAILED(hr)) {
        if (errors) std::snprintf(last_error, sizeof(last_error),
            "D3DCompile low residual: %s",
            static_cast<const char*>(errors->GetBufferPointer()));
        return hr;
    }
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 2;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = 3;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_code;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                    root_code.GetAddressOf(), nullptr);
    if (SUCCEEDED(hr)) hr = bridge->device->CreateRootSignature(0,
        root_code->GetBufferPointer(), root_code->GetBufferSize(),
        IID_PPV_ARGS(bridge->root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = bridge->root.Get();
    pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    hr = bridge->device->CreateComputePipelineState(&pso,
        IID_PPV_ARGS(bridge->pipeline.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_DESCRIPTOR_HEAP_DESC desc{};
    desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    desc.NumDescriptors = 3;
    desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = bridge->device->CreateDescriptorHeap(&desc,
        IID_PPV_ARGS(bridge->descriptors.GetAddressOf()));
    if (FAILED(hr)) return hr;
    auto cpu = bridge->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += 2 * bridge->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    bridge->device->CreateUnorderedAccessView(bridge->output.Get(), nullptr,
                                               nullptr, cpu);
    return S_OK;
}
}

extern "C" GPU_LOW_RESIDUAL_API uint32_t gpu_low_residual_abi_version(void) {
    return 1;
}
extern "C" GPU_LOW_RESIDUAL_API const char* gpu_low_residual_last_error(void) {
    return last_error;
}
extern "C" GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_get_transfer_counters(
    GpuLowResidualBridge* bridge, GpuLowResidualCounters* counters) {
    if (!bridge || !counters || counters->struct_size != sizeof(*counters))
        return 1;
    *counters = GpuLowResidualCounters{sizeof(*counters),
        bridge->process_count, bridge->working_device_copy_bytes, 0};
    return 0;
}
extern "C" GPU_LOW_RESIDUAL_API void gpu_low_residual_close(
    GpuLowResidualBridge* bridge) {
    if (!bridge) return;
    if (bridge->color) cuda_d3d12_color_destroy(bridge->color);
    if (bridge->event) CloseHandle(bridge->event);
    delete bridge;
}
extern "C" GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_open(
    void* device, uint32_t width, uint32_t height, GpuLowResidualBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    if (!device || !out || width < 1 || height < 1 || width > 16384 ||
        height > 16384) { fail("invalid low residual open", 0); return 1; }
    auto* bridge = new (std::nothrow) GpuLowResidualBridge;
    if (!bridge) { fail("allocate low residual bridge", 0); return 2; }
    bridge->device = static_cast<ID3D12Device*>(device);
    bridge->width = width;
    bridge->height = height;
    D3D12_COMMAND_QUEUE_DESC queue{};
    queue.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = bridge->device->CreateCommandQueue(&queue,
        IID_PPV_ARGS(bridge->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = bridge->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,
        IID_PPV_ARGS(bridge->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = bridge->device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_DIRECT, bridge->allocator.Get(), nullptr,
        IID_PPV_ARGS(bridge->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = bridge->list->Close();
    if (SUCCEEDED(hr)) hr = bridge->device->CreateFence(0,
        D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(bridge->fence.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = open_resources(bridge);
    if (SUCCEEDED(hr)) hr = open_shader(bridge);
    if (FAILED(hr)) {
        if (!last_error[0]) fail("initialize low residual bridge", hr);
        gpu_low_residual_close(bridge);
        return 3;
    }
    bridge->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!bridge->event) {
        fail("low residual CreateEvent", GetLastError());
        gpu_low_residual_close(bridge);
        return 4;
    }
    *out = bridge;
    return 0;
}

extern "C" GPU_LOW_RESIDUAL_API int32_t gpu_low_residual_process(
    GpuLowResidualBridge* bridge, void* dlss_raw, void* dlss_fence,
    uint64_t dlss_value, void* cuda_context, uint64_t working_ptr,
    size_t working_pitch, void** out_texture, void** out_fence,
    uint64_t* out_fence_value) {
    last_error[0] = 0;
    if (out_texture) *out_texture = nullptr;
    if (out_fence) *out_fence = nullptr;
    if (out_fence_value) *out_fence_value = 0;
    if (!bridge || bridge->failed || !dlss_raw || !dlss_fence ||
        !cuda_context || !working_ptr ||
        working_pitch < size_t(bridge->width) * 4 || !out_texture ||
        !out_fence || !out_fence_value) {
        fail("invalid low residual process", 0); return 1;
    }
    auto* dlss = static_cast<ID3D12Resource*>(dlss_raw);
    ComPtr<ID3D12Device> owner;
    HRESULT hr = dlss->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
    const auto desc = dlss->GetDesc();
    if (FAILED(hr) || owner.Get() != bridge->device.Get() ||
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Width != bridge->width || desc.Height != bridge->height ||
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        fail("low residual DLSS resource mismatch", FAILED(hr) ? hr : 0);
        return 2;
    }
    if (!bridge->color) {
        if (cuda_d3d12_color_create(bridge->device.Get(), cuda_context,
            bridge->width, bridge->height, aligned_pitch(bridge->width),
            &bridge->color)) {
            std::snprintf(last_error, sizeof(last_error), "low residual color: %s",
                          nvof_d3d12_last_error());
            return 3;
        }
        bridge->cuda_context = cuda_context;
    }
    if (bridge->cuda_context != cuda_context) {
        fail("low residual CUDA context changed", 0); return 4;
    }
    uint64_t color_value = 0;
    if (cuda_d3d12_color_copy(bridge->color, working_ptr, working_pitch,
                             &color_value)) {
        std::snprintf(last_error, sizeof(last_error), "low residual copy: %s",
                      nvof_d3d12_last_error());
        bridge->failed = true; return 5;
    }
    void *color_raw = nullptr, *color_fence = nullptr;
    cuda_d3d12_color_get_resources(bridge->color, &color_raw, &color_fence);
    auto* color = static_cast<ID3D12Resource*>(color_raw);
    hr = bridge->queue->Wait(static_cast<ID3D12Fence*>(dlss_fence), dlss_value);
    if (SUCCEEDED(hr)) hr = bridge->queue->Wait(
        static_cast<ID3D12Fence*>(color_fence), color_value);
    if (FAILED(hr)) { fail("low residual queue wait", hr);
        bridge->failed = true; return 6; }
    auto cpu = bridge->descriptors->GetCPUDescriptorHandleForHeapStart();
    const uint32_t step = bridge->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC color_view{};
    color_view.Format = DXGI_FORMAT_R32_TYPELESS;
    color_view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    color_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    color_view.Buffer.NumElements = aligned_pitch(bridge->width) * bridge->height / 4;
    color_view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    bridge->device->CreateShaderResourceView(color, &color_view, cpu);
    cpu.ptr += step;
    D3D12_SHADER_RESOURCE_VIEW_DESC dlss_view{};
    dlss_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dlss_view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    dlss_view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    dlss_view.Texture2D.MipLevels = 1;
    bridge->device->CreateShaderResourceView(dlss, &dlss_view, cpu);
    hr = bridge->allocator->Reset();
    if (SUCCEEDED(hr)) hr = bridge->list->Reset(bridge->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset low residual list", hr);
        bridge->failed = true; return 7; }
    auto* list = bridge->list.Get();
    transition(list, color, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, dlss, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, bridge->output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {bridge->descriptors.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(bridge->root.Get());
    list->SetPipelineState(bridge->pipeline.Get());
    auto gpu = bridge->descriptors->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += 2 * step;
    list->SetComputeRootDescriptorTable(1, gpu);
    const uint32_t values[] = {bridge->width, bridge->height,
                               aligned_pitch(bridge->width)};
    list->SetComputeRoot32BitConstants(2, 3, values, 0);
    list->Dispatch((bridge->width + 7) / 8, (bridge->height + 7) / 8, 1);
    transition(list, bridge->output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list, dlss, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list, color, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    hr = list->Close();
    if (FAILED(hr)) { fail("close low residual list", hr);
        bridge->failed = true; return 8; }
    ID3D12CommandList* commands[] = {list};
    bridge->queue->ExecuteCommandLists(1, commands);
    hr = bridge->queue->Signal(bridge->fence.Get(), ++bridge->fence_value);
    if (FAILED(hr) || !wait_gpu(bridge)) {
        if (FAILED(hr)) fail("signal low residual fence", hr);
        bridge->failed = true; return 9;
    }
    *out_texture = bridge->output.Get();
    *out_fence = bridge->fence.Get();
    *out_fence_value = bridge->fence_value;
    bridge->process_count++;
    bridge->working_device_copy_bytes +=
        uint64_t(bridge->width) * bridge->height * 4;
    return 0;
}
