#define GPU_LANCZOS_EXPORTS
#include "gpu_lanczos_bridge.h"
#include "nvof_d3d12_interop.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <new>

using Microsoft::WRL::ComPtr;
namespace {
thread_local char last_error[384] = "";
void fail(const char* step, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", step,
                  static_cast<unsigned long>(code));
}
uint32_t row_pitch(uint32_t width) { return (width * 4 + 255u) & ~255u; }
static constexpr char shader[] = R"hlsl(
// These are the exact float32 bits from the CPU Lanczos-2 axis plan at 50%.
float wt(uint i) {
    if (i == 0 || i == 7) return asfloat(0xbc113782);
    if (i == 1 || i == 6) return asfloat(0xbd2bc950);
    if (i == 2 || i == 5) return asfloat(0x3dee979a);
    return asfloat(0x3ede5d00);
}
int bound(int value, int maximum) { return min(max(value, 0), maximum); }
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, LowWidth, LowHeight;
    uint FullPitch, LowPitch, Padding0, Padding1;
};
#ifdef VERTICAL
ByteAddressBuffer Source : register(t0);
RWStructuredBuffer<float4> Destination : register(u0);
[numthreads(16, 8, 1)]
void vertical(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= LowHeight) return;
    int first = int(id.y * 2) - 3;
    precise float4 acc = float4(0, 0, 0, 0);
    [unroll] for (uint t = 0; t < 8; ++t) {
        int sy = bound(first + int(t), int(FullHeight) - 1);
        uint packed = Source.Load(uint(sy) * FullPitch + id.x * 4);
        float4 color = float4(packed & 255u, (packed >> 8) & 255u,
            (packed >> 16) & 255u, 0);
        precise float4 term = wt(t) * color;
        acc = acc + term;
    }
    Destination[id.y * FullWidth + id.x] = acc;
}
#else
StructuredBuffer<float4> Source : register(t0);
RWByteAddressBuffer Destination : register(u0);
[numthreads(16, 8, 1)]
void horizontal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= LowWidth || id.y >= LowHeight) return;
    int first = int(id.x * 2) - 3;
    precise float4 acc = float4(0, 0, 0, 0);
    [unroll] for (uint t = 0; t < 8; ++t) {
        int sx = bound(first + int(t), int(FullWidth) - 1);
        precise float4 term = wt(t) * Source[id.y * FullWidth + uint(sx)];
        acc = acc + term;
    }
    uint3 rgb = uint3(round(clamp(acc.rgb, 0.0, 255.0)));
    uint packed = rgb.r | (rgb.g << 8) | (rgb.b << 16) | 0xff000000u;
    Destination.Store(id.y * LowPitch + id.x * 4, packed);
}
#endif
)hlsl";
// The scaled pair takes its taps from the caller's AxisPlan instead of from the
// constants above, so any downscale ratio can be served. The window is always
// walked to GPU_LANCZOS_MAX_TAPS - the same number this #define must carry, which
// the caller checks its plan against before it opens - and the plan pads every
// shorter row with weights of 0.0f at a valid index, which is exactly what the
// CPU reference's _apply_axis_plan sums over, so the two add the same terms in
// the same order rather than the kernel quietly dropping the padding.
static constexpr char scaled_shader[] = R"hlsl(
#define TAPS 24u
struct TapRow { float W[TAPS]; int I[TAPS]; };
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, LowWidth, LowHeight;
    uint FullPitch, LowPitch, Padding0, Padding1;
};
int bound(int value, int maximum) { return min(max(value, 0), maximum); }
#ifdef VERTICAL
ByteAddressBuffer Source : register(t0);
StructuredBuffer<TapRow> Plan : register(t1);
RWStructuredBuffer<float4> Destination : register(u0);
[numthreads(16, 8, 1)]
void vertical(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= LowHeight) return;
    TapRow row = Plan[id.y];
    precise float4 acc = float4(0, 0, 0, 0);
    [unroll] for (uint t = 0; t < TAPS; ++t) {
        int sy = bound(row.I[t], int(FullHeight) - 1);
        uint packed = Source.Load(uint(sy) * FullPitch + id.x * 4);
        float4 color = float4(packed & 255u, (packed >> 8) & 255u,
            (packed >> 16) & 255u, 0);
        precise float4 term = row.W[t] * color;
        acc = acc + term;
    }
    Destination[id.y * FullWidth + id.x] = acc;
}
#else
StructuredBuffer<float4> Source : register(t0);
StructuredBuffer<TapRow> Plan : register(t1);
RWByteAddressBuffer Destination : register(u0);
[numthreads(16, 8, 1)]
void horizontal(uint3 id : SV_DispatchThreadID) {
    if (id.x >= LowWidth || id.y >= LowHeight) return;
    TapRow row = Plan[id.x];
    precise float4 acc = float4(0, 0, 0, 0);
    [unroll] for (uint t = 0; t < TAPS; ++t) {
        int sx = bound(row.I[t], int(FullWidth) - 1);
        precise float4 term = row.W[t] * Source[id.y * FullWidth + uint(sx)];
        acc = acc + term;
    }
    uint3 rgb = uint3(round(clamp(acc.rgb, 0.0, 255.0)));
    uint packed = rgb.r | (rgb.g << 8) | (rgb.b << 16) | 0xff000000u;
    Destination.Store(id.y * LowPitch + id.x * 4, packed);
}
#endif
)hlsl";
void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* r,
    D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = r;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}
}

struct GpuLanczosBridge {
    uint32_t full_w = 0, full_h = 0, low_w = 0, low_h = 0;
    bool scaled = false;
    uint32_t tap_rows_y = 0, tap_rows_x = 0, tap_stride = 0;
    void* cuda_context = nullptr;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t fence_value = 0;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> vertical;
    ComPtr<ID3D12PipelineState> horizontal;
    ComPtr<ID3D12RootSignature> scaled_root;
    ComPtr<ID3D12PipelineState> scaled_vertical;
    ComPtr<ID3D12PipelineState> scaled_horizontal;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12Resource> scratch;
    ComPtr<ID3D12Resource> taps;
    CudaD3D12ColorBridge* source = nullptr;
    CudaD3D12ColorBridge* output = nullptr;
    bool failed = false;
    GpuLanczosTransferCounters counters{sizeof(GpuLanczosTransferCounters)};
};

namespace {
HRESULT make_shader(GpuLanczosBridge* b) {
    D3D12_DESCRIPTOR_RANGE srv{}, uav{};
    srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv.NumDescriptors = 1;
    uav.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 8;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(), nullptr);
    if (SUCCEEDED(hr)) hr = b->device->CreateRootSignature(0,
        blob->GetBufferPointer(), blob->GetBufferSize(),
        IID_PPV_ARGS(b->root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    for (int i = 0; i < 2; ++i) {
        D3D_SHADER_MACRO macros[] = {{"VERTICAL", "1"}, {nullptr, nullptr}};
        ComPtr<ID3DBlob> code, errors;
        hr = D3DCompile(shader, sizeof(shader) - 1, "gpu_lanczos",
            i == 0 ? macros : nullptr, nullptr,
            i == 0 ? "vertical" : "horizontal", "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
            0, code.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(hr)) {
            if (errors) std::snprintf(last_error, sizeof(last_error),
                "D3DCompile Lanczos: %s",
                static_cast<const char*>(errors->GetBufferPointer()));
            return hr;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = b->root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = b->device->CreateComputePipelineState(&pso,
            IID_PPV_ARGS(i == 0 ? b->vertical.GetAddressOf() :
                                b->horizontal.GetAddressOf()));
        if (FAILED(hr)) return hr;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = 4;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    return b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->descriptors.GetAddressOf()));
}
HRESULT make_scratch(GpuLanczosBridge* b) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = uint64_t(b->full_w) * b->low_h * 16;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(b->scratch.GetAddressOf()));
}
void descriptors(GpuLanczosBridge* b, ID3D12Resource* source,
                 ID3D12Resource* output) {
    const uint32_t step = b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.NumElements = row_pitch(b->full_w) * b->full_h / 4;
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(source, &raw, cpu);
    cpu.ptr += step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC structured_uav{};
    structured_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    structured_uav.Buffer.NumElements = b->full_w * b->low_h;
    structured_uav.Buffer.StructureByteStride = 16;
    b->device->CreateUnorderedAccessView(b->scratch.Get(), nullptr,
        &structured_uav, cpu);
    cpu.ptr += step;
    D3D12_SHADER_RESOURCE_VIEW_DESC structured_srv{};
    structured_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    structured_srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    structured_srv.Buffer.NumElements = b->full_w * b->low_h;
    structured_srv.Buffer.StructureByteStride = 16;
    b->device->CreateShaderResourceView(b->scratch.Get(), &structured_srv, cpu);
    cpu.ptr += step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw_uav{};
    raw_uav.Format = DXGI_FORMAT_R32_TYPELESS;
    raw_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    raw_uav.Buffer.NumElements = row_pitch(b->low_w) * b->low_h / 4;
    raw_uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(output, nullptr, &raw_uav, cpu);
}
D3D12_GPU_DESCRIPTOR_HANDLE offset(D3D12_GPU_DESCRIPTOR_HANDLE start,
                                   uint32_t step, uint32_t slot) {
    start.ptr += uint64_t(slot) * step;
    return start;
}
bool wait_gpu(GpuLanczosBridge* b) {
    HRESULT hr = b->fence->SetEventOnCompletion(b->fence_value, b->event);
    if (FAILED(hr)) { fail("SetEventOnCompletion", hr); return false; }
    if (WaitForSingleObject(b->event, 15000) != WAIT_OBJECT_0) {
        fail("GPU Lanczos fence timeout", ERROR_TIMEOUT); return false;
    }
    return true;
}
HRESULT make_scaled_shader(GpuLanczosBridge* b) {
    D3D12_DESCRIPTOR_RANGE srv{}, uav{};
    srv.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    srv.NumDescriptors = 2;
    uav.RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    uav.NumDescriptors = 1;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &srv;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &uav;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 8;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 3;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    HRESULT hr = D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(), nullptr);
    if (SUCCEEDED(hr)) hr = b->device->CreateRootSignature(0,
        blob->GetBufferPointer(), blob->GetBufferSize(),
        IID_PPV_ARGS(b->scaled_root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    for (int i = 0; i < 2; ++i) {
        D3D_SHADER_MACRO macros[] = {{"VERTICAL", "1"}, {nullptr, nullptr}};
        ComPtr<ID3DBlob> code, errors;
        hr = D3DCompile(scaled_shader, sizeof(scaled_shader) - 1,
            "gpu_lanczos_scaled", i == 0 ? macros : nullptr, nullptr,
            i == 0 ? "vertical" : "horizontal", "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
            0, code.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(hr)) {
            if (errors) std::snprintf(last_error, sizeof(last_error),
                "D3DCompile scaled Lanczos: %s",
                static_cast<const char*>(errors->GetBufferPointer()));
            return hr;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = b->scaled_root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = b->device->CreateComputePipelineState(&pso,
            IID_PPV_ARGS(i == 0 ? b->scaled_vertical.GetAddressOf() :
                                  b->scaled_horizontal.GetAddressOf()));
        if (FAILED(hr)) return hr;
    }
    // Six slots, two contiguous SRV pairs so each pass can bind {source, plan}
    // from one table: 0 source, 1 vertical plan rows, 2 scratch UAV,
    // 3 scratch, 4 horizontal plan rows, 5 output UAV.
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = 6;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    return b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->descriptors.GetAddressOf()));
}
HRESULT create_tap_buffer(GpuLanczosBridge* b) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = uint64_t(b->tap_rows_y + b->tap_rows_x) * b->tap_stride;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(b->taps.GetAddressOf()));
}
void plan_srv(GpuLanczosBridge* b, uint32_t first, uint32_t rows,
              D3D12_CPU_DESCRIPTOR_HANDLE cpu) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.Buffer.FirstElement = first;
    srv.Buffer.NumElements = rows;
    srv.Buffer.StructureByteStride = b->tap_stride;
    b->device->CreateShaderResourceView(b->taps.Get(), &srv, cpu);
}
void descriptors_scaled(GpuLanczosBridge* b, ID3D12Resource* source,
                        ID3D12Resource* output) {
    const uint32_t step = b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const auto start = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    const auto slot = [&](uint32_t n) {
        D3D12_CPU_DESCRIPTOR_HANDLE h = start; h.ptr += n * step; return h; };
    D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
    raw.Format = DXGI_FORMAT_R32_TYPELESS;
    raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    raw.Buffer.NumElements = row_pitch(b->full_w) * b->full_h / 4;
    raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(source, &raw, slot(0));
    plan_srv(b, 0, b->tap_rows_y, slot(1));
    D3D12_UNORDERED_ACCESS_VIEW_DESC scratch_uav{};
    scratch_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    scratch_uav.Buffer.NumElements = b->full_w * b->low_h;
    scratch_uav.Buffer.StructureByteStride = 16;
    b->device->CreateUnorderedAccessView(b->scratch.Get(), nullptr,
        &scratch_uav, slot(2));
    D3D12_SHADER_RESOURCE_VIEW_DESC structured_srv{};
    structured_srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    structured_srv.Shader4ComponentMapping =
        D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    structured_srv.Buffer.NumElements = b->full_w * b->low_h;
    structured_srv.Buffer.StructureByteStride = 16;
    b->device->CreateShaderResourceView(b->scratch.Get(), &structured_srv,
        slot(3));
    plan_srv(b, b->tap_rows_y, b->tap_rows_x, slot(4));
    D3D12_UNORDERED_ACCESS_VIEW_DESC raw_uav{};
    raw_uav.Format = DXGI_FORMAT_R32_TYPELESS;
    raw_uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    raw_uav.Buffer.NumElements = row_pitch(b->low_w) * b->low_h / 4;
    raw_uav.Buffer.Flags = D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(output, nullptr, &raw_uav, slot(5));
}
/* One upload for the life of the bridge: the extents, and so the taps, are
 * fixed for an export job. Filter coefficients, not a frame - this is why no
 * GpuLanczosTransferCounters field moves here. */
HRESULT upload_taps(GpuLanczosBridge* b, const GpuLanczosTapTables* plan) {
    const uint64_t bytes = uint64_t(b->tap_rows_y + b->tap_rows_x) * b->tap_stride;
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC staging{};
    staging.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    staging.Width = bytes;
    staging.Height = 1;
    staging.DepthOrArraySize = 1;
    staging.MipLevels = 1;
    staging.SampleDesc.Count = 1;
    staging.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> staging_buffer;
    HRESULT hr = b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &staging, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
        IID_PPV_ARGS(staging_buffer.GetAddressOf()));
    if (FAILED(hr)) { fail("create Lanczos plan staging", hr); return hr; }
    uint8_t* mapped = nullptr;
    hr = staging_buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map Lanczos plan staging", hr); return hr; }
    const uint32_t taps = plan->taps;
    for (int axis = 0; axis < 2; ++axis) {
        const uint32_t first_row = axis ? b->tap_rows_y : 0;
        const uint32_t rows = axis ? b->tap_rows_x : b->tap_rows_y;
        const int32_t* indices = axis ? plan->horizontal_indices
            : plan->vertical_indices;
        const float* weights = axis ? plan->horizontal_weights
            : plan->vertical_weights;
        for (uint32_t row = 0; row < rows; ++row) {
            uint8_t* target = mapped + uint64_t(first_row + row) * b->tap_stride;
            for (uint32_t t = 0; t < GPU_LANCZOS_MAX_TAPS; ++t) {
                const bool real = t < taps;
                const uint64_t source = uint64_t(row) * taps + t;
                const float weight = real ? weights[source] : 0.0f;
                const int32_t index = real ? indices[source] : 0;
                std::memcpy(target + t * 4, &weight, 4);
                std::memcpy(target + GPU_LANCZOS_MAX_TAPS * 4 + t * 4,
                    &index, 4);
            }
        }
    }
    staging_buffer->Unmap(0, nullptr);
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset plan upload list", hr); return hr; }
    b->list->CopyBufferRegion(b->taps.Get(), 0, staging_buffer.Get(), 0, bytes);
    transition(b->list.Get(), b->taps.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    hr = b->list->Close();
    if (FAILED(hr)) { fail("close plan upload list", hr); return hr; }
    ID3D12CommandList* commands[] = {b->list.Get()};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal plan upload fence", hr); return hr; }
    if (!wait_gpu(b)) return HRESULT_FROM_WIN32(ERROR_TIMEOUT);
    return S_OK;
}
int32_t check_plan(const GpuLanczosTapTables* plan, uint32_t full_w,
                   uint32_t full_h, uint32_t low_w, uint32_t low_h) {
    if (!plan || plan->struct_size != sizeof(GpuLanczosTapTables)) {
        fail("Lanczos plan struct size does not match this header", 0); return 1; }
    if (plan->taps < 1 || plan->taps > GPU_LANCZOS_MAX_TAPS) {
        fail("Lanczos plan tap window is outside 1..GPU_LANCZOS_MAX_TAPS", 0);
        return 1; }
    if (plan->vertical_src_len != full_h || plan->vertical_dst_len != low_h ||
        plan->horizontal_src_len != full_w || plan->horizontal_dst_len != low_w) {
        fail("Lanczos plan axes disagree with the opened extents", 0); return 1; }
    if (!plan->vertical_indices || !plan->vertical_weights ||
        !plan->horizontal_indices || !plan->horizontal_weights) {
        fail("Lanczos plan has a null table", 0); return 1; }
    const uint32_t axis[2] = {full_h, full_w};
    const uint32_t rows[2] = {low_h, low_w};
    const int32_t* tables[2] = {plan->vertical_indices, plan->horizontal_indices};
    const float* weights[2] = {plan->vertical_weights, plan->horizontal_weights};
    for (int kind = 0; kind < 2; ++kind) {
        for (uint32_t row = 0; row < rows[kind]; ++row) {
            for (uint32_t t = 0; t < plan->taps; ++t) {
                const uint64_t at = uint64_t(row) * plan->taps + t;
                if (tables[kind][at] < 0 ||
                    uint32_t(tables[kind][at]) >= axis[kind]) {
                    std::snprintf(last_error, sizeof(last_error),
                        "Lanczos plan index %d is outside 0..%u at row %u tap %u",
                        tables[kind][at], axis[kind] - 1, row, t);
                    return 1;
                }
                if (!std::isfinite(weights[kind][at])) {
                    std::snprintf(last_error, sizeof(last_error),
                        "Lanczos plan weight is not finite at row %u tap %u",
                        row, t);
                    return 1;
                }
            }
        }
    }
    return 0;
}
/* Both entry points come here; what differs is which kernels and which
 * descriptor layout the bridge binds, and whether a plan is uploaded. */
int32_t bring_up(GpuLanczosBridge* b, void* context,
                 const GpuLanczosTapTables* plan) {
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
        IID_PPV_ARGS(b->device.GetAddressOf()));
    if (FAILED(hr)) { fail("D3D12CreateDevice", hr); return 3; }
    if (cuda_d3d12_color_create(b->device.Get(), context, b->full_w, b->full_h,
            row_pitch(b->full_w), &b->source) ||
        cuda_d3d12_color_create_output(b->device.Get(), context,
            b->low_w, b->low_h, row_pitch(b->low_w), &b->output)) {
        std::snprintf(last_error, sizeof(last_error), "CUDA/D3D12 colors: %s",
            nvof_d3d12_last_error()); return 4;
    }
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = b->device->CreateCommandQueue(&q,
        IID_PPV_ARGS(b->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(b->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_DIRECT, b->allocator.Get(), nullptr,
        IID_PPV_ARGS(b->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->list->Close();
    if (SUCCEEDED(hr)) hr = b->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(b->fence.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = make_scratch(b);
    if (SUCCEEDED(hr)) hr = b->scaled ? make_scaled_shader(b) : make_shader(b);
    if (SUCCEEDED(hr) && b->scaled) hr = create_tap_buffer(b);
    if (FAILED(hr)) { if (!last_error[0]) fail("initialize Lanczos", hr); return 5; }
    b->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!b->event) { fail("CreateEvent", GetLastError()); return 6; }
    if (b->scaled && FAILED(upload_taps(b, plan))) return 7;
    return 0;
}
}

extern "C" GPU_LANCZOS_API uint32_t gpu_lanczos_abi_version(void) {
    return GPU_LANCZOS_ABI_VERSION;
}
extern "C" GPU_LANCZOS_API uint32_t gpu_lanczos_max_taps(void) {
    return GPU_LANCZOS_MAX_TAPS;
}
extern "C" GPU_LANCZOS_API const char* gpu_lanczos_last_error(void) {
    return last_error;
}
extern "C" GPU_LANCZOS_API void gpu_lanczos_close(GpuLanczosBridge* b) {
    if (!b) return;
    if (b->source) cuda_d3d12_color_destroy(b->source);
    if (b->output) cuda_d3d12_color_destroy(b->output);
    if (b->event) CloseHandle(b->event);
    delete b;
}
extern "C" GPU_LANCZOS_API int32_t gpu_lanczos_open(void* context,
    uint32_t width, uint32_t height, GpuLanczosBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    if (!out || !context || width < 2 || height < 2 ||
        (width & 1) || (height & 1) || width > 16384 || height > 16384 ||
        height > UINT32_MAX / width / 16) {
        fail("invalid Lanczos dimensions", 0); return 1;
    }
    auto* b = new (std::nothrow) GpuLanczosBridge;
    if (!b) { fail("allocate Lanczos bridge", 0); return 2; }
    b->full_w = width; b->full_h = height;
    b->low_w = width / 2; b->low_h = height / 2;
    b->cuda_context = context;
    const int32_t code = bring_up(b, context, nullptr);
    if (code) { gpu_lanczos_close(b); return code; }
    *out = b;
    return 0;
}
extern "C" GPU_LANCZOS_API int32_t gpu_lanczos_open_scaled(void* context,
    uint32_t full_width, uint32_t full_height, uint32_t working_width,
    uint32_t working_height, const GpuLanczosTapTables* plan,
    GpuLanczosBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    if (!out || !context || !plan || full_width < 2 || full_height < 2 ||
        working_width < 2 || working_height < 2 || full_width > 16384 ||
        full_height > 16384 || working_width > full_width ||
        working_height > full_height ||
        working_height > UINT32_MAX / full_width / 16) {
        fail("invalid scaled Lanczos dimensions", 0); return 1;
    }
    if (check_plan(plan, full_width, full_height, working_width,
                   working_height)) {
        return 1;
    }
    auto* b = new (std::nothrow) GpuLanczosBridge;
    if (!b) { fail("allocate Lanczos bridge", 0); return 2; }
    b->scaled = true;
    b->full_w = full_width; b->full_h = full_height;
    b->low_w = working_width; b->low_h = working_height;
    b->tap_rows_y = plan->vertical_dst_len;
    b->tap_rows_x = plan->horizontal_dst_len;
    b->tap_stride = GPU_LANCZOS_MAX_TAPS * 2 * sizeof(uint32_t);
    b->cuda_context = context;
    const int32_t code = bring_up(b, context, plan);
    if (code) { gpu_lanczos_close(b); return code; }
    *out = b;
    return 0;
}
extern "C" GPU_LANCZOS_API int32_t gpu_lanczos_process(GpuLanczosBridge* b,
    uint64_t full_ptr, size_t full_pitch, uint64_t* out_ptr,
    uint32_t* out_pitch) {
    last_error[0] = 0;
    if (!b || b->failed || !full_ptr || full_pitch < size_t(b->full_w) * 4 ||
        !out_ptr || !out_pitch) { fail("invalid Lanczos frame", 0); return 1; }
    uint64_t ready = 0;
    if (cuda_d3d12_color_copy(b->source, full_ptr, full_pitch, &ready)) {
        std::snprintf(last_error, sizeof(last_error), "source color copy: %s",
            nvof_d3d12_last_error()); b->failed = true; return 2;
    }
    void *source_raw = nullptr, *output_raw = nullptr;
    void *source_fence = nullptr, *output_fence = nullptr;
    cuda_d3d12_color_get_resources(b->source, &source_raw, &source_fence);
    cuda_d3d12_color_get_resources(b->output, &output_raw, &output_fence);
    auto* source = static_cast<ID3D12Resource*>(source_raw);
    auto* output = static_cast<ID3D12Resource*>(output_raw);
    HRESULT hr = b->queue->Wait(static_cast<ID3D12Fence*>(source_fence), ready);
    if (FAILED(hr)) { fail("queue wait CUDA source", hr);
        b->failed = true; return 3; }
    if (b->scaled) descriptors_scaled(b, source, output);
    else descriptors(b, source, output);
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset Lanczos list", hr);
        b->failed = true; return 4; }
    auto* list = b->list.Get();
    transition(list, source, D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ID3D12DescriptorHeap* heaps[] = {b->descriptors.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(b->scaled ? b->scaled_root.Get() : b->root.Get());
    struct Constants { uint32_t full_w, full_h, low_w, low_h;
        uint32_t full_pitch, low_pitch, zero0, zero1;
    } constants{b->full_w, b->full_h, b->low_w, b->low_h,
        row_pitch(b->full_w), row_pitch(b->low_w), 0, 0};
    list->SetComputeRoot32BitConstants(2, 8, &constants, 0);
    auto gpu = b->descriptors->GetGPUDescriptorHandleForHeapStart();
    const uint32_t step = b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    // The two layouts differ only in where each pass finds its table: the exact
    // 2x kernels need one SRV per pass, the plan kernels need their source plus
    // their tap rows, so the scratch and the output move up the heap by two.
    uint32_t vertical_uav = 1, horizontal_srv = 2, horizontal_uav = 3;
    if (b->scaled) { vertical_uav = 2; horizontal_srv = 3; horizontal_uav = 5; }
    list->SetPipelineState(b->scaled ? b->scaled_vertical.Get()
                                     : b->vertical.Get());
    list->SetComputeRootDescriptorTable(0, offset(gpu, step, 0));
    list->SetComputeRootDescriptorTable(1, offset(gpu, step, vertical_uav));
    list->Dispatch((b->full_w + 15) / 16, (b->low_h + 7) / 8, 1);
    transition(list, b->scratch.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(list, output, D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    list->SetPipelineState(b->scaled ? b->scaled_horizontal.Get()
                                     : b->horizontal.Get());
    list->SetComputeRootDescriptorTable(0, offset(gpu, step, horizontal_srv));
    list->SetComputeRootDescriptorTable(1, offset(gpu, step, horizontal_uav));
    list->Dispatch((b->low_w + 15) / 16, (b->low_h + 7) / 8, 1);
    transition(list, output, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON);
    transition(list, b->scratch.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list, source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_COMMON);
    hr = list->Close();
    if (FAILED(hr)) { fail("close Lanczos list", hr);
        b->failed = true; return 5; }
    ID3D12CommandList* commands[] = {list};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr) || !wait_gpu(b)) {
        if (FAILED(hr)) fail("signal Lanczos fence", hr);
        b->failed = true; return 6;
    }
    if (cuda_d3d12_color_get_device_pointer(b->output, out_ptr, out_pitch)) {
        fail("mapped CUDA working pointer", 0); b->failed = true; return 7;
    }
    // The four counts below describe the call just made, so they are written
    // after it succeeded and never before. Each is a payload figure the code
    // determines: the source copy moves width*4 bytes of each row, the vertical
    // pass writes every float4 element of the scratch once, and the horizontal
    // pass writes one packed RGBA8 pixel per output thread.
    b->counters.process_count += 1;
    b->counters.source_device_copy_bytes +=
        static_cast<uint64_t>(b->full_w) * b->full_h * 4;
    b->counters.scratch_device_write_bytes +=
        static_cast<uint64_t>(b->full_w) * b->low_h * 16;
    b->counters.output_device_write_bytes +=
        static_cast<uint64_t>(b->low_w) * b->low_h * 4;
    return 0;
}

extern "C" GPU_LANCZOS_API int32_t gpu_lanczos_get_transfer_counters(
    GpuLanczosBridge* b, GpuLanczosTransferCounters* counters) {
    if (!b || !counters ||
        counters->struct_size != sizeof(GpuLanczosTransferCounters)) return -1;
    *counters = b->counters;
    return 0;
}
