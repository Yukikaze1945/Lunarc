#define GPU_RESIDUAL_EXPORTS
#include "gpu_residual_bridge.h"
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
uint32_t aligned_pitch(uint32_t width) { return (width * 4 + 255u) & ~255u; }

// The Catmull-Rom weights at 2x are exact binary fractions. The tap order
// mirrors the native CPU residual: horizontal first, then vertical.
static constexpr char shader[] = R"hlsl(
ByteAddressBuffer FullColor : register(t0);
ByteAddressBuffer WorkingColor : register(t1);
Texture2D<float4> DlssColor : register(t2);
RWTexture2D<float4> Output : register(u0);
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, WorkingWidth, WorkingHeight;
    uint FullPitch, WorkingPitch;
    float Strength;
    uint Padding;
};
int bound(int value, int maximum) { return min(max(value, 0), maximum); }
float weight(uint phase, uint tap) {
    if (phase == 0) {
        if (tap == 0) return -0.0234375;
        if (tap == 1) return 0.2265625;
        if (tap == 2) return 0.8671875;
        return -0.0703125;
    }
    if (tap == 0) return -0.0703125;
    if (tap == 1) return 0.8671875;
    if (tap == 2) return 0.2265625;
    return -0.0234375;
}
float3 rgb(uint packed) {
    return float3(packed & 255u, (packed >> 8) & 255u,
                  (packed >> 16) & 255u);
}
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= FullHeight) return;
    uint px = id.x & 1u, py = id.y & 1u;
    int baseX = int(id.x >> 1) - (px == 0 ? 2 : 1);
    int baseY = int(id.y >> 1) - (py == 0 ? 2 : 1);
    precise float3 vertical = float3(0, 0, 0);
    [unroll] for (uint j = 0; j < 4; ++j) {
        int sy = bound(baseY + int(j), int(WorkingHeight) - 1);
        precise float3 horizontal = float3(0, 0, 0);
        [unroll] for (uint i = 0; i < 4; ++i) {
            int sx = bound(baseX + int(i), int(WorkingWidth) - 1);
            uint source = WorkingColor.Load(uint(sy) * WorkingPitch + uint(sx) * 4);
            float3 original = rgb(source);
            float3 denoised = round(DlssColor.Load(int3(sx, sy, 0)).rgb * 255.0);
            precise float3 delta = denoised - original;
            precise float3 term = weight(px, i) * delta;
            horizontal = horizontal + term;
        }
        precise float3 rowTerm = weight(py, j) * horizontal;
        vertical = vertical + rowTerm;
    }
    uint full = FullColor.Load(id.y * FullPitch + id.x * 4);
    precise float3 residual = abs(Strength - 1.0) > 0.0001 ?
        vertical * Strength : vertical;
    precise float3 composite = rgb(full) + residual;
    Output[id.xy] = float4(round(clamp(composite, 0.0, 255.0)) / 255.0, 1.0);
}
)hlsl";

// The plan-driven upscale for ratios the table above cannot express. Only where
// a tap's weight and index come from changes; the two loops and the composite
// are the exact-2x kernel's, written out again so that the ABI 1 path keeps
// compiling the source it always compiled. A horizontal axis plan built for a
// double selects the same indices and the same weights as the phase trick does.
static constexpr char plan_shader[] = R"hlsl(
#define TAPS 4u
struct TapRow { float W[TAPS]; int I[TAPS]; };
ByteAddressBuffer FullColor : register(t0);
ByteAddressBuffer WorkingColor : register(t1);
Texture2D<float4> DlssColor : register(t2);
StructuredBuffer<TapRow> Plan : register(t3);
RWTexture2D<float4> Output : register(u0);
cbuffer Frame : register(b0) {
    uint FullWidth, FullHeight, WorkingWidth, WorkingHeight;
    uint FullPitch, WorkingPitch;
    float Strength;
    uint Padding;
};
int bound(int value, int maximum) { return min(max(value, 0), maximum); }
float3 rgb(uint packed) {
    return float3(packed & 255u, (packed >> 8) & 255u,
                  (packed >> 16) & 255u);
}
[numthreads(8, 8, 1)]
void main_plan(uint3 id : SV_DispatchThreadID) {
    if (id.x >= FullWidth || id.y >= FullHeight) return;
    const uint column = id.x;
    const uint row = FullWidth + id.y;
    precise float3 vertical = float3(0, 0, 0);
    [unroll] for (uint j = 0; j < TAPS; ++j) {
        int sy = bound(Plan[row].I[j], int(WorkingHeight) - 1);
        precise float3 horizontal = float3(0, 0, 0);
        [unroll] for (uint i = 0; i < TAPS; ++i) {
            int sx = bound(Plan[column].I[i], int(WorkingWidth) - 1);
            uint source = WorkingColor.Load(uint(sy) * WorkingPitch + uint(sx) * 4);
            float3 original = rgb(source);
            float3 denoised = round(DlssColor.Load(int3(sx, sy, 0)).rgb * 255.0);
            precise float3 delta = denoised - original;
            precise float3 term = Plan[column].W[i] * delta;
            horizontal = horizontal + term;
        }
        precise float3 rowTerm = Plan[row].W[j] * horizontal;
        vertical = vertical + rowTerm;
    }
    uint full = FullColor.Load(id.y * FullPitch + id.x * 4);
    precise float3 residual = abs(Strength - 1.0) > 0.0001 ?
        vertical * Strength : vertical;
    precise float3 composite = rgb(full) + residual;
    Output[id.xy] = float4(round(clamp(composite, 0.0, 255.0)) / 255.0, 1.0);
}
)hlsl";

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
             D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}
}

struct GpuResidualBridge {
    uint32_t low_w = 0, low_h = 0, full_w = 0, full_h = 0;
    bool planned = false;
    uint32_t tap_stride = 0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t fence_value = 0;
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Resource> readback;
    ComPtr<ID3D12Resource> taps;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT readback_layout{};
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    CudaD3D12ColorBridge* full_bridge = nullptr;
    CudaD3D12ColorBridge* low_bridge = nullptr;
    void* full_context = nullptr;
    void* low_context = nullptr;
    bool processed = false;
    bool failed = false;
    uint64_t full_device_copy_bytes = 0;
    uint64_t working_device_copy_bytes = 0;
    uint64_t output_readback_bytes = 0;
};

namespace {
bool wait_gpu(GpuResidualBridge* b) {
    if (b->fence->GetCompletedValue() >= b->fence_value) return true;
    HRESULT hr = b->fence->SetEventOnCompletion(b->fence_value, b->event);
    if (FAILED(hr)) { fail("SetEventOnCompletion", hr); return false; }
    if (WaitForSingleObject(b->event, 15000) != WAIT_OBJECT_0) {
        fail("GPU residual fence timeout", ERROR_TIMEOUT); return false;
    }
    return true;
}
HRESULT make_resources(GpuResidualBridge* b) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = b->full_w;
    desc.Height = b->full_h;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr,
        IID_PPV_ARGS(b->output.GetAddressOf()));
    if (FAILED(hr)) return hr;
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    b->device->GetCopyableFootprints(&desc, 0, 1, 0,
        &b->readback_layout, &rows, &row_bytes, &total);
    heap.Type = D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = 1;
    buffer.DepthOrArraySize = 1;
    buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(b->readback.GetAddressOf()));
}
HRESULT make_shader(GpuResidualBridge* b) {
    ComPtr<ID3DBlob> code, errors;
    const bool planned = b->planned;
    HRESULT hr = D3DCompile(planned ? plan_shader : shader,
        planned ? sizeof(plan_shader) - 1 : sizeof(shader) - 1,
        planned ? "gpu_residual_plan" : "gpu_residual", nullptr, nullptr,
        planned ? "main_plan" : "main", "cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
        0, code.GetAddressOf(), errors.GetAddressOf());
    if (FAILED(hr)) {
        if (errors) std::snprintf(last_error, sizeof(last_error),
            "D3DCompile residual: %s",
            static_cast<const char*>(errors->GetBufferPointer()));
        return hr;
    }
    // The plan kernel reads its taps through a fourth SRV; everything else about
    // the binding is the ABI 1 shape.
    const uint32_t srv_count = planned ? 4 : 3;
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = srv_count;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister = 0;
    params[2].Constants.Num32BitValues = 8;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ComPtr<ID3DBlob> root_code;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
        root_code.GetAddressOf(), nullptr);
    if (SUCCEEDED(hr)) hr = b->device->CreateRootSignature(0,
        root_code->GetBufferPointer(), root_code->GetBufferSize(),
        IID_PPV_ARGS(b->root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature = b->root.Get();
    pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
    hr = b->device->CreateComputePipelineState(&pso,
        IID_PPV_ARGS(b->pipeline.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = srv_count + 1;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->descriptors.GetAddressOf()));
    if (FAILED(hr)) return hr;
    const uint32_t step = b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    if (planned) {
        // Slot 3, inside the SRV run the root signature expects. The table is
        // written once, at open, so its view is made once too.
        D3D12_SHADER_RESOURCE_VIEW_DESC table{};
        table.Format = DXGI_FORMAT_UNKNOWN;
        table.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        table.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        table.Buffer.NumElements = b->full_w + b->full_h;
        table.Buffer.StructureByteStride = b->tap_stride;
        auto slot = b->descriptors->GetCPUDescriptorHandleForHeapStart();
        slot.ptr += 3 * step;
        b->device->CreateShaderResourceView(b->taps.Get(), &table, slot);
    }
    cpu.ptr += srv_count * step;
    b->device->CreateUnorderedAccessView(b->output.Get(), nullptr, nullptr, cpu);
    return S_OK;
}
void set_raw_srv(GpuResidualBridge* b, ID3D12Resource* resource,
                 uint32_t bytes, uint32_t slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = bytes / 4;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    auto cpu = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += slot * b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    b->device->CreateShaderResourceView(resource, &srv, cpu);
}

// The plan's axes are validated against the extents the bridge is about to be
// opened with, so a table built for one pair can never be read as if it belonged
// to another: that would sample outside the working frame and be a wrong picture
// rather than an error.
int32_t check_plan(const GpuResidualTapTable* plan, uint32_t full_w,
                   uint32_t full_h, uint32_t low_w, uint32_t low_h) {
    if (!plan || plan->struct_size != sizeof(GpuResidualTapTable)) {
        fail("residual plan struct size does not match this header", 0);
        return 1; }
    if (plan->taps < 1 || plan->taps > GPU_RESIDUAL_MAX_TAPS) {
        fail("residual plan tap window is outside 1..GPU_RESIDUAL_MAX_TAPS", 0);
        return 1; }
    if (plan->vertical_src_len != low_h || plan->vertical_dst_len != full_h ||
        plan->horizontal_src_len != low_w || plan->horizontal_dst_len != full_w) {
        fail("residual plan axes disagree with the opened extents", 0);
        return 1; }
    if (!plan->vertical_indices || !plan->vertical_weights ||
        !plan->horizontal_indices || !plan->horizontal_weights) {
        fail("residual plan has a null table", 0); return 1; }
    const uint32_t axis[2] = {low_h, low_w};
    const uint32_t rows[2] = {full_h, full_w};
    const int32_t* tables[2] = {plan->vertical_indices, plan->horizontal_indices};
    const float* weights[2] = {plan->vertical_weights, plan->horizontal_weights};
    for (int kind = 0; kind < 2; ++kind) {
        for (uint32_t row = 0; row < rows[kind]; ++row) {
            for (uint32_t t = 0; t < plan->taps; ++t) {
                const uint64_t at = uint64_t(row) * plan->taps + t;
                if (tables[kind][at] < 0 ||
                    uint32_t(tables[kind][at]) >= axis[kind]) {
                    std::snprintf(last_error, sizeof(last_error),
                        "residual plan index %d is outside 0..%u at row %u tap %u",
                        tables[kind][at], axis[kind] - 1, row, t);
                    return 1;
                }
                if (!std::isfinite(weights[kind][at])) {
                    std::snprintf(last_error, sizeof(last_error),
                        "residual plan weight is not finite at row %u tap %u",
                        row, t);
                    return 1;
                }
            }
        }
    }
    return 0;
}

// Horizontal rows first, then vertical: the kernel finds a column at element x
// and a row at element FullWidth + y, which is why one SRV covers the whole
// table. Rows are always GPU_RESIDUAL_MAX_TAPS wide; a narrower plan is padded
// with weight 0.0f at index 0, the padding the CPU _apply_axis_plan sums over,
// so widening it changes no sum.
HRESULT create_tap_buffer(GpuResidualBridge* b) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = uint64_t(b->full_w + b->full_h) * b->tap_stride;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return b->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
        IID_PPV_ARGS(b->taps.GetAddressOf()));
}

HRESULT upload_plan(GpuResidualBridge* b, const GpuResidualTapTable* plan) {
    const uint64_t bytes = uint64_t(b->full_w + b->full_h) * b->tap_stride;
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
    if (FAILED(hr)) { fail("create residual plan staging", hr); return hr; }
    uint8_t* mapped = nullptr;
    hr = staging_buffer->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map residual plan staging", hr); return hr; }
    const uint32_t taps = plan->taps;
    for (int axis = 0; axis < 2; ++axis) {
        // axis 0 lays out the horizontal columns at the head of the buffer,
        // which is the order the kernel indexes them in, not the order the
        // struct lists its axes in.
        const uint32_t first_row = axis ? b->full_w : 0;
        const uint32_t rows = axis ? b->full_h : b->full_w;
        const int32_t* indices = axis ? plan->vertical_indices
            : plan->horizontal_indices;
        const float* weights = axis ? plan->vertical_weights
            : plan->horizontal_weights;
        for (uint32_t row = 0; row < rows; ++row) {
            uint8_t* target = mapped + uint64_t(first_row + row) * b->tap_stride;
            for (uint32_t t = 0; t < GPU_RESIDUAL_MAX_TAPS; ++t) {
                const bool real = t < taps;
                const uint64_t source = uint64_t(row) * taps + t;
                const float weight = real ? weights[source] : 0.0f;
                const int32_t index = real ? indices[source] : 0;
                std::memcpy(target + t * 4, &weight, 4);
                std::memcpy(target + GPU_RESIDUAL_MAX_TAPS * 4 + t * 4,
                    &index, 4);
            }
        }
    }
    staging_buffer->Unmap(0, nullptr);
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset residual plan list", hr); return hr; }
    b->list->CopyBufferRegion(b->taps.Get(), 0, staging_buffer.Get(), 0, bytes);
    barrier(b->list.Get(), b->taps.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    hr = b->list->Close();
    if (FAILED(hr)) { fail("close residual plan list", hr); return hr; }
    ID3D12CommandList* commands[] = {b->list.Get()};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal residual plan fence", hr); return hr; }
    if (!wait_gpu(b)) return HRESULT_FROM_WIN32(ERROR_TIMEOUT);
    return S_OK;
}

/* Both entry points come here; what differs is which kernel and which
 * descriptor layout the bridge builds, and whether a table is uploaded. */
int32_t bring_up(GpuResidualBridge* b, void* device,
                 const GpuResidualTapTable* plan) {
    b->device = static_cast<ID3D12Device*>(device);
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = b->device->CreateCommandQueue(&q,
        IID_PPV_ARGS(b->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(b->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_DIRECT, b->allocator.Get(), nullptr,
        IID_PPV_ARGS(b->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = b->list->Close();
    if (SUCCEEDED(hr)) hr = b->device->CreateFence(0,
        D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(b->fence.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = make_resources(b);
    if (SUCCEEDED(hr) && b->planned) hr = create_tap_buffer(b);
    if (SUCCEEDED(hr) && b->planned) hr = upload_plan(b, plan);
    if (SUCCEEDED(hr)) hr = make_shader(b);
    if (FAILED(hr)) { if (!last_error[0]) fail("initialize GPU residual", hr);
        gpu_residual_close(b); return 3; }
    b->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!b->event) { fail("CreateEvent", GetLastError());
        gpu_residual_close(b); return 4; }
    return 0;
}
}

extern "C" GPU_RESIDUAL_API uint32_t gpu_residual_abi_version(void) {
    return GPU_RESIDUAL_ABI_VERSION;
}
extern "C" GPU_RESIDUAL_API const char* gpu_residual_last_error(void) {
    return last_error;
}
extern "C" GPU_RESIDUAL_API void gpu_residual_close(GpuResidualBridge* b) {
    if (!b) return;
    if (b->full_bridge) cuda_d3d12_color_destroy(b->full_bridge);
    if (b->low_bridge) cuda_d3d12_color_destroy(b->low_bridge);
    if (b->event) CloseHandle(b->event);
    delete b;
}
extern "C" GPU_RESIDUAL_API int32_t gpu_residual_open(void* device,
    uint32_t low_w, uint32_t low_h, GpuResidualBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    if (!device || !out || !low_w || !low_h || low_w > UINT32_MAX / 8 ||
        low_h > UINT32_MAX / 2) { fail("invalid residual dimensions", 0); return 1; }
    auto* b = new (std::nothrow) GpuResidualBridge;
    if (!b) { fail("allocate residual bridge", 0); return 2; }
    b->low_w = low_w; b->low_h = low_h;
    b->full_w = low_w * 2; b->full_h = low_h * 2;
    const int32_t code = bring_up(b, device, nullptr);
    if (code) return code;
    *out = b;
    return 0;
}

extern "C" GPU_RESIDUAL_API int32_t gpu_residual_open_scaled(void* device,
    uint32_t full_w, uint32_t full_h, uint32_t low_w, uint32_t low_h,
    const GpuResidualTapTable* plan, GpuResidualBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    // An upscale to the same size is a copy, and the bridge that composes a
    // residual onto a copy has no reason to exist: the caller would do better to
    // encode the DLSS output directly, which is what the 1.0 chain does.
    if (!device || !out || full_w < 2 || full_h < 2 || low_w < 1 || low_h < 1 ||
        low_w > full_w || low_h > full_h || full_w > 16384 || full_h > 16384 ||
        uint64_t(aligned_pitch(full_w)) * full_h > UINT32_MAX ||
        uint64_t(aligned_pitch(low_w)) * low_h > UINT32_MAX) {
        fail("invalid scaled residual dimensions", 0); return 1; }
    if (check_plan(plan, full_w, full_h, low_w, low_h)) return 1;
    auto* b = new (std::nothrow) GpuResidualBridge;
    if (!b) { fail("allocate residual bridge", 0); return 2; }
    b->low_w = low_w; b->low_h = low_h;
    b->full_w = full_w; b->full_h = full_h;
    b->planned = true;
    b->tap_stride = GPU_RESIDUAL_MAX_TAPS * 2 * sizeof(uint32_t);
    const int32_t code = bring_up(b, device, plan);
    if (code) return code;
    *out = b;
    return 0;
}

extern "C" GPU_RESIDUAL_API int32_t gpu_residual_process(GpuResidualBridge* b,
    void* dlss_raw, void* dlss_fence_raw, uint64_t dlss_value,
    void* full_context, uint64_t full_ptr, size_t full_pitch,
    void* low_context, uint64_t low_ptr, size_t low_pitch,
    float strength, void** out_texture) {
    last_error[0] = 0;
    if (out_texture) *out_texture = nullptr;
    if (!b || b->failed || !dlss_raw || !dlss_fence_raw || !out_texture ||
        !full_context || !full_ptr || full_pitch < size_t(b->full_w) * 4 ||
        !low_context || !low_ptr || low_pitch < size_t(b->low_w) * 4 ||
        !std::isfinite(strength) || strength < 0.0f) {
        fail("invalid GPU residual arguments", 0); return 1;
    }
    auto* dlss = static_cast<ID3D12Resource*>(dlss_raw);
    ComPtr<ID3D12Device> owner;
    HRESULT hr = dlss->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
    const auto desc = dlss->GetDesc();
    if (FAILED(hr) || owner.Get() != b->device.Get() ||
        desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Width != b->low_w || desc.Height != b->low_h ||
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM) {
        fail("DLSS output resource mismatch", FAILED(hr) ? hr : 0); return 2;
    }
    if (!b->full_bridge) {
        if (cuda_d3d12_color_create(b->device.Get(), full_context,
                b->full_w, b->full_h, aligned_pitch(b->full_w),
                &b->full_bridge)) {
            std::snprintf(last_error, sizeof(last_error), "full color bridge: %s",
                nvof_d3d12_last_error()); return 3;
        }
        b->full_context = full_context;
    }
    if (!b->low_bridge) {
        if (cuda_d3d12_color_create(b->device.Get(), low_context,
                b->low_w, b->low_h, aligned_pitch(b->low_w),
                &b->low_bridge)) {
            std::snprintf(last_error, sizeof(last_error), "working color bridge: %s",
                nvof_d3d12_last_error()); return 4;
        }
        b->low_context = low_context;
    }
    if (b->full_context != full_context || b->low_context != low_context) {
        fail("CUDA context changed", 0); return 5;
    }
    uint64_t full_ready = 0, low_ready = 0;
    if (cuda_d3d12_color_copy(b->full_bridge, full_ptr, full_pitch,
            &full_ready) || cuda_d3d12_color_copy(b->low_bridge, low_ptr,
            low_pitch, &low_ready)) {
        std::snprintf(last_error, sizeof(last_error), "CUDA color copy: %s",
            nvof_d3d12_last_error()); b->failed = true; return 6;
    }
    b->full_device_copy_bytes += uint64_t(b->full_w) * b->full_h * 4;
    b->working_device_copy_bytes += uint64_t(b->low_w) * b->low_h * 4;
    void *full_raw = nullptr, *low_raw = nullptr;
    void *full_fence_raw = nullptr, *low_fence_raw = nullptr;
    cuda_d3d12_color_get_resources(b->full_bridge, &full_raw, &full_fence_raw);
    cuda_d3d12_color_get_resources(b->low_bridge, &low_raw, &low_fence_raw);
    auto* full = static_cast<ID3D12Resource*>(full_raw);
    auto* low = static_cast<ID3D12Resource*>(low_raw);
    hr = b->queue->Wait(static_cast<ID3D12Fence*>(dlss_fence_raw), dlss_value);
    if (SUCCEEDED(hr)) hr = b->queue->Wait(
        static_cast<ID3D12Fence*>(full_fence_raw), full_ready);
    if (SUCCEEDED(hr)) hr = b->queue->Wait(
        static_cast<ID3D12Fence*>(low_fence_raw), low_ready);
    if (FAILED(hr)) { fail("queue wait", hr); b->failed = true; return 7; }
    set_raw_srv(b, full, aligned_pitch(b->full_w) * b->full_h, 0);
    set_raw_srv(b, low, aligned_pitch(b->low_w) * b->low_h, 1);
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Texture2D.MipLevels = 1;
    auto cpu = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += 2 * b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    b->device->CreateShaderResourceView(dlss, &srv, cpu);
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset residual command list", hr);
        b->failed = true; return 8; }
    auto* list = b->list.Get();
    barrier(list, full, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(list, low, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(list, dlss, D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(list, b->output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ID3D12DescriptorHeap* heaps[] = {b->descriptors.Get()};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(b->root.Get());
    list->SetPipelineState(b->pipeline.Get());
    auto gpu = b->descriptors->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    // The plan occupies the fourth SRV slot, so the output UAV sits one past
    // where the exact-2x layout leaves it.
    const uint32_t srv_count = b->planned ? 4 : 3;
    gpu.ptr += srv_count * b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    list->SetComputeRootDescriptorTable(1, gpu);
    struct Constants {
        uint32_t full_w, full_h, low_w, low_h, full_pitch, low_pitch;
        float strength; uint32_t padding;
    } constants{b->full_w, b->full_h, b->low_w, b->low_h,
        aligned_pitch(b->full_w), aligned_pitch(b->low_w), strength, 0};
    list->SetComputeRoot32BitConstants(2, 8, &constants, 0);
    list->Dispatch((b->full_w + 7) / 8, (b->full_h + 7) / 8, 1);
    barrier(list, b->output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(list, dlss, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(list, low, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    barrier(list, full, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    hr = list->Close();
    if (FAILED(hr)) { fail("close residual command list", hr);
        b->failed = true; return 9; }
    ID3D12CommandList* commands[] = {list};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr) || !wait_gpu(b)) {
        if (FAILED(hr)) fail("signal residual fence", hr);
        b->failed = true; return 10;
    }
    b->processed = true;
    *out_texture = b->output.Get();
    return 0;
}

extern "C" GPU_RESIDUAL_API int32_t gpu_residual_readback(GpuResidualBridge* b,
    uint8_t* rgba, size_t stride) {
    last_error[0] = 0;
    if (!b || b->failed || !b->processed || !rgba ||
        stride < size_t(b->full_w) * 4) {
        fail("invalid residual readback", 0); return 1;
    }
    HRESULT hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset readback list", hr); b->failed = true; return 2; }
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = b->readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = b->readback_layout;
    src.pResource = b->output.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    b->list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    hr = b->list->Close();
    if (FAILED(hr)) { fail("close readback list", hr); b->failed = true; return 3; }
    ID3D12CommandList* commands[] = {b->list.Get()};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr) || !wait_gpu(b)) {
        if (FAILED(hr)) fail("signal readback fence", hr);
        b->failed = true; return 4;
    }
    void* mapped = nullptr;
    hr = b->readback->Map(0, nullptr, &mapped);
    if (FAILED(hr)) { fail("map residual readback", hr);
        b->failed = true; return 5; }
    for (uint32_t y = 0; y < b->full_h; ++y)
        std::memcpy(rgba + size_t(y) * stride,
            static_cast<const uint8_t*>(mapped) +
                size_t(y) * b->readback_layout.Footprint.RowPitch,
            size_t(b->full_w) * 4);
    b->readback->Unmap(0, nullptr);
    b->output_readback_bytes += uint64_t(b->full_w) * b->full_h * 4;
    return 0;
}

extern "C" GPU_RESIDUAL_API int32_t gpu_residual_get_transfer_counters(
    GpuResidualBridge* b, GpuResidualTransferCounters* counters) {
    if (!b || !counters || counters->struct_size != sizeof(*counters)) return 1;
    *counters = GpuResidualTransferCounters{sizeof(*counters),
        b->full_device_copy_bytes, b->working_device_copy_bytes,
        b->output_readback_bytes};
    return 0;
}
