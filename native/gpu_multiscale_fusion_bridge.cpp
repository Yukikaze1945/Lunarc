#include "gpu_multiscale_fusion_bridge.h"
#include "gpu_multiscale_fusion_shader.h"
#include "nvof_d3d12_interop.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <new>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
thread_local char last_error[768] = "";
void fail(const char* operation, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", operation,
                  static_cast<unsigned long>(code));
}
constexpr uint32_t kMaxTaps = 24;
constexpr uint32_t kSrvCount = 8, kUavCount = 4, kDescriptors = 12;
constexpr uint32_t kDescriptorCapacity = 4096;
uint32_t pitch_for(uint32_t width) { return (width * 4u + 255u) & ~255u; }
/* The HDR transport's row: a half sample is 8 bytes, so a row that is width * 4
 * bytes wide carries half of it and the tail of every row is silently lost.
 * Rounded to the same 256-byte boundary a placed footprint requires. */
uint32_t pitch16_for(uint32_t width) { return (width * 8u + 255u) & ~255u; }
/* Which of the bridge's own textures is a store's final picture: it is created
 * and left in COPY_SOURCE rather than NON_PIXEL_SHADER_RESOURCE, because the
 * only thing that ever reads it is a copy out. */
bool is_final_format(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R8G8B8A8_UNORM ||
           format == DXGI_FORMAT_R16G16B16A16_FLOAT;
}
struct TapRow { float weights[kMaxTaps]; int32_t indices[kMaxTaps]; float norm_square = 0; };
struct Edge { uint32_t destination; float weight; };
struct AxisBuffers {
    ComPtr<ID3D12Resource> rows, offsets, edges;
    uint32_t row_count = 0, offset_count = 0, edge_count = 0;
    uint32_t taps = 0;
    double max_row_abs = 0, max_column_abs = 0;
    float max_norm_square = 0;
};
enum Kernel : size_t { kForwardY, kForwardX, kAdjointX, kAdjointY,
                       kCatmull, kVector, kComposite, kDot, kUnpackHalf,
                       kCompositeHdr, kReduce, kCgControl, kKernelCount };
/* The two kernels that store a picture's final pixels write u2 rather than u0,
 * and u2 is bound to the one texture make_texture created in COPY_SOURCE. Both
 * halves of that rule read one predicate, so an R16G16B16A16_FLOAT final is
 * transitioned as the store it is instead of as a work texture, and the state a
 * caller is told about is the state the code actually left it in. */
bool stores_final(Kernel kernel) {
    return kernel == kComposite || kernel == kCompositeHdr;
}
struct CgPacket {
    float square, initial, alpha, gamma;
    uint32_t status, iterations;
    float tolerance, reserved;
};
static_assert(sizeof(CgPacket) == 32, "CG packet layout");
struct Texture {
    ComPtr<ID3D12Resource> resource;
    uint32_t width = 0, height = 0;
};
struct LayerBuffers {
    uint32_t width = 0, height = 0;
    float weight = 0;
    AxisBuffers x, y;
    Texture scratch, down, iterate, gradient, direction;
    /* What this layer contributes to the objective as its r_i. The SDR path
     * never fills it: that path's observation arrives already as an
     * R32G32B32A32_FLOAT texture owned by the caller and is borrowed for the
     * call. The HDR path widens the borrowed half plane into this texture,
     * because a CUDA plane has no shader-visible layout to bind and an R32 view
     * over half samples would be a reinterpretation, not a conversion. */
    Texture observed;
};
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
bool valid_axis(const GpuFusionAxisPlan& plan, uint32_t source,
                uint32_t destination, uint32_t capacity) {
    if (plan.src_len != source || plan.dst_len != destination ||
        !plan.indices || !plan.weights || plan.taps < 1 ||
        plan.taps > capacity) return false;
    for (uint32_t row = 0; row < destination; ++row)
        for (uint32_t tap = 0; tap < plan.taps; ++tap) {
            const uint64_t at = uint64_t(row) * plan.taps + tap;
            if (plan.indices[at] < 0 || uint32_t(plan.indices[at]) >= source ||
                !std::isfinite(plan.weights[at])) return false;
        }
    return true;
}
}

struct GpuFusionBridge {
    GpuFusionOptions options{};
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    std::array<ComPtr<ID3D12PipelineState>, kKernelCount> kernels;
    std::array<ComPtr<ID3D12PipelineState>, kMaxTaps+1> forward_x, forward_y;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    uint32_t descriptor_cursor = 0, descriptor_step = 0;
    HANDLE event = nullptr;
    uint64_t fence_value = 0;
    CudaD3D12ColorBridge* color = nullptr;
    void* cuda_context = nullptr;
    std::array<LayerBuffers, 3> layers;
    AxisBuffers baseline_x, baseline_y;
    Texture r0, b[2], x[2], g[2], p[2], h[2], work, final;
    Texture probe_output;
    D3D12_RESOURCE_STATES probe_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    ComPtr<ID3D12Resource> partial, partial_readback, reduction_a, reduction_b;
    uint32_t partial_count = 0;
    uint64_t process_count = 0, full_device_copy_bytes = 0;
    uint64_t scalar_readback_bytes = 0;
    uint64_t workspace_logical_bytes = 0, workspace_committed_bytes = 0;
    uint64_t scalar_sync_count = 0;
    ComPtr<ID3D12Resource> control, control_readback;
    D3D12_RESOURCE_STATES control_state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    ID3D12Resource* last_scalar = nullptr;
    bool gpu_cg = true, cg_guard = false, observation_space = false, jacobi = false;
    float control_tolerance = 0;
    uint32_t batch_size = 4;
    double cpu_wait_ms = 0;
    bool failed = false;
    /* --- typed HDR (scRGB) contract -------------------------------------- */
    /* Which colour contract this bridge was opened for. It is decided once, at
     * open, and it decides what the samples in and out of this bridge mean: an
     * HDR bridge's planes are linear scRGB half and its store is an FP16
     * texture, and neither entry will run the other's chain. Nothing is
     * inferred per call and nothing falls back. */
    bool hdr = false;
    /* The observation planes come in one per layer, because each layer's own
     * gpu_hdr_observe plane has that layer's own extent and pitch and one
     * transport bridge can only describe one row. Their buffers, fences and
     * fence values are asked of the transport each call, as the full-resolution
     * plane's already are, so nothing here can go stale. */
    CudaD3D12ColorBridge* obs_color[3] = {};
    uint64_t observation_device_copy_bytes = 0;
    uint64_t unrepresentable_components = 0;
};

namespace {
bool wait_gpu(GpuFusionBridge* bridge) {
    if (bridge->fence->GetCompletedValue() >= bridge->fence_value) return true;
    const auto started = std::chrono::steady_clock::now();
    HRESULT hr = bridge->fence->SetEventOnCompletion(bridge->fence_value,
                                                     bridge->event);
    if (FAILED(hr)) { fail("fusion fence event", hr); return false; }
    const auto waited = WaitForSingleObject(bridge->event, 30000);
    bridge->cpu_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    if (waited != WAIT_OBJECT_0) {
        fail("fusion fence timeout", ERROR_TIMEOUT); return false;
    }
    return true;
}
HRESULT make_buffer(GpuFusionBridge* bridge, uint64_t bytes,
                    D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                    ComPtr<ID3D12Resource>* output) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = std::max<uint64_t>(bytes, 4);
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (type == D3D12_HEAP_TYPE_DEFAULT)
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return bridge->device->CreateCommittedResource(&heap,
        D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
        IID_PPV_ARGS(output->GetAddressOf()));
}
template<class T>
HRESULT upload(GpuFusionBridge* bridge, const std::vector<T>& values,
               ComPtr<ID3D12Resource>* resource) {
    HRESULT hr = make_buffer(bridge, uint64_t(values.size()) * sizeof(T),
        D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, resource);
    if (FAILED(hr)) return hr;
    void* mapped = nullptr;
    hr = (*resource)->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return hr;
    if (!values.empty()) std::memcpy(mapped, values.data(),
                                     values.size() * sizeof(T));
    (*resource)->Unmap(0, nullptr);
    return S_OK;
}
HRESULT upload_axis(GpuFusionBridge* bridge, const GpuFusionAxisPlan& plan,
                    AxisBuffers* result) {
    std::vector<TapRow> rows(plan.dst_len);
    std::vector<std::vector<Edge>> inverse(plan.src_len);
    for (uint32_t row = 0; row < plan.dst_len; ++row)
        for (uint32_t tap = 0; tap < kMaxTaps; ++tap) {
            const bool real = tap < plan.taps;
            const uint64_t at = uint64_t(row) * plan.taps + tap;
            const float weight = real ? plan.weights[at] : 0.0f;
            const int32_t source = real ? plan.indices[at] : 0;
            rows[row].weights[tap] = weight;
            rows[row].indices[tap] = source;
            if (real && weight != 0.0f)
                inverse[source].push_back(Edge{row, weight});
        }
    std::vector<double> column_abs(plan.src_len, 0.0);
    for (uint32_t row = 0; row < plan.dst_len; ++row) {
        double row_abs = 0, row_square = 0;
        for (uint32_t tap = 0; tap < plan.taps; ++tap) {
            const auto index = rows[row].indices[tap];
            bool first = true;
            for (uint32_t previous = 0; previous < tap; ++previous)
                if (rows[row].indices[previous] == index) first = false;
            if (!first) continue;
            double merged = 0;
            for (uint32_t next = tap; next < plan.taps; ++next)
                if (rows[row].indices[next] == index) merged += rows[row].weights[next];
            row_square += merged * merged;
            row_abs += std::abs(merged);
            column_abs[index] += std::abs(merged);
        }
        rows[row].norm_square = static_cast<float>(row_square);
        result->max_norm_square = (std::max)(result->max_norm_square, rows[row].norm_square);
        result->max_row_abs = (std::max)(result->max_row_abs, row_abs);
    }
    for (double value : column_abs)
        result->max_column_abs = (std::max)(result->max_column_abs, value);
    std::vector<uint32_t> offsets(plan.src_len + 1);
    std::vector<Edge> edges;
    for (uint32_t source = 0; source < plan.src_len; ++source) {
        offsets[source] = static_cast<uint32_t>(edges.size());
        edges.insert(edges.end(), inverse[source].begin(), inverse[source].end());
    }
    offsets[plan.src_len] = static_cast<uint32_t>(edges.size());
    result->row_count = plan.dst_len;
    // Remove trailing all-zero padding only. Every non-zero tap is retained.
    result->taps = plan.taps;
    while (result->taps > 1) {
        bool used = false;
        for (const auto& row : rows) if (row.weights[result->taps-1] != 0) used = true;
        if (used) break;
        --result->taps;
    }
    result->offset_count = static_cast<uint32_t>(offsets.size());
    result->edge_count = static_cast<uint32_t>(edges.size());
    HRESULT hr = upload(bridge, rows, &result->rows);
    if (SUCCEEDED(hr)) hr = upload(bridge, offsets, &result->offsets);
    if (SUCCEEDED(hr)) hr = upload(bridge, edges, &result->edges);
    return hr;
}
HRESULT make_texture(GpuFusionBridge* bridge, uint32_t width, uint32_t height,
                     DXGI_FORMAT format, Texture* output) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    const auto state = (is_final_format(format)
        ? D3D12_RESOURCE_STATE_COPY_SOURCE
        : D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    HRESULT hr = bridge->device->CreateCommittedResource(&heap,
        D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
        IID_PPV_ARGS(output->resource.GetAddressOf()));
    if (SUCCEEDED(hr)) {
        output->width = width; output->height = height;
        if (format == DXGI_FORMAT_R32G32B32A32_FLOAT) {
            bridge->workspace_logical_bytes += uint64_t(width) * height * 16u;
            bridge->workspace_committed_bytes += bridge->device->GetResourceAllocationInfo(
                0, 1, &desc).SizeInBytes;
        }
    }
    return hr;
}
HRESULT make_resources(GpuFusionBridge* bridge) {
    const uint32_t width = bridge->options.full_width;
    const uint32_t height = bridge->options.full_height;
    auto full = [&](Texture* texture) {
        return make_texture(bridge, width, height,
            DXGI_FORMAT_R32G32B32A32_FLOAT, texture);
    };
    HRESULT hr = full(&bridge->r0);
    if (SUCCEEDED(hr) && !bridge->observation_space) hr = full(&bridge->b[0]);
    if (SUCCEEDED(hr)) {
        if (bridge->gpu_cg) bridge->b[1] = bridge->b[0];
        else hr = full(&bridge->b[1]);
    }
    if (SUCCEEDED(hr)) hr = full(&bridge->x[0]);
    if (SUCCEEDED(hr)) {
        if (bridge->gpu_cg) bridge->x[1] = bridge->x[0];
        else hr = full(&bridge->x[1]);
    }
    if (SUCCEEDED(hr) && !bridge->observation_space) hr = full(&bridge->g[0]);
    if (SUCCEEDED(hr)) {
        if (bridge->gpu_cg) bridge->g[1] = bridge->g[0];
        else hr = full(&bridge->g[1]);
    }
    if (SUCCEEDED(hr) && !bridge->observation_space) hr = full(&bridge->p[0]);
    if (SUCCEEDED(hr)) {
        if (bridge->gpu_cg) bridge->p[1] = bridge->p[0];
        else hr = full(&bridge->p[1]);
    }
    if (SUCCEEDED(hr)) hr = full(&bridge->h[0]);
    if (SUCCEEDED(hr)) {
        if (bridge->gpu_cg) bridge->h[1] = bridge->h[0];
        else hr = full(&bridge->h[1]);
    }
    if (SUCCEEDED(hr)) hr = full(&bridge->work);
    /* The store's pixel format is the colour contract, not a per-call choice:
     * RGBA8 bytes for the chain that has always run here, a half scRGB picture
     * for the one that hands its output to the HDR bridge's composite. Every
     * other texture in this function is FP32 work on both paths. */
    if (SUCCEEDED(hr)) hr = make_texture(bridge, width, height,
        bridge->hdr ? DXGI_FORMAT_R16G16B16A16_FLOAT
                    : DXGI_FORMAT_R8G8B8A8_UNORM, &bridge->final);
    for (uint32_t i = 0; i < bridge->options.layer_count && SUCCEEDED(hr); ++i) {
        auto& layer = bridge->layers[i];
        hr = make_texture(bridge, width, layer.height,
            DXGI_FORMAT_R32G32B32A32_FLOAT, &layer.scratch);
        if (SUCCEEDED(hr)) hr = make_texture(bridge, layer.width, layer.height,
            DXGI_FORMAT_R32G32B32A32_FLOAT, &layer.down);
        if (SUCCEEDED(hr) && bridge->observation_space) {
            for (Texture* texture : {&layer.iterate, &layer.gradient, &layer.direction})
                if (SUCCEEDED(hr)) hr = make_texture(bridge, layer.width, layer.height,
                    DXGI_FORMAT_R32G32B32A32_FLOAT, texture);
        }
        /* The HDR path needs a place of its own to widen each borrowed half
         * plane into, at that layer's extent and in the precision the CG
         * kernels already work in. The SDR path borrows the caller's FP32
         * texture and allocates nothing. */
        if (SUCCEEDED(hr) && bridge->hdr)
            hr = make_texture(bridge, layer.width, layer.height,
                DXGI_FORMAT_R32G32B32A32_FLOAT, &layer.observed);
    }
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, sizeof(CgPacket),
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &bridge->control);
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, sizeof(CgPacket),
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, &bridge->control_readback);
    if (SUCCEEDED(hr)) transition(bridge->list.Get(), bridge->control.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    bridge->partial_count = ((width + 7) / 8) * ((height + 7) / 8);
    const uint64_t partial_bytes = uint64_t(bridge->partial_count) * sizeof(float);
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, partial_bytes,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        &bridge->partial);
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, sizeof(float),
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
        &bridge->partial_readback);
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, partial_bytes,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        &bridge->reduction_a);
    if (SUCCEEDED(hr)) hr = make_buffer(bridge, partial_bytes,
        D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON,
        &bridge->reduction_b);
    /* D3D12 buffers are created in COMMON regardless of the requested initial
     * state. Establish UAV explicitly so the dot and reduction descriptors are
     * valid from their first dispatch, including under the debug layer. */
    if (SUCCEEDED(hr)) transition(bridge->list.Get(), bridge->partial.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (SUCCEEDED(hr)) transition(bridge->list.Get(), bridge->reduction_a.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (SUCCEEDED(hr)) transition(bridge->list.Get(), bridge->reduction_b.Get(),
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    return hr;
}
HRESULT make_kernels(GpuFusionBridge* bridge) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = kSrvCount;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = kUavCount;
    D3D12_ROOT_PARAMETER parameters[3]{};
    for (uint32_t i = 0; i < 2; ++i) {
        parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameters[i].DescriptorTable.NumDescriptorRanges = 1;
        parameters[i].DescriptorTable.pDescriptorRanges = &ranges[i];
    }
    parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    parameters[2].Constants.ShaderRegister = 0;
    parameters[2].Constants.Num32BitValues = 8;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = parameters;
    ComPtr<ID3DBlob> root_blob;
    HRESULT hr = D3D12SerializeRootSignature(&root_desc,
        D3D_ROOT_SIGNATURE_VERSION_1, root_blob.GetAddressOf(), nullptr);
    if (SUCCEEDED(hr)) hr = bridge->device->CreateRootSignature(0,
        root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
        IID_PPV_ARGS(bridge->root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    const char* names[kKernelCount] = {"ForwardY", "ForwardX", "AdjointX",
        "AdjointY", "CatmullBaseline", "Vector", "Composite", "Dot",
        "UnpackHalf", "CompositeHdr", "Reduce", "CgControl"};
    for (size_t i = 0; i < kKernelCount; ++i) {
        ComPtr<ID3DBlob> code, errors;
        hr = D3DCompile(fusion_shader, sizeof(fusion_shader) - 1,
            "gpu_multiscale_fusion", nullptr, nullptr, names[i], "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
            0, code.GetAddressOf(), errors.GetAddressOf());
        if (FAILED(hr)) {
            if (errors) std::snprintf(last_error, sizeof(last_error),
                "D3DCompile %s: %s", names[i],
                static_cast<const char*>(errors->GetBufferPointer()));
            return hr;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = bridge->root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = bridge->device->CreateComputePipelineState(&pso,
            IID_PPV_ARGS(bridge->kernels[i].GetAddressOf()));
        if (FAILED(hr)) return hr;
    }
    for (uint32_t layer=0; layer<bridge->options.layer_count; ++layer) {
        for (uint32_t taps : {bridge->layers[layer].x.taps, bridge->layers[layer].y.taps}) {
            for (size_t direction=0; direction<2; ++direction) {
                auto& target = direction ? bridge->forward_x[taps] : bridge->forward_y[taps];
                if (target) continue;
                char count[16]; std::snprintf(count,sizeof(count),"%u",taps);
                const D3D_SHADER_MACRO macros[]={{"FUSION_TAP_COUNT",count},{nullptr,nullptr}};
                ComPtr<ID3DBlob> code, errors;
                hr=D3DCompile(fusion_shader,sizeof(fusion_shader)-1,"gpu_multiscale_fusion",
                    macros,nullptr,direction ? "ForwardX" : "ForwardY","cs_5_0",
                    D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS,
                    0,code.GetAddressOf(),errors.GetAddressOf());
                if (FAILED(hr)) return hr;
                D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
                pso.pRootSignature=bridge->root.Get();
                pso.CS={code->GetBufferPointer(),code->GetBufferSize()};
                hr=bridge->device->CreateComputePipelineState(&pso,IID_PPV_ARGS(target.GetAddressOf()));
                if (FAILED(hr)) return hr;
            }
        }
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors = kDescriptorCapacity;
    heap.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr = bridge->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(bridge->descriptors.GetAddressOf()));
    if (SUCCEEDED(hr)) bridge->descriptor_step =
        bridge->device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    return hr;
}
void texture_srv(GpuFusionBridge* bridge, ID3D12Resource* resource,
                 D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
    view.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Texture2D.MipLevels = 1;
    bridge->device->CreateShaderResourceView(resource, &view, slot);
}
void structured_srv(GpuFusionBridge* bridge, ID3D12Resource* resource,
                    uint32_t count, uint32_t stride,
                    D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_UNKNOWN;
    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.NumElements = (std::max)(count, 1u);
    view.Buffer.StructureByteStride = stride;
    bridge->device->CreateShaderResourceView(resource, &view, slot);
}
void raw_srv(GpuFusionBridge* bridge, ID3D12Resource* resource,
             D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format = DXGI_FORMAT_R32_TYPELESS;
    view.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    view.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.NumElements = static_cast<uint32_t>(resource->GetDesc().Width / 4);
    view.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    bridge->device->CreateShaderResourceView(resource, &view, slot);
}
void texture_uav(GpuFusionBridge* bridge, ID3D12Resource* resource,
                 D3D12_CPU_DESCRIPTOR_HANDLE slot) {
    bridge->device->CreateUnorderedAccessView(resource, nullptr, nullptr, slot);
}
struct FrameConstants {
    uint32_t out_width, out_height, tap_count, beta_bits;
    uint32_t full_pitch, group_width, mode;
    float scale;
};
bool record(GpuFusionBridge* bridge, Kernel kernel, Texture* input_a,
            Texture* input_b, Texture* output, AxisBuffers* axis,
            AxisBuffers* vertical, ID3D12Resource* full_color,
            uint32_t full_pitch, uint32_t width, uint32_t height,
            float scale = 0.0f, uint32_t mode = 0) {
    if (bridge->descriptor_cursor + kDescriptors > kDescriptorCapacity) {
        fail("fusion descriptor heap exhausted", 0); return false;
    }
    if (!input_a || !input_a->resource || !axis || !vertical ||
        !full_color || !output || !output->resource) {
        fail("fusion dispatch resource missing", 0); return false;
    }
    input_b = input_b ? input_b : input_a;
    const uint32_t first = bridge->descriptor_cursor;
    bridge->descriptor_cursor += kDescriptors;
    auto cpu = bridge->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr += size_t(first) * bridge->descriptor_step;
    const auto at = [&](uint32_t index) {
        D3D12_CPU_DESCRIPTOR_HANDLE slot = cpu;
        slot.ptr += size_t(index) * bridge->descriptor_step;
        return slot;
    };
    texture_srv(bridge, input_a->resource.Get(), at(0));
    texture_srv(bridge, input_b->resource.Get(), at(1));
    structured_srv(bridge, axis->rows.Get(), axis->row_count,
                   sizeof(TapRow), at(2));
    structured_srv(bridge, axis->offsets.Get(), axis->offset_count,
                   sizeof(uint32_t), at(3));
    structured_srv(bridge, axis->edges.Get(), axis->edge_count,
                   sizeof(Edge), at(4));
    raw_srv(bridge, full_color, at(5));
    structured_srv(bridge, vertical->rows.Get(), vertical->row_count,
                   sizeof(TapRow), at(6));
    const auto desired_control_state = kernel==kCgControl ? D3D12_RESOURCE_STATE_UNORDERED_ACCESS :
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (bridge->control_state != desired_control_state) {
        transition(bridge->list.Get(),bridge->control.Get(),bridge->control_state,desired_control_state);
        bridge->control_state=desired_control_state;
    }
    raw_srv(bridge, kernel==kCgControl ? bridge->last_scalar : bridge->control.Get(), at(7));
    texture_uav(bridge, output->resource.Get(), at(kSrvCount));
    D3D12_UNORDERED_ACCESS_VIEW_DESC partial_view{};
    partial_view.Format = DXGI_FORMAT_UNKNOWN;
    partial_view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    partial_view.Buffer.NumElements = bridge->partial_count;
    partial_view.Buffer.StructureByteStride = sizeof(float);
    bridge->device->CreateUnorderedAccessView(bridge->partial.Get(), nullptr,
                                              &partial_view, at(kSrvCount + 1));
    texture_uav(bridge, bridge->final.resource.Get(), at(kSrvCount + 2));
    D3D12_UNORDERED_ACCESS_VIEW_DESC control_view{};
    control_view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    control_view.Buffer.NumElements = 1;
    control_view.Buffer.StructureByteStride = sizeof(CgPacket);
    bridge->device->CreateUnorderedAccessView(bridge->control.Get(), nullptr,
        &control_view, at(kSrvCount + 3));
    auto* list = bridge->list.Get();
    if (kernel != kDot) {
        const bool final = stores_final(kernel);
        transition(list, output->resource.Get(),
            final ? D3D12_RESOURCE_STATE_COPY_SOURCE :
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    list->SetPipelineState(kernel==kForwardX ? bridge->forward_x[axis->taps].Get() :
        kernel==kForwardY ? bridge->forward_y[axis->taps].Get() : bridge->kernels[kernel].Get());
    auto gpu = bridge->descriptors->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr += size_t(first) * bridge->descriptor_step;
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += size_t(kSrvCount) * bridge->descriptor_step;
    list->SetComputeRootDescriptorTable(1, gpu);
    const bool guarded = bridge->cg_guard &&
        (kernel == kForwardX || kernel == kForwardY || kernel == kAdjointX ||
         kernel == kAdjointY || kernel == kVector || kernel == kDot);
    uint32_t beta_bits = 0;
    std::memcpy(&beta_bits, &bridge->options.beta, sizeof(beta_bits));
    FrameConstants constants{width, height, axis->taps, beta_bits,
        full_pitch, (width + 7) / 8, mode | (guarded ? 256u : 0u), scale};
    list->SetComputeRoot32BitConstants(2, 8, &constants, 0);
    list->Dispatch((width + 7) / 8, (height + 7) / 8, 1);
    if (kernel != kDot) {
        const bool final = stores_final(kernel);
        transition(list, output->resource.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            final ? D3D12_RESOURCE_STATE_COPY_SOURCE :
                    D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    return true;
}
bool flush(GpuFusionBridge* bridge) {
    HRESULT hr = bridge->list->Close();
    if (FAILED(hr)) { fail("close fusion command list", hr); return false; }
    ID3D12CommandList* lists[] = {bridge->list.Get()};
    bridge->queue->ExecuteCommandLists(1, lists);
    hr = bridge->queue->Signal(bridge->fence.Get(), ++bridge->fence_value);
    if (FAILED(hr)) { fail("signal fusion fence", hr); return false; }
    if (!wait_gpu(bridge)) return false;
    hr = bridge->allocator->Reset();
    if (SUCCEEDED(hr)) hr = bridge->list->Reset(bridge->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset fusion command list", hr); return false; }
    bridge->descriptor_cursor = 0;
    ID3D12DescriptorHeap* heaps[] = {bridge->descriptors.Get()};
    bridge->list->SetDescriptorHeaps(1, heaps);
    bridge->list->SetComputeRootSignature(bridge->root.Get());
    return true;
}
bool vector(GpuFusionBridge* bridge, Texture* a, Texture* b, Texture* output,
            ID3D12Resource* full, uint32_t pitch, float scale,
            bool scale_only = false) {
    return record(bridge, kVector, a, b, output, &bridge->baseline_x,
                  &bridge->baseline_y, full, pitch,
                  bridge->options.full_width, bridge->options.full_height,
                  scale, scale_only ? 1u : 0u);
}
bool down(GpuFusionBridge* bridge, uint32_t layer_index, Texture* input,
          ID3D12Resource* full, uint32_t pitch) {
    auto& layer = bridge->layers[layer_index];
    return record(bridge, kForwardY, input, nullptr, &layer.scratch,
        &layer.y, &bridge->baseline_y, full, pitch,
        bridge->options.full_width, layer.height) &&
        record(bridge, kForwardX, &layer.scratch, nullptr, &layer.down,
        &layer.x, &bridge->baseline_y, full, pitch,
        layer.width, layer.height);
}
bool adjoint(GpuFusionBridge* bridge, uint32_t layer_index, Texture* input,
             ID3D12Resource* full, uint32_t pitch) {
    auto& layer = bridge->layers[layer_index];
    return record(bridge, kAdjointX, input, nullptr, &layer.scratch,
        &layer.x, &bridge->baseline_y, full, pitch,
        bridge->options.full_width, layer.height) &&
        record(bridge, kAdjointY, &layer.scratch, nullptr, &bridge->work,
        &layer.y, &bridge->baseline_y, full, pitch,
        bridge->options.full_width, bridge->options.full_height);
}
Texture* h_apply(GpuFusionBridge* bridge, Texture* input,
                 ID3D12Resource* full, uint32_t pitch) {
    const auto& options = bridge->options;
    if (!vector(bridge, input, nullptr, &bridge->h[0], full, pitch,
                options.beta, true)) return nullptr;
    uint32_t current = 0;
    for (uint32_t i = 0; i < options.layer_count; ++i) {
        if (!down(bridge, i, input, full, pitch) ||
            !adjoint(bridge, i, &bridge->layers[i].down, full, pitch) ||
            !(bridge->gpu_cg ? record(bridge, kVector, &bridge->work, nullptr,
                &bridge->h[0], &bridge->baseline_x, &bridge->baseline_y, full,
                pitch, options.full_width, options.full_height, bridge->layers[i].weight, 5u)
                : vector(bridge, &bridge->h[current], &bridge->work,
                    &bridge->h[1-current], full, pitch, bridge->layers[i].weight))) return nullptr;
        current = bridge->gpu_cg ? 0 : 1-current;
    }
    return &bridge->h[current];
}
/* Reduce group partials on the GPU through distinct ping-pong UAV buffers.
 * Only the final FP32 scalar is copied to the host. */
ID3D12Resource* reduce_gpu(GpuFusionBridge* bridge, uint32_t count = 0) {
    auto* list = bridge->list.Get();
    ID3D12Resource* input = bridge->partial.Get();
    ID3D12Resource* output = bridge->reduction_a.Get();
    uint32_t input_count = count ? count : bridge->partial_count;
    while (input_count > 1) {
        const uint32_t output_count = (input_count + 255u) / 256u;
        if (bridge->descriptor_cursor + kDescriptors > kDescriptorCapacity) {
            fail("fusion descriptor heap exhausted during reduction", 0);
            return nullptr;
        }
        const uint32_t first = bridge->descriptor_cursor;
        bridge->descriptor_cursor += kDescriptors;
        auto cpu = bridge->descriptors->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += size_t(first) * bridge->descriptor_step;
        D3D12_CPU_DESCRIPTOR_HANDLE slot = cpu;
        slot.ptr += size_t(5) * bridge->descriptor_step;
        raw_srv(bridge, input, slot);
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_UNKNOWN;
        view.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        view.Buffer.NumElements = output_count;
        view.Buffer.StructureByteStride = sizeof(float);
        slot = cpu;
        slot.ptr += size_t(kSrvCount + 1) * bridge->descriptor_step;
        bridge->device->CreateUnorderedAccessView(output, nullptr, &view, slot);

        /* Every pass reads input as an SRV and writes a different UAV. The
         * UAV->SRV transition orders the prior producer; returning input to UAV
         * after the dispatch both retires that read and makes it legal to reuse
         * this buffer as a later ping-pong destination. A and B never alias the
         * input of the same pass. */
        transition(list, input, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        list->SetPipelineState(bridge->kernels[kReduce].Get());
        auto gpu = bridge->descriptors->GetGPUDescriptorHandleForHeapStart();
        gpu.ptr += size_t(first) * bridge->descriptor_step;
        list->SetComputeRootDescriptorTable(0, gpu);
        gpu.ptr += size_t(kSrvCount) * bridge->descriptor_step;
        list->SetComputeRootDescriptorTable(1, gpu);
        FrameConstants constants{input_count, 1, input_count, 1, 0,
                                 output_count, 0, 0.0f};
        list->SetComputeRoot32BitConstants(2, 8, &constants, 0);
        list->Dispatch(output_count, 1, 1);
        transition(list, input, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        input = output;
        input_count = output_count;
        output = output == bridge->reduction_a.Get()
            ? bridge->reduction_b.Get() : bridge->reduction_a.Get();
    }
    bridge->last_scalar = input;
    return input;
}
double reduce_partial(GpuFusionBridge* bridge) {
    auto* input = reduce_gpu(bridge);
    if (!input) return NAN;
    auto* list = bridge->list.Get();
    transition(list, input, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(bridge->partial_readback.Get(), 0, input, 0,
                           sizeof(float));
    transition(list, input, D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!flush(bridge)) return NAN;
    float* mapped = nullptr;
    HRESULT hr = bridge->partial_readback->Map(0, nullptr,
                                               reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map fusion scalar reductions", hr); return NAN; }
    const double result = mapped[0];
    bridge->partial_readback->Unmap(0, nullptr);
    bridge->scalar_readback_bytes += sizeof(float);
    ++bridge->scalar_sync_count;
    return result;
}
double gpu_dot(GpuFusionBridge* bridge, Texture* a, Texture* b,
               ID3D12Resource* full, uint32_t pitch) {
    if (!record(bridge, kDot, a, b, &bridge->work,
                &bridge->baseline_x, &bridge->baseline_y, full, pitch,
                bridge->options.full_width, bridge->options.full_height))
        return NAN;
    return reduce_partial(bridge);
}
/* Widens one layer's borrowed linear scRGB half plane into the FP32 texture the
 * CG reads. A half sample is exactly representable in float, so this changes
 * storage and not value. InputA is not a sample this kernel touches - its
 * payload comes through the raw view of the transport buffer - but record()
 * binds a texture in every slot it has, so it names one that exists and is
 * already shader-visible. */
bool unpack_half(GpuFusionBridge* bridge, uint32_t layer_index,
                 ID3D12Resource* plane, uint32_t pitch) {
    auto& layer = bridge->layers[layer_index];
    return record(bridge, kUnpackHalf, &bridge->r0, nullptr, &layer.observed,
        &bridge->baseline_x, &bridge->baseline_y, plane, pitch,
        layer.width, layer.height);
}
/* What a solve produced, before either entry writes it into its own report
 * struct. The fields mean the same on both paths because the arithmetic does. */
struct SolveEvidence {
    uint32_t iterations = 0;
    double initial = 0.0, final_norm = 0.0, relative = 0.0;
    bool converged = false;
};
/* The objective and its CG, in one function so the two entries cannot become
 * two solvers. Everything that differs between the contracts is passed in: the
 * residuals, at their layer extents, and the full-resolution plane as a raw
 * buffer with its byte pitch. What is not passed in and does not differ is the
 * maths - the float32 taps, centres and clamp edges, the retained duplicated
 * border taps, the true transposes, the Catmull baseline, beta, the layer
 * weights, the tolerance and the iteration budget - and every kernel this
 * dispatches is the kernel the SDR path has always dispatched.
 *
 * Returns 0, or the stage code gpu_fusion_process has always returned for that
 * stage: 8 a dispatch that could not be recorded, 9 an unusable initial norm,
 * 10 a CG step that could not be recorded, 11 a lost positive definiteness or
 * an unusable norm mid-iteration, 12 a budget that ran out before the
 * tolerance. 11 and 12 are failures rather than outcomes: the bridge is marked
 * and the caller gets an error code, so there is no path here that hands back a
 * picture which is really the baseline while saying it was solved. */
int32_t run_solve_legacy(GpuFusionBridge* bridge, Texture* observed,
                  ID3D12Resource* full, uint32_t pitch,
                  SolveEvidence* evidence, uint32_t* out_x_index) {
    const uint32_t count = bridge->options.layer_count;
    if (!record(bridge, kCatmull, &observed[0], nullptr, &bridge->r0,
                &bridge->baseline_x, &bridge->baseline_y, full, pitch,
                bridge->options.full_width, bridge->options.full_height) ||
        !vector(bridge, &bridge->r0, nullptr, &bridge->b[0], full, pitch,
                bridge->options.beta, true)) {
        bridge->failed = true; return 8;
    }
    uint32_t b_index = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!adjoint(bridge, i, &observed[i], full, pitch) ||
            !vector(bridge, &bridge->b[b_index], &bridge->work,
                    &bridge->b[1-b_index], full, pitch,
                    bridge->layers[i].weight)) {
            bridge->failed = true; return 8;
        }
        b_index = 1-b_index;
    }
    Texture* rhs = &bridge->b[b_index];
    if (!vector(bridge, &bridge->r0, nullptr, &bridge->x[0], full, pitch,
                1.0f, true)) { bridge->failed = true; return 8; }
    Texture* applied = h_apply(bridge, &bridge->x[0], full, pitch);
    if (!applied ||
        !vector(bridge, rhs, applied, &bridge->g[0], full, pitch, -1.0f) ||
        !vector(bridge, &bridge->g[0], nullptr, &bridge->p[0], full, pitch,
                1.0f, true)) { bridge->failed = true; return 8; }
    double square = gpu_dot(bridge, &bridge->g[0], &bridge->g[0], full, pitch);
    if (!std::isfinite(square) || square < 0) {
        if (!last_error[0]) fail("fusion initial norm is invalid", 0);
        bridge->failed = true; return 9;
    }
    const double initial = std::sqrt(square);
    evidence->initial = initial;
    uint32_t x_index = 0, g_index = 0, p_index = 0, iterations = 0;
    double relative = initial == 0 ? 0 : 1;
    bool converged = initial == 0;
    for (uint32_t step = 1; !converged &&
         step <= bridge->options.max_iterations; ++step) {
        Texture* hp = h_apply(bridge, &bridge->p[p_index], full, pitch);
        if (!hp) { bridge->failed = true; return 10; }
        const double denominator = gpu_dot(bridge, &bridge->p[p_index], hp,
                                           full, pitch);
        if (!std::isfinite(denominator) || denominator <= 0) {
            std::snprintf(last_error, sizeof(last_error),
                "fusion CG lost positive definiteness at iteration %u", step);
            bridge->failed = true; return 11;
        }
        const float alpha = static_cast<float>(square / denominator);
        if (!vector(bridge, &bridge->x[x_index], &bridge->p[p_index],
                    &bridge->x[1-x_index], full, pitch, alpha) ||
            !vector(bridge, &bridge->g[g_index], hp,
                    &bridge->g[1-g_index], full, pitch, -alpha)) {
            bridge->failed = true; return 10;
        }
        x_index = 1-x_index;
        g_index = 1-g_index;
        const double next_square = gpu_dot(bridge, &bridge->g[g_index],
                                           &bridge->g[g_index], full, pitch);
        if (!std::isfinite(next_square) || next_square < 0) {
            if (!last_error[0]) fail("fusion CG norm is invalid", step);
            bridge->failed = true; return 11;
        }
        relative = std::sqrt(next_square) / initial;
        iterations = step;
        if (relative <= bridge->options.relative_tolerance) {
            converged = true;
            square = next_square;
            break;
        }
        const float gamma = static_cast<float>(next_square / square);
        if (!vector(bridge, &bridge->g[g_index], &bridge->p[p_index],
                    &bridge->p[1-p_index], full, pitch, gamma)) {
            bridge->failed = true; return 10;
        }
        p_index = 1-p_index;
        square = next_square;
    }
    evidence->iterations = iterations;
    if (converged) {
        Texture* truth_h=h_apply(bridge,&bridge->x[x_index],full,pitch);
        if (!truth_h || !vector(bridge,rhs,truth_h,&bridge->g[g_index],full,pitch,-1)) {
            bridge->failed=true; return 10;
        }
        square=gpu_dot(bridge,&bridge->g[g_index],&bridge->g[g_index],full,pitch);
        if (!std::isfinite(square) || square<0) {
            fail("fusion true equation residual is invalid",0); bridge->failed=true; return 11;
        }
        relative=initial==0 ? (square==0 ? 0 : INFINITY) : std::sqrt(square)/initial;
        converged=relative<=bridge->options.relative_tolerance;
    }
    evidence->final_norm = std::sqrt(square);
    evidence->relative = relative;
    evidence->converged = converged;
    if (!converged) {
        std::snprintf(last_error, sizeof(last_error),
            "fusion CG did not converge in %u iterations: relative residual "
            "%.8g > required %.8g", bridge->options.max_iterations,
            relative, bridge->options.relative_tolerance);
        bridge->failed = true; return 12;
    }
    *out_x_index = x_index;
    return 0;
}

void order_control(GpuFusionBridge* bridge) {
    if (bridge->control_state != D3D12_RESOURCE_STATE_UNORDERED_ACCESS) {
        transition(bridge->list.Get(),bridge->control.Get(),bridge->control_state,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        bridge->control_state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    barrier.UAV.pResource = bridge->control.Get();
    bridge->list->ResourceBarrier(1, &barrier);
}
bool control_step(GpuFusionBridge* bridge, ID3D12Resource* scalar,
                  ID3D12Resource* full, uint32_t pitch, uint32_t mode) {
    if (!scalar) return false;
    bridge->last_scalar = scalar;
    order_control(bridge);
    transition(bridge->list.Get(), scalar, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const bool ok = record(bridge, kCgControl, &bridge->work, nullptr, &bridge->r0,
        &bridge->baseline_x, &bridge->baseline_y, full, pitch, 1, 1,
        bridge->control_tolerance, mode);
    transition(bridge->list.Get(), scalar, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    order_control(bridge);
    return ok;
}
ID3D12Resource* dot_device(GpuFusionBridge* bridge, Texture* a, Texture* b,
                          ID3D12Resource* full, uint32_t pitch) {
    if (!record(bridge, kDot, a, b, &bridge->work, &bridge->baseline_x,
        &bridge->baseline_y, full, pitch, a->width, a->height)) return nullptr;
    return reduce_gpu(bridge, ((a->width+7)/8)*((a->height+7)/8));
}
bool read_control(GpuFusionBridge* bridge, CgPacket* packet) {
    transition(bridge->list.Get(), bridge->control.Get(),
        bridge->control_state, D3D12_RESOURCE_STATE_COPY_SOURCE);
    bridge->list->CopyBufferRegion(bridge->control_readback.Get(), 0,
        bridge->control.Get(), 0, sizeof(*packet));
    transition(bridge->list.Get(), bridge->control.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    bridge->control_state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (!flush(bridge)) return false;
    const D3D12_RANGE range{0, sizeof(*packet)};
    void* mapped = nullptr;
    const HRESULT hr = bridge->control_readback->Map(0, &range, &mapped);
    if (FAILED(hr)) { fail("map fusion CG state", hr); return false; }
    std::memcpy(packet, mapped, sizeof(*packet));
    const D3D12_RANGE written{0, 0};
    bridge->control_readback->Unmap(0, &written);
    bridge->scalar_readback_bytes += sizeof(*packet);
    ++bridge->scalar_sync_count;
    return true;
}
bool controlled_vector(GpuFusionBridge* bridge, Texture* input, Texture* output,
                       ID3D12Resource* full, uint32_t pitch, uint32_t mode) {
    return record(bridge, kVector, input, nullptr, output, &bridge->baseline_x,
        &bridge->baseline_y, full, pitch, output->width,
        output->height, 0, mode);
}
int32_t run_solve_gpu(GpuFusionBridge* bridge, Texture* observed,
                     ID3D12Resource* full, uint32_t pitch,
                     SolveEvidence* evidence, uint32_t* out_x_index) {
    bridge->cg_guard = false;
    const uint32_t count = bridge->options.layer_count;
    if (!record(bridge, kCatmull, &observed[0], nullptr, &bridge->r0,
            &bridge->baseline_x, &bridge->baseline_y, full, pitch,
            bridge->options.full_width, bridge->options.full_height) ||
        !vector(bridge, &bridge->r0, nullptr, &bridge->b[0], full, pitch,
            bridge->options.beta, true)) { bridge->failed = true; return 8; }
    uint32_t rhs_index = 0;
    for (uint32_t i = 0; i < count; ++i) {
        if (!adjoint(bridge, i, &observed[i], full, pitch) ||
            !record(bridge, kVector, &bridge->work, nullptr, &bridge->b[0],
                &bridge->baseline_x, &bridge->baseline_y, full, pitch,
                bridge->options.full_width, bridge->options.full_height,
                bridge->layers[i].weight, 5u)) {
            bridge->failed = true; return 8;
        }
    }
    Texture* rhs = &bridge->b[rhs_index];
    if (!vector(bridge, &bridge->r0, nullptr, &bridge->x[0], full, pitch, 1, true)) {
        bridge->failed = true; return 8;
    }
    Texture* hx = h_apply(bridge, &bridge->x[0], full, pitch);
    if (!hx || !vector(bridge, rhs, hx, &bridge->g[0], full, pitch, -1) ||
        !vector(bridge, &bridge->g[0], nullptr, &bridge->p[0], full, pitch, 1, true) ||
        !control_step(bridge, dot_device(bridge, &bridge->g[0], &bridge->g[0], full, pitch),
            full, pitch, 0)) { bridge->failed = true; return 8; }
    CgPacket packet{};
    uint32_t scheduled = 0;
    while (scheduled < bridge->options.max_iterations) {
        bridge->cg_guard = bridge->batch_size > 1;
        const uint32_t batch = (std::min)(bridge->batch_size,
            bridge->options.max_iterations - scheduled);
        for (uint32_t i = 0; i < batch; ++i) {
            Texture* hp = h_apply(bridge, &bridge->p[0], full, pitch);
            if (!hp || !control_step(bridge, dot_device(bridge, &bridge->p[0], hp, full, pitch),
                    full, pitch, 1) ||
                !controlled_vector(bridge, &bridge->p[0], &bridge->x[0], full, pitch, 2) ||
                !controlled_vector(bridge, hp, &bridge->g[0], full, pitch, 3) ||
                !control_step(bridge, dot_device(bridge, &bridge->g[0], &bridge->g[0], full, pitch),
                    full, pitch, 2) ||
                !controlled_vector(bridge, &bridge->g[0], &bridge->p[0], full, pitch, 4)) {
                bridge->failed = true; return 10;
            }
        }
        scheduled += batch;
        bridge->cg_guard = false;
        if (!read_control(bridge, &packet)) { bridge->failed = true; return 10; }
        if (packet.status >= 2 || !std::isfinite(packet.initial) ||
            !std::isfinite(packet.square)) {
            fail(packet.status == 2 ? "fusion initial norm is invalid" :
                "fusion CG lost positive definiteness or has non-finite arithmetic", packet.iterations);
            bridge->failed = true; return packet.status == 2 ? 9 : 11;
        }
        evidence->initial = std::sqrt(double(packet.initial));
        evidence->iterations = packet.iterations;
        if (packet.status != 1 && scheduled < bridge->options.max_iterations) continue;
        // Certify the original equation, not only the recursive CG residual.
        hx = h_apply(bridge, &bridge->x[0], full, pitch);
        if (!hx || !vector(bridge, rhs, hx, &bridge->g[0], full, pitch, -1)) {
            bridge->failed = true; return 10;
        }
        const double truth = gpu_dot(bridge, &bridge->g[0], &bridge->g[0], full, pitch);
        if (!std::isfinite(truth) || truth < 0) {
            fail("fusion true equation residual is invalid", 0);
            bridge->failed = true; return 11;
        }
        evidence->final_norm = std::sqrt(truth);
        evidence->relative = evidence->initial == 0 ? (truth == 0 ? 0 : INFINITY)
            : evidence->final_norm / evidence->initial;
        if (evidence->relative <= bridge->options.relative_tolerance) {
            evidence->converged = true; *out_x_index = 0; return 0;
        }
        if (scheduled < bridge->options.max_iterations) {
            if (!vector(bridge, &bridge->g[0], nullptr, &bridge->p[0], full, pitch, 1, true) ||
                !control_step(bridge, bridge->last_scalar, full, pitch, 3)) {
                bridge->failed = true; return 10;
            }
        }
    }
    std::snprintf(last_error, sizeof(last_error),
        "fusion CG did not converge in %u iterations: true relative residual %.8g > %.8g",
        evidence->iterations, evidence->relative, bridge->options.relative_tolerance);
    bridge->failed = true; return 12;
}

bool layer_vector(GpuFusionBridge* bridge, uint32_t index, Texture* a,
                  Texture* b, Texture* output, ID3D12Resource* full,
                  uint32_t pitch, float scale, uint32_t mode) {
    auto& layer = bridge->layers[index];
    return record(bridge, kVector, a, b, output, &layer.x, &layer.y, full, pitch,
        layer.width, layer.height, scale, mode);
}
bool observation_transpose(GpuFusionBridge* bridge, uint32_t which,
                           ID3D12Resource* full, uint32_t pitch) {
    if (!vector(bridge, &bridge->r0, nullptr, &bridge->h[0], full, pitch, 0, true)) return false;
    for (uint32_t i = 0; i < bridge->options.layer_count; ++i) {
        auto& layer = bridge->layers[i];
        Texture* input = which == 0 ? &layer.iterate : which == 1 ? &layer.direction : &layer.gradient;
        if (!adjoint(bridge, i, input, full, pitch) ||
            !record(bridge, kVector, &bridge->work, nullptr, &bridge->h[0],
                &bridge->baseline_x, &bridge->baseline_y, full, pitch,
                bridge->options.full_width, bridge->options.full_height,
                std::sqrt(layer.weight), 5u)) return false;
    }
    return true;
}
bool observation_apply(GpuFusionBridge* bridge, uint32_t which,
                       ID3D12Resource* full, uint32_t pitch) {
    if (!observation_transpose(bridge, which, full, pitch)) return false;
    for (uint32_t i = 0; i < bridge->options.layer_count; ++i) {
        auto& layer = bridge->layers[i];
        Texture* input = which == 0 ? &layer.iterate : &layer.direction;
        if (!down(bridge, i, &bridge->h[0], full, pitch) ||
            !layer_vector(bridge, i, input, nullptr, &layer.down, full, pitch,
                std::sqrt(layer.weight), 6u)) return false;
    }
    return true;
}
bool observation_dot(GpuFusionBridge* bridge, uint32_t mode,
                     ID3D12Resource* full, uint32_t pitch) {
    if (!control_step(bridge, bridge->last_scalar, full, pitch, 4)) return false;
    for (uint32_t i = 0; i < bridge->options.layer_count; ++i) {
        auto& layer = bridge->layers[i];
        Texture* a = mode == 1 ? &layer.direction : &layer.gradient;
        Texture* b = mode == 1 ? &layer.down : mode == 6 ? &layer.direction :
            bridge->jacobi ? &layer.down : &layer.gradient;
        if (!control_step(bridge, dot_device(bridge, a, b, full, pitch), full, pitch, 5)) return false;
    }
    return control_step(bridge, bridge->last_scalar, full, pitch, mode | 512u);
}
int32_t run_solve_observation(GpuFusionBridge* bridge, Texture* observed,
                            ID3D12Resource* full, uint32_t pitch,
                            SolveEvidence* evidence, uint32_t* out_x_index) {
    bridge->cg_guard = false;
    const uint32_t count = bridge->options.layer_count;
    if (!record(bridge, kCatmull, &observed[0], nullptr, &bridge->r0,
            &bridge->baseline_x, &bridge->baseline_y, full, pitch,
            bridge->options.full_width, bridge->options.full_height)) {
        bridge->failed = true; return 8;
    }
    for (uint32_t i = 0; i < count; ++i) {
        auto& layer = bridge->layers[i];
        if (!down(bridge, i, &bridge->r0, full, pitch) ||
            !layer_vector(bridge, i, &observed[i], &layer.down, &layer.gradient,
                full, pitch, -1, 0) ||
            !layer_vector(bridge, i, &layer.gradient, nullptr, &layer.direction,
                full, pitch, std::sqrt(layer.weight), 1) ||
            !layer_vector(bridge, i, &layer.direction, nullptr, &layer.gradient,
                full, pitch, 1, 1) ||
            !layer_vector(bridge, i, &layer.gradient, nullptr, &layer.iterate,
                full, pitch, 0, 1)) { bridge->failed = true; return 8; }
    }
    // All layers contribute to one A^T sum; no independent per-layer solve.
    if (!observation_transpose(bridge, 2, full, pitch) ||
        !control_step(bridge, dot_device(bridge, &bridge->h[0], &bridge->h[0], full, pitch),
            full, pitch, 0)) { bridge->failed = true; return 8; }
    for (uint32_t i = 0; i < count; ++i) {
        auto& layer = bridge->layers[i];
        if (!layer_vector(bridge, i, &layer.gradient, nullptr, &layer.direction, full,
                pitch, bridge->jacobi ? layer.weight : 1.f, bridge->jacobi ? 8u : 1u)) {
            bridge->failed = true; return 8;
        }
    }
    if (!observation_dot(bridge, 6, full, pitch)) { bridge->failed = true; return 8; }
    CgPacket packet{};
    uint32_t scheduled = 0;
    while (scheduled < bridge->options.max_iterations) {
        bridge->cg_guard = bridge->batch_size > 1;
        const uint32_t batch = (std::min)(bridge->batch_size,
            bridge->options.max_iterations - scheduled);
        for (uint32_t step = 0; step < batch; ++step) {
            if (!observation_apply(bridge, 1, full, pitch) ||
                !observation_dot(bridge, 1, full, pitch)) {
                bridge->failed = true; return 10;
            }
            for (uint32_t i = 0; i < count; ++i) {
                auto& layer = bridge->layers[i];
                if (!controlled_vector(bridge, &layer.direction, &layer.iterate, full, pitch, 2) ||
                    !controlled_vector(bridge, &layer.down, &layer.gradient, full, pitch, 3)) {
                    bridge->failed = true; return 10;
                }
                if (bridge->jacobi && !layer_vector(bridge, i, &layer.gradient, nullptr,
                        &layer.down, full, pitch, layer.weight, 8)) {
                    bridge->failed = true; return 10;
                }
            }
            if (!observation_dot(bridge, 2, full, pitch)) { bridge->failed = true; return 10; }
            for (uint32_t i = 0; i < count; ++i) {
                auto& layer = bridge->layers[i];
                if (!controlled_vector(bridge, bridge->jacobi ? &layer.down : &layer.gradient,
                        &layer.direction, full, pitch, 4)) { bridge->failed = true; return 10; }
            }
        }
        scheduled += batch;
        bridge->cg_guard = false;
        if (!read_control(bridge, &packet)) { bridge->failed = true; return 10; }
        if (packet.status >= 2 || !std::isfinite(packet.initial)) {
            fail(packet.status == 2 ? "fusion initial norm is invalid" :
                "fusion observation CG lost positive definiteness or has non-finite arithmetic",
                packet.iterations);
            bridge->failed = true; return packet.status == 2 ? 9 : 11;
        }
        evidence->initial = std::sqrt(double(packet.initial));
        evidence->iterations = packet.iterations;
        if (packet.status != 1 && scheduled < bridge->options.max_iterations) continue;
        // q=e-(beta I+A A^T)z, then original residual=b-Hr=A^T q.
        if (!observation_apply(bridge, 0, full, pitch)) { bridge->failed = true; return 10; }
        for (uint32_t i = 0; i < count; ++i) {
            auto& layer = bridge->layers[i];
            if (!layer_vector(bridge, i, &layer.down, nullptr, &layer.gradient, full, pitch, 1, 1) ||
                !down(bridge, i, &bridge->r0, full, pitch) ||
                !layer_vector(bridge, i, &observed[i], &layer.down, &layer.gradient, full, pitch,
                    std::sqrt(layer.weight), 7)) { bridge->failed = true; return 10; }
        }
        if (!observation_transpose(bridge, 2, full, pitch)) { bridge->failed = true; return 10; }
        const double truth = gpu_dot(bridge, &bridge->h[0], &bridge->h[0], full, pitch);
        if (!std::isfinite(truth) || truth < 0) {
            fail("fusion original equation residual is invalid", 0);
            bridge->failed = true; return 11;
        }
        evidence->final_norm = std::sqrt(truth);
        evidence->relative = evidence->initial == 0 ? (truth == 0 ? 0 : INFINITY)
            : evidence->final_norm/evidence->initial;
        if (evidence->relative <= bridge->options.relative_tolerance) {
            if (!observation_transpose(bridge, 0, full, pitch) ||
                !vector(bridge, &bridge->r0, &bridge->h[0], &bridge->x[0], full, pitch, 1)) {
                bridge->failed = true; return 10;
            }
            evidence->converged = true; *out_x_index = 0; return 0;
        }
        if (scheduled < bridge->options.max_iterations) {
            for (uint32_t i = 0; i < count; ++i) {
                auto& layer = bridge->layers[i];
                if (!layer_vector(bridge, i, &layer.gradient, nullptr, &layer.direction, full,
                        pitch, bridge->jacobi ? layer.weight : 1.f, bridge->jacobi ? 8u : 1u)) {
                    bridge->failed = true; return 10;
                }
            }
            if (!observation_dot(bridge, 6, full, pitch)) { bridge->failed = true; return 10; }
        }
    }
    std::snprintf(last_error, sizeof(last_error),
        "fusion observation CG did not converge in %u iterations: true relative residual %.8g > %.8g",
        evidence->iterations, evidence->relative, bridge->options.relative_tolerance);
    bridge->failed = true; return 12;
}

int32_t run_solve(GpuFusionBridge* bridge, Texture* observed,
                 ID3D12Resource* full, uint32_t pitch,
                 SolveEvidence* evidence, uint32_t* out_x_index) {
    if (bridge->observation_space)
        return run_solve_observation(bridge, observed, full, pitch, evidence, out_x_index);
    return bridge->gpu_cg ? run_solve_gpu(bridge, observed, full, pitch, evidence, out_x_index)
        : run_solve_legacy(bridge, observed, full, pitch, evidence, out_x_index);
}

}

extern "C" GPU_FUSION_API uint32_t gpu_fusion_abi_version(void) { return 1; }
extern "C" GPU_FUSION_API const char* gpu_fusion_last_error(void) {
    return last_error;
}
extern "C" GPU_FUSION_API int32_t gpu_fusion_get_solver_diagnostics(
    GpuFusionBridge* bridge, GpuFusionSolverDiagnostics* report) {
    if (!bridge || !report || report->struct_size != sizeof(*report)) return 1;
    *report = GpuFusionSolverDiagnostics{sizeof(*report), 1,
        bridge->observation_space ? (bridge->jacobi ? 4u : 3u) : bridge->gpu_cg ? 2u : 1u,
        bridge->gpu_cg ? bridge->batch_size : 1u,
        bridge->workspace_logical_bytes, bridge->workspace_committed_bytes,
        bridge->scalar_sync_count, bridge->cpu_wait_ms, {}, {}};
    for (uint32_t i = 0; i < bridge->options.layer_count; ++i) {
        report->horizontal_taps[i] = bridge->layers[i].x.taps;
        report->vertical_taps[i] = bridge->layers[i].y.taps;
    }
    return 0;
}
extern "C" GPU_FUSION_API int32_t gpu_fusion_get_transfer_counters(
    GpuFusionBridge* bridge, GpuFusionTransferCounters* counters) {
    if (!bridge || !counters || counters->struct_size != sizeof(*counters))
        return 1;
    *counters = GpuFusionTransferCounters{sizeof(*counters),
        bridge->process_count, bridge->full_device_copy_bytes,
        bridge->scalar_readback_bytes, 0};
    return 0;
}
extern "C" GPU_FUSION_API int32_t gpu_fusion_probe_operator(GpuFusionBridge* bridge,
    uint32_t layer_index, uint32_t transpose, void* texture,
    void* input_fence, uint64_t input_value,
    void** out_texture, void** out_fence, uint64_t* out_value) {
    if (out_texture) *out_texture = nullptr;
    if (out_fence) *out_fence = nullptr;
    if (out_value) *out_value = 0;
    if (!bridge || bridge->failed || layer_index >= bridge->options.layer_count ||
        transpose > 1 || !texture || !out_texture || !out_fence || !out_value) return 1;
    auto* resource = static_cast<ID3D12Resource*>(texture);
    const auto desc = resource->GetDesc();
    const auto& layer = bridge->layers[layer_index];
    const uint32_t width = transpose ? layer.width : bridge->options.full_width;
    const uint32_t height = transpose ? layer.height : bridge->options.full_height;
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Format != DXGI_FORMAT_R32G32B32A32_FLOAT || desc.Width != width ||
        desc.Height != height || desc.MipLevels != 1) return 2;
    if (input_fence && FAILED(bridge->queue->Wait(static_cast<ID3D12Fence*>(input_fence), input_value))) return 3;
    Texture input; input.resource = resource; input.width = width; input.height = height;
    bridge->cg_guard = false;
    if (!(transpose ? adjoint(bridge, layer_index, &input, bridge->partial.Get(), 0)
                    : down(bridge, layer_index, &input, bridge->partial.Get(), 0))) return 4;
    Texture* computed = transpose ? &bridge->work : &bridge->layers[layer_index].down;
    if (!bridge->probe_output.resource || bridge->probe_output.width != computed->width ||
        bridge->probe_output.height != computed->height) {
        if (!wait_gpu(bridge)) return 5;
        bridge->probe_output.resource.Reset();
        if (FAILED(make_texture(bridge, computed->width, computed->height,
            DXGI_FORMAT_R32G32B32A32_FLOAT, &bridge->probe_output))) return 5;
        bridge->probe_state = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    }
    transition(bridge->list.Get(), computed->resource.Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(bridge->list.Get(), bridge->probe_output.resource.Get(),
        bridge->probe_state, D3D12_RESOURCE_STATE_COPY_DEST);
    bridge->list->CopyResource(bridge->probe_output.resource.Get(), computed->resource.Get());
    transition(bridge->list.Get(), computed->resource.Get(),
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(bridge->list.Get(), bridge->probe_output.resource.Get(),
        D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    bridge->probe_state = D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (!flush(bridge)) return 6;
    *out_texture = bridge->probe_output.resource.Get();
    *out_fence = bridge->fence.Get(); *out_value = bridge->fence_value;
    return 0;
}
extern "C" GPU_FUSION_API void gpu_fusion_close(GpuFusionBridge* bridge) {
    if (!bridge) return;
    for (auto& transport : bridge->obs_color)
        if (transport) cuda_d3d12_color_destroy(transport);
    if (bridge->color) cuda_d3d12_color_destroy(bridge->color);
    if (bridge->event) CloseHandle(bridge->event);
    delete bridge;
}
/* One bridge, whichever colour contract it is opened for. The options mean the
 * same thing either way - the same float32 taps, centres and clamp edges, the
 * same Catmull baseline, beta, layer weights, tolerance and iteration budget -
 * and what the contract decides is the pixel format of the store, the row width
 * of the transport and which of the two entries may later feed this bridge. The
 * solver itself is shared, so opening cannot change a number in it. */
static int32_t create_bridge(void* device, const GpuFusionOptions* options,
                             bool hdr, GpuFusionBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    const uint32_t transport_pitch = !options ? 0u : (hdr
        ? pitch16_for(options->full_width) : pitch_for(options->full_width));
    if (!device || !options || !out ||
        options->struct_size != sizeof(GpuFusionOptions) ||
        options->full_width < 2 || options->full_height < 2 ||
        options->full_width > 16384 || options->full_height > 16384 ||
        options->layer_count < 2 || options->layer_count > 3 ||
        !std::isfinite(options->beta) || options->beta <= 0 ||
        !std::isfinite(options->residual_strength) ||
        options->residual_strength < 0 ||
        !std::isfinite(options->relative_tolerance) ||
        options->relative_tolerance <= 0 ||
        options->relative_tolerance >= 1 ||
        options->max_iterations < 1 || options->max_iterations > 1000 ||
        uint64_t(transport_pitch) * options->full_height > UINT32_MAX) {
        fail("invalid fusion options", 0); return 1;
    }
    for (uint32_t i = 0; i < options->layer_count; ++i) {
        const auto& layer = options->layers[i];
        if (layer.width < 1 || layer.height < 1 ||
            layer.width > options->full_width ||
            layer.height > options->full_height ||
            !std::isfinite(layer.weight) || layer.weight <= 0 ||
            !valid_axis(layer.horizontal, options->full_width,
                        layer.width, kMaxTaps) ||
            !valid_axis(layer.vertical, options->full_height,
                        layer.height, kMaxTaps)) {
            fail("invalid fusion layer plan", i); return 2;
        }
    }
    if (!valid_axis(options->baseline_horizontal, options->layers[0].width,
                    options->full_width, 4) ||
        !valid_axis(options->baseline_vertical, options->layers[0].height,
                    options->full_height, 4)) {
        fail("invalid Catmull baseline plan", 0); return 3;
    }
    auto* bridge = new (std::nothrow) GpuFusionBridge;
    if (!bridge) { fail("allocate fusion bridge", 0); return 4; }
    bridge->device = static_cast<ID3D12Device*>(device);
    bridge->options = *options;
    bridge->hdr = hdr;
    char mode[32] = {};
    GetEnvironmentVariableA("LUNARC_FUSION_SOLVER", mode, sizeof(mode));
    if (mode[0] && std::strcmp(mode,"legacy") && std::strcmp(mode,"primal") &&
        std::strcmp(mode,"observation")) {
        delete bridge; fail("unknown fusion solver selection",0); return 3;
    }
    const bool large_grid=uint64_t(options->full_width)*options->full_height >= uint64_t(3840)*2160;
    if (std::strcmp(mode, "legacy") == 0) bridge->gpu_cg = false;
    // Full-frame dot reductions still run their group barriers for guarded
    // iterations. Avoid dispatching extra large-grid reductions after convergence.
    if (large_grid)
        bridge->batch_size=1;
    // Observation CG remains explicit: lower vector length alone did not improve
    // measured throughput on this GPU. Preserve the coupled solver for research.
    bridge->observation_space = bridge->gpu_cg && std::strcmp(mode,"observation")==0;
    char preconditioner[32] = {};
    GetEnvironmentVariableA("LUNARC_FUSION_PRECONDITIONER", preconditioner, sizeof(preconditioner));
    if ((preconditioner[0] && std::strcmp(preconditioner,"jacobi") &&
         std::strcmp(preconditioner,"none")) ||
        (!bridge->observation_space && std::strcmp(preconditioner,"jacobi")==0)) {
        delete bridge; fail("invalid fusion preconditioner selection; Jacobi requires observation space",0); return 3;
    }
    bridge->jacobi = bridge->observation_space && std::strcmp(preconditioner, "jacobi") == 0;
    bridge->control_tolerance = options->relative_tolerance;

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
    if (SUCCEEDED(hr)) hr = bridge->device->CreateFence(0,
        D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(bridge->fence.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = upload_axis(bridge,
        options->baseline_horizontal, &bridge->baseline_x);
    if (SUCCEEDED(hr)) hr = upload_axis(bridge,
        options->baseline_vertical, &bridge->baseline_y);
    for (uint32_t i = 0; i < options->layer_count && SUCCEEDED(hr); ++i) {
        auto& layer = bridge->layers[i];
        layer.width = options->layers[i].width;
        layer.height = options->layers[i].height;
        layer.weight = options->layers[i].weight;
        hr = upload_axis(bridge, options->layers[i].horizontal, &layer.x);
        if (SUCCEEDED(hr)) hr = upload_axis(bridge,
            options->layers[i].vertical, &layer.y);
    }
    if (SUCCEEDED(hr) && bridge->observation_space) {
        double row_bound = 0, column_bound = 0, max_diagonal = bridge->jacobi ? 0 : 1;
        for (uint32_t i = 0; i < options->layer_count; ++i) {
            const auto& layer = bridge->layers[i];
            const double weight = std::sqrt(double(layer.weight));
            row_bound = (std::max)(row_bound, weight*layer.x.max_row_abs*layer.y.max_row_abs);
            column_bound += weight*layer.x.max_column_abs*layer.y.max_column_abs;
            if (bridge->jacobi) max_diagonal = (std::max)(max_diagonal,
                double(options->beta)+layer.weight*layer.x.max_norm_square*layer.y.max_norm_square);
        }
        const double bound = (std::max)(1e-30, std::sqrt(row_bound*column_bound*max_diagonal));
        bridge->control_tolerance = static_cast<float>(options->relative_tolerance / bound);
    }
    if (SUCCEEDED(hr)) hr = make_resources(bridge);
    if (SUCCEEDED(hr)) hr = make_kernels(bridge);
    if (FAILED(hr)) {
        if (!last_error[0]) fail("initialize fusion bridge", hr);
        gpu_fusion_close(bridge); return 5;
    }
    bridge->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!bridge->event) { fail("fusion CreateEvent", GetLastError());
        gpu_fusion_close(bridge); return 6; }
    ID3D12DescriptorHeap* heaps[] = {bridge->descriptors.Get()};
    bridge->list->SetDescriptorHeaps(1, heaps);
    bridge->list->SetComputeRootSignature(bridge->root.Get());
    *out = bridge;
    return 0;
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_open(void* device,
    const GpuFusionOptions* options, GpuFusionBridge** out) {
    return create_bridge(device, options, false, out);
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_update_parameters(GpuFusionBridge* bridge,
    const float* weights, uint32_t layer_count, float beta, float strength,
    float tolerance, uint32_t max_iterations) {
    last_error[0] = 0;
    if (!bridge || bridge->failed || !weights || layer_count != bridge->options.layer_count ||
        !std::isfinite(beta) || beta <= 0 || !std::isfinite(strength) || strength < 0 ||
        !std::isfinite(tolerance) || tolerance <= 0 || tolerance >= 1 ||
        max_iterations < 1 || max_iterations > 1000) {
        fail("invalid fusion runtime parameters", 0); return 1;
    }
    for (uint32_t i = 0; i < layer_count; ++i) {
        if (!std::isfinite(weights[i]) || weights[i] <= 0) {
            fail("invalid fusion runtime weight", i); return 1;
        }
    }
    bridge->options.beta = beta;
    bridge->options.residual_strength = strength;
    bridge->options.relative_tolerance = tolerance;
    bridge->options.max_iterations = max_iterations;
    bridge->control_tolerance = tolerance;
    double row_bound = 0, column_bound = 0, max_diagonal = bridge->jacobi ? 0 : 1;
    for (uint32_t i = 0; i < layer_count; ++i) {
        auto& layer = bridge->layers[i];
        layer.weight = bridge->options.layers[i].weight = weights[i];
        if (bridge->observation_space) {
            const double weight = std::sqrt(double(layer.weight));
            row_bound = (std::max)(row_bound, weight*layer.x.max_row_abs*layer.y.max_row_abs);
            column_bound += weight*layer.x.max_column_abs*layer.y.max_column_abs;
            if (bridge->jacobi) max_diagonal = (std::max)(max_diagonal,
                double(beta)+layer.weight*layer.x.max_norm_square*layer.y.max_norm_square);
        }
    }
    if (bridge->observation_space) {
        const double bound = (std::max)(1e-30, std::sqrt(row_bound*column_bound*max_diagonal));
        bridge->control_tolerance = static_cast<float>(tolerance / bound);
    }
    return 0;
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_process(GpuFusionBridge* bridge,
    void* cuda_context, uint64_t full_rgba_ptr, size_t full_pitch,
    const GpuFusionObservation* observations, uint32_t observation_count,
    void** out_rgba_texture, GpuFusionSolveReport* report) {
    last_error[0] = 0;
    if (out_rgba_texture) *out_rgba_texture = nullptr;
    if (!bridge || bridge->failed || !cuda_context || !full_rgba_ptr ||
        full_pitch < size_t(bridge->options.full_width) * 4 ||
        !observations || observation_count != bridge->options.layer_count ||
        !out_rgba_texture || !report ||
        report->struct_size != sizeof(GpuFusionSolveReport)) {
        fail("invalid fusion process arguments", 0); return 1;
    }
    /* Refused before a sample is read: a bridge opened for scRGB has an FP16
     * store and a width * 8 transport, and moving half samples through this
     * entry would read the two bytes of one component as the four of a RGBA8
     * pixel. There is no fallback between the contracts, in either direction. */
    if (bridge->hdr) {
        fail("gpu_fusion_process moves RGBA8 planes and composites to bytes in "
             "[0,255]; this bridge was opened for linear scRGB, so use "
             "gpu_fusion_process_hdr", 0);
        return 1;
    }
    const uint32_t count = bridge->options.layer_count;
    std::array<Texture, 3> observed;
    for (uint32_t i = 0; i < count; ++i) {
        const auto& descriptor = observations[i];
        if (!descriptor.signed_texture || !descriptor.producer_fence) {
            fail("missing signed residual observation", i); return 2;
        }
        auto* resource = static_cast<ID3D12Resource*>(descriptor.signed_texture);
        ComPtr<ID3D12Device> owner;
        HRESULT hr = resource->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
        const auto dims = resource->GetDesc();
        if (FAILED(hr) || owner.Get() != bridge->device.Get() ||
            dims.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
            dims.Width != bridge->layers[i].width ||
            dims.Height != bridge->layers[i].height ||
            dims.Format != DXGI_FORMAT_R32G32B32A32_FLOAT) {
            fail("signed residual resource mismatch", i); return 3;
        }
        observed[i].resource = resource;
        observed[i].width = bridge->layers[i].width;
        observed[i].height = bridge->layers[i].height;
    }
    if (!bridge->color) {
        if (cuda_d3d12_color_create(bridge->device.Get(), cuda_context,
            bridge->options.full_width, bridge->options.full_height,
            pitch_for(bridge->options.full_width), &bridge->color)) {
            std::snprintf(last_error, sizeof(last_error), "fusion full color: %s",
                          nvof_d3d12_last_error()); return 4;
        }
        bridge->cuda_context = cuda_context;
    }
    if (bridge->cuda_context != cuda_context) {
        fail("fusion CUDA context changed", 0); return 5;
    }
    uint64_t full_ready = 0;
    if (cuda_d3d12_color_copy(bridge->color, full_rgba_ptr, full_pitch,
                             &full_ready)) {
        std::snprintf(last_error, sizeof(last_error), "fusion full copy: %s",
                      nvof_d3d12_last_error());
        bridge->failed = true; return 6;
    }
    bridge->full_device_copy_bytes +=
        uint64_t(bridge->options.full_width) * bridge->options.full_height * 4;
    void *full_raw = nullptr, *full_fence = nullptr;
    cuda_d3d12_color_get_resources(bridge->color, &full_raw, &full_fence);
    auto* full = static_cast<ID3D12Resource*>(full_raw);
    HRESULT hr = bridge->queue->Wait(static_cast<ID3D12Fence*>(full_fence),
                                     full_ready);
    for (uint32_t i = 0; i < count && SUCCEEDED(hr); ++i)
        hr = bridge->queue->Wait(static_cast<ID3D12Fence*>(
            observations[i].producer_fence),
            observations[i].producer_fence_value);
    if (FAILED(hr)) { fail("fusion producer wait", hr);
        bridge->failed = true; return 7; }
    auto* list = bridge->list.Get();
    transition(list, full, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for (uint32_t i = 0; i < count; ++i)
        transition(list, observed[i].resource.Get(),
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const uint32_t width = bridge->options.full_width;
    const uint32_t height = bridge->options.full_height;
    const uint32_t pitch = pitch_for(width);
    SolveEvidence evidence;
    uint32_t x_index = 0;
    const int32_t status = run_solve(bridge, observed.data(), full, pitch,
                                     &evidence, &x_index);
    report->iterations = evidence.iterations;
    report->initial_residual_norm = static_cast<float>(evidence.initial);
    report->final_residual_norm = static_cast<float>(evidence.final_norm);
    report->relative_residual = static_cast<float>(evidence.relative);
    report->converged = evidence.converged ? 1u : 0u;
    if (status) return status;
    if (!record(bridge, kComposite, &bridge->x[x_index], nullptr,
                &bridge->final, &bridge->baseline_x,
                &bridge->baseline_y, full, pitch, width, height,
                bridge->options.residual_strength)) {
        bridge->failed = true; return 13;
    }
    for (uint32_t i = 0; i < count; ++i)
        transition(list, observed[i].resource.Get(),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list, full, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    if (!flush(bridge)) { bridge->failed = true; return 14; }
    *out_rgba_texture = bridge->final.resource.Get();
    bridge->process_count++;
    return 0;
}

/* --- Typed HDR (scRGB) path ---------------------------------------------
 *
 * The stage codes gpu_fusion_process_hdr returns are the ones its own header
 * lists, and 8 to 12 are the solver's codes because run_solve is the solver both
 * entries call. */

extern "C" GPU_FUSION_API uint32_t gpu_fusion_hdr_abi_version(void) {
    return GPU_FUSION_HDR_ABI_VERSION;
}

namespace {
/* What the transport says it moves, checked against what this contract needs.
 * The older entry points all moved width * 4 bytes a row; a half plane's row is
 * width * 8, and a transport that quietly took the narrower one would deliver
 * every row with its tail half missing and still report success. So the bridge
 * asks the transport its own geometry after creating it instead of assuming the
 * numbers it was handed. */
bool half_row_transport(CudaD3D12ColorBridge* transport, uint32_t width,
                        uint32_t height, const char* what) {
    CudaD3D12ColorLayout layout{sizeof(CudaD3D12ColorLayout)};
    if (cuda_d3d12_color_get_layout(transport, &layout)) {
        std::snprintf(last_error, sizeof(last_error), "%s transport layout: %s",
                      what, nvof_d3d12_last_error());
        return false;
    }
    if (layout.pixel_bytes != 8 || layout.width != width ||
        layout.height != height || layout.bytes_per_row != width * 8u ||
        (layout.row_pitch & 255u)) {
        std::snprintf(last_error, sizeof(last_error),
            "%s transport does not describe an 8-byte scRGB row: it reports %u "
            "bytes per sample, %u bytes of payload, %u x %u at pitch %u",
            what, layout.pixel_bytes, layout.bytes_per_row, layout.width,
            layout.height, layout.row_pitch);
        return false;
    }
    return true;
}
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_open_hdr(void* device,
    const GpuFusionOptions* options, const GpuFusionHdrOptions* hdr_options,
    GpuFusionBridge** out) {
    last_error[0] = 0;
    if (out) *out = nullptr;
    /* A struct guessed from a stale header is refused by size and version before
     * a single byte of it is read past its own declared end. */
    if (!hdr_options || hdr_options->struct_size != sizeof(*hdr_options) ||
        hdr_options->abi_version != GPU_FUSION_HDR_ABI_VERSION ||
        hdr_options->reserved) {
        fail("the scRGB options are not this build's: struct size, ABI version "
             "or reserved field does not match", 0);
        return 1;
    }
    /* There is no 4 here on purpose. Accepting it would mean an HDR bridge that
     * reads half samples out of an RGBA8 row, which is the SDR contract with a
     * different name on it. */
    if (hdr_options->pixel_bytes != 8) {
        fail("the scRGB contract carries R16G16B16A16_FLOAT samples, so "
             "pixel_bytes has to be 8; an RGBA8 chain is gpu_fusion_open and "
             "gpu_fusion_process", 0);
        return 1;
    }
    const uint32_t capabilities = cuda_d3d12_color_row_capabilities();
    if (!(capabilities & CUDA_D3D12_COLOR_ROW_RGBA16F)) {
        std::snprintf(last_error, sizeof(last_error),
            "the CUDA/D3D12 interop build in use cannot transport an RGBA16F row "
            "(row capabilities 0x%x): the scRGB contract is refused rather than "
            "moved through a width * 4 row", capabilities);
        return 3;
    }
    return create_bridge(device, options, true, out);
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_process_hdr(GpuFusionBridge* bridge,
    void* cuda_context, uint64_t full_scrgb_ptr, size_t full_scrgb_pitch,
    const GpuFusionHdrObservation* observations, uint32_t observation_count,
    void** out_texture, void** out_fence, uint64_t* out_fence_value,
    GpuFusionHdrReport* report) {
    last_error[0] = 0;
    if (out_texture) *out_texture = nullptr;
    if (out_fence) *out_fence = nullptr;
    if (out_fence_value) *out_fence_value = 0;
    if (!bridge || bridge->failed || !bridge->hdr || !cuda_context ||
        !full_scrgb_ptr ||
        full_scrgb_pitch < size_t(bridge->options.full_width) * 8 ||
        !observations || observation_count != bridge->options.layer_count ||
        !out_texture || !out_fence || !out_fence_value || !report ||
        report->struct_size != sizeof(GpuFusionHdrReport) ||
        report->abi_version != GPU_FUSION_HDR_ABI_VERSION) {
        fail("invalid scRGB fusion arguments: a bridge opened by "
             "gpu_fusion_open_hdr needs full-extent R16G16B16A16_FLOAT rows (at "
             "least width * 8 bytes of pitch), one observation per layer, a "
             "sized report and all three output slots", 0);
        return 1;
    }
    const uint32_t count = bridge->options.layer_count;
    const uint32_t width = bridge->options.full_width;
    const uint32_t height = bridge->options.full_height;
    const uint32_t pitch = pitch16_for(width);
    const uint64_t scalars_before = bridge->scalar_readback_bytes;
    /* Each residual is named by the extent the layer measured it at. A plane of
     * some other size is not that layer's r_i, and reading it as one would
     * produce a number that looks valid. */
    for (uint32_t i = 0; i < count; ++i) {
        const auto& plane = observations[i];
        const auto& layer = bridge->layers[i];
        if (!plane.residual_ptr || plane.residual_pitch < layer.width * 8u ||
            plane.width != layer.width || plane.height != layer.height) {
            std::snprintf(last_error, sizeof(last_error),
                "scRGB observation %u does not describe that layer's residual "
                "plane: it is %u x %u with %u bytes in a row and %s, where this "
                "layer's own gpu_hdr_observe plane is %u x %u with at least %u",
                i, plane.width, plane.height, plane.residual_pitch,
                plane.residual_ptr ? "a pointer" : "no pointer",
                layer.width, layer.height, layer.width * 8u);
            return 2;
        }
    }
    if (!bridge->color) {
        if (cuda_d3d12_color_create2(bridge->device.Get(), cuda_context, width,
            height, pitch, 8, 0, &bridge->color)) {
            std::snprintf(last_error, sizeof(last_error),
                "scRGB source transport: %s", nvof_d3d12_last_error());
            return 3;
        }
        if (!half_row_transport(bridge->color, width, height, "source")) {
            cuda_d3d12_color_destroy(bridge->color);
            bridge->color = nullptr;
            return 3;
        }
        bridge->cuda_context = cuda_context;
    }
    if (bridge->cuda_context != cuda_context) {
        fail("scRGB fusion CUDA context changed", 0); return 4;
    }
    uint64_t full_ready = 0;
    if (cuda_d3d12_color_copy(bridge->color, full_scrgb_ptr, full_scrgb_pitch,
                             &full_ready)) {
        std::snprintf(last_error, sizeof(last_error), "scRGB source copy: %s",
                      nvof_d3d12_last_error());
        bridge->failed = true; return 5;
    }
    bridge->full_device_copy_bytes += uint64_t(width) * height * 8;
    /* The observation copies are device-to-device and each transport retires
     * its own work inside its call, so a plane the producer has already retired
     * - which is what gpu_hdr_observe hands back - only has to survive this
     * call. A producer still running is the caller's to complete first; there
     * is no fence here to pretend otherwise. */
    std::array<uint64_t, 3> ready{};
    std::array<void*, 3> raw{}, fence{};
    uint64_t observation_bytes = 0;
    for (uint32_t i = 0; i < count; ++i) {
        auto& layer = bridge->layers[i];
        if (!bridge->obs_color[i]) {
            if (cuda_d3d12_color_create2(bridge->device.Get(), cuda_context,
                layer.width, layer.height, pitch16_for(layer.width), 8, 0,
                &bridge->obs_color[i])) {
                std::snprintf(last_error, sizeof(last_error),
                    "scRGB observation %u transport: %s",
                    i, nvof_d3d12_last_error());
                return 3;
            }
            if (!half_row_transport(bridge->obs_color[i], layer.width,
                                    layer.height, "observation")) return 3;
        }
        if (cuda_d3d12_color_copy(bridge->obs_color[i],
                observations[i].residual_ptr, observations[i].residual_pitch,
                &ready[i])) {
            std::snprintf(last_error, sizeof(last_error),
                "scRGB observation %u copy: %s", i, nvof_d3d12_last_error());
            bridge->failed = true; return 6;
        }
        const uint64_t bytes = uint64_t(layer.width) * layer.height * 8;
        bridge->observation_device_copy_bytes += bytes;
        observation_bytes += bytes;
    }
    void* full_raw = nullptr;
    void* full_fence = nullptr;
    cuda_d3d12_color_get_resources(bridge->color, &full_raw, &full_fence);
    auto* full = static_cast<ID3D12Resource*>(full_raw);
    for (uint32_t i = 0; i < count; ++i)
        cuda_d3d12_color_get_resources(bridge->obs_color[i], &raw[i],
                                       &fence[i]);
    HRESULT hr = bridge->queue->Wait(static_cast<ID3D12Fence*>(full_fence),
                                     full_ready);
    for (uint32_t i = 0; i < count && SUCCEEDED(hr); ++i)
        hr = bridge->queue->Wait(static_cast<ID3D12Fence*>(fence[i]), ready[i]);
    if (FAILED(hr)) { fail("scRGB fusion producer wait", hr);
        bridge->failed = true; return 7; }
    auto* list = bridge->list.Get();
    transition(list, full, D3D12_RESOURCE_STATE_COMMON,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for (uint32_t i = 0; i < count; ++i) {
        transition(list, static_cast<ID3D12Resource*>(raw[i]),
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }
    /* Widen, do not reinterpret: each half plane becomes the solver's own FP32
     * texture through one pass per layer, so every kernel from here on is the
     * kernel the SDR path runs on the same float32 plans. */
    std::array<Texture, 3> observed;
    for (uint32_t i = 0; i < count; ++i) {
        if (!unpack_half(bridge, i, static_cast<ID3D12Resource*>(raw[i]),
                         pitch16_for(bridge->layers[i].width))) {
            bridge->failed = true; return 8;
        }
        observed[i] = bridge->layers[i].observed;
    }
    SolveEvidence evidence;
    uint32_t x_index = 0;
    const int32_t status = run_solve(bridge, observed.data(), full, pitch,
                                     &evidence, &x_index);
    report->iterations = evidence.iterations;
    report->initial_residual_norm = static_cast<float>(evidence.initial);
    report->final_residual_norm = static_cast<float>(evidence.final_norm);
    report->relative_residual = static_cast<float>(evidence.relative);
    report->converged = evidence.converged ? 1u : 0u;
    if (status) return status;
    if (!record(bridge, kCompositeHdr, &bridge->x[x_index], nullptr,
                &bridge->final, &bridge->baseline_x, &bridge->baseline_y, full,
                pitch, width, height,
                bridge->options.residual_strength)) {
        bridge->failed = true; return 13;
    }
    /* The store's own count of what it could not hold, read back as the scalars
     * it is: one number per group, reduced here, and never a picture. */
    const double reduced = reduce_partial(bridge);
    if (!std::isfinite(reduced)) {
        if (!last_error[0])
            fail("scRGB composite counted fallback is unreadable", 0);
        bridge->failed = true; return 14;
    }
    /* A group writes 1e30 when even the master sample it fell back to is
     * outside the half grid, which means the source plane itself carried a
     * non-finite or unrepresentable component. Storing that would be a picture
     * that reports success while holding a value no encode can read, so the
     * call fails and the bridge is retired with it. */
    if (reduced > 1.0e20) {
        fail("the fused scRGB picture fell back to a master sample the half grid "
             "cannot hold: the source plane itself carries a non-finite or "
             "out-of-range component, so no composite is handed back", 0);
        bridge->failed = true; return 15;
    }
    const uint32_t fell_back = static_cast<uint32_t>(std::llround(reduced));
    report->unrepresentable_components = fell_back;
    bridge->unrepresentable_components += fell_back;
    for (uint32_t i = 0; i < count; ++i)
        transition(list, static_cast<ID3D12Resource*>(raw[i]),
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    transition(list, full, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COMMON);
    if (!flush(bridge)) { bridge->failed = true; return 16; }
    const auto store = bridge->final.resource->GetDesc();
    /* Asked of the texture, not asserted: the format and extent the caller
     * validates against its own enum are the ones this store actually has, and
     * the state is the one the two transitions above left it in - which is the
     * state make_texture created a store in and the only state it is ever put
     * in here. */
    report->output_format = static_cast<uint32_t>(store.Format);
    report->output_width = bridge->final.width;
    report->output_height = bridge->final.height;
    report->output_state = uint32_t(D3D12_RESOURCE_STATE_COPY_SOURCE);
    report->source_device_copy_bytes = uint64_t(width) * height * 8;
    report->observation_device_copy_bytes = observation_bytes;
    report->scalar_readback_bytes =
        bridge->scalar_readback_bytes - scalars_before;
    *out_texture = bridge->final.resource.Get();
    /* The fence the solver's own queue signalled for the work that wrote this
     * texture, and the value that names it. Another queue on this device waits
     * it before it takes the texture, which is what gpu_hdr_set_composite does;
     * nothing here forges a completion the queue never reported. */
    *out_fence = bridge->fence.Get();
    *out_fence_value = bridge->fence_value;
    bridge->process_count++;
    return 0;
}

extern "C" GPU_FUSION_API int32_t gpu_fusion_get_hdr_counters(
    GpuFusionBridge* bridge, GpuFusionHdrCounters* counters) {
    last_error[0] = 0;
    if (!bridge || !counters || counters->struct_size != sizeof(*counters) ||
        counters->abi_version != GPU_FUSION_HDR_ABI_VERSION) {
        fail("gpu_fusion_get_hdr_counters needs an open bridge and a counters "
             "struct of this build's size and ABI version", 0);
        return 1;
    }
    if (!bridge->hdr) {
        fail("this bridge was opened for the RGBA8 contract and has no scRGB "
             "totals; gpu_fusion_get_transfer_counters reports its own", 0);
        return 2;
    }
    *counters = GpuFusionHdrCounters{sizeof(*counters),
        GPU_FUSION_HDR_ABI_VERSION, bridge->process_count,
        bridge->full_device_copy_bytes, bridge->observation_device_copy_bytes,
        bridge->scalar_readback_bytes, bridge->unrepresentable_components};
    return 0;
}
