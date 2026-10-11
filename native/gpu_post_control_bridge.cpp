#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gpu_post_control_bridge.h"
#include "gpu_post_control_shader.h"
#include "nvof_d3d12_interop.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>

using Microsoft::WRL::ComPtr;
static_assert(sizeof(GpuPostControlOptions) == 84, "ABI options layout changed");
static_assert(sizeof(GpuPostCandidate) == 48, "ABI candidate layout changed");
static_assert(sizeof(GpuPostControlCounters) == 48, "ABI counters layout changed");
static_assert(sizeof(size_t) == 8, "ABI 1 is Windows x64");

namespace {
thread_local char last_error[1024] = "";
void fail(const char* where, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", where,
                  static_cast<unsigned long>(code));
}
uint32_t row_pitch(uint32_t width, uint32_t pixel_bytes) {
    const uint64_t bytes = uint64_t(width) * pixel_bytes;
    return uint32_t((bytes + 255u) & ~uint64_t(255u));
}
DXGI_FORMAT pixel_format(uint32_t bytes) {
    return bytes == 4 ? DXGI_FORMAT_R8G8B8A8_UNORM :
                        DXGI_FORMAT_R16G16B16A16_FLOAT;
}
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
void uav_barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource = resource;
    list->ResourceBarrier(1, &b);
}
bool finite_options(const GpuPostControlOptions& o) {
    const float gains[] = {o.total_gain,o.lightness_gain,o.chroma_gain,
        o.low_frequency_gain,o.high_frequency_gain};
    for (float v : gains) if (!std::isfinite(v) || std::abs(v) > 32.0f) return false;
    const float protections[] = {o.hue_protection,o.dark_protection,
        o.highlight_protection,o.soft_compression,o.roi_feather};
    for (float v : protections) if (!std::isfinite(v) || v < 0.0f || v > 1.0f)
        return false;
    for (float v : o.roi_bounds) if (!std::isfinite(v) || v < 0.0f || v > 1.0f)
        return false;
    if (o.roi_kind > GPU_POST_ROI_ELLIPSE || o.debug_mode > GPU_POST_DEBUG_WEIGHT)
        return false;
    return o.roi_kind == GPU_POST_ROI_NONE ||
        (o.roi_bounds[0] < o.roi_bounds[2] &&
         o.roi_bounds[1] < o.roi_bounds[3]);
}
HRESULT make_buffer(ID3D12Device* device, uint64_t bytes, D3D12_HEAP_TYPE type,
                    D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                    ComPtr<ID3D12Resource>* out) {
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
    desc.Flags = flags;
    return device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
        state,nullptr,IID_PPV_ARGS(out->GetAddressOf()));
}
HRESULT compile_entry(const char* entry, ComPtr<ID3DBlob>* code) {
    ComPtr<ID3DBlob> errors;
    HRESULT hr = D3DCompile(gpu_post_control_shader,
        sizeof(gpu_post_control_shader)-1,"gpu_post_control",nullptr,nullptr,
        entry,"cs_5_0",D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS,
        0,code->GetAddressOf(),errors.GetAddressOf());
    if (FAILED(hr) && errors)
        std::snprintf(last_error,sizeof(last_error),"D3DCompile %s: %s",entry,
            static_cast<const char*>(errors->GetBufferPointer()));
    return hr;
}
}

struct GpuPostControlBridge {
    uint32_t width=0,height=0,pixel_bytes=0,pitch=0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> output,analysis,fallback,readback;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> analyze_pipeline,apply_pipeline;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    CudaD3D12ColorBridge* source_bridge=nullptr;
    CudaD3D12ColorBridge* candidate_bridge=nullptr;
    void* cuda_context=nullptr;
    HANDLE event=nullptr;
    uint64_t fence_value=0;
    uint64_t process_count=0,source_transport_bytes=0,candidate_transport_bytes=0;
    uint64_t scalar_readback_bytes=0,unrepresentable_pixels=0;
    bool failed=false;
    std::mutex mutex;
};

namespace {
bool wait_bridge(GpuPostControlBridge* b) {
    if (b->fence->GetCompletedValue() >= b->fence_value) return true;
    HRESULT hr=b->fence->SetEventOnCompletion(b->fence_value,b->event);
    if (FAILED(hr)) { fail("post-control fence event",hr); return false; }
    if (WaitForSingleObject(b->event,30000)!=WAIT_OBJECT_0) {
        fail("post-control fence timeout",ERROR_TIMEOUT); return false;
    }
    return true;
}
HRESULT make_resources(GpuPostControlBridge* b) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC texture{};
    texture.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texture.Width=b->width; texture.Height=b->height;
    texture.DepthOrArraySize=1; texture.MipLevels=1;
    texture.Format=pixel_format(b->pixel_bytes);
    texture.SampleDesc.Count=1;
    texture.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr=b->device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
        &texture,D3D12_RESOURCE_STATE_COPY_SOURCE,nullptr,
        IID_PPV_ARGS(b->output.GetAddressOf()));
    if (FAILED(hr)) return hr;
    hr=make_buffer(b->device.Get(),16,D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,&b->analysis);
    if (SUCCEEDED(hr)) hr=make_buffer(b->device.Get(),16,D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,&b->fallback);
    if (FAILED(hr)) return hr;
    return make_buffer(b->device.Get(),32,D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,&b->readback);
}
HRESULT make_pipeline(GpuPostControlBridge* b) {
    ComPtr<ID3DBlob> analyze,apply;
    HRESULT hr=compile_entry("Analyze",&analyze);
    if (SUCCEEDED(hr)) hr=compile_entry("Apply",&apply);
    if (FAILED(hr)) return hr;
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors=3;
    ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors=3;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges=1;
    params[0].DescriptorTable.pDescriptorRanges=&ranges[0];
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges=1;
    params[1].DescriptorTable.pDescriptorRanges=&ranges[1];
    params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister=0;
    params[2].Constants.Num32BitValues=sizeof(GpuPostControlOptions)/4;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters=3; root_desc.pParameters=params;
    ComPtr<ID3DBlob> root_blob,errors;
    hr=D3D12SerializeRootSignature(&root_desc,D3D_ROOT_SIGNATURE_VERSION_1,
        root_blob.GetAddressOf(),errors.GetAddressOf());
    if (FAILED(hr)) {
        if (errors) std::snprintf(last_error,sizeof(last_error),"post-control root: %s",
            static_cast<const char*>(errors->GetBufferPointer()));
        return hr;
    }
    hr=b->device->CreateRootSignature(0,root_blob->GetBufferPointer(),
        root_blob->GetBufferSize(),IID_PPV_ARGS(b->root.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
    pso.pRootSignature=b->root.Get();
    pso.CS={analyze->GetBufferPointer(),analyze->GetBufferSize()};
    hr=b->device->CreateComputePipelineState(&pso,
        IID_PPV_ARGS(b->analyze_pipeline.GetAddressOf()));
    if (FAILED(hr)) return hr;
    pso.CS={apply->GetBufferPointer(),apply->GetBufferSize()};
    hr=b->device->CreateComputePipelineState(&pso,
        IID_PPV_ARGS(b->apply_pipeline.GetAddressOf()));
    if (FAILED(hr)) return hr;
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors=6; heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    hr=b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->descriptors.GetAddressOf()));
    if (FAILED(hr)) return hr;
    const UINT step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr+=3*step;
    D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
    view.Format=pixel_format(b->pixel_bytes);
    view.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
    b->device->CreateUnorderedAccessView(b->output.Get(),nullptr,&view,cpu);
    cpu.ptr+=step;
    view={}; view.Format=DXGI_FORMAT_R32_TYPELESS;
    view.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    view.Buffer.NumElements=4; view.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(b->fallback.Get(),nullptr,&view,cpu);
    cpu.ptr+=step;
    b->device->CreateUnorderedAccessView(b->analysis.Get(),nullptr,&view,cpu);
    return S_OK;
}
void set_raw_srv(GpuPostControlBridge* b,ID3D12Resource* resource,uint32_t slot) {
    auto cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    cpu.ptr+=slot*b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format=DXGI_FORMAT_R32_TYPELESS;
    view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.NumElements=b->pitch*b->height/4;
    view.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(resource,&view,cpu);
}
void set_stats_srv(GpuPostControlBridge* b) {
    auto cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    const UINT step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    cpu.ptr+=2*step;
    D3D12_SHADER_RESOURCE_VIEW_DESC view{};
    view.Format=DXGI_FORMAT_R32_TYPELESS;
    view.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
    view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    view.Buffer.NumElements=4; view.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(b->analysis.Get(),&view,cpu);
}
void set_compute_bindings(GpuPostControlBridge* b,
                          const GpuPostControlOptions* options,
                          ID3D12PipelineState* pipeline) {
    ID3D12DescriptorHeap* heaps[]={b->descriptors.Get()};
    b->list->SetDescriptorHeaps(1,heaps);
    b->list->SetComputeRootSignature(b->root.Get());
    b->list->SetPipelineState(pipeline);
    const UINT step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto gpu=b->descriptors->GetGPUDescriptorHandleForHeapStart();
    b->list->SetComputeRootDescriptorTable(0,gpu);
    gpu.ptr+=3*step;
    b->list->SetComputeRootDescriptorTable(1,gpu);
    b->list->SetComputeRoot32BitConstants(2,
        sizeof(GpuPostControlOptions)/4,options,0);
}
bool reset_list(GpuPostControlBridge* b) {
    HRESULT hr=b->allocator->Reset();
    if (SUCCEEDED(hr)) hr=b->list->Reset(b->allocator.Get(),nullptr);
    if (FAILED(hr)) { fail("post-control list reset",hr); return false; }
    return true;
}
bool execute_wait(GpuPostControlBridge* b,const char* label) {
    HRESULT hr=b->list->Close();
    if (FAILED(hr)) { fail(label,hr); return false; }
    ID3D12CommandList* lists[]={b->list.Get()};
    b->queue->ExecuteCommandLists(1,lists);
    hr=b->queue->Signal(b->fence.Get(),++b->fence_value);
    if (FAILED(hr)) { fail("post-control queue signal",hr); return false; }
    return wait_bridge(b);
}
}

extern "C" GPU_POST_CONTROL_API uint32_t gpu_post_control_abi_version(void) {
    return GPU_POST_CONTROL_ABI_VERSION;
}
extern "C" GPU_POST_CONTROL_API const char* gpu_post_control_last_error(void) {
    return last_error;
}
extern "C" GPU_POST_CONTROL_API void gpu_post_control_close(
    GpuPostControlBridge* b) {
    if (!b) return;
    if (b->queue && b->fence && b->event) {
        if (SUCCEEDED(b->queue->Signal(b->fence.Get(),++b->fence_value)))
            wait_bridge(b);
    }
    if (b->event) CloseHandle(b->event);
    if (b->candidate_bridge) cuda_d3d12_color_destroy(b->candidate_bridge);
    if (b->source_bridge) cuda_d3d12_color_destroy(b->source_bridge);
    delete b;
}
extern "C" GPU_POST_CONTROL_API int32_t gpu_post_control_open(
    void* raw_device,void* cuda_context,uint32_t width,uint32_t height,
    uint32_t pixel_bytes,GpuPostControlBridge** out) {
    last_error[0]=0;
    if (out) *out=nullptr;
    if (!raw_device || !cuda_context || !out || !width || !height ||
        width>16384 || height>16384 || (pixel_bytes!=4 && pixel_bytes!=8) ||
        uint64_t(width)*pixel_bytes>UINT32_MAX-255u) {
        fail("invalid post-control open",0); return 1;
    }
    auto* b=new(std::nothrow) GpuPostControlBridge;
    if (!b) { fail("allocate post-control bridge",0); return 2; }
    b->device=static_cast<ID3D12Device*>(raw_device);
    b->width=width; b->height=height; b->pixel_bytes=pixel_bytes;
    b->pitch=row_pitch(width,pixel_bytes); b->cuda_context=cuda_context;
    D3D12_COMMAND_QUEUE_DESC q{}; q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr=b->device->CreateCommandQueue(&q,IID_PPV_ARGS(b->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr=b->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(b->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr=b->device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_DIRECT,b->allocator.Get(),nullptr,
        IID_PPV_ARGS(b->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr=b->list->Close();
    if (SUCCEEDED(hr)) hr=b->device->CreateFence(0,D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(b->fence.GetAddressOf()));
    if (SUCCEEDED(hr)) hr=make_resources(b);
    if (SUCCEEDED(hr)) hr=make_pipeline(b);
    if (FAILED(hr)) {
        if (!last_error[0]) fail("initialize post-control bridge",hr);
        gpu_post_control_close(b); return 3;
    }
    int32_t rc=cuda_d3d12_color_create2(b->device.Get(),cuda_context,width,height,
        b->pitch,pixel_bytes,0,&b->source_bridge);
    if (!rc) rc=cuda_d3d12_color_create2(b->device.Get(),cuda_context,width,height,
        b->pitch,pixel_bytes,0,&b->candidate_bridge);
    if (rc) {
        std::snprintf(last_error,sizeof(last_error),"post-control CUDA interop: %s",
                      nvof_d3d12_last_error());
        gpu_post_control_close(b); return 4;
    }
    b->event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if (!b->event) {
        fail("post-control CreateEvent",GetLastError());
        gpu_post_control_close(b); return 5;
    }
    *out=b; return 0;
}

extern "C" GPU_POST_CONTROL_API int32_t gpu_post_control_process(
    GpuPostControlBridge* b,uint64_t source_ptr,size_t source_pitch,
    const GpuPostCandidate* candidate,const GpuPostControlOptions* options,
    void** out_texture,void** out_fence,uint64_t* out_fence_value,
    GpuPostControlCounters* counters) {
    last_error[0]=0;
    if (out_texture) *out_texture=nullptr;
    if (out_fence) *out_fence=nullptr;
    if (out_fence_value) *out_fence_value=0;
    if (!b || b->failed || !source_ptr || !candidate || !options ||
        !out_texture || !out_fence || !out_fence_value || !counters ||
        counters->struct_size!=sizeof(*counters) ||
        counters->abi_version!=GPU_POST_CONTROL_ABI_VERSION) {
        fail("invalid post-control process arguments",0); return 1;
    }
    std::lock_guard<std::mutex> guard(b->mutex);
    if (options->struct_size!=sizeof(*options) ||
        options->abi_version!=GPU_POST_CONTROL_ABI_VERSION ||
        options->width!=b->width || options->height!=b->height ||
        options->pixel_bytes!=b->pixel_bytes || !finite_options(*options)) {
        fail("post-control options mismatch",0); return 2;
    }
    if (source_pitch<size_t(b->width)*b->pixel_bytes) {
        fail("post-control source pitch",0); return 3;
    }
    auto* candidate_texture=static_cast<ID3D12Resource*>(candidate->d3d_texture);
    auto* candidate_fence=static_cast<ID3D12Fence*>(candidate->producer_fence);
    if (candidate->kind==GPU_POST_CANDIDATE_CUDA) {
        if (!candidate->cuda_ptr || candidate->cuda_pitch<size_t(b->width)*b->pixel_bytes ||
            candidate->d3d_state || candidate->d3d_texture || candidate->producer_fence ||
            candidate->producer_fence_value) {
            fail("invalid CUDA post-control candidate",0); return 4;
        }
    } else if (candidate->kind==GPU_POST_CANDIDATE_D3D12 ||
               candidate->kind==GPU_POST_CANDIDATE_D3D12_COMPLETED) {
        const bool fenced=candidate->kind==GPU_POST_CANDIDATE_D3D12;
        if (candidate->cuda_ptr || candidate->cuda_pitch || !candidate_texture ||
            (fenced && !candidate_fence) || (!fenced && (candidate_fence ||
                candidate->producer_fence_value)) ||
            candidate->d3d_state!=D3D12_RESOURCE_STATE_COPY_SOURCE) {
            fail("invalid D3D12 post-control candidate",0); return 4;
        }
        ComPtr<ID3D12Device> owner;
        HRESULT hr=candidate_texture->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
        const auto d=candidate_texture->GetDesc();
        ComPtr<ID3D12Device> fence_owner;
        if (SUCCEEDED(hr) && candidate_fence)
            hr=candidate_fence->GetDevice(IID_PPV_ARGS(fence_owner.GetAddressOf()));
        if (FAILED(hr) || owner.Get()!=b->device.Get() ||
            (candidate_fence && fence_owner.Get()!=b->device.Get()) ||
            d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.Width!=b->width ||
            d.Height!=b->height || d.DepthOrArraySize!=1 || d.MipLevels!=1 ||
            d.SampleDesc.Count!=1 || d.Format!=pixel_format(b->pixel_bytes)) {
            fail("D3D12 candidate adapter/shape/format mismatch",FAILED(hr)?hr:0); return 5;
        }
    } else { fail("unknown post-control candidate kind",0); return 4; }

    uint64_t source_ready=0,candidate_ready=0;
    if (cuda_d3d12_color_copy(b->source_bridge,source_ptr,source_pitch,&source_ready)) {
        std::snprintf(last_error,sizeof(last_error),"post-control source copy: %s",
                      nvof_d3d12_last_error()); b->failed=true; return 6;
    }
    if (candidate->kind==GPU_POST_CANDIDATE_CUDA) {
        if (cuda_d3d12_color_copy(b->candidate_bridge,candidate->cuda_ptr,
                                  candidate->cuda_pitch,&candidate_ready)) {
            std::snprintf(last_error,sizeof(last_error),"post-control candidate copy: %s",
                          nvof_d3d12_last_error()); b->failed=true; return 6;
        }
    }
    void *source_raw=nullptr,*source_fence_raw=nullptr,*candidate_raw=nullptr,
         *candidate_fence_raw=nullptr;
    cuda_d3d12_color_get_resources(b->source_bridge,&source_raw,&source_fence_raw);
    cuda_d3d12_color_get_resources(b->candidate_bridge,&candidate_raw,&candidate_fence_raw);
    auto* source=static_cast<ID3D12Resource*>(source_raw);
    auto* candidate_buffer=static_cast<ID3D12Resource*>(candidate_raw);
    HRESULT hr=b->queue->Wait(static_cast<ID3D12Fence*>(source_fence_raw),source_ready);
    if (SUCCEEDED(hr) && candidate->kind==GPU_POST_CANDIDATE_CUDA)
        hr=b->queue->Wait(static_cast<ID3D12Fence*>(candidate_fence_raw),candidate_ready);
    if (SUCCEEDED(hr) && candidate->kind==GPU_POST_CANDIDATE_D3D12)
        hr=b->queue->Wait(candidate_fence,candidate->producer_fence_value);
    if (FAILED(hr)) { fail("post-control producer fence wait",hr); b->failed=true; return 7; }
    set_raw_srv(b,source,0); set_raw_srv(b,candidate_buffer,1); set_stats_srv(b);

    if (!reset_list(b)) { b->failed=true; return 8; }
    ID3D12DescriptorHeap* heaps[]={b->descriptors.Get()};
    b->list->SetDescriptorHeaps(1,heaps);
    const UINT step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    auto gpu=b->descriptors->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr+=5*step; gpu.ptr+=5*step;
    const UINT clear[4]={0,0,0,0};
    b->list->ClearUnorderedAccessViewUint(gpu,cpu,b->analysis.Get(),clear,0,nullptr);
    if (candidate->kind==GPU_POST_CANDIDATE_D3D12 ||
        candidate->kind==GPU_POST_CANDIDATE_D3D12_COMPLETED) {
        barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST);
        D3D12_TEXTURE_COPY_LOCATION dst{},src{};
        dst.pResource=candidate_buffer; dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Offset=0;
        dst.PlacedFootprint.Footprint.Format=pixel_format(b->pixel_bytes);
        dst.PlacedFootprint.Footprint.Width=b->width;
        dst.PlacedFootprint.Footprint.Height=b->height;
        dst.PlacedFootprint.Footprint.Depth=1;
        dst.PlacedFootprint.Footprint.RowPitch=b->pitch;
        src.pResource=candidate_texture; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        b->list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    } else barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    uav_barrier(b->list.Get(),b->analysis.Get());
    set_compute_bindings(b,options,b->analyze_pipeline.Get());
    b->list->Dispatch((b->width+7)/8,(b->height+7)/8,1);
    uav_barrier(b->list.Get(),b->analysis.Get());
    barrier(b->list.Get(),b->analysis.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    b->list->CopyBufferRegion(b->readback.Get(),0,b->analysis.Get(),0,16);
    barrier(b->list.Get(),b->analysis.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!execute_wait(b,"post-control analyze list")) { b->failed=true; return 9; }
    uint32_t* mapped=nullptr;
    D3D12_RANGE read_range{0,16};
    hr=b->readback->Map(0,&read_range,reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map post-control analysis",hr); b->failed=true; return 10; }
    const uint32_t invalid_source=mapped[1];
    b->readback->Unmap(0,nullptr);
    b->scalar_readback_bytes+=16;
    if (invalid_source) { fail("post-control source contains non-finite pixels",0);
        b->failed=true; return 11; }

    if (!reset_list(b)) { b->failed=true; return 12; }
    b->list->SetDescriptorHeaps(1,heaps);
    cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    gpu=b->descriptors->GetGPUDescriptorHandleForHeapStart();
    cpu.ptr+=4*step; gpu.ptr+=4*step;
    b->list->ClearUnorderedAccessViewUint(gpu,cpu,b->fallback.Get(),clear,0,nullptr);
    barrier(b->list.Get(),b->analysis.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    barrier(b->list.Get(),b->output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    set_compute_bindings(b,options,b->apply_pipeline.Get());
    b->list->Dispatch((b->width+7)/8,(b->height+7)/8,1);
    barrier(b->list.Get(),b->output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(b->list.Get(),b->analysis.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(b->list.Get(),b->fallback.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    b->list->CopyBufferRegion(b->readback.Get(),16,b->fallback.Get(),0,16);
    barrier(b->list.Get(),b->analysis.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(b->list.Get(),b->fallback.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    if (!execute_wait(b,"post-control apply list")) { b->failed=true; return 13; }
    D3D12_RANGE fallback_range{16,32};
    hr=b->readback->Map(0,&fallback_range,reinterpret_cast<void**>(&mapped));
    if (FAILED(hr)) { fail("map post-control counters",hr); b->failed=true; return 14; }
    const uint32_t fallback_pixels=mapped[4];
    b->readback->Unmap(0,nullptr);
    b->scalar_readback_bytes+=16;
    b->process_count++;
    b->source_transport_bytes+=uint64_t(b->width)*b->height*b->pixel_bytes;
    b->candidate_transport_bytes+=uint64_t(b->width)*b->height*b->pixel_bytes;
    b->unrepresentable_pixels+=fallback_pixels;
    *out_texture=b->output.Get(); *out_fence=b->fence.Get();
    *out_fence_value=b->fence_value;
    *counters=GpuPostControlCounters{sizeof(*counters),GPU_POST_CONTROL_ABI_VERSION,
        b->process_count,b->source_transport_bytes,b->candidate_transport_bytes,
        b->scalar_readback_bytes,b->unrepresentable_pixels};
    return 0;
}
