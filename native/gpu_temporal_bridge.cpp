#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "gpu_temporal_bridge.h"
#include "gpu_temporal_shader.h"
#include "nvof_d3d12_interop.h"
#include "gpu_dred_diagnostics.h"

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
static_assert(sizeof(GpuTemporalObserveReport)==64,"observe report ABI changed");
static_assert(sizeof(GpuTemporalProcessReport)==56,"process report ABI changed");
static_assert(sizeof(GpuPostCandidate)==48,"post candidate ABI changed");

namespace {
constexpr uint32_t kGuideWidth=32, kGuideHeight=18, kGuideCount=576;
constexpr uint32_t kSrvCount=6, kUavCount=3, kDescriptors=9;
constexpr uint32_t kDescriptorCapacity=64;
thread_local char last_error[1024]="";
void fail(const char* operation,long code) {
    std::snprintf(last_error,sizeof(last_error),"%s: 0x%08lx",operation,
                  static_cast<unsigned long>(code));
}
uint32_t row_pitch(uint32_t width,uint32_t pixel_bytes) {
    return uint32_t((uint64_t(width)*pixel_bytes+255u)&~uint64_t(255u));
}
DXGI_FORMAT pixel_format(uint32_t bytes) {
    return bytes==4 ? DXGI_FORMAT_R8G8B8A8_UNORM :
                      DXGI_FORMAT_R16G16B16A16_FLOAT;
}
void barrier(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,
             D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=resource;
    b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore=before;
    b.Transition.StateAfter=after;
    list->ResourceBarrier(1,&b);
}
void uav_barrier(ID3D12GraphicsCommandList* list,ID3D12Resource* resource) {
    D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;
    b.UAV.pResource=resource; list->ResourceBarrier(1,&b);
}
HRESULT make_buffer(ID3D12Device* device,uint64_t bytes,D3D12_HEAP_TYPE type,
                    D3D12_RESOURCE_FLAGS flags,D3D12_RESOURCE_STATES state,
                    ComPtr<ID3D12Resource>* result) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type=type;
    D3D12_RESOURCE_DESC d{};
    d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER; d.Width=std::max<uint64_t>(bytes,4);
    d.Height=1; d.DepthOrArraySize=1; d.MipLevels=1; d.SampleDesc.Count=1;
    d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR; d.Flags=flags;
    return device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,state,
        nullptr,IID_PPV_ARGS(result->GetAddressOf()));
}
HRESULT make_texture(ID3D12Device* device,uint32_t width,uint32_t height,
                     DXGI_FORMAT format,ComPtr<ID3D12Resource>* result) {
    D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC d{}; d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width=width; d.Height=height; d.DepthOrArraySize=1; d.MipLevels=1;
    d.Format=format; d.SampleDesc.Count=1;
    d.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    return device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&d,
        D3D12_RESOURCE_STATE_COPY_SOURCE,nullptr,IID_PPV_ARGS(result->GetAddressOf()));
}
HRESULT compile_entry(const char* entry,ComPtr<ID3DBlob>* output) {
    ComPtr<ID3DBlob> errors;
    HRESULT hr=D3DCompile(gpu_temporal_shader,sizeof(gpu_temporal_shader)-1,
        "gpu_temporal",nullptr,nullptr,entry,"cs_5_0",
        D3DCOMPILE_ENABLE_STRICTNESS|D3DCOMPILE_IEEE_STRICTNESS,0,
        output->GetAddressOf(),errors.GetAddressOf());
    if (FAILED(hr) && errors)
        std::snprintf(last_error,sizeof(last_error),"D3DCompile %s: %s",entry,
            static_cast<const char*>(errors->GetBufferPointer()));
    return hr;
}
}

struct GpuTemporalBridge {
    uint32_t width=0,height=0,pixel_bytes=0,pitch=0;
    uint32_t guide_index=0,history_index=0,scene_reset_count=0;
    uint32_t descriptor_cursor=0;
    bool guide_valid=false,history_valid=false,failed=false;
    bool guide_srv[2]={false,false},history_srv[2]={false,false};
    void* cuda_context=nullptr;
    CudaD3D12ColorBridge* source_bridge=nullptr;
    CudaD3D12ColorBridge* candidate_bridge=nullptr;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    std::array<ComPtr<ID3D12PipelineState>,3> pipelines;
    ComPtr<ID3D12DescriptorHeap> descriptors;
    ComPtr<ID3D12DescriptorHeap> stats_clear_heap;
    ComPtr<ID3D12Resource> output,guide[2],history[2],stats,readback;
    HANDLE event=nullptr;
    uint64_t fence_value=0;
    std::mutex mutex;
};

namespace {
bool wait_gpu(GpuTemporalBridge* b) {
    const uint64_t completed = b->fence->GetCompletedValue();
    if (completed == UINT64_MAX || FAILED(b->device->GetDeviceRemovedReason())) {
        describe_gpu_removal(b->device.Get(), "temporal fence device removed",
                             DXGI_ERROR_DEVICE_REMOVED, last_error, sizeof(last_error));
        return false;
    }
    if (completed>=b->fence_value) return true;
    HRESULT hr=b->fence->SetEventOnCompletion(b->fence_value,b->event);
    if (FAILED(hr)) { fail("temporal fence event",hr); return false; }
    if (WaitForSingleObject(b->event,30000)!=WAIT_OBJECT_0) {
        fail("temporal fence timeout",ERROR_TIMEOUT); return false;
    }
    const HRESULT removed = b->device->GetDeviceRemovedReason();
    if (FAILED(removed) || b->fence->GetCompletedValue() == UINT64_MAX) {
        describe_gpu_removal(b->device.Get(), "temporal wait device removed",
                             removed, last_error, sizeof(last_error));
        return false;
    }
    return true;
}
bool reset_list(GpuTemporalBridge* b) {
    HRESULT hr=b->allocator->Reset();
    if (SUCCEEDED(hr)) hr=b->list->Reset(b->allocator.Get(),nullptr);
    if (FAILED(hr)) { fail("temporal command-list reset",hr); return false; }
    /* The previous submission is always fence-complete before this is called,
       so descriptor slots may be recycled along with the command allocator. */
    b->descriptor_cursor=0;
    return true;
}
bool execute_wait(GpuTemporalBridge* b,const char* where) {
    HRESULT hr=b->list->Close();
    if (FAILED(hr)) { fail(where,hr); return false; }
    ID3D12CommandList* lists[]={b->list.Get()};
    b->queue->ExecuteCommandLists(1,lists);
    hr=b->queue->Signal(b->fence.Get(),++b->fence_value);
    if (FAILED(hr)) { fail("temporal queue signal",hr); return false; }
    return wait_gpu(b);
}
void create_raw_srv(GpuTemporalBridge* b,ID3D12Resource* r,
                    D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_R32_TYPELESS; d.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements=static_cast<uint32_t>(r->GetDesc().Width/4);
    d.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(r,&d,h);
}
void create_raw_uav(GpuTemporalBridge* b,ID3D12Resource* r,
                    D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_R32_TYPELESS; d.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements=static_cast<uint32_t>(r->GetDesc().Width/4);
    d.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(r,nullptr,&d,h);
}
void create_null_raw_uav(GpuTemporalBridge* b,D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_UNORDERED_ACCESS_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_R32_TYPELESS; d.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;
    d.Buffer.NumElements=1; d.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;
    b->device->CreateUnorderedAccessView(nullptr,nullptr,&d,h);
}
void create_null_raw_srv(GpuTemporalBridge* b,D3D12_CPU_DESCRIPTOR_HANDLE h) {
    D3D12_SHADER_RESOURCE_VIEW_DESC d{};
    d.Format=DXGI_FORMAT_R32_TYPELESS; d.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;
    d.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    d.Buffer.NumElements=1; d.Buffer.Flags=D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(nullptr,&d,h);
}
struct Constants {
    uint32_t width,height,pitch,pixel_bytes;
    uint32_t guide_width,guide_height,history_valid,reset;
    float history_weight;
};
struct DispatchBinding {
    ID3D12Resource* srv[kSrvCount]{};
    ID3D12Resource* output_texture=nullptr;
    ID3D12Resource* state_output=nullptr;
    ID3D12Resource* stats=nullptr;
};
uint32_t bind_dispatch(GpuTemporalBridge* b,const DispatchBinding& binding) {
    const uint32_t first=static_cast<uint32_t>(b->descriptor_cursor);
    b->descriptor_cursor+=kDescriptors;
    auto cpu=b->descriptors->GetCPUDescriptorHandleForHeapStart();
    auto at=[&](uint32_t i) { auto h=cpu; h.ptr+=size_t(first+i)*b->device->
        GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV); return h; };
    for (uint32_t i=0;i<kSrvCount;++i) {
        if (binding.srv[i]) create_raw_srv(b,binding.srv[i],at(i));
        else create_null_raw_srv(b,at(i));
    }
    D3D12_UNORDERED_ACCESS_VIEW_DESC tex{};
    tex.Format=pixel_format(b->pixel_bytes); tex.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
    b->device->CreateUnorderedAccessView(binding.output_texture,nullptr,
        &tex,at(kSrvCount));
    if (binding.state_output) create_raw_uav(b,binding.state_output,at(kSrvCount+1));
    else create_null_raw_uav(b,at(kSrvCount+1));
    create_raw_uav(b,binding.stats ? binding.stats : b->stats.Get(),at(kSrvCount+2));
    return first;
}
void bind_roots(GpuTemporalBridge* b,uint32_t first,const Constants& c) {
    const uint32_t step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    auto gpu=b->descriptors->GetGPUDescriptorHandleForHeapStart();
    gpu.ptr+=size_t(first)*step;
    b->list->SetComputeRootDescriptorTable(0,gpu);
    gpu.ptr+=size_t(kSrvCount)*step;
    b->list->SetComputeRootDescriptorTable(1,gpu);
    b->list->SetComputeRoot32BitConstants(2,9,&c,0);
}
bool clear_stats(GpuTemporalBridge* b,uint32_t first) {
    auto cpu=b->stats_clear_heap->GetCPUDescriptorHandleForHeapStart();
    auto gpu=b->descriptors->GetGPUDescriptorHandleForHeapStart();
    const uint32_t step=b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    gpu.ptr+=size_t(first+kSrvCount+2)*step;
    const UINT zeros[4]={0,0,0,0};
    b->list->ClearUnorderedAccessViewUint(gpu,cpu,b->stats.Get(),zeros,0,nullptr);
    return true;
}
bool make_root_and_pipeline(GpuTemporalBridge* b) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors=kSrvCount;
    ranges[1].RangeType=D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors=kUavCount;
    D3D12_ROOT_PARAMETER params[3]{};
    for (uint32_t i=0;i<2;++i) {
        params[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        params[i].DescriptorTable.NumDescriptorRanges=1;
        params[i].DescriptorTable.pDescriptorRanges=&ranges[i];
    }
    params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.ShaderRegister=0; params[2].Constants.Num32BitValues=9;
    D3D12_ROOT_SIGNATURE_DESC root{}; root.NumParameters=3; root.pParameters=params;
    ComPtr<ID3DBlob> blob,errors;
    HRESULT hr=D3D12SerializeRootSignature(&root,D3D_ROOT_SIGNATURE_VERSION_1,
        blob.GetAddressOf(),errors.GetAddressOf());
    if (SUCCEEDED(hr)) hr=b->device->CreateRootSignature(0,blob->GetBufferPointer(),
        blob->GetBufferSize(),IID_PPV_ARGS(b->root.GetAddressOf()));
    if (FAILED(hr)) return false;
    const char* names[]={"BuildGuide","CompareGuide","TemporalProcess"};
    for (size_t i=0;i<3;++i) {
        ComPtr<ID3DBlob> code;
        hr=compile_entry(names[i],&code);
        if (FAILED(hr)) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature=b->root.Get();
        pso.CS={code->GetBufferPointer(),code->GetBufferSize()};
        hr=b->device->CreateComputePipelineState(&pso,
            IID_PPV_ARGS(b->pipelines[i].GetAddressOf()));
        if (FAILED(hr)) return false;
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap{};
    heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap.NumDescriptors=kDescriptorCapacity;
    heap.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (FAILED(b->device->CreateDescriptorHeap(&heap,
        IID_PPV_ARGS(b->descriptors.GetAddressOf())))) return false;
    return true;
}
bool make_resources(GpuTemporalBridge* b) {
    HRESULT hr=S_OK;
    for (auto& r:b->guide) if (SUCCEEDED(hr)) hr=make_buffer(b->device.Get(),
        uint64_t(kGuideCount)*16,D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON,&r);

    if (SUCCEEDED(hr)) hr=make_buffer(b->device.Get(),32,D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON,&b->stats);
    if (SUCCEEDED(hr)) hr=make_buffer(b->device.Get(),32,D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST,&b->readback);
    if (FAILED(hr)) return false;
    D3D12_DESCRIPTOR_HEAP_DESC clear_heap{};
    clear_heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    clear_heap.NumDescriptors=1;
    if (FAILED(b->device->CreateDescriptorHeap(&clear_heap,
        IID_PPV_ARGS(b->stats_clear_heap.GetAddressOf())))) return false;
    create_raw_uav(b,b->stats.Get(),b->stats_clear_heap->GetCPUDescriptorHandleForHeapStart());
    hr=b->allocator->Reset();
    if (SUCCEEDED(hr)) hr=b->list->Reset(b->allocator.Get(),nullptr);
    if (FAILED(hr)) return false;
    barrier(b->list.Get(),b->stats.Get(),D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    for (auto& r:b->guide) barrier(b->list.Get(),r.Get(),
        D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    if (FAILED(b->list->Close())) return false;
    ID3D12CommandList* lists[]={b->list.Get()};
    b->queue->ExecuteCommandLists(1,lists);
    hr=b->queue->Signal(b->fence.Get(),++b->fence_value);
    return SUCCEEDED(hr) && wait_gpu(b);
}
// Only temporal stabilization needs full-size histories and a candidate copy.
bool ensure_processing_resources(GpuTemporalBridge* b) {
    if (b->output) return true;
    ComPtr<ID3D12Resource> output, history[2];
    HRESULT hr=make_texture(b->device.Get(),b->width,b->height,
        pixel_format(b->pixel_bytes),&output);
    for (auto& r:history) if (SUCCEEDED(hr)) hr=make_buffer(b->device.Get(),
        uint64_t(b->width)*b->height*32,D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON,&r);
    if (FAILED(hr)) { fail("allocate temporal processing resources",hr); return false; }
    if (cuda_d3d12_color_create2(b->device.Get(),b->cuda_context,b->width,b->height,
        b->pitch,b->pixel_bytes,0,&b->candidate_bridge)) {
        std::snprintf(last_error,sizeof(last_error),"temporal candidate interop: %s",nvof_d3d12_last_error());
        return false;
    }
    if (!reset_list(b)) return false;
    for (auto& r:history) barrier(b->list.Get(),r.Get(),D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    // Hold ownership even if execution fails: cleanup must precede resource release.
    b->output=output; b->history[0]=history[0]; b->history[1]=history[1];
    return execute_wait(b,"initialize temporal histories");
}
bool get_transport_resources(CudaD3D12ColorBridge* bridge,
    ID3D12Resource** resource,ID3D12Fence** fence) {
    void* r=nullptr; void* f=nullptr;
    if (cuda_d3d12_color_get_resources(bridge,&r,&f)) return false;
    *resource=static_cast<ID3D12Resource*>(r);
    *fence=static_cast<ID3D12Fence*>(f);
    return *resource && *fence;
}
void set_heap_and_root(GpuTemporalBridge* b) {
    ID3D12DescriptorHeap* heaps[]={b->descriptors.Get()};
    b->list->SetDescriptorHeaps(1,heaps);
    b->list->SetComputeRootSignature(b->root.Get());
}
bool read_stats(GpuTemporalBridge* b,uint32_t bytes,uint32_t* words) {
    if (!reset_list(b)) return false;
    barrier(b->list.Get(),b->stats.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    b->list->CopyBufferRegion(b->readback.Get(),0,b->stats.Get(),0,bytes);
    barrier(b->list.Get(),b->stats.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if (!execute_wait(b,"temporal stats readback")) return false;
    void* mapped=nullptr; D3D12_RANGE range{0,bytes};
    HRESULT hr=b->readback->Map(0,&range,&mapped);
    if (FAILED(hr)) {
        describe_gpu_removal(b->device.Get(), "map temporal scalar readback", hr,
                             last_error, sizeof(last_error));
        return false;
    }
    std::memcpy(words,mapped,bytes);
    D3D12_RANGE no_write{0,0}; b->readback->Unmap(0,&no_write);
    return true;
}
void invalidate(GpuTemporalBridge* b) {
    b->guide_valid=false; b->history_valid=false;
}
float bits_float(uint32_t bits) { float v; std::memcpy(&v,&bits,4); return v; }
uint32_t float_bits(float value) { uint32_t v; std::memcpy(&v,&value,4); return v; }
float clamp01(float v) { return (std::max)(0.0f,(std::min)(1.0f,v)); }
}

extern "C" GPU_TEMPORAL_API uint32_t gpu_temporal_abi_version(void) {
    return GPU_TEMPORAL_ABI_VERSION;
}
extern "C" GPU_TEMPORAL_API const char* gpu_temporal_last_error(void) {
    return last_error;
}
extern "C" GPU_TEMPORAL_API void gpu_temporal_close(GpuTemporalBridge* b) {
    if (!b) return;
    if (b->queue && b->fence && b->event) {
        if (SUCCEEDED(b->queue->Signal(b->fence.Get(),++b->fence_value))) wait_gpu(b);
    }
    if (b->event) CloseHandle(b->event);
    if (b->candidate_bridge) cuda_d3d12_color_destroy(b->candidate_bridge);
    if (b->source_bridge) cuda_d3d12_color_destroy(b->source_bridge);
    delete b;
}

extern "C" GPU_TEMPORAL_API uint64_t gpu_temporal_allocated_device_bytes(GpuTemporalBridge* b) {
    if (!b) return 0;
    std::lock_guard<std::mutex> guard(b->mutex);
    uint64_t bytes=0;
    const auto add=[&](ID3D12Resource* r) {
        if (r) { auto desc=r->GetDesc(); bytes+=b->device->GetResourceAllocationInfo(0,1,&desc).SizeInBytes; }
    };
    add(b->output.Get()); add(b->stats.Get());
    for (auto& r:b->guide) add(r.Get());
    for (auto& r:b->history) add(r.Get());
    for (auto* bridge : {b->source_bridge,b->candidate_bridge}) if (bridge) {
        ID3D12Resource* r=nullptr; ID3D12Fence* f=nullptr;
        if (get_transport_resources(bridge,&r,&f)) add(r);
    }
    return bytes;
}
extern "C" GPU_TEMPORAL_API int32_t gpu_temporal_open(void* raw_device,
    void* cuda_context,uint32_t width,uint32_t height,uint32_t pixel_bytes,
    GpuTemporalBridge** out) {
    last_error[0]=0;
    if (out) *out=nullptr;
    if (!raw_device || !cuda_context || !out || !width || !height ||
        width>16384 || height>16384 || (pixel_bytes!=4 && pixel_bytes!=8) ||
        uint64_t(width)*pixel_bytes>UINT32_MAX-255u ||
        uint64_t(width)*height>UINT32_MAX/32u) {
        fail("invalid temporal open arguments",0); return 1;
    }
    auto* b=new(std::nothrow) GpuTemporalBridge;
    if (!b) { fail("allocate temporal bridge",0); return 2; }
    b->device=static_cast<ID3D12Device*>(raw_device); b->cuda_context=cuda_context;
    b->width=width; b->height=height; b->pixel_bytes=pixel_bytes;
    b->pitch=row_pitch(width,pixel_bytes);
    b->event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if (!b->event) { fail("temporal CreateEvent",GetLastError());
        gpu_temporal_close(b); return 5; }
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
    if (SUCCEEDED(hr)) {
        b->queue->SetName(L"Lunarc temporal queue");
        b->list->SetName(L"Lunarc temporal commands");
    }
    if (SUCCEEDED(hr) && !make_root_and_pipeline(b)) hr=E_FAIL;
    if (SUCCEEDED(hr) && !make_resources(b)) hr=E_FAIL;
    if (FAILED(hr)) {
        if (!last_error[0]) fail("initialize temporal resources",hr);
        gpu_temporal_close(b); return 3;
    }
    int32_t rc=cuda_d3d12_color_create2(b->device.Get(),cuda_context,width,height,
        b->pitch,pixel_bytes,0,&b->source_bridge);

    if (rc) {
        std::snprintf(last_error,sizeof(last_error),"temporal CUDA interop: %s",
            nvof_d3d12_last_error()); gpu_temporal_close(b); return 4;
    }
    *out=b; return 0;
}

extern "C" GPU_TEMPORAL_API int32_t gpu_temporal_observe(
    GpuTemporalBridge* b,uint64_t source_ptr,size_t source_pitch,uint32_t reset,
    uint32_t scene_enabled,GpuTemporalObserveReport* report) {
    last_error[0]=0;
    if (!b || b->failed || !source_ptr || !report ||
        report->struct_size!=sizeof(*report) ||
        report->abi_version!=GPU_TEMPORAL_ABI_VERSION || reset>1 ||
        scene_enabled>1 || source_pitch<size_t(b->width)*b->pixel_bytes) {
        fail("invalid temporal observe arguments",0); return 1;
    }
    std::lock_guard<std::mutex> guard(b->mutex);
    if (reset) invalidate(b);
    if (!scene_enabled) {
        b->guide_valid=false; b->history_valid=false;
        report->classification=GPU_TEMPORAL_DISABLED; report->scene_cut=0;
        report->similarity=0; report->normalized_delta=0; report->chroma_delta=0;
        report->exposure_ratio=1; report->shift_x=0; report->shift_y=0;
        report->scene_reset_count=b->scene_reset_count;
        report->source_transport_bytes=0; report->scalar_readback_bytes=0;
        return 0;
    }
    uint64_t ready=0;
    if (cuda_d3d12_color_copy(b->source_bridge,source_ptr,source_pitch,&ready)) {
        std::snprintf(last_error,sizeof(last_error),"temporal source copy: %s",
            nvof_d3d12_last_error()); b->failed=true; return 2;
    }
    ID3D12Resource* source=nullptr; ID3D12Fence* producer=nullptr;
    if (!get_transport_resources(b->source_bridge,&source,&producer)) {
        fail("temporal source transport resource",0); b->failed=true; return 2;
    }
    HRESULT hr=b->queue->Wait(producer,ready);
    if (FAILED(hr)) { fail("temporal source producer wait",hr); b->failed=true; return 2; }
    const uint32_t previous=b->guide_index,target=1u-previous;
    if (!reset_list(b)) { b->failed=true; return 3; }
    ID3D12DescriptorHeap* heaps[]={b->descriptors.Get()};
    b->list->SetDescriptorHeaps(1,heaps); b->list->SetComputeRootSignature(b->root.Get());
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    if (b->guide_srv[target]) barrier(b->list.Get(),b->guide[target].Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchBinding build{}; build.srv[0]=source;
    // Guide shaders never access the image output UAV (typed null descriptor).
    build.state_output=b->guide[target].Get(); build.stats=b->stats.Get();
    const uint32_t build_first=bind_dispatch(b,build);
    clear_stats(b,build_first);
    uav_barrier(b->list.Get(),b->stats.Get());
    Constants c{b->width,b->height,b->pitch,b->pixel_bytes,
        kGuideWidth,kGuideHeight,b->guide_valid?1u:0u,reset,0.0f};
    b->list->SetPipelineState(b->pipelines[0].Get()); bind_roots(b,build_first,c);
    b->list->Dispatch(1,1,1);
    barrier(b->list.Get(),b->guide[target].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b->guide_srv[target]=true;
    DispatchBinding compare{};
    /* On the first dispatch there is no previous guide yet. Keep its SRV null
       instead of binding a UAV-state ping-pong resource, even though the
       shader's HistoryValid branch would not read it. */
    if (b->guide_valid && !reset)
        compare.srv[3]=b->guide[previous].Get();
    compare.srv[5]=b->guide[target].Get();
    compare.stats=b->stats.Get();
    const uint32_t compare_first=bind_dispatch(b,compare);
    c.history_valid=b->guide_valid?1u:0u;
    b->list->SetPipelineState(b->pipelines[1].Get()); bind_roots(b,compare_first,c);
    b->list->Dispatch(1,1,1);
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);

    if (!execute_wait(b,"temporal observe dispatch")) { b->failed=true; return 4; }
    uint32_t words[8]{};
    if (!read_stats(b,32,words)) { b->failed=true; return 5; }
    const float mean=bits_float(words[0]);
    const float support=bits_float(words[1]);
    const float chroma=bits_float(words[2]);
    const float ratio=b->guide_valid ? bits_float(words[3]) : 1.0f;
    const float dx=bits_float(words[4]),dy=bits_float(words[5]);
    uint32_t classification=GPU_TEMPORAL_AMBIGUOUS;
    uint32_t cut=0;
    if (!b->guide_valid || reset) {
        classification=GPU_TEMPORAL_FIRST; cut=1;
    } else if (support>=0.70f && mean<=0.10f &&
               std::abs(std::log((std::max)(ratio,1.0e-8f)))>0.08f) {
        classification=GPU_TEMPORAL_EXPOSURE_FLASH;
    } else if (support>=0.70f && mean<=0.10f && (dx!=0.0f || dy!=0.0f)) {
        classification=GPU_TEMPORAL_TRANSLATION;
    } else if (support>=0.88f && mean<=0.025f) {
        classification=GPU_TEMPORAL_DUPLICATE_STATIC;
    /* Guide distances are normalized RGB L1 means; support is the fraction
       below the shader's 0.08 per-sample match distance. Require both a
       substantial mean disagreement and <=10% local matches so achromatic
       cuts can register without treating exposure/flash or searched shifts as
       cuts. This remains a sampled heuristic, especially for large motion. */
    } else if (mean>=0.15f && support<=0.10f) {
        /* Strong structural disagreement plus almost no local guide matches
           can be a cut even when the incoming scene is achromatic. */
        classification=GPU_TEMPORAL_CUT; cut=1;
    } else if (mean>=0.15f && chroma>=0.08f) {
        classification=GPU_TEMPORAL_CUT; cut=1;
    }
    const bool discontinuity=cut || classification==GPU_TEMPORAL_AMBIGUOUS ||
        classification==GPU_TEMPORAL_EXPOSURE_FLASH ||
        classification==GPU_TEMPORAL_TRANSLATION;
    b->guide_index=target; b->guide_valid=true;
    if (discontinuity) {
        b->history_valid=false;
        if (cut) ++b->scene_reset_count;
    }
    report->classification=classification; report->scene_cut=cut;
    report->similarity=support; report->normalized_delta=mean;
    report->chroma_delta=chroma; report->exposure_ratio=ratio;
    report->shift_x=dx; report->shift_y=dy;
    report->scene_reset_count=b->scene_reset_count;
    report->source_transport_bytes=uint64_t(b->width)*b->height*b->pixel_bytes;
    report->scalar_readback_bytes=32;
    return 0;
}

extern "C" GPU_TEMPORAL_API int32_t gpu_temporal_process(
    GpuTemporalBridge* b,uint64_t source_ptr,size_t source_pitch,
    const GpuPostCandidate* candidate,float history_weight,uint32_t reset,
    void** out_texture,void** out_fence,uint64_t* out_fence_value,
    GpuTemporalProcessReport* report) {
    last_error[0]=0;
    if (out_texture) *out_texture=nullptr; if (out_fence) *out_fence=nullptr;
    if (out_fence_value) *out_fence_value=0;
    if (!b || b->failed || !source_ptr || !candidate || !out_texture ||
        !out_fence || !out_fence_value || !report ||
        report->struct_size!=sizeof(*report) ||
        report->abi_version!=GPU_TEMPORAL_ABI_VERSION || reset>1 ||
        !std::isfinite(history_weight) || history_weight<0.0f || history_weight>1.0f ||
        source_pitch<size_t(b->width)*b->pixel_bytes) {
        fail("invalid temporal process arguments",0); return 1;
    }
    std::lock_guard<std::mutex> guard(b->mutex);
    /* Observe runs before all temporal layers. Its reset clears stale guides,
       then process(reset=1) must preserve the just-observed current guide so
       the following frame is compared against this one rather than FIRST. */
    if (reset) b->history_valid=false;
    auto* candidate_texture=static_cast<ID3D12Resource*>(candidate->d3d_texture);
    auto* producer_fence=static_cast<ID3D12Fence*>(candidate->producer_fence);
    bool fenced=false;
    if (candidate->kind==GPU_POST_CANDIDATE_CUDA) {
        if (!candidate->cuda_ptr || candidate->cuda_pitch<size_t(b->width)*b->pixel_bytes ||
            candidate->d3d_state || candidate_texture || producer_fence ||
            candidate->producer_fence_value) {
            fail("invalid CUDA temporal candidate",0); return 2;
        }
    } else if (candidate->kind==GPU_POST_CANDIDATE_D3D12 ||
               candidate->kind==GPU_POST_CANDIDATE_D3D12_COMPLETED) {
        fenced=candidate->kind==GPU_POST_CANDIDATE_D3D12;
        if (candidate->cuda_ptr || candidate->cuda_pitch || !candidate_texture ||
            (fenced && !producer_fence) || (!fenced && (producer_fence ||
                candidate->producer_fence_value)) ||
            candidate->d3d_state!=D3D12_RESOURCE_STATE_COPY_SOURCE) {
            fail("invalid D3D12 temporal candidate",0); return 2;
        }
        ComPtr<ID3D12Device> owner;
        HRESULT hr=candidate_texture->GetDevice(IID_PPV_ARGS(owner.GetAddressOf()));
        const auto d=candidate_texture->GetDesc();
        ComPtr<ID3D12Device> fence_owner;
        if (SUCCEEDED(hr) && producer_fence)
            hr=producer_fence->GetDevice(IID_PPV_ARGS(fence_owner.GetAddressOf()));
        if (FAILED(hr) || owner.Get()!=b->device.Get() ||
            (producer_fence && fence_owner.Get()!=b->device.Get()) ||
            d.Dimension!=D3D12_RESOURCE_DIMENSION_TEXTURE2D || d.Width!=b->width ||
            d.Height!=b->height || d.DepthOrArraySize!=1 || d.MipLevels!=1 ||
            d.SampleDesc.Count!=1 || d.Format!=pixel_format(b->pixel_bytes)) {
            fail("D3D12 temporal candidate adapter/shape/format mismatch",FAILED(hr)?hr:0);
            return 3;
        }
    } else { fail("unknown temporal candidate kind",0); return 2; }

    if (!ensure_processing_resources(b)) { b->failed=true; return 4; }
    uint64_t source_ready=0,candidate_ready=0;
    if (cuda_d3d12_color_copy(b->source_bridge,source_ptr,source_pitch,&source_ready)) {
        std::snprintf(last_error,sizeof(last_error),"temporal source copy: %s",
            nvof_d3d12_last_error()); b->failed=true; return 4;
    }
    if (candidate->kind==GPU_POST_CANDIDATE_CUDA &&
        cuda_d3d12_color_copy(b->candidate_bridge,candidate->cuda_ptr,
                              candidate->cuda_pitch,&candidate_ready)) {
        std::snprintf(last_error,sizeof(last_error),"temporal candidate copy: %s",
            nvof_d3d12_last_error()); b->failed=true; return 4;
    }
    ID3D12Resource *source=nullptr,*candidate_buffer=nullptr;
    ID3D12Fence *source_fence=nullptr,*candidate_copy_fence=nullptr;
    if (!get_transport_resources(b->source_bridge,&source,&source_fence) ||
        !get_transport_resources(b->candidate_bridge,&candidate_buffer,
                                 &candidate_copy_fence)) {
        fail("temporal transport resources",0); b->failed=true; return 5;
    }
    HRESULT hr=b->queue->Wait(source_fence,source_ready);
    if (SUCCEEDED(hr) && candidate->kind==GPU_POST_CANDIDATE_CUDA)
        hr=b->queue->Wait(candidate_copy_fence,candidate_ready);
    if (SUCCEEDED(hr) && fenced)
        hr=b->queue->Wait(producer_fence,candidate->producer_fence_value);
    if (FAILED(hr)) { fail("temporal producer fence wait",hr); b->failed=true; return 5; }

    if (!reset_list(b)) { b->failed=true; return 6; }
    ID3D12DescriptorHeap* heaps[]={b->descriptors.Get()};
    b->list->SetDescriptorHeaps(1,heaps); b->list->SetComputeRootSignature(b->root.Get());
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if (candidate->kind==GPU_POST_CANDIDATE_CUDA) {
        barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    } else {
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
    }
    const uint32_t previous=b->history_index,target=1u-previous;
    if (b->history_srv[target]) barrier(b->list.Get(),b->history[target].Get(),
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    barrier(b->list.Get(),b->output.Get(),D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    DispatchBinding binding{};
    binding.srv[0]=source; binding.srv[1]=candidate_buffer;
    if (b->history_valid && !reset) binding.srv[2]=b->history[previous].Get();
    binding.output_texture=b->output.Get(); binding.state_output=b->history[target].Get();
    binding.stats=b->stats.Get();
    const uint32_t first=bind_dispatch(b,binding); clear_stats(b,first);
    uav_barrier(b->list.Get(),b->stats.Get());
    Constants c{b->width,b->height,b->pitch,b->pixel_bytes,kGuideWidth,kGuideHeight,
        (b->history_valid && !reset)?1u:0u,reset,history_weight};
    b->list->SetPipelineState(b->pipelines[2].Get()); bind_roots(b,first,c);
    b->list->Dispatch((b->width+7u)/8u,(b->height+7u)/8u,1);
    barrier(b->list.Get(),b->output.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE);
    barrier(b->list.Get(),b->history[target].Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    b->history_srv[target]=true;
    barrier(b->list.Get(),source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    barrier(b->list.Get(),candidate_buffer,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            D3D12_RESOURCE_STATE_COMMON);
    if (!execute_wait(b,"temporal process dispatch")) { b->failed=true; return 7; }
    uint32_t words[4]{};
    if (!read_stats(b,16,words)) { b->failed=true; return 8; }
    if (words[0]) {
        fail("temporal source or candidate contains invalid/unrepresentable components",0);
        b->failed=true; return 9;
    }
    b->history_index=target; b->history_valid=true;
    *out_texture=b->output.Get(); *out_fence=b->fence.Get();
    *out_fence_value=b->fence_value;
    report->history_used=words[2]; report->history_rejected=words[3];
    report->unrepresentable_pixels=words[1]; report->effective_history_weight=
        words[2] ? history_weight : 0.0f;
    report->scene_reset_count=b->scene_reset_count;
    report->source_transport_bytes=uint64_t(b->width)*b->height*b->pixel_bytes;
    report->candidate_transport_bytes=report->source_transport_bytes;
    report->scalar_readback_bytes=16;
    return 0;
}
