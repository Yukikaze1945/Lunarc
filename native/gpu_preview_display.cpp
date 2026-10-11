// Original display-only CUDA/D3D12 area resampling and SDR presentation.
// Processing/export colours are never modified. Only viewport BGRA8 is read back.
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include "nvof_d3d12_interop.h"
using Microsoft::WRL::ComPtr;
namespace {
thread_local char error[512]{};
struct Display {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> output, readback;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    CudaD3D12ColorBridge* input = nullptr;
    ID3D12Resource* source = nullptr;
    ID3D12Fence* producer = nullptr;
    UINT w=0,h=0,ow=0,oh=0,pitch=0,bytes=0;
    UINT64 serial=0;
    bool poisoned=false;
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    ~Display(){if(input)cuda_d3d12_color_destroy(input);if(event)CloseHandle(event);}
};
int fail(const char* text,HRESULT hr){std::snprintf(error,sizeof(error),"%s: 0x%08lx",text,static_cast<unsigned long>(hr));return -1;}
const char* shader=R"(
cbuffer Geometry : register(b0) { uint W,H,OW,OH,Pitch,Bytes,HDR; };
ByteAddressBuffer Source : register(t0);
RWByteAddressBuffer Output : register(u0);
float3 sample(uint x,uint y) {
    uint address=y*Pitch+x*Bytes;
    if(Bytes==4){uint v=Source.Load(address);return float3(v&255,(v>>8)&255,(v>>16)&255)/255.;}
    uint2 v=Source.Load2(address);
    return float3(f16tof32(v.x&65535),f16tof32(v.x>>16),f16tof32(v.y&65535));
}
[numthreads(16,16,1)] void main(uint3 id:SV_DispatchThreadID) {
    if(id.x>=OW||id.y>=OH)return;
    float2 start=float2(id.xy)*float2(W,H)/float2(OW,OH);
    float2 stop=float2(id.xy+1)*float2(W,H)/float2(OW,OH);
    float3 sum=0;float weight=0;
    for(uint y=(uint)floor(start.y);y<min(H,(uint)ceil(stop.y));++y)
    for(uint x=(uint)floor(start.x);x<min(W,(uint)ceil(stop.x));++x){
        float a=max(0.,min(stop.x,x+1.)-max(start.x,(float)x))*max(0.,min(stop.y,y+1.)-max(start.y,(float)y));
        sum+=sample(x,y)*a;weight+=a;
    }
    float3 rgb=sum/max(weight,1e-8);
    if(Bytes==8){
        if(HDR!=0)rgb/=1.+max(dot(rgb,float3(.2126,.7152,.0722)),0.);
        rgb=saturate(rgb);
        rgb=float3(rgb.r<.018?4.5*rgb.r:1.099*pow(rgb.r,.45)-.099,
                   rgb.g<.018?4.5*rgb.g:1.099*pow(rgb.g,.45)-.099,
                   rgb.b<.018?4.5*rgb.b:1.099*pow(rgb.b,.45)-.099);
    }
    uint3 v=(uint3)round(saturate(rgb)*255.);
    Output.Store((id.y*OW+id.x)*4,v.b|(v.g<<8)|(v.r<<16)|(255u<<24));
})";
}
#define API extern "C" __declspec(dllexport)
API const char* gpu_preview_display_error(){return error;}
API void gpu_preview_display_close(Display* d){
    // Failed completion is not proof that submitted GPU resources are idle.
    if(d&&d->poisoned&&SUCCEEDED(d->device->GetDeviceRemovedReason()))return;
    delete d;
}
API int gpu_preview_display_open(ID3D12Device* device,void* context,UINT w,UINT h,
                                 UINT ow,UINT oh,UINT bytes,Display** out){
    if(!out)return fail("null display output",E_INVALIDARG);
    *out=nullptr;
    if(!device||!context||!w||!h||!ow||!oh||ow>w||oh>h||w>16384||h>16384||(bytes!=4&&bytes!=8))
        return fail("display geometry",E_INVALIDARG);
    auto d=std::make_unique<Display>();d->device=device;d->w=w;d->h=h;d->ow=ow;d->oh=oh;d->bytes=bytes;d->pitch=(w*bytes+255)&~255u;
    if(!d->event)return fail("display event",HRESULT_FROM_WIN32(GetLastError()));
    if(cuda_d3d12_color_create2(device,context,w,h,d->pitch,bytes,0,&d->input))return fail(nvof_d3d12_last_error(),E_FAIL);
    void* source=nullptr;void* producer=nullptr;
    if(cuda_d3d12_color_get_resources(d->input,&source,&producer))return fail(nvof_d3d12_last_error(),E_FAIL);
    d->source=static_cast<ID3D12Resource*>(source);d->producer=static_cast<ID3D12Fence*>(producer);
    D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr=device->CreateCommandQueue(&q,IID_PPV_ARGS(&d->queue));
    if(SUCCEEDED(hr))hr=device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&d->allocator));
    if(SUCCEEDED(hr))hr=device->CreateCommandList(0,q.Type,d->allocator.Get(),nullptr,IID_PPV_ARGS(&d->list));
    if(SUCCEEDED(hr))hr=d->list->Close();
    if(SUCCEEDED(hr))hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&d->fence));
    if(FAILED(hr))return fail("display queue",hr);
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=UINT64(ow)*oh*4;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;b.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&d->output));
    heap.Type=D3D12_HEAP_TYPE_READBACK;b.Flags=D3D12_RESOURCE_FLAG_NONE;
    if(SUCCEEDED(hr))hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&d->readback));
    if(FAILED(hr))return fail("display buffers",hr);
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[0].Constants.Num32BitValues=7;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC desc{};desc.NumParameters=3;desc.pParameters=params;
    ComPtr<ID3DBlob> serialized,errors,code;
    hr=D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&errors);
    if(SUCCEEDED(hr))hr=device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&d->root));
    if(FAILED(hr))return fail("display root signature",hr);
    hr=D3DCompile(shader,std::strlen(shader),"preview_display",nullptr,nullptr,"main","cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
    if(FAILED(hr))return fail(errors?static_cast<const char*>(errors->GetBufferPointer()):"display shader",hr);
    D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};ps.pRootSignature=d->root.Get();ps.CS={code->GetBufferPointer(),code->GetBufferSize()};
    hr=device->CreateComputePipelineState(&ps,IID_PPV_ARGS(&d->pipeline));
    if(FAILED(hr))return fail("display pipeline",hr);
    *out=d.release();return 0;
}
API int gpu_preview_display_read(Display* d,UINT64 pointer,size_t pitch,UINT hdr,uint8_t* host,size_t stride){
    if(!d||!pointer||pitch<size_t(d->w)*d->bytes||!host||stride<size_t(d->ow)*4||hdr>1||d->poisoned)
        return fail("display read arguments",E_INVALIDARG);
    UINT64 value=0;
    if(cuda_d3d12_color_copy(d->input,pointer,pitch,&value))return fail(nvof_d3d12_last_error(),E_FAIL);
    HRESULT hr=d->queue->Wait(d->producer,value);
    if(SUCCEEDED(hr))hr=d->allocator->Reset();
    if(SUCCEEDED(hr))hr=d->list->Reset(d->allocator.Get(),d->pipeline.Get());
    if(FAILED(hr))return fail("display begin",hr);
    D3D12_RESOURCE_BARRIER barriers[2]{};
    for(auto& b:barriers){b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;}
    barriers[0].Transition={d->source,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE};
    d->list->ResourceBarrier(1,barriers);
    const UINT constants[]={d->w,d->h,d->ow,d->oh,d->pitch,d->bytes,hdr};
    d->list->SetComputeRootSignature(d->root.Get());d->list->SetComputeRoot32BitConstants(0,7,constants,0);
    d->list->SetComputeRootShaderResourceView(1,d->source->GetGPUVirtualAddress());d->list->SetComputeRootUnorderedAccessView(2,d->output->GetGPUVirtualAddress());
    d->list->Dispatch((d->ow+15)/16,(d->oh+15)/16,1);
    barriers[0].Transition.StateBefore=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;barriers[0].Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;
    barriers[1].Transition={d->output.Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COPY_SOURCE};
    d->list->ResourceBarrier(2,barriers);d->list->CopyBufferRegion(d->readback.Get(),0,d->output.Get(),0,UINT64(d->ow)*d->oh*4);
    barriers[1].Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;barriers[1].Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    d->list->ResourceBarrier(1,&barriers[1]);hr=d->list->Close();
    if(FAILED(hr))return fail("display submit",hr);
    ID3D12CommandList* lists[]={d->list.Get()};d->poisoned=true;d->queue->ExecuteCommandLists(1,lists);
    hr=d->queue->Signal(d->fence.Get(),++d->serial);
    if(SUCCEEDED(hr))hr=d->fence->SetEventOnCompletion(d->serial,d->event);
    if(FAILED(hr))return fail("display completion",hr);
    if(WaitForSingleObject(d->event,30000)!=WAIT_OBJECT_0)return fail("display timeout",HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    d->poisoned=false;
    void* mapped=nullptr;D3D12_RANGE range{0,size_t(d->ow)*d->oh*4};hr=d->readback->Map(0,&range,&mapped);
    if(FAILED(hr))return fail("display map",hr);
    for(UINT row=0;row<d->oh;++row)std::memcpy(host+row*stride,static_cast<uint8_t*>(mapped)+size_t(row)*d->ow*4,size_t(d->ow)*4);
    D3D12_RANGE none{0,0};d->readback->Unmap(0,&none);return 0;
}
