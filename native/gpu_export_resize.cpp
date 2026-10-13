// Original separable export resampler. FP32 accumulation; FP16 stays signed.
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
struct Resize {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> scratch;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> horizontal,vertical;
    CudaD3D12ColorBridge *input=nullptr,*output=nullptr;
    ID3D12Resource *source=nullptr,*destination=nullptr;
    ID3D12Fence* producer=nullptr;
    UINT w=0,h=0,ow=0,oh=0,pitch=0,opitch=0,bytes=0,algorithm=0;
    UINT64 serial=0,pointer=0;
    bool poisoned=false;
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    ~Resize(){if(output)cuda_d3d12_color_destroy(output);if(input)cuda_d3d12_color_destroy(input);if(event)CloseHandle(event);}
};
int fail(const char* text,HRESULT hr){std::snprintf(error,sizeof(error),"%s: 0x%08lx",text,static_cast<unsigned long>(hr));return -1;}
const char* shader=R"(
cbuffer Geometry : register(b0) { uint W,H,OW,OH,Pitch,OPitch,Bytes,Algorithm; };
ByteAddressBuffer Source : register(t0);
RWByteAddressBuffer Output : register(u0);
float4 sampleInput(uint x,uint y) {
    uint address=y*Pitch+x*Bytes;
    if(Bytes==4){uint v=Source.Load(address);return float4(v&255,(v>>8)&255,(v>>16)&255,v>>24)/255.;}
    uint2 v=Source.Load2(address);
    return float4(f16tof32(v.x&65535),f16tof32(v.x>>16),f16tof32(v.y&65535),f16tof32(v.y>>16));
}
float kernel(float x) {
    x=abs(x);
    if(Algorithm==1)return max(1.-x,0.);
    if(Algorithm==2){if(x<1.)return 1.5*x*x*x-2.5*x*x+1.;if(x<2.)return -.5*x*x*x+2.5*x*x-4.*x+2.;return 0.;}
    if(x<1e-6)return 1.;if(x>=2.)return 0.;
    float p=3.141592653589793*x;return sin(p)*sin(p*.5)/(p*p*.5);
}
[numthreads(16,16,1)] void Horizontal(uint3 id:SV_DispatchThreadID) {
    if(id.x>=OW||id.y>=H)return;
    float centre=(id.x+.5)*W/OW-.5;
    float4 sum=0;float norm=0;
    if(Algorithm==0)sum=sampleInput(min(W-1,(uint)floor((id.x+.5)*W/OW)),id.y);
    else{
        float scale=max(1.,(float)W/OW),radius=(Algorithm==1?1.:2.)*scale;
        for(int x=(int)ceil(centre-radius);x<=(int)floor(centre+radius);++x){
            float a=kernel((x-centre)/scale);if(a!=0.)sum+=a*sampleInput((uint)clamp(x,0,(int)W-1),id.y);norm+=a;
        }
        sum/=norm;
    }
    Output.Store4((id.y*OW+id.x)*16,asuint(sum));
}
[numthreads(16,16,1)] void Vertical(uint3 id:SV_DispatchThreadID) {
    if(id.x>=OW||id.y>=OH)return;
    float centre=(id.y+.5)*H/OH-.5;
    float4 sum=0;float norm=0;
    if(Algorithm==0)sum=asfloat(Source.Load4((min(H-1,(uint)floor((id.y+.5)*H/OH))*OW+id.x)*16));
    else{
        float scale=max(1.,(float)H/OH),radius=(Algorithm==1?1.:2.)*scale;
        for(int y=(int)ceil(centre-radius);y<=(int)floor(centre+radius);++y){
            float a=kernel((y-centre)/scale);if(a!=0.)sum+=a*asfloat(Source.Load4(((uint)clamp(y,0,(int)H-1)*OW+id.x)*16));norm+=a;
        }
        sum/=norm;
    }
    uint address=id.y*OPitch+id.x*Bytes;
    if(Bytes==4){uint4 v=(uint4)round(saturate(sum)*255.);Output.Store(address,v.x|(v.y<<8)|(v.z<<16)|(v.w<<24));}
    else{uint4 v=f32tof16(sum);Output.Store2(address,uint2(v.x|(v.y<<16),v.z|(v.w<<16)));}
})";
void transition(ID3D12GraphicsCommandList* list,ID3D12Resource* resource,D3D12_RESOURCE_STATES before,D3D12_RESOURCE_STATES after){
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition={resource,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,before,after};list->ResourceBarrier(1,&b);
}
}
#define API extern "C" __declspec(dllexport)
API const char* gpu_export_resize_error(){return error;}
API void gpu_export_resize_close(Resize* d){
    if(d&&d->poisoned&&SUCCEEDED(d->device->GetDeviceRemovedReason()))return;
    delete d;
}
API int gpu_export_resize_open(ID3D12Device* device,void* context,UINT w,UINT h,
                                 UINT ow,UINT oh,UINT bytes,UINT algorithm,Resize** out){
    if(!out)return fail("null resize output",E_INVALIDARG);*out=nullptr;
    if(!device||!context||!w||!h||!ow||!oh||w>16384||h>16384||ow>16384||oh>16384||(bytes!=4&&bytes!=8)||algorithm>3)
        return fail("resize geometry/format/algorithm",E_INVALIDARG);
    auto d=std::make_unique<Resize>();d->device=device;d->w=w;d->h=h;d->ow=ow;d->oh=oh;d->bytes=bytes;
    d->pitch=(w*bytes+255)&~255u;d->opitch=(ow*bytes+255)&~255u;d->algorithm=algorithm;
    if(!d->event)return fail("resize event",HRESULT_FROM_WIN32(GetLastError()));
    if(cuda_d3d12_color_create2(device,context,w,h,d->pitch,bytes,0,&d->input)||
       cuda_d3d12_color_create2(device,context,ow,oh,d->opitch,bytes,1,&d->output))return fail(nvof_d3d12_last_error(),E_FAIL);
    void *source=nullptr,*producer=nullptr,*destination=nullptr,*output_fence=nullptr;
    if(cuda_d3d12_color_get_resources(d->input,&source,&producer)||
       cuda_d3d12_color_get_resources(d->output,&destination,&output_fence)||
       cuda_d3d12_color_get_device_pointer(d->output,&d->pointer,&d->opitch))return fail(nvof_d3d12_last_error(),E_FAIL);
    d->source=static_cast<ID3D12Resource*>(source);d->producer=static_cast<ID3D12Fence*>(producer);d->destination=static_cast<ID3D12Resource*>(destination);
    D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr=device->CreateCommandQueue(&q,IID_PPV_ARGS(&d->queue));
    if(SUCCEEDED(hr))hr=device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&d->allocator));
    if(SUCCEEDED(hr))hr=device->CreateCommandList(0,q.Type,d->allocator.Get(),nullptr,IID_PPV_ARGS(&d->list));
    if(SUCCEEDED(hr))hr=d->list->Close();
    if(SUCCEEDED(hr))hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&d->fence));
    if(FAILED(hr))return fail("resize queue",hr);
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC b{};b.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;b.Width=UINT64(ow)*h*16;b.Height=1;b.DepthOrArraySize=1;b.MipLevels=1;b.SampleDesc.Count=1;b.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;b.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&b,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,nullptr,IID_PPV_ARGS(&d->scratch));
    if(FAILED(hr))return fail("resize scratch",hr);
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;params[0].Constants.Num32BitValues=8;
    params[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_UAV;
    D3D12_ROOT_SIGNATURE_DESC desc{};desc.NumParameters=3;desc.pParameters=params;
    ComPtr<ID3DBlob> serialized,errors;
    hr=D3D12SerializeRootSignature(&desc,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&errors);
    if(SUCCEEDED(hr))hr=device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&d->root));
    if(FAILED(hr))return fail("resize root signature",hr);
    const char* entries[]={"Horizontal","Vertical"};
    for(unsigned i=0;i<2;++i){
        ComPtr<ID3DBlob> code;
        hr=D3DCompile(shader,std::strlen(shader),"export_resize",nullptr,nullptr,entries[i],"cs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
        if(FAILED(hr))return fail(errors?static_cast<const char*>(errors->GetBufferPointer()):"resize shader",hr);
        D3D12_COMPUTE_PIPELINE_STATE_DESC ps{};ps.pRootSignature=d->root.Get();ps.CS={code->GetBufferPointer(),code->GetBufferSize()};
        if(i==0)hr=device->CreateComputePipelineState(&ps,IID_PPV_ARGS(&d->horizontal));
        else hr=device->CreateComputePipelineState(&ps,IID_PPV_ARGS(&d->vertical));
        if(FAILED(hr))return fail("resize pipeline",hr);
    }
    *out=d.release();return 0;
}
API int gpu_export_resize_run(Resize* d,UINT64 pointer,size_t pitch,UINT64* output,UINT* opitch){
    if(!d||!pointer||pitch<size_t(d->w)*d->bytes||!output||!opitch||d->poisoned)return fail("resize run arguments",E_INVALIDARG);
    UINT64 value=0;if(cuda_d3d12_color_copy(d->input,pointer,pitch,&value))return fail(nvof_d3d12_last_error(),E_FAIL);
    HRESULT hr=d->queue->Wait(d->producer,value);
    if(SUCCEEDED(hr))hr=d->allocator->Reset();
    if(SUCCEEDED(hr))hr=d->list->Reset(d->allocator.Get(),d->horizontal.Get());
    if(FAILED(hr))return fail("resize begin",hr);
    transition(d->list.Get(),d->source,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const UINT constants[]={d->w,d->h,d->ow,d->oh,d->pitch,d->opitch,d->bytes,d->algorithm};
    d->list->SetComputeRootSignature(d->root.Get());d->list->SetComputeRoot32BitConstants(0,8,constants,0);
    d->list->SetComputeRootShaderResourceView(1,d->source->GetGPUVirtualAddress());d->list->SetComputeRootUnorderedAccessView(2,d->scratch->GetGPUVirtualAddress());
    d->list->Dispatch((d->ow+15)/16,(d->h+15)/16,1);
    transition(d->list.Get(),d->scratch.Get(),D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    transition(d->list.Get(),d->destination,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    d->list->SetPipelineState(d->vertical.Get());d->list->SetComputeRootShaderResourceView(1,d->scratch->GetGPUVirtualAddress());d->list->SetComputeRootUnorderedAccessView(2,d->destination->GetGPUVirtualAddress());
    d->list->Dispatch((d->ow+15)/16,(d->oh+15)/16,1);
    transition(d->list.Get(),d->source,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON);
    transition(d->list.Get(),d->scratch.Get(),D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(d->list.Get(),d->destination,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_COMMON);
    hr=d->list->Close();if(FAILED(hr))return fail("resize submit",hr);
    ID3D12CommandList* lists[]={d->list.Get()};d->poisoned=true;d->queue->ExecuteCommandLists(1,lists);
    hr=d->queue->Signal(d->fence.Get(),++d->serial);
    if(SUCCEEDED(hr))hr=d->fence->SetEventOnCompletion(d->serial,d->event);
    if(FAILED(hr))return fail("resize completion",hr);
    if(WaitForSingleObject(d->event,30000)!=WAIT_OBJECT_0)return fail("resize timeout",HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    d->poisoned=false;*output=d->pointer;*opitch=d->opitch;return 0;
}
