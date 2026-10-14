#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
#include "nvof_d3d12_interop.h"
#include "fidelityfx_flow_bridge.h"
#include <FidelityFX/host/ffx_opticalflow.h>
#include <FidelityFX/host/backends/dx12/ffx_dx12.h>

using Microsoft::WRL::ComPtr;
static thread_local std::string g_error;
static void set_error(const char* where, HRESULT hr = S_OK) {
    char b[320]{}; if (FAILED(hr)) std::snprintf(b,sizeof(b),"%s (HRESULT 0x%08lx)",where,(unsigned long)hr);
    else std::snprintf(b,sizeof(b),"%s",where); g_error=b;
}
static uint32_t align256(uint32_t n) { return (n + 255u) & ~255u; }

struct FfxFlowHandle {
    ID3D12Device* device{}; void* cuda{}; uint32_t w{},h{},analysisW{},analysisH{},gridW{},gridH{},inPitch{},outPitch{};
    CudaD3D12ColorBridge* input{}; CudaD3D12ColorBridge* output{};
    ComPtr<ID3D12Resource> inputTexture, analysisTexture, flowTexture, scdTexture, readback;
    ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list; ComPtr<ID3D12Fence> fence; HANDLE eventHandle{}; uint64_t fenceValue{};
    ComPtr<ID3D12RootSignature> root; ComPtr<ID3D12PipelineState> pso, preparePso;
    ComPtr<ID3D12DescriptorHeap> heap; UINT descriptorStride{};
    FfxInterface backend{}; FfxOpticalflowContext context{}; bool contextCreated{}; std::vector<uint8_t> scratch;
    FfxResource flowResource{},scdResource{}; uint64_t calls{},dispatches{},resets{},scdChanges{}; uint32_t scdHistory{},scdWorkgroups{};
    std::mutex mutex;
    bool poisoned{};
};

static const char* kExpand = R"(
Texture2D<int2> flowGrid : register(t0);
RWByteAddressBuffer outputFlow : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint pitch; uint pad; uint analysisWidth; uint analysisHeight; uint pad2; uint pad3; };
[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID) {
  if(id.x>=width || id.y>=height) return;
  uint2 pixel = min(uint2((float2(id.xy)+0.5)*float2(analysisWidth,analysisHeight)/float2(width,height)), uint2(analysisWidth-1,analysisHeight-1));
  float2 v=(float2)flowGrid.Load(int3(pixel/8,0))*float2(width,height)/float2(analysisWidth,analysisHeight);
  uint a=(id.y*pitch)+(id.x*8);
  outputFlow.Store(a,asuint((float)v.x)); outputFlow.Store(a+4,asuint((float)v.y));
})";
// Only motion guidance is averaged; the NR picture plane stays at its own extent.
static const char* kPrepare = R"(
Texture2D<float4> picture : register(t0);
RWTexture2D<float4> analysis : register(u0);
cbuffer Params : register(b0) { uint width; uint height; uint pitch; uint pad; uint analysisWidth; uint analysisHeight; uint pad2; uint pad3; };
[numthreads(8,8,1)] void main(uint3 id:SV_DispatchThreadID) {
  if(id.x>=analysisWidth || id.y>=analysisHeight) return;
  uint2 lo=id.xy*uint2(width,height)/uint2(analysisWidth,analysisHeight);
  uint2 hi=min(max(lo+1,(id.xy+1)*uint2(width,height)/uint2(analysisWidth,analysisHeight)),uint2(width,height));
  float4 sum=0; uint count=0;
  for(uint y=lo.y;y<hi.y;++y) for(uint x=lo.x;x<hi.x;++x) {sum+=picture.Load(int3(x,y,0));++count;}
  analysis[id.xy]=sum/count;
})";

static bool wait_gpu(FfxFlowHandle* h) {
    const uint64_t v=++h->fenceValue; HRESULT hr=h->queue->Signal(h->fence.Get(),v); if(FAILED(hr)){set_error("D3D12 queue signal failed",hr);return false;}
    if(h->fence->GetCompletedValue()<v){ hr=h->fence->SetEventOnCompletion(v,h->eventHandle); if(FAILED(hr)){set_error("fence event setup failed",hr);return false;} DWORD wr=WaitForSingleObject(h->eventHandle,30000); if(wr!=WAIT_OBJECT_0){set_error("GPU fence wait timed out");return false;} }
    return true;
}
static bool make_texture(ID3D12Device* d,uint32_t w,uint32_t h,DXGI_FORMAT fmt,D3D12_RESOURCE_STATES state,ComPtr<ID3D12Resource>& out) {
    D3D12_RESOURCE_DESC rd{}; rd.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D; rd.Width=w; rd.Height=h; rd.DepthOrArraySize=1; rd.MipLevels=1; rd.Format=fmt; rd.SampleDesc.Count=1; rd.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN; rd.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    D3D12_HEAP_PROPERTIES hp{}; hp.Type=D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr=d->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&rd,state,nullptr,IID_PPV_ARGS(&out)); if(FAILED(hr)){set_error("create optical-flow texture failed",hr);return false;} return true;
}
static bool make_pipeline(FfxFlowHandle* h) {
    D3D12_DESCRIPTOR_RANGE ranges[2]{}; ranges[0]={D3D12_DESCRIPTOR_RANGE_TYPE_SRV,1,0,0,0}; ranges[1]={D3D12_DESCRIPTOR_RANGE_TYPE_UAV,1,0,0,0};
    D3D12_ROOT_PARAMETER p[3]{}; for(int i=0;i<2;i++){p[i].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;p[i].DescriptorTable.NumDescriptorRanges=1;p[i].DescriptorTable.pDescriptorRanges=&ranges[i];p[i].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;}
    p[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;p[2].Constants={0,0,8};p[2].ShaderVisibility=D3D12_SHADER_VISIBILITY_ALL;
    D3D12_ROOT_SIGNATURE_DESC rs{};rs.NumParameters=3;rs.pParameters=p;rs.Flags=D3D12_ROOT_SIGNATURE_FLAG_NONE;
    ComPtr<ID3DBlob> blob,err; HRESULT hr=D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&err);if(FAILED(hr)){set_error("root signature serialization failed",hr);return false;}
    hr=h->device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&h->root));if(FAILED(hr)){set_error("root signature creation failed",hr);return false;}
    ComPtr<ID3DBlob> cs; hr=D3DCompile(kExpand,strlen(kExpand),"ffx_flow_expand",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&cs,&err);if(FAILED(hr)){set_error("flow expansion shader compile failed",hr);return false;}
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};pd.pRootSignature=h->root.Get();pd.CS={cs->GetBufferPointer(),cs->GetBufferSize()};hr=h->device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&h->pso));if(FAILED(hr)){set_error("flow expansion pipeline creation failed",hr);return false;}
    D3D12_DESCRIPTOR_HEAP_DESC hd{};hd.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;hd.NumDescriptors=4;hd.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;hr=h->device->CreateDescriptorHeap(&hd,IID_PPV_ARGS(&h->heap));if(FAILED(hr)){set_error("flow expansion descriptor heap failed",hr);return false;}h->descriptorStride=h->device->GetDescriptorHandleIncrementSize(hd.Type);
    D3D12_SHADER_RESOURCE_VIEW_DESC sv{};sv.Format=DXGI_FORMAT_R16G16_SINT;sv.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;sv.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;sv.Texture2D.MipLevels=1;h->device->CreateShaderResourceView(h->flowTexture.Get(),&sv,h->heap->GetCPUDescriptorHandleForHeapStart());
    D3D12_UNORDERED_ACCESS_VIEW_DESC uv{};uv.Format=DXGI_FORMAT_R32_TYPELESS;uv.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;uv.Buffer.NumElements=(h->outPitch*h->h)/4;uv.Buffer.Flags=D3D12_BUFFER_UAV_FLAG_RAW;D3D12_CPU_DESCRIPTOR_HANDLE uh=h->heap->GetCPUDescriptorHandleForHeapStart();uh.ptr+=h->descriptorStride;void* ob{};void* of{}; if(cuda_d3d12_color_get_resources(h->output,&ob,&of)!=0){set_error("get output shared buffer failed");return false;}h->device->CreateUnorderedAccessView(static_cast<ID3D12Resource*>(ob),nullptr,&uv,uh);
    if(h->analysisTexture) {
        cs.Reset(); err.Reset();
        hr=D3DCompile(kPrepare,strlen(kPrepare),"ffx_flow_prepare",nullptr,nullptr,"main","cs_5_0",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&cs,&err);
        if(FAILED(hr)){set_error("analysis shader compile failed",hr);return false;}
        pd.CS={cs->GetBufferPointer(),cs->GetBufferSize()};
        hr=h->device->CreateComputePipelineState(&pd,IID_PPV_ARGS(&h->preparePso));
        if(FAILED(hr)){set_error("analysis pipeline creation failed",hr);return false;}
        sv.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
        auto handle=h->heap->GetCPUDescriptorHandleForHeapStart(); handle.ptr+=2*h->descriptorStride;
        h->device->CreateShaderResourceView(h->inputTexture.Get(),&sv,handle);
        D3D12_UNORDERED_ACCESS_VIEW_DESC target{};target.Format=DXGI_FORMAT_R8G8B8A8_UNORM;target.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
        handle.ptr+=h->descriptorStride;h->device->CreateUnorderedAccessView(h->analysisTexture.Get(),nullptr,&target,handle);
    }
    return true;
}
static void cleanup(FfxFlowHandle* h) {
    if(!h)return; if(h->contextCreated){ ffxOpticalflowContextDestroy(&h->context);h->contextCreated=false; }
    if(h->list)h->list.Reset(); if(h->allocator)h->allocator.Reset();
    if(h->input)cuda_d3d12_color_destroy(h->input); if(h->output)cuda_d3d12_color_destroy(h->output);
    if(h->eventHandle)CloseHandle(h->eventHandle); if(h->device)h->device->Release();
}
extern "C" {
uint32_t ffx_flow_abi_version(void){return 1;}
const char* ffx_flow_last_error(void){return g_error.c_str();}
int32_t ffx_flow_open(void* dev,void* cuda,uint32_t w,uint32_t h,FfxFlowHandle** out){
    return ffx_flow_open2(dev,cuda,w,h,1,out);
}
int32_t ffx_flow_open2(void* dev,void* cuda,uint32_t w,uint32_t h,uint32_t mode,FfxFlowHandle** out){
    if(!dev||!cuda||!out||!w||!h||w>16384||h>16384||mode>1){set_error("invalid flow open arguments");return -1;}*out=nullptr;
    auto* x=new(std::nothrow) FfxFlowHandle();if(!x){set_error("allocation failed");return -2;}x->device=static_cast<ID3D12Device*>(dev);x->device->AddRef();x->cuda=cuda;x->w=w;x->h=h;x->analysisW=mode ? w : (w+1)/2;x->analysisH=mode ? h : (h+1)/2;x->gridW=(x->analysisW+7)/8;x->gridH=(x->analysisH+7)/8;x->inPitch=align256(w*4);x->outPitch=align256(w*8);
    D3D12_COMMAND_QUEUE_DESC qd{}; qd.Type=D3D12_COMMAND_LIST_TYPE_COMPUTE; HRESULT hr=S_OK;
    FfxOpticalflowContextDescription cd{}; FfxErrorCode ec=FFX_OK; FfxOpticalflowSharedResourceDescriptions sr{};
    D3D12_RESOURCE_DESC rb{}; D3D12_HEAP_PROPERTIES rh{};
    hr=x->device->CreateCommandQueue(&qd,IID_PPV_ARGS(&x->queue));if(FAILED(hr)){set_error("create compute queue failed",hr);goto fail;}
    hr=x->device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&x->fence));if(FAILED(hr)){set_error("create fence failed",hr);goto fail;}x->eventHandle=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!x->eventHandle){set_error("create fence event failed",HRESULT_FROM_WIN32(GetLastError()));goto fail;}
    hr=x->device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,IID_PPV_ARGS(&x->allocator));if(FAILED(hr)){set_error("create command allocator failed",hr);goto fail;}hr=x->device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_COMPUTE,x->allocator.Get(),nullptr,IID_PPV_ARGS(&x->list));if(FAILED(hr)){set_error("create command list failed",hr);goto fail;}x->list->Close();
    if(cuda_d3d12_color_create2(dev,cuda,w,h,x->inPitch,4,0,&x->input)!=0){set_error("create CUDA RGBA8 input interop failed");goto fail;}if(cuda_d3d12_color_create2(dev,cuda,w,h,x->outPitch,8,1,&x->output)!=0){set_error("create CUDA float2 output interop failed");goto fail;}
    if(!make_texture(x->device,w,h,DXGI_FORMAT_R8G8B8A8_UNORM,D3D12_RESOURCE_STATE_COPY_DEST,x->inputTexture)||!make_texture(x->device,x->gridW,x->gridH,DXGI_FORMAT_R16G16_SINT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,x->flowTexture)||!make_texture(x->device,3,1,DXGI_FORMAT_R32_UINT,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,x->scdTexture))goto fail;
    if(mode==0 && !make_texture(x->device,x->analysisW,x->analysisH,DXGI_FORMAT_R8G8B8A8_UNORM,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,x->analysisTexture))goto fail;
    cd.backendInterface={};cd.resolution={x->analysisW,x->analysisH};x->scratch.resize(ffxGetScratchMemorySizeDX12(FFX_OPTICALFLOW_CONTEXT_COUNT));ec=ffxGetInterfaceDX12(&cd.backendInterface,ffxGetDeviceDX12(x->device),x->scratch.data(),x->scratch.size(),FFX_OPTICALFLOW_CONTEXT_COUNT);if(ec!=FFX_OK){set_error("FidelityFX DX12 interface creation failed");goto fail;}
    ec=ffxOpticalflowContextCreate(&x->context,&cd);if(ec!=FFX_OK){set_error("FidelityFX Optical Flow context creation failed");goto fail;}x->contextCreated=true;
    ec=ffxOpticalflowGetSharedResourceDescriptions(&x->context,&sr);if(ec!=FFX_OK){set_error("FidelityFX resource query failed");goto fail;}
    x->flowResource=ffxGetResourceDX12(x->flowTexture.Get(),sr.opticalFlowVector.resourceDescription,sr.opticalFlowVector.name,FFX_RESOURCE_STATE_UNORDERED_ACCESS);x->scdResource=ffxGetResourceDX12(x->scdTexture.Get(),sr.opticalFlowSCD.resourceDescription,sr.opticalFlowSCD.name,FFX_RESOURCE_STATE_UNORDERED_ACCESS);
    rb.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;rb.Width=256;rb.Height=1;rb.DepthOrArraySize=1;rb.MipLevels=1;rb.SampleDesc.Count=1;rb.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;rh.Type=D3D12_HEAP_TYPE_READBACK;hr=x->device->CreateCommittedResource(&rh,D3D12_HEAP_FLAG_NONE,&rb,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&x->readback));if(FAILED(hr)){set_error("create SCD readback buffer failed",hr);goto fail;}
    if(!make_pipeline(x))goto fail;*out=x;return 0;
fail: cleanup(x);delete x;return -3;
}
int32_t ffx_flow_process(FfxFlowHandle* x,uint64_t src,uint32_t srcPitch,int32_t reset,uint64_t* out,uint32_t* outPitch){
    if(!x||!src||!out||!outPitch||srcPitch<x->w*4){set_error("invalid flow process arguments");return -1;}std::lock_guard<std::mutex> lock(x->mutex);++x->calls;
    if(x->poisoned){set_error("flow queue lost completion proof; reopen required");return -1;}
    uint64_t stageValue=0;if(cuda_d3d12_color_copy(x->input,src,srcPitch,&stageValue)!=0){set_error("CUDA input transfer failed");return -2;}void* stageBuf{};void* stageFence{};void* outputBuf{};void* ignored{};if(cuda_d3d12_color_get_resources(x->input,&stageBuf,&stageFence)!=0||cuda_d3d12_color_get_resources(x->output,&outputBuf,&ignored)!=0){set_error("interop resource lookup failed");return -3;}
    HRESULT hr=x->allocator->Reset();if(FAILED(hr)){set_error("command allocator reset failed",hr);return -4;}hr=x->list->Reset(x->allocator.Get(),nullptr);if(FAILED(hr)){set_error("command list reset failed",hr);return -4;}
    hr=x->queue->Wait(static_cast<ID3D12Fence*>(stageFence),stageValue);
    if(FAILED(hr)){set_error("input producer wait failed",hr);x->list->Close();return -4;}
    D3D12_RESOURCE_BARRIER prep[2]{};UINT prepCount=0;
    if(x->calls>1){prep[prepCount].Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;prep[prepCount].Transition.pResource=x->scdTexture.Get();prep[prepCount].Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;prep[prepCount].Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;prep[prepCount].Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;++prepCount;}
    if(x->calls>1){prep[prepCount].Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;prep[prepCount].Transition.pResource=static_cast<ID3D12Resource*>(stageBuf);prep[prepCount].Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;prep[prepCount].Transition.StateBefore=D3D12_RESOURCE_STATE_COMMON;prep[prepCount].Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;++prepCount;}
    if(prepCount)x->list->ResourceBarrier(prepCount,prep);
    else {D3D12_RESOURCE_BARRIER tr{};tr.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;tr.Transition.pResource=static_cast<ID3D12Resource*>(stageBuf);tr.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;tr.Transition.StateBefore=D3D12_RESOURCE_STATE_COMMON;tr.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;x->list->ResourceBarrier(1,&tr);}
    D3D12_TEXTURE_COPY_LOCATION sl{};sl.pResource=static_cast<ID3D12Resource*>(stageBuf);sl.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;sl.PlacedFootprint.Footprint.Format=DXGI_FORMAT_R8G8B8A8_UNORM;sl.PlacedFootprint.Footprint.Width=x->w;sl.PlacedFootprint.Footprint.Height=x->h;sl.PlacedFootprint.Footprint.Depth=1;sl.PlacedFootprint.Footprint.RowPitch=x->inPitch;
    D3D12_TEXTURE_COPY_LOCATION dl{};dl.pResource=x->inputTexture.Get();dl.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;x->list->CopyTextureRegion(&dl,0,0,0,&sl,nullptr);
    D3D12_RESOURCE_BARRIER toRead{};toRead.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;toRead.Transition.pResource=x->inputTexture.Get();toRead.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;toRead.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;toRead.Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;x->list->ResourceBarrier(1,&toRead);
    uint32_t constants[8]={x->w,x->h,x->outPitch,0,x->analysisW,x->analysisH,0,0};
    ID3D12Resource* sdkInput=x->inputTexture.Get();
    if(x->analysisTexture) {
        x->list->SetPipelineState(x->preparePso.Get());x->list->SetComputeRootSignature(x->root.Get());x->list->SetDescriptorHeaps(1,x->heap.GetAddressOf());
        auto gh=x->heap->GetGPUDescriptorHandleForHeapStart();gh.ptr+=2*x->descriptorStride;x->list->SetComputeRootDescriptorTable(0,gh);gh.ptr+=x->descriptorStride;x->list->SetComputeRootDescriptorTable(1,gh);
        x->list->SetComputeRoot32BitConstants(2,8,constants,0);x->list->Dispatch((x->analysisW+7)/8,(x->analysisH+7)/8,1);
        auto barrier=toRead;barrier.Transition.pResource=x->analysisTexture.Get();barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        x->list->ResourceBarrier(1,&barrier);sdkInput=x->analysisTexture.Get();
    }
    FfxOpticalflowDispatchDescription dd{};dd.commandList=ffxGetCommandListDX12(x->list.Get());dd.color=ffxGetResourceDX12(sdkInput,ffxGetResourceDescriptionDX12(sdkInput),L"input",FFX_RESOURCE_STATE_COMPUTE_READ);dd.opticalFlowVector=x->flowResource;dd.opticalFlowSCD=x->scdResource;dd.reset=reset!=0;dd.backbufferTransferFunction=0;dd.minMaxLuminance={0.0f,1.0f};FfxErrorCode ec=ffxOpticalflowContextDispatch(&x->context,&dd);if(ec!=FFX_OK){set_error("FidelityFX Optical Flow dispatch failed");x->list->Close();return -5;}
    D3D12_RESOURCE_BARRIER readFlow{};readFlow.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;readFlow.Transition.pResource=x->flowTexture.Get();readFlow.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;readFlow.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;readFlow.Transition.StateAfter=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;x->list->ResourceBarrier(1,&readFlow);
    D3D12_RESOURCE_BARRIER outputBarrier{};outputBarrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;outputBarrier.Transition.pResource=static_cast<ID3D12Resource*>(outputBuf);outputBarrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;outputBarrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COMMON;outputBarrier.Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;x->list->ResourceBarrier(1,&outputBarrier);
    x->list->SetPipelineState(x->pso.Get());x->list->SetComputeRootSignature(x->root.Get());x->list->SetDescriptorHeaps(1,x->heap.GetAddressOf());D3D12_GPU_DESCRIPTOR_HANDLE gh=x->heap->GetGPUDescriptorHandleForHeapStart();x->list->SetComputeRootDescriptorTable(0,gh);gh.ptr+=x->descriptorStride;x->list->SetComputeRootDescriptorTable(1,gh);x->list->SetComputeRoot32BitConstants(2,8,constants,0);x->list->Dispatch((x->w+7)/8,(x->h+7)/8,1);
    outputBarrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;outputBarrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;x->list->ResourceBarrier(1,&outputBarrier);
    readFlow.Transition.StateBefore=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;readFlow.Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;x->list->ResourceBarrier(1,&readFlow);
    D3D12_RESOURCE_BARRIER returnInput{};returnInput.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;returnInput.Transition.pResource=x->inputTexture.Get();returnInput.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;returnInput.Transition.StateBefore=D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;returnInput.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_DEST;x->list->ResourceBarrier(1,&returnInput);
    if(x->analysisTexture) {
        auto barrier=returnInput;barrier.Transition.pResource=x->analysisTexture.Get();barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;x->list->ResourceBarrier(1,&barrier);
    }
    D3D12_RESOURCE_BARRIER scdBarrier{};scdBarrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;scdBarrier.Transition.pResource=x->scdTexture.Get();scdBarrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;scdBarrier.Transition.StateBefore=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;scdBarrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;x->list->ResourceBarrier(1,&scdBarrier);
    D3D12_TEXTURE_COPY_LOCATION scdSrc{};scdSrc.pResource=x->scdTexture.Get();scdSrc.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;D3D12_TEXTURE_COPY_LOCATION scdDst{};scdDst.pResource=x->readback.Get();scdDst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;scdDst.PlacedFootprint.Footprint.Format=DXGI_FORMAT_R32_UINT;scdDst.PlacedFootprint.Footprint.Width=3;scdDst.PlacedFootprint.Footprint.Height=1;scdDst.PlacedFootprint.Footprint.Depth=1;scdDst.PlacedFootprint.Footprint.RowPitch=256;x->list->CopyTextureRegion(&scdDst,0,0,0,&scdSrc,nullptr);
    D3D12_RESOURCE_BARRIER returnStage{};returnStage.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;returnStage.Transition.pResource=static_cast<ID3D12Resource*>(stageBuf);returnStage.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;returnStage.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;returnStage.Transition.StateAfter=D3D12_RESOURCE_STATE_COMMON;x->list->ResourceBarrier(1,&returnStage);
    hr=x->list->Close();if(FAILED(hr)){set_error("close command list failed",hr);return -6;}ID3D12CommandList* lists[]={x->list.Get()};x->poisoned=true;x->queue->ExecuteCommandLists(1,lists);if(!wait_gpu(x))return -7;x->poisoned=false;void* mapped{};D3D12_RANGE rr{0,12};hr=x->readback->Map(0,&rr,&mapped);if(FAILED(hr)){set_error("map scalar SCD diagnostics failed",hr);return -8;}uint32_t vals[3]{};memcpy(vals,mapped,12);D3D12_RANGE wr{0,0};x->readback->Unmap(0,&wr);x->scdChanges+=(vals[1]&1u)!=0;x->scdHistory=vals[1];x->scdWorkgroups=vals[2];++x->dispatches;if(reset)++x->resets;
    uint64_t op{};uint32_t pitch{};if(cuda_d3d12_color_get_device_pointer(x->output,&op,&pitch)!=0){set_error("get CUDA output pointer failed");return -9;}*out=op;*outPitch=pitch;return 0;
}
int32_t ffx_flow_get_counters(FfxFlowHandle* x,FfxFlowCounters* o){if(!x||!o){set_error("invalid counter arguments");return -1;}std::lock_guard<std::mutex> l(x->mutex);*o={x->calls,x->dispatches,x->resets,x->scdChanges,x->w,x->h,x->gridW,x->gridH,x->outPitch,1,x->scdHistory,x->scdWorkgroups};return 0;}
void ffx_flow_close(FfxFlowHandle* x){
    if(!x)return;
    // Failed waits/signals leave queued SDK resources in use. Retain the
    // context on a healthy device rather than destroying pending resources.
    if(x->poisoned && SUCCEEDED(x->device->GetDeviceRemovedReason()))return;
    cleanup(x);delete x;
}
}
