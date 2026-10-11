#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "include/cuda.h"
#include "gpu_temporal_bridge.h"

using Microsoft::WRL::ComPtr;
namespace {
constexpr uint32_t kWidth=128,kHeight=72;
int g_failures=0;
void check(bool ok,const char* name) {
    std::printf("[%s] %s\n",ok?" ok ":"FAIL",name);
    if (!ok) ++g_failures;
}
struct Cuda {
    HMODULE lib=nullptr;
    CUresult(CUDAAPI* init)(unsigned)=nullptr;
    CUresult(CUDAAPI* count)(int*)=nullptr;
    CUresult(CUDAAPI* get)(CUdevice*,int)=nullptr;
    CUresult(CUDAAPI* luid)(char*,unsigned*,CUdevice)=nullptr;
    CUresult(CUDAAPI* create)(CUcontext*,unsigned,CUdevice)=nullptr;
    CUresult(CUDAAPI* destroy)(CUcontext)=nullptr;
    CUresult(CUDAAPI* alloc)(CUdeviceptr*,size_t)=nullptr;
    CUresult(CUDAAPI* free_mem)(CUdeviceptr)=nullptr;
    CUresult(CUDAAPI* copy_htod)(CUdeviceptr,const void*,size_t)=nullptr;
    bool open(ID3D12Device* device,CUcontext* context) {
        lib=LoadLibraryA("nvcuda.dll"); if (!lib) return false;
        auto s=[&](const char* n){return GetProcAddress(lib,n);};
        init=reinterpret_cast<decltype(init)>(s("cuInit"));
        count=reinterpret_cast<decltype(count)>(s("cuDeviceGetCount"));
        get=reinterpret_cast<decltype(get)>(s("cuDeviceGet"));
        luid=reinterpret_cast<decltype(luid)>(s("cuDeviceGetLuid"));
        create=reinterpret_cast<decltype(create)>(s("cuCtxCreate_v2"));
        destroy=reinterpret_cast<decltype(destroy)>(s("cuCtxDestroy_v2"));
        alloc=reinterpret_cast<decltype(alloc)>(s("cuMemAlloc_v2"));
        free_mem=reinterpret_cast<decltype(free_mem)>(s("cuMemFree_v2"));
        copy_htod=reinterpret_cast<decltype(copy_htod)>(s("cuMemcpyHtoD_v2"));
        if (!init||!count||!get||!luid||!create||!destroy||!alloc||!free_mem||!copy_htod||
            init(0)!=CUDA_SUCCESS) return false;
        int n=0; if (count(&n)!=CUDA_SUCCESS) return false;
        const LUID target=device->GetAdapterLuid();
        for(int i=0;i<n;++i) {
            CUdevice d=0; char id[8]{}; unsigned mask=0;
            if (get(&d,i)!=CUDA_SUCCESS || luid(id,&mask,d)!=CUDA_SUCCESS ||
                std::memcmp(id,&target,8)!=0) continue;
            return create(context,0,d)==CUDA_SUCCESS;
        }
        return false;
    }
};
struct DevicePlane {
    CUdeviceptr ptr=0; size_t bytes=0; uint32_t pitch=0;
    bool allocate(Cuda& cuda,size_t size,uint32_t row_pitch) {
        bytes=size; pitch=row_pitch; return cuda.alloc(&ptr,size)==CUDA_SUCCESS;
    }
    bool upload(Cuda& cuda,const std::vector<uint8_t>& data) {
        return data.size()<=bytes && cuda.copy_htod(ptr,data.data(),data.size())==CUDA_SUCCESS;
    }
    void close(Cuda& cuda) { if(ptr) cuda.free_mem(ptr); ptr=0; }
};
uint16_t to_half(float value) {
    uint32_t bits=0; std::memcpy(&bits,&value,4);
    const uint32_t sign=(bits>>16)&0x8000u;
    const uint32_t exponent=(bits>>23)&0xffu;
    uint32_t mantissa=bits&0x7fffffu;
    if(exponent==255u) return uint16_t(sign|0x7c00u|(mantissa?0x0200u:0u));
    int e=int(exponent)-127+15;
    if(e<=0) return uint16_t(sign);
    if(e>=31) return uint16_t(sign|0x7c00u);
    mantissa+=0x1000u;
    if(mantissa&0x800000u) { mantissa=0; if(++e>=31)return uint16_t(sign|0x7c00u); }
    return uint16_t(sign|(uint32_t(e)<<10)|(mantissa>>13));
}
float from_half(uint16_t half) {
    const uint32_t sign=uint32_t(half&0x8000u)<<16;
    uint32_t e=(half>>10)&31u,m=half&1023u,bits=0;
    if(e==0) {
        if(m==0) bits=sign;
        else { int shift=0; while((m&1024u)==0){m<<=1;++shift;}
            m&=1023u; bits=sign|uint32_t(127-15-shift)<<23|m<<13; }
    } else if(e==31) bits=sign|0x7f800000u|m<<13;
    else bits=sign|((e+112u)<<23)|(m<<13);
    float result; std::memcpy(&result,&bits,4); return result;
}
struct Pixel { float r,g,b,a; };
std::vector<uint8_t> encode(uint32_t bytes,uint32_t width,uint32_t height,
    const std::function<Pixel(uint32_t,uint32_t)>& make) {
    std::vector<uint8_t> out(size_t(width)*height*bytes);
    for(uint32_t y=0;y<height;++y) for(uint32_t x=0;x<width;++x) {
        Pixel p=make(x,y); size_t at=(size_t(y)*width+x)*bytes;
        if(bytes==4) {
            out[at]=uint8_t(std::lround(std::clamp(p.r,0.0f,1.0f)*255));
            out[at+1]=uint8_t(std::lround(std::clamp(p.g,0.0f,1.0f)*255));
            out[at+2]=uint8_t(std::lround(std::clamp(p.b,0.0f,1.0f)*255)); out[at+3]=255;
        } else {
            const uint16_t h[4]={to_half(p.r),to_half(p.g),to_half(p.b),to_half(p.a)};
            std::memcpy(out.data()+at,h,sizeof(h));
        }
    }
    return out;
}
Pixel decode(const std::vector<uint8_t>& data,uint32_t bytes,uint32_t x,uint32_t y) {
    const size_t at=(size_t(y)*kWidth+x)*bytes;
    if(bytes==4) return {data[at]/255.0f,data[at+1]/255.0f,
        data[at+2]/255.0f,data[at+3]/255.0f};
    uint16_t h[4]; std::memcpy(h,data.data()+at,8);
    return {from_half(h[0]),from_half(h[1]),from_half(h[2]),from_half(h[3])};
}
std::vector<uint8_t> scene(uint32_t bytes,bool alternate=false) {
    return encode(bytes,kWidth,kHeight,[=](uint32_t x,uint32_t y) {
        if(bytes==4) {
            if(alternate) {
                const float a=((x*17u+y*13u)%29u<14u)?0.78f:0.06f;
                const float b=((x/4u+y/3u)%2u)?0.72f:0.12f;
                const float c=((x*7u+y*11u)%37u<10u)?0.88f:0.04f;
                return Pixel{a,b,c,1};
            }
            const float stripe=((x/13u+y/9u+(alternate?1u:0u))%2u)?0.08f:0.0f;
            return Pixel{0.12f+0.38f*float(x)/kWidth+stripe,
                0.10f+0.36f*float(y)/kHeight+0.5f*stripe,
                0.08f+0.17f*float((x+y)%31u)/31.0f+stripe,1};
        }
        if(alternate) {
            const float a=((x*17u+y*13u)%29u<14u)?-3.0f:2.0f;
            const float b=((x/4u+y/3u)%2u)?4.0f:-1.0f;
            const float c=((x*7u+y*11u)%37u<10u)?20.0f:-0.5f;
            return Pixel{a,b,c,1};
        }
        const float spot=(x%32u<5u && y%18u<5u)?10.0f:0.2f;
        const float stripe=((x/13u+y/9u+(alternate?1u:0u))%2u)?0.15f:0.0f;
        return Pixel{-0.45f+2.1f*float(x)/kWidth+stripe,
            -0.25f+0.7f*float(y)/kHeight+stripe,spot,1};
    });
}
std::vector<uint8_t> achromatic_scene(uint32_t bytes) {
    return encode(bytes,kWidth,kHeight,[](uint32_t x,uint32_t y) {
        const uint32_t pattern=(x*73u+y*151u+x*y*31u)%997u;
        const float value=0.04f+0.92f*float(pattern)/996.0f;
        return Pixel{value,value,value,1};
    });
}
std::vector<uint8_t> scale_scene(const std::vector<uint8_t>& input,uint32_t bytes,float scale) {
    return encode(bytes,kWidth,kHeight,[&](uint32_t x,uint32_t y) {
        Pixel p=decode(input,bytes,x,y); p.r*=scale;p.g*=scale;p.b*=scale;return p;
    });
}
std::vector<uint8_t> translate_scene(const std::vector<uint8_t>& input,uint32_t bytes,
                                    int dx,int dy) {
    return encode(bytes,kWidth,kHeight,[&](uint32_t x,uint32_t y) {
        const uint32_t sx=uint32_t(std::clamp(int(x)-dx,0,int(kWidth)-1));
        const uint32_t sy=uint32_t(std::clamp(int(y)-dy,0,int(kHeight)-1));
        return decode(input,bytes,sx,sy);
    });
}
std::vector<uint8_t> residual_candidate(const std::vector<uint8_t>& source,
    uint32_t bytes,float base,bool detail=false) {
    return encode(bytes,kWidth,kHeight,[&](uint32_t x,uint32_t y) {
        Pixel p=decode(source,bytes,x,y);
        float r=base,g=base*0.6f,b=-base*0.3f;
        if(detail) { const float h=((x+y)&1u)?0.08f:-0.08f;r+=h;g-=h;b+=h*0.5f; }
        p.r+=r;p.g+=g;p.b+=b;return p;
    });
}
struct Machine {
    ComPtr<ID3D12Device> device; ComPtr<ID3D12InfoQueue> info;
    Cuda cuda; CUcontext context=nullptr;
    bool open() {
        ComPtr<ID3D12Debug> debug;
        if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) debug->EnableDebugLayer();
        if(FAILED(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,IID_PPV_ARGS(&device)))) return false;
        device.As(&info);
        return cuda.open(device.Get(),&context);
    }
    void close() { if(context)cuda.destroy(context);context=nullptr; }
};
void check_debug_messages(Machine& machine) {
    if(!machine.info) return;
    const UINT64 count=machine.info->GetNumStoredMessagesAllowedByRetrievalFilter();
    for(UINT64 i=0;i<count;++i) {
        SIZE_T size=0; machine.info->GetMessage(i,nullptr,&size);
        std::vector<uint8_t> storage(size);
        auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if(FAILED(machine.info->GetMessage(i,message,&size))) continue;
        if(message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING) {
            std::fprintf(stderr,"D3D12[%u]: %s\n",unsigned(message->Severity),message->pDescription);
            check(false,"D3D12 debug layer has no warnings/errors");
        }
    }
}
struct Probe {
    Machine machine; GpuTemporalBridge* bridge=nullptr;
    DevicePlane source,candidate;
    uint32_t pixel_bytes=0,pitch=0;
    bool open(uint32_t bytes) {
        pixel_bytes=bytes; pitch=kWidth*bytes;
        if(!machine.open()) return false;
        if(!source.allocate(machine.cuda,size_t(pitch)*kHeight,pitch) ||
           !candidate.allocate(machine.cuda,size_t(pitch)*kHeight,pitch)) return false;
        return gpu_temporal_open(machine.device.Get(),machine.context,kWidth,kHeight,
            bytes,&bridge)==0;
    }
    void close() {
        if(bridge)gpu_temporal_close(bridge);bridge=nullptr;
        source.close(machine.cuda);candidate.close(machine.cuda);machine.close();
    }
    bool upload(DevicePlane& d,const std::vector<uint8_t>& data) {
        return d.upload(machine.cuda,data);
    }
    bool observe(const std::vector<uint8_t>& data,uint32_t reset,
                 GpuTemporalObserveReport* report,uint32_t enabled=1) {
        if(!upload(source,data)) return false;
        *report={};report->struct_size=sizeof(*report);report->abi_version=GPU_TEMPORAL_ABI_VERSION;
        const int32_t rc=gpu_temporal_observe(bridge,source.ptr,pitch,reset,enabled,report);
        if(rc) std::fprintf(stderr,"observe rc=%d bridge=%p ptr=%llx pitch=%u report=%u/%u err=%s\n",
            rc,static_cast<void*>(bridge),static_cast<unsigned long long>(source.ptr),
            pitch,report->struct_size,report->abi_version,gpu_temporal_last_error());
        return rc==0;
    }
    bool process(const std::vector<uint8_t>& src,const std::vector<uint8_t>& cand,
                 float weight,uint32_t reset,std::vector<uint8_t>* output,
                 GpuTemporalProcessReport* report,const GpuPostCandidate* override_candidate=nullptr) {
        if(!upload(source,src) || (!override_candidate && !upload(candidate,cand))) return false;
        GpuPostCandidate descriptor{};
        if(override_candidate) descriptor=*override_candidate;
        else { descriptor.kind=GPU_POST_CANDIDATE_CUDA;descriptor.cuda_ptr=candidate.ptr;
            descriptor.cuda_pitch=pitch; }
        void *texture=nullptr,*fence=nullptr;uint64_t value=0;
        *report={};report->struct_size=sizeof(*report);report->abi_version=GPU_TEMPORAL_ABI_VERSION;
        const int32_t rc=gpu_temporal_process(bridge,source.ptr,pitch,&descriptor,
            weight,reset,&texture,&fence,&value,report);
        if(rc || !texture) {
            std::fprintf(stderr,"temporal process %d: %s\n",rc,gpu_temporal_last_error());
            return false;
        }
        return harvest(static_cast<ID3D12Resource*>(texture),
            static_cast<ID3D12Fence*>(fence),value,output);
    }
    bool harvest(ID3D12Resource* texture,ID3D12Fence* producer,uint64_t producer_value,
                 std::vector<uint8_t>* output) {
        const uint32_t out_pitch=(kWidth*pixel_bytes+255u)&~255u;
        const uint64_t size=uint64_t(out_pitch)*kHeight;
        D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width=size;bd.Height=1;bd.DepthOrArraySize=1;bd.MipLevels=1;
        bd.SampleDesc.Count=1;bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> readback;
        if(FAILED(machine.device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&bd,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)))) return false;
        D3D12_COMMAND_QUEUE_DESC qd{};qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        ComPtr<ID3D12CommandQueue> queue;ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;ComPtr<ID3D12Fence> fence;
        if(FAILED(machine.device->CreateCommandQueue(&qd,IID_PPV_ARGS(&queue)))||
           FAILED(machine.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)))||
           FAILED(machine.device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)))||
           FAILED(list->Close())||FAILED(machine.device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)))) return false;
        allocator->Reset(); if(FAILED(list->Reset(allocator.Get(),nullptr))) return false;
        if(FAILED(queue->Wait(producer,producer_value))) return false;
        D3D12_TEXTURE_COPY_LOCATION dst{},src{};
        dst.pResource=readback.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint.Footprint.Format=pixel_bytes==4?DXGI_FORMAT_R8G8B8A8_UNORM:DXGI_FORMAT_R16G16B16A16_FLOAT;
        dst.PlacedFootprint.Footprint.Width=kWidth;dst.PlacedFootprint.Footprint.Height=kHeight;
        dst.PlacedFootprint.Footprint.Depth=1;dst.PlacedFootprint.Footprint.RowPitch=out_pitch;
        src.pResource=texture;src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        if(FAILED(list->Close()))return false;
        ID3D12CommandList* lists[]={list.Get()};queue->ExecuteCommandLists(1,lists);
        if(FAILED(queue->Signal(fence.Get(),1)))return false;
        HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);
        if(!ev)return false;
        HRESULT hr=fence->SetEventOnCompletion(1,ev);
        const bool waited=SUCCEEDED(hr)&&WaitForSingleObject(ev,30000)==WAIT_OBJECT_0;
        CloseHandle(ev);if(!waited)return false;
        void* mapped=nullptr;D3D12_RANGE range{0,size};
        if(FAILED(readback->Map(0,&range,&mapped)))return false;
        output->resize(size_t(kWidth)*kHeight*pixel_bytes);
        for(uint32_t y=0;y<kHeight;++y)
            std::memcpy(output->data()+size_t(y)*kWidth*pixel_bytes,
                static_cast<uint8_t*>(mapped)+size_t(y)*out_pitch,
                size_t(kWidth)*pixel_bytes);
        D3D12_RANGE no_write{0,0};readback->Unmap(0,&no_write);
        return true;
    }
};

bool make_d3d_candidate(ID3D12Device* device,uint32_t bytes,
    const std::vector<uint8_t>& pixels,ComPtr<ID3D12Resource>* texture,
    ComPtr<ID3D12Fence>* fence,ComPtr<ID3D12CommandQueue>* queue,
    ComPtr<ID3D12Resource>* upload,uint64_t* fence_value,bool wait) {
    const uint32_t pitch=(kWidth*bytes+255u)&~255u;const uint64_t size=uint64_t(pitch)*kHeight;
    D3D12_HEAP_PROPERTIES hp{};hp.Type=D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC td{};td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    td.Width=kWidth;td.Height=kHeight;td.DepthOrArraySize=1;td.MipLevels=1;
    td.Format=bytes==4?DXGI_FORMAT_R8G8B8A8_UNORM:DXGI_FORMAT_R16G16B16A16_FLOAT;
    td.SampleDesc.Count=1;td.Flags=D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    if(FAILED(device->CreateCommittedResource(&hp,D3D12_HEAP_FLAG_NONE,&td,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(texture->GetAddressOf()))))return false;
    D3D12_HEAP_PROPERTIES up{};up.Type=D3D12_HEAP_TYPE_UPLOAD;
    D3D12_RESOURCE_DESC bd{};bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;bd.Width=size;
    bd.Height=1;bd.DepthOrArraySize=1;bd.MipLevels=1;bd.SampleDesc.Count=1;
    bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if(FAILED(device->CreateCommittedResource(&up,D3D12_HEAP_FLAG_NONE,&bd,
        D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(upload->GetAddressOf()))))return false;
    void* mapped=nullptr;D3D12_RANGE no_read{0,0};
    if(FAILED((*upload)->Map(0,&no_read,&mapped)))return false;
    for(uint32_t y=0;y<kHeight;++y)std::memcpy(static_cast<uint8_t*>(mapped)+size_t(y)*pitch,
        pixels.data()+size_t(y)*kWidth*bytes,size_t(kWidth)*bytes);
    D3D12_RANGE written{0,size};(*upload)->Unmap(0,&written);
    D3D12_COMMAND_QUEUE_DESC qd{};qd.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandAllocator> allocator;ComPtr<ID3D12GraphicsCommandList> list;
    if(FAILED(device->CreateCommandQueue(&qd,IID_PPV_ARGS(queue->GetAddressOf())))||
       FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,IID_PPV_ARGS(&allocator)))||
       FAILED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,allocator.Get(),nullptr,IID_PPV_ARGS(&list)))||
       FAILED(list->Close())||FAILED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(fence->GetAddressOf()))))return false;
    allocator->Reset();if(FAILED(list->Reset(allocator.Get(),nullptr)))return false;
    D3D12_TEXTURE_COPY_LOCATION dst{},src{};dst.pResource=texture->Get();
    dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.pResource=upload->Get();src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    src.PlacedFootprint.Footprint.Format=td.Format;src.PlacedFootprint.Footprint.Width=kWidth;
    src.PlacedFootprint.Footprint.Height=kHeight;src.PlacedFootprint.Footprint.Depth=1;
    src.PlacedFootprint.Footprint.RowPitch=pitch;
    list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource=texture->Get();barrier.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
    barrier.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;list->ResourceBarrier(1,&barrier);
    if(FAILED(list->Close()))return false;ID3D12CommandList* lists[]={list.Get()};
    queue->Get()->ExecuteCommandLists(1,lists);*fence_value=1;
    if(FAILED(queue->Get()->Signal(fence->Get(),1)))return false;
    if(wait){HANDLE ev=CreateEventW(nullptr,FALSE,FALSE,nullptr);if(!ev)return false;
        HRESULT hr=fence->Get()->SetEventOnCompletion(1,ev);
        bool okay=SUCCEEDED(hr)&&WaitForSingleObject(ev,30000)==WAIT_OBJECT_0;CloseHandle(ev);
        if(!okay)return false;}
    return true;
}

void report_observe(Probe& p,const std::vector<uint8_t>& image,uint32_t reset,
                    uint32_t expected,const char* label) {
    GpuTemporalObserveReport r{};
    const bool called=p.observe(image,reset,&r);
    if(!called) std::fprintf(stderr,"observe error: %s\n",gpu_temporal_last_error());
    if(!called && p.machine.info) {
        const UINT64 count=p.machine.info->GetNumStoredMessagesAllowedByRetrievalFilter();
        for(UINT64 i=0;i<count;++i) {
            SIZE_T size=0; p.machine.info->GetMessage(i,nullptr,&size);
            std::vector<uint8_t> storage(size);
            auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if(SUCCEEDED(p.machine.info->GetMessage(i,message,&size)) &&
               message->Severity<=D3D12_MESSAGE_SEVERITY_WARNING)
                std::fprintf(stderr,"D3D12[%u]: %s\n",unsigned(message->Severity),message->pDescription);
        }
    }
    const bool match=called&&r.classification==expected;
    std::printf("observe %-18s class=%u cut=%u sim=%.4f diff=%.4f chroma=%.4f ratio=%.4f shift=(%.0f,%.0f) reset_count=%u bytes=%llu/%llu\n",
        label,r.classification,r.scene_cut,r.similarity,r.normalized_delta,r.chroma_delta,
        r.exposure_ratio,r.shift_x,r.shift_y,r.scene_reset_count,
        static_cast<unsigned long long>(r.source_transport_bytes),
        static_cast<unsigned long long>(r.scalar_readback_bytes));
    check(match,label);
    if(called) check(r.scalar_readback_bytes==32,"observe reads back 32 scalar bytes");
}
double mean_delta(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b,
                  uint32_t bytes,uint32_t x0=32,uint32_t y0=16,
                  uint32_t x1=96,uint32_t y1=56) {
    double sum=0;uint64_t n=0;
    for(uint32_t y=y0;y<y1;++y)for(uint32_t x=x0;x<x1;++x){
        Pixel p=decode(a,bytes,x,y),q=decode(b,bytes,x,y);
        sum+=std::abs(p.r-q.r)+std::abs(p.g-q.g)+std::abs(p.b-q.b);n+=3;}
    return n?sum/n:0;
}
double highband_error(const std::vector<uint8_t>& source,
                      const std::vector<uint8_t>& candidate,
                      const std::vector<uint8_t>& output,uint32_t bytes) {
    constexpr float weights[3]={1.0f,2.0f,1.0f};
    double error=0;uint64_t count=0;
    auto residual=[&](const std::vector<uint8_t>& image,uint32_t x,uint32_t y,int c) {
        const Pixel a=decode(image,bytes,x,y),b=decode(source,bytes,x,y);
        const float av=c==0?a.r:(c==1?a.g:a.b);
        const float bv=c==0?b.r:(c==1?b.g:b.b);
        return av-bv;
    };
    for(uint32_t y=17;y+1<kHeight-16;++y)for(uint32_t x=33;x+1<kWidth-32;++x)
        for(int c=0;c<3;++c) {
            float low_candidate=0,low_output=0;
            for(int oy=-1;oy<=1;++oy)for(int ox=-1;ox<=1;++ox) {
                const float w=weights[oy+1]*weights[ox+1]*(1.0f/16.0f);
                low_candidate+=w*residual(candidate,uint32_t(int(x)+ox),uint32_t(int(y)+oy),c);
                low_output+=w*residual(output,uint32_t(int(x)+ox),uint32_t(int(y)+oy),c);
            }
            const float candidate_high=residual(candidate,x,y,c)-low_candidate;
            const float output_high=residual(output,x,y,c)-low_output;
            error+=std::abs(candidate_high-output_high);++count;
        }
    return count?error/count:0;
}

} // namespace

int main() {
    std::setvbuf(stdout,nullptr,_IONBF,0);
    for(uint32_t bytes:{4u,8u}) {
        Probe p;if(!p.open(bytes)){std::fprintf(stderr,"GPU/CUDA setup failed for %u-byte format\n",bytes);return 2;}
        const std::string kind=bytes==4?"RGBA8":"scRGB FP16";
        auto base=scene(bytes), duplicate=base;
        auto prime=residual_candidate(base,bytes,0.08f,true), jitter=residual_candidate(base,bytes,0.06f,true);
        report_observe(p,base,1,GPU_TEMPORAL_FIRST,(kind+" first frame").c_str());
        std::vector<uint8_t> out;GpuTemporalProcessReport pr{};
        bool ok=p.process(base,prime,0.0f,1,&out,&pr);
        GpuTemporalObserveReport after_first_reset{};
        const bool first_reset_continues=p.observe(duplicate,0,&after_first_reset);
        check(ok&&out==prime&&first_reset_continues&&
            after_first_reset.classification==GPU_TEMPORAL_DUPLICATE_STATIC&&
            after_first_reset.scene_reset_count==1,
            (kind+" process(first reset) preserves observed guide").c_str());
        report_observe(p,duplicate,0,GPU_TEMPORAL_DUPLICATE_STATIC,(kind+" duplicate").c_str());
        auto translated=translate_scene(base,bytes,4,4);
        report_observe(p,translated,0,GPU_TEMPORAL_TRANSLATION,(kind+" translation").c_str());
        auto exposure=scale_scene(base,bytes,1.25f);
        report_observe(p,exposure,0,GPU_TEMPORAL_EXPOSURE_FLASH,(kind+" exposure").c_str());
        auto cut=scene(bytes,true);
        report_observe(p,cut,0,GPU_TEMPORAL_CUT,(kind+" synthetic cut").c_str());
        ok=p.process(cut,residual_candidate(cut,bytes,0.03f,true),0.0f,1,&out,&pr);
        GpuTemporalObserveReport after_cut_reset{};
        const bool cut_reset_continues=p.observe(cut,0,&after_cut_reset);
        check(ok&&cut_reset_continues&&
            after_cut_reset.classification==GPU_TEMPORAL_DUPLICATE_STATIC&&
            after_cut_reset.scene_reset_count==2,
            (kind+" process(cut reset) preserves observed guide/count").c_str());
        auto achromatic=achromatic_scene(bytes);
        GpuTemporalObserveReport gray_cut{};
        const bool gray_cut_ok=p.observe(achromatic,0,&gray_cut);
        check(gray_cut_ok&&gray_cut.classification==GPU_TEMPORAL_CUT&&
            gray_cut.scene_cut&&gray_cut.scene_reset_count==3,
            (kind+" achromatic structural change detects cut").c_str());
        ok=p.process(achromatic,residual_candidate(achromatic,bytes,0.03f,true),
            0.0f,1,&out,&pr);
        GpuTemporalObserveReport after_gray_cut{};
        const bool gray_continues=p.observe(achromatic,0,&after_gray_cut);
        check(ok&&gray_continues&&
            after_gray_cut.classification==GPU_TEMPORAL_DUPLICATE_STATIC&&
            after_gray_cut.scene_reset_count==3,
            (kind+" achromatic cut survives process reset").c_str());
        GpuTemporalObserveReport disabled{};
        const bool disabled_ok=p.observe(base,0,&disabled,0);
        check(disabled_ok&&disabled.classification==GPU_TEMPORAL_DISABLED&&
            disabled.source_transport_bytes==0&&disabled.scalar_readback_bytes==0,
            "disabled scene observation performs no GPU readback");

        /* Prime one static residual, then perturb its low frequency with the
         * same sign while retaining a high-frequency checker component. */
        ok=p.process(base,prime,0.0f,1,&out,&pr);
        check(ok&&out==prime,"history-disabled output equals current candidate");
        ok=p.process(base,jitter,0.8f,0,&out,&pr);
        const double lowband_change=mean_delta(jitter,out,bytes);
        const double flicker_before=mean_delta(prime,jitter,bytes);
        const double flicker_after=mean_delta(prime,out,bytes);
        const double checker_error=highband_error(base,jitter,out,bytes);
        std::printf("process %s history_used=%u rejected=%u weight=%.3f jitter_delta=%.7g filtered_delta=%.7g highband_error=%.7g bytes=%llu/%llu/%llu\n",
            kind.c_str(),pr.history_used,pr.history_rejected,pr.effective_history_weight,
            flicker_before,flicker_after,checker_error,
            static_cast<unsigned long long>(pr.source_transport_bytes),
            static_cast<unsigned long long>(pr.candidate_transport_bytes),
            static_cast<unsigned long long>(pr.scalar_readback_bytes));
        check(ok&&pr.history_used>0&&pr.effective_history_weight==0.8f,
            (kind+" static residual history accepted").c_str());
        check(ok&&flicker_after<flicker_before*0.65,
            (kind+" static flicker step is reduced").c_str());
        check(ok&&checker_error<(bytes==4?0.0035:0.0007),
            (kind+" accepted filter preserves checker high frequency").c_str());
        check(ok&&lowband_change>0,
            (kind+" filtered output remains responsive to current residual").c_str());
        check(pr.scalar_readback_bytes==16,"process reads back 16 scalar bytes");
        auto step=residual_candidate(base,bytes,0.55f,true);
        ok=p.process(base,step,0.8f,0,&out,&pr);
        check(ok&&pr.history_used==0&&pr.history_rejected>0,
            (kind+" material residual amplitude step rejects history").c_str());
        check(ok&&mean_delta(step,out,bytes)<(bytes==4?0.004:0.001),
            "detail-step rejection preserves current candidate");
        auto reverse=residual_candidate(base,bytes,-0.06f,true);
        ok=p.process(base,reverse,0.8f,0,&out,&pr);
        check(ok&&pr.history_used==0,"low residual sign reversal rejects history");
        check(mean_delta(reverse,out,bytes)<(bytes==4?0.004:0.001),
            "rejected history preserves the current high-detail candidate");

        /* A changing HUD/text patch must remain exactly the current candidate
         * there even while unchanged pixels elsewhere remain eligible. */
        auto text=encode(bytes,kWidth,kHeight,[&](uint32_t x,uint32_t y){
            Pixel q=decode(base,bytes,x,y);
            if(x>=56&&x<72&&y>=28&&y<44){q.r=bytes==4?0.95f:5.0f;q.g=0.02f;q.b=0.02f;}
            return q;});
        auto text_candidate=residual_candidate(text,bytes,0.04f,true);
        ok=p.process(text,text_candidate,0.8f,0,&out,&pr);
        const bool roi_same=mean_delta(text_candidate,out,bytes,56,28,72,44)<
            (bytes==4?0.004:0.001);
        check(ok&&roi_same,"changed HUD/text region rejects stale history");
        check(pr.history_rejected>0,"motion/change rejection is reported");

        /* A real D3D12 candidate carries a COPY_SOURCE declaration and producer
         * fence. The completed variant explicitly waits that fence first. */
        ComPtr<ID3D12Resource> candidate_texture,upload;ComPtr<ID3D12Fence> candidate_fence;
        ComPtr<ID3D12CommandQueue> candidate_queue;uint64_t candidate_value=0;
        auto exact=residual_candidate(base,bytes,0.02f,false);
        bool made=make_d3d_candidate(p.machine.device.Get(),bytes,exact,
            &candidate_texture,&candidate_fence,&candidate_queue,&upload,&candidate_value,false);
        check(made,"create fenced D3D12 candidate");
        if(made){
            GpuPostCandidate d{};d.kind=GPU_POST_CANDIDATE_D3D12;
            d.d3d_state=D3D12_RESOURCE_STATE_COPY_SOURCE;d.d3d_texture=candidate_texture.Get();
            d.producer_fence=candidate_fence.Get();d.producer_fence_value=candidate_value;
            ok=p.process(base,exact,0.0f,1,&out,&pr,&d);
            check(ok&&out==exact,"fenced D3D12 candidate copied and preserved");
            const bool waited=candidate_fence->GetCompletedValue()>=candidate_value;
            d.kind=GPU_POST_CANDIDATE_D3D12_COMPLETED;d.producer_fence=nullptr;d.producer_fence_value=0;
            if(waited){ok=p.process(base,exact,0.0f,0,&out,&pr,&d);
                check(ok&&out==exact,"completed D3D12 candidate copied and preserved");}
        }
        if(bytes==8) {
            Pixel signed_sample=decode(base,bytes,2,2);
            check(signed_sample.r<0.0f&&signed_sample.b>1.0f,
                "HDR fixture contains signed values and highlight headroom");
            std::vector<uint8_t> bad=base;
            const uint16_t inf=0x7c00;std::memcpy(bad.data()+size_t(4)*8,&inf,2);
            ok=p.process(bad,base,0.0f,0,&out,&pr);
            check(!ok,"invalid HDR source is refused without an output texture");
        }
        check_debug_messages(p.machine);
        p.close();
    }
    std::printf("%s temporal GPU probe: %d failure(s)\n",g_failures?"[FAIL]":"[ ok ]",g_failures);
    return g_failures?1:0;
}
