#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
#include "include/cuda.h"
#include "gpu_post_control_bridge.h"

using Microsoft::WRL::ComPtr;
namespace {
int failures=0;
void check(bool ok,const char* label) {
    std::printf("%s %s\n",ok?"[ok]":"[FAIL]",label);
    if (!ok) ++failures;
}
uint16_t half(float value) {
    uint32_t bits=0; std::memcpy(&bits,&value,4);
    const uint32_t sign=(bits>>16)&0x8000u;
    int32_t exponent=int32_t((bits>>23)&255u)-127+15;
    uint32_t mantissa=bits&0x7fffffu;
    if (((bits>>23)&255u)==255u)
        return uint16_t(sign|0x7c00u|(mantissa?0x0200u:0u));
    if (exponent<=0) {
        if (exponent < -10) return uint16_t(sign);
        mantissa=(mantissa|0x800000u)>>(1-exponent);
        if ((mantissa&0x1fffu)>0x1000u ||
            ((mantissa&0x3fffu)==0x3000u)) mantissa+=0x2000u;
        return uint16_t(sign|(mantissa>>13));
    }
    if (exponent>=31) return uint16_t(sign|0x7c00u);
    if ((mantissa&0x1fffu)>0x1000u ||
        ((mantissa&0x3fffu)==0x3000u)) {
        mantissa+=0x2000u;
        if (mantissa&0x800000u) { mantissa=0; ++exponent; }
    }
    if (exponent>=31) return uint16_t(sign|0x7c00u);
    return uint16_t(sign|(uint32_t(exponent)<<10)|(mantissa>>13));
}
float unhalf(uint16_t h) {
    const uint32_t sign=uint32_t(h&0x8000u)<<16;
    uint32_t exp=(h>>10)&31u, mant=h&1023u, bits=0;
    if (!exp) {
        if (!mant) bits=sign;
        else {
            int e=-14;
            while ((mant&1024u)==0) { mant<<=1; --e; }
            mant&=1023u;
            bits=sign|(uint32_t(e+127)<<23)|(mant<<13);
        }
    } else if (exp==31) bits=sign|0x7f800000u|(mant<<13);
    else bits=sign|((exp-15u+127u)<<23)|(mant<<13);
    float value; std::memcpy(&value,&bits,4); return value;
}
struct Pixel { float r,g,b,a; };
std::vector<uint8_t> pack(const std::vector<Pixel>& pixels,uint32_t bpp) {
    std::vector<uint8_t> out(pixels.size()*bpp);
    for (size_t i=0;i<pixels.size();++i) {
        if (bpp==8) {
            uint16_t p[4]={half(pixels[i].r),half(pixels[i].g),
                           half(pixels[i].b),half(pixels[i].a)};
            std::memcpy(out.data()+i*bpp,p,8);
        } else {
            const float v[4]={pixels[i].r,pixels[i].g,pixels[i].b,pixels[i].a};
            for (unsigned c=0;c<4;++c)
                out[i*bpp+c]=uint8_t(std::lround(std::clamp(v[c],0.0f,1.0f)*255.0f));
        }
    }
    return out;
}
Pixel pixel_at(const std::vector<uint8_t>& bytes,uint32_t index,uint32_t bpp) {
    if (bpp==8) {
        uint16_t p[4]; std::memcpy(p,bytes.data()+size_t(index)*8,8);
        return {unhalf(p[0]),unhalf(p[1]),unhalf(p[2]),unhalf(p[3])};
    }
    const uint8_t* p=bytes.data()+size_t(index)*4;
    return {p[0]/255.0f,p[1]/255.0f,p[2]/255.0f,p[3]/255.0f};
}
struct Cuda {
    HMODULE dll=nullptr;
    CUresult(CUDAAPI* init)(unsigned)=nullptr;
    CUresult(CUDAAPI* count)(int*)=nullptr;
    CUresult(CUDAAPI* get)(CUdevice*,int)=nullptr;
    CUresult(CUDAAPI* luid)(char*,unsigned*,CUdevice)=nullptr;
    CUresult(CUDAAPI* context_create)(CUcontext*,unsigned,CUdevice)=nullptr;
    CUresult(CUDAAPI* context_destroy)(CUcontext)=nullptr;
    CUresult(CUDAAPI* alloc)(CUdeviceptr*,size_t)=nullptr;
    CUresult(CUDAAPI* free)(CUdeviceptr)=nullptr;
    CUresult(CUDAAPI* h2d)(CUdeviceptr,const void*,size_t)=nullptr;
    bool load() {
        dll=LoadLibraryW(L"nvcuda.dll"); if (!dll) return false;
#define GET(name,symbol) name=reinterpret_cast<decltype(name)>(GetProcAddress(dll,symbol))
        GET(init,"cuInit"); GET(count,"cuDeviceGetCount"); GET(get,"cuDeviceGet");
        GET(luid,"cuDeviceGetLuid"); GET(context_create,"cuCtxCreate_v2");
        GET(context_destroy,"cuCtxDestroy_v2"); GET(alloc,"cuMemAlloc_v2");
        GET(free,"cuMemFree_v2"); GET(h2d,"cuMemcpyHtoD_v2");
#undef GET
        return init&&count&&get&&luid&&context_create&&context_destroy&&alloc&&free&&h2d;
    }
};
struct Machine {
    ComPtr<ID3D12Device> device;
    Cuda cuda;
    CUcontext context=nullptr;
    bool open() {
        if (FAILED(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_11_0,
                                     IID_PPV_ARGS(&device)))) return false;
        if (!cuda.load() || cuda.init(0)!=CUDA_SUCCESS) return false;
        int n=0; if (cuda.count(&n)!=CUDA_SUCCESS) return false;
        LUID dx=device->GetAdapterLuid();
        for (int i=0;i<n;++i) {
            CUdevice d=0; char l[8]{}; unsigned node=0;
            if (cuda.get(&d,i)!=CUDA_SUCCESS || cuda.luid(l,&node,d)!=CUDA_SUCCESS ||
                std::memcmp(l,&dx,8)!=0) continue;
            return cuda.context_create(&context,0,d)==CUDA_SUCCESS;
        }
        return false;
    }
    void close() {
        if (context) cuda.context_destroy(context);
        context=nullptr; device.Reset();
        if (cuda.dll) FreeLibrary(cuda.dll);
    }
};
struct Producer {
    ComPtr<ID3D12Resource> texture;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    uint64_t value=0;
    HANDLE event=nullptr;
    bool create(ID3D12Device* device,uint32_t width,uint32_t height,uint32_t bpp,
                const std::vector<uint8_t>& bytes) {
        D3D12_RESOURCE_DESC td{}; td.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        td.Width=width; td.Height=height; td.DepthOrArraySize=1; td.MipLevels=1;
        td.Format=bpp==8?DXGI_FORMAT_R16G16B16A16_FLOAT:DXGI_FORMAT_R8G8B8A8_UNORM;
        td.SampleDesc.Count=1;
        D3D12_HEAP_PROPERTIES def{}; def.Type=D3D12_HEAP_TYPE_DEFAULT;
        if (FAILED(device->CreateCommittedResource(&def,D3D12_HEAP_FLAG_NONE,&td,
            D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&texture)))) return false;
        const uint32_t pitch=(width*bpp+255u)&~255u;
        D3D12_HEAP_PROPERTIES upload_heap{}; upload_heap.Type=D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC bd{}; bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
        bd.Width=uint64_t(pitch)*height; bd.Height=1; bd.DepthOrArraySize=1;
        bd.MipLevels=1; bd.SampleDesc.Count=1; bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        ComPtr<ID3D12Resource> upload;
        if (FAILED(device->CreateCommittedResource(&upload_heap,D3D12_HEAP_FLAG_NONE,&bd,
            D3D12_RESOURCE_STATE_GENERIC_READ,nullptr,IID_PPV_ARGS(&upload)))) return false;
        uint8_t* mapped=nullptr;
        if (FAILED(upload->Map(0,nullptr,reinterpret_cast<void**>(&mapped)))) return false;
        std::memset(mapped,0,size_t(pitch)*height);
        for (uint32_t y=0;y<height;++y)
            std::memcpy(mapped+size_t(y)*pitch,bytes.data()+size_t(y)*width*bpp,
                        size_t(width)*bpp);
        upload->Unmap(0,nullptr);
        D3D12_COMMAND_QUEUE_DESC q{}; q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue))) ||
            FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocator))) ||
            FAILED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
                allocator.Get(),nullptr,IID_PPV_ARGS(&list)))) return false;
        D3D12_TEXTURE_COPY_LOCATION dst{},src{};
        dst.pResource=texture.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.pResource=upload.Get(); src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        src.PlacedFootprint.Footprint.Format=td.Format;
        src.PlacedFootprint.Footprint.Width=width; src.PlacedFootprint.Footprint.Height=height;
        src.PlacedFootprint.Footprint.Depth=1; src.PlacedFootprint.Footprint.RowPitch=pitch;
        list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
        D3D12_RESOURCE_BARRIER b{}; b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource=texture.Get(); b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        b.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1,&b);
        if (FAILED(list->Close())) return false;
        if (FAILED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)))) return false;
        event=CreateEventW(nullptr,FALSE,FALSE,nullptr); if (!event) return false;
        ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
        if (FAILED(queue->Signal(fence.Get(),++value))) return false;
        if (fence->GetCompletedValue()<value) {
            if (FAILED(fence->SetEventOnCompletion(value,event)) ||
                WaitForSingleObject(event,30000)!=WAIT_OBJECT_0) return false;
        }
        return true;
    }
    ~Producer() { if (event) CloseHandle(event); }
};
bool read_output(ID3D12Device* device,ID3D12Resource* texture,ID3D12Fence* producer,
                 uint64_t producer_value,uint32_t width,uint32_t height,uint32_t bpp,
                 std::vector<uint8_t>* tight) {
    const uint32_t pitch=(width*bpp+255u)&~255u;
    D3D12_HEAP_PROPERTIES heap{}; heap.Type=D3D12_HEAP_TYPE_READBACK;
    D3D12_RESOURCE_DESC bd{}; bd.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;
    bd.Width=uint64_t(pitch)*height; bd.Height=1; bd.DepthOrArraySize=1;
    bd.MipLevels=1; bd.SampleDesc.Count=1; bd.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    ComPtr<ID3D12Resource> readback;
    if (FAILED(device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&bd,
        D3D12_RESOURCE_STATE_COPY_DEST,nullptr,IID_PPV_ARGS(&readback)))) return false;
    D3D12_COMMAND_QUEUE_DESC q{}; q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue; ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list; ComPtr<ID3D12Fence> fence;
    if (FAILED(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0,D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(),nullptr,IID_PPV_ARGS(&list))) ||
        FAILED(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)))) return false;
    if (FAILED(queue->Wait(producer,producer_value))) return false;
    D3D12_TEXTURE_COPY_LOCATION dst{},src{};
    dst.pResource=readback.Get(); dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Footprint.Format=bpp==8?DXGI_FORMAT_R16G16B16A16_FLOAT:
                                             DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width=width; dst.PlacedFootprint.Footprint.Height=height;
    dst.PlacedFootprint.Footprint.Depth=1; dst.PlacedFootprint.Footprint.RowPitch=pitch;
    src.pResource=texture; src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[]={list.Get()}; queue->ExecuteCommandLists(1,lists);
    if (FAILED(queue->Signal(fence.Get(),1))) return false;
    HANDLE event=CreateEventW(nullptr,FALSE,FALSE,nullptr); if (!event) return false;
    if (fence->GetCompletedValue()<1 &&
        (FAILED(fence->SetEventOnCompletion(1,event)) ||
         WaitForSingleObject(event,30000)!=WAIT_OBJECT_0)) { CloseHandle(event); return false; }
    CloseHandle(event);
    uint8_t* mapped=nullptr; D3D12_RANGE range{0,size_t(pitch)*height};
    if (FAILED(readback->Map(0,&range,reinterpret_cast<void**>(&mapped)))) return false;
    tight->resize(size_t(width)*height*bpp);
    for (uint32_t y=0;y<height;++y)
        std::memcpy(tight->data()+size_t(y)*width*bpp,mapped+size_t(y)*pitch,size_t(width)*bpp);
    readback->Unmap(0,nullptr); return true;
}
}

int main() {
    constexpr uint32_t width=5,height=5,bpp=8;
    Machine machine;
    check(machine.open(),"matching D3D12/CUDA adapter and context");
    if (!machine.context) return 2;
    GpuPostControlBridge* bridge=nullptr;
    const int opened=gpu_post_control_open(machine.device.Get(),machine.context,
        width,height,bpp,&bridge);
    check(opened==0,"open FP16 post-control bridge");
    if (opened) { std::printf("error=%s\n",gpu_post_control_last_error()); machine.close(); return 2; }

    std::vector<Pixel> source(width*height,{0.25f,0.5f,0.75f,1.0f});
    std::vector<Pixel> candidate=source;
    for (size_t i=0;i<candidate.size();++i)
        candidate[i]={source[i].r-0.5f,source[i].g+0.25f,source[i].b+1.5f,1.0f};
    const auto src_bytes=pack(source,bpp), cand_bytes=pack(candidate,bpp);
    CUdeviceptr src_ptr=0,cand_ptr=0;
    const size_t plane_bytes=src_bytes.size();
    check(machine.cuda.alloc(&src_ptr,plane_bytes)==CUDA_SUCCESS &&
          machine.cuda.alloc(&cand_ptr,plane_bytes)==CUDA_SUCCESS,"allocate synthetic CUDA planes");
    check(machine.cuda.h2d(src_ptr,src_bytes.data(),plane_bytes)==CUDA_SUCCESS &&
          machine.cuda.h2d(cand_ptr,cand_bytes.data(),plane_bytes)==CUDA_SUCCESS,
          "upload signed synthetic planes");
    auto upload_case=[&](const std::vector<Pixel>& s,const std::vector<Pixel>& cand) {
        const auto sb=pack(s,bpp), cb=pack(cand,bpp);
        return machine.cuda.h2d(src_ptr,sb.data(),sb.size())==CUDA_SUCCESS &&
               machine.cuda.h2d(cand_ptr,cb.data(),cb.size())==CUDA_SUCCESS;
    };
    GpuPostControlOptions o{};
    o.struct_size=sizeof(o); o.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    o.width=width; o.height=height; o.pixel_bytes=bpp; o.color_mode=bpp==8 ? 1u : 0u;
    o.total_gain=o.lightness_gain=o.chroma_gain=1.0f;
    o.low_frequency_gain=o.high_frequency_gain=1.0f;
    GpuPostCandidate c{}; c.kind=GPU_POST_CANDIDATE_CUDA;
    c.cuda_ptr=cand_ptr; c.cuda_pitch=width*bpp;
    GpuPostControlCounters counters{}; counters.struct_size=sizeof(counters);
    counters.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    uint64_t smoke_calls=0;
    auto process=[&](GpuPostControlOptions options,const GpuPostCandidate& input,
                     std::vector<uint8_t>* result) {
        void* texture=nullptr; void* fence=nullptr; uint64_t value=0;
        GpuPostControlCounters latest{}; latest.struct_size=sizeof(latest);
        latest.abi_version=GPU_POST_CONTROL_ABI_VERSION;
        const int rc=gpu_post_control_process(bridge,src_ptr,width*bpp,&input,
            &options,&texture,&fence,&value,&latest);
        if (rc) { std::printf("process_error=%s\n",gpu_post_control_last_error()); return false; }
        ++smoke_calls;
        counters=latest;
        return read_output(machine.device.Get(),static_cast<ID3D12Resource*>(texture),
            static_cast<ID3D12Fence*>(fence),value,width,height,bpp,result);
    };
    std::vector<uint8_t> result;
    check(process(o,c,&result),"neutral CUDA candidate process/readback");
    bool identical=result==cand_bytes;
    check(identical,"neutral control exactly preserves representable signed FP16 candidate");
    Pixel p=pixel_at(result,12,bpp);
    check(p.r<0.0f && p.b>1.0f,"neutral HDR output retains negative and above-one values");

    o.total_gain=0.0f;
    check(process(o,c,&result),"zero-total control process");
    check(pixel_at(result,12,bpp).r==source[12].r &&
          pixel_at(result,12,bpp).g==source[12].g,"zero total returns source");
    o.total_gain=1.0f; o.roi_kind=GPU_POST_ROI_RECT;
    o.roi_bounds[0]=0.0f; o.roi_bounds[1]=0.0f;
    o.roi_bounds[2]=0.2f; o.roi_bounds[3]=0.2f;
    check(process(o,c,&result),"ROI control process");
    check(pixel_at(result,0,bpp).r==source[0].r &&
          pixel_at(result,0,bpp).g==source[0].g &&
          pixel_at(result,24,bpp).r==pixel_at(cand_bytes,24,bpp).r,
          "rectangular ROI protects its interior and leaves outside unchanged");
    o.roi_bounds[0]=0.2f; o.roi_bounds[1]=0.2f;
    o.roi_bounds[2]=0.8f; o.roi_bounds[3]=0.8f;
    check(process(o,c,&result),"deep rectangular ROI process");
    check(pixel_at(result,12,bpp).r==source[12].r &&
          pixel_at(result,12,bpp).g==source[12].g,
          "deep rectangular ROI returns exact source");
    o.roi_kind=GPU_POST_ROI_ELLIPSE;
    check(process(o,c,&result),"elliptical ROI process");
    check(pixel_at(result,12,bpp).r==source[12].r &&
          pixel_at(result,12,bpp).g==source[12].g &&
          pixel_at(result,24,bpp).r==pixel_at(cand_bytes,24,bpp).r,
          "elliptical ROI protects center and leaves outside unchanged");
    o.roi_kind=GPU_POST_ROI_RECT; o.roi_feather=0.4f;
    check(process(o,c,&result),"feathered ROI process");
    const float feather_r=pixel_at(result,11,bpp).r;
    check(feather_r>pixel_at(cand_bytes,11,bpp).r &&
          feather_r<source[11].r,
          "ROI feather smoothly attenuates correction near the protected edge");
    o.roi_feather=0.0f;
    o.roi_kind=GPU_POST_ROI_NONE;

    std::vector<Pixel> flat_source(width*height,{0.2f,0.2f,0.2f,1.0f});
    std::vector<Pixel> flat_candidate=flat_source;
    for (auto& q:flat_candidate) q.r+=0.2f;
    check(upload_case(flat_source,flat_candidate),"upload lightness/chroma fixture");
    o={}; o.struct_size=sizeof(o); o.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    o.width=width; o.height=height; o.pixel_bytes=bpp; o.color_mode=bpp==8 ? 1u : 0u;
    o.total_gain=o.lightness_gain=o.chroma_gain=1.0f;
    o.low_frequency_gain=o.high_frequency_gain=1.0f;
    o.lightness_gain=0.0f;
    check(process(o,c,&result),"chroma-only process");
    p=pixel_at(result,12,bpp);
    const float chroma_y=(p.r-flat_source[12].r)*0.2126f+
        (p.g-flat_source[12].g)*0.7152f+(p.b-flat_source[12].b)*0.0722f;
    check(std::abs(chroma_y)<0.001f,"chroma-only delta has zero Rec.709 lightness");
    o.lightness_gain=1.0f; o.chroma_gain=0.0f;
    check(process(o,c,&result),"lightness-only process");
    p=pixel_at(result,12,bpp);
    check(std::abs((p.r-flat_source[12].r)-(p.g-flat_source[12].g))<0.001f &&
          std::abs((p.g-flat_source[12].g)-(p.b-flat_source[12].b))<0.001f,
          "lightness-only delta is equal in RGB");

    std::vector<Pixel> impulse_source(width*height,{0.25f,0.25f,0.25f,1.0f});
    std::vector<Pixel> impulse_candidate=impulse_source;
    impulse_candidate[12].r+=0.4f; impulse_candidate[12].g+=0.4f;
    impulse_candidate[12].b+=0.4f;
    check(upload_case(impulse_source,impulse_candidate),"upload frequency impulse fixture");
    o={}; o.struct_size=sizeof(o); o.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    o.width=width; o.height=height; o.pixel_bytes=bpp; o.color_mode=bpp==8 ? 1u : 0u;
    o.total_gain=o.lightness_gain=o.chroma_gain=1.0f;
    o.low_frequency_gain=1.0f; o.high_frequency_gain=0.0f;
    check(process(o,c,&result),"low-frequency-only process");
    const float low_center=pixel_at(result,12,bpp).r-impulse_source[12].r;
    o.low_frequency_gain=0.0f; o.high_frequency_gain=1.0f;
    check(process(o,c,&result),"high-frequency-only process");
    const float high_center=pixel_at(result,12,bpp).r-impulse_source[12].r;
    check(std::abs(low_center-0.1f)<0.002f && std::abs(high_center-0.3f)<0.002f,
          "binomial residual split matches signed low/high reference");

    std::vector<Pixel> dark_source(width*height,{0.5f,0.5f,0.5f,1.0f});
    std::vector<Pixel> dark_candidate=dark_source;
    dark_source[0]={0.001f,0.001f,0.001f,1.0f};
    dark_candidate[0]={0.051f,0.051f,0.051f,1.0f};
    check(upload_case(dark_source,dark_candidate),"upload dark protection fixture");
    o.low_frequency_gain=o.high_frequency_gain=1.0f; o.dark_protection=0.0f;
    check(process(o,c,&result),"dark baseline process");
    const float dark_open=pixel_at(result,0,bpp).r-dark_source[0].r;
    o.dark_protection=1.0f;
    check(process(o,c,&result),"dark-protected process");
    const float dark_closed=pixel_at(result,0,bpp).r-dark_source[0].r;
    check(dark_closed<dark_open*0.5f,"dark protection reduces near-black correction");

    std::vector<Pixel> peak_source(width*height,{124.9f,124.9f,124.9f,1.0f});
    std::vector<Pixel> peak_candidate(width*height,{125.9f,125.9f,125.9f,1.0f});
    check(upload_case(peak_source,peak_candidate),"upload 10000 nit highlight fixture");
    o.dark_protection=0.0f; o.highlight_protection=0.0f;
    check(process(o,c,&result),"unprotected highlight process");
    const float high_open=pixel_at(result,12,bpp).r;
    o.highlight_protection=1.0f;
    check(process(o,c,&result),"highlight-protected process");
    const float high_closed=pixel_at(result,12,bpp).r;
    const float stored_peak=pixel_at(pack(peak_source,bpp),12,bpp).r;
    check(high_closed<high_open && high_closed>=stored_peak,
          "highlight protection suppresses outward BT.2020 125 scRGB approach");
    std::vector<Pixel> above_source(width*height,{130.0f,130.0f,130.0f,1.0f});
    std::vector<Pixel> above_candidate(width*height,{131.0f,131.0f,131.0f,1.0f});
    check(upload_case(above_source,above_candidate),"upload over-bound source fixture");
    check(process(o,c,&result),"over-bound source process");
    check(pixel_at(result,12,bpp).r>=above_source[12].r,
          "existing over-bound source is not pulled below its source value");

    std::vector<Pixel> soft_source(width*height,{0.5f,0.5f,0.5f,1.0f});
    std::vector<Pixel> soft_candidate(width*height,{1.5f,1.5f,1.5f,1.0f});
    check(upload_case(soft_source,soft_candidate),"upload soft compression fixture");
    o.highlight_protection=0.0f; o.soft_compression=0.0f;
    check(process(o,c,&result),"uncompressed residual process");
    const float soft_open=pixel_at(result,12,bpp).r;
    o.soft_compression=1.0f;
    check(process(o,c,&result),"soft-compressed residual process");
    const float soft_closed=pixel_at(result,12,bpp).r;
    check(soft_closed>soft_source[12].r && soft_closed<soft_open,
          "soft compression preserves sign and reduces large correction");

    check(upload_case(soft_source,soft_candidate),"restore diagnostic residual fixture");
    o.soft_compression=0.0f;
    std::vector<Pixel> hue_source(width*height,{0.8f,0.2f,0.1f,1.0f});
    std::vector<Pixel> hue_candidate=hue_source;
    for (auto& q:hue_candidate) q.g+=0.2f;
    check(upload_case(hue_source,hue_candidate),"upload hue projection fixture");
    o.soft_compression=0.0f; o.hue_protection=0.0f;
    check(process(o,c,&result),"unprotected hue process");
    const Pixel hue_open=pixel_at(result,12,bpp);
    o.hue_protection=1.0f;
    check(process(o,c,&result),"hue-protected process");
    const Pixel hue_closed=pixel_at(result,12,bpp);
    check(std::abs(hue_open.g-hue_closed.g)>0.01f &&
          std::abs(hue_closed.r-hue_source[12].r)>0.01f,
          "hue protection projects chroma correction onto source-chroma direction");
    o.hue_protection=0.0f;
    check(upload_case(soft_source,soft_candidate),"restore signed diagnostic fixture");

    o.debug_mode=GPU_POST_DEBUG_RAW;
    check(process(o,c,&result),"raw diagnostic process");
    const float debug_raw=pixel_at(result,12,bpp).r;
    check(std::abs(debug_raw-(0.5f+0.5f*1.0f/2.0f))<0.002f,
          "raw diagnostic uses bounded signed mapping");
    o.debug_mode=GPU_POST_DEBUG_CONTROLLED; o.total_gain=0.0f;
    check(process(o,c,&result),"controlled diagnostic process");
    check(std::abs(pixel_at(result,12,bpp).r-0.5f)<0.002f,
          "controlled diagnostic distinguishes zero controlled residual");
    o.debug_mode=GPU_POST_DEBUG_WEIGHT; o.total_gain=1.0f;
    o.roi_kind=GPU_POST_ROI_RECT; o.roi_bounds[0]=0.4f; o.roi_bounds[1]=0.4f;
    o.roi_bounds[2]=0.6f; o.roi_bounds[3]=0.6f;
    check(process(o,c,&result),"protection weight diagnostic process");
    check(pixel_at(result,12,bpp).r<0.001f && pixel_at(result,24,bpp).r>0.99f,
          "weight diagnostic reports protected retention and unprotected outside");
    o.debug_mode=GPU_POST_DEBUG_OFF; o.roi_kind=GPU_POST_ROI_NONE;

    std::vector<Pixel> invalid_candidate=source;
    invalid_candidate[12].r=std::numeric_limits<float>::quiet_NaN();
    check(upload_case(source,invalid_candidate),"upload unrepresentable candidate fixture");
    const uint64_t fallback_before=counters.unrepresentable_pixels;
    check(process(o,c,&result),"candidate fallback process");
    p=pixel_at(result,12,bpp);
    check(std::abs(p.r-source[12].r)<0.001f &&
          counters.unrepresentable_pixels>fallback_before,
          "non-finite candidate falls back to source and increments counter");

    check(upload_case(source,candidate),"restore baseline CUDA candidate");
    o={}; o.struct_size=sizeof(o); o.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    o.width=width; o.height=height; o.pixel_bytes=bpp; o.color_mode=bpp==8 ? 1u : 0u;
    o.total_gain=o.lightness_gain=o.chroma_gain=1.0f;
    o.low_frequency_gain=o.high_frequency_gain=1.0f;

    Producer producer;
    check(producer.create(machine.device.Get(),width,height,bpp,cand_bytes),
          "create completed D3D12 candidate producer");
    GpuPostCandidate d{}; d.kind=GPU_POST_CANDIDATE_D3D12_COMPLETED;
    d.d3d_state=D3D12_RESOURCE_STATE_COPY_SOURCE; d.d3d_texture=producer.texture.Get();
    check(process(o,d,&result),"completed D3D12 candidate process");
    check(result==cand_bytes,"completed D3D12 candidate matches CUDA candidate");
    d.kind=GPU_POST_CANDIDATE_D3D12; d.producer_fence=producer.fence.Get();
    d.producer_fence_value=producer.value;
    check(process(o,d,&result),"producer-fenced D3D12 candidate process");
    check(result==cand_bytes,"fenced D3D12 candidate matches CUDA candidate");

    check(counters.process_count==smoke_calls && counters.source_transport_bytes==
        uint64_t(width)*height*bpp*smoke_calls && counters.candidate_transport_bytes==
        uint64_t(width)*height*bpp*smoke_calls && counters.scalar_readback_bytes==16u*2u*smoke_calls,
        "transport and scalar counters match actual process paths");

    gpu_post_control_close(bridge); bridge=nullptr;

    /* SDR remains normalized RGBA8. This is a separate bridge specialization. */
    check(gpu_post_control_open(machine.device.Get(),machine.context,width,height,
        4,&bridge)==0,"open SDR RGBA8 post-control bridge");
    std::vector<Pixel> sdr_source(width*height,{0.2f,0.4f,0.6f,1.0f});
    std::vector<Pixel> sdr_candidate=sdr_source;
    for (auto& q:sdr_candidate) q={0.1f,0.8f,1.2f,1.0f};
    const auto sdr_src=pack(sdr_source,4),sdr_cand=pack(sdr_candidate,4);
    CUdeviceptr sdr_src_ptr=0,sdr_cand_ptr=0;
    check(machine.cuda.alloc(&sdr_src_ptr,sdr_src.size())==CUDA_SUCCESS &&
          machine.cuda.alloc(&sdr_cand_ptr,sdr_cand.size())==CUDA_SUCCESS &&
          machine.cuda.h2d(sdr_src_ptr,sdr_src.data(),sdr_src.size())==CUDA_SUCCESS &&
          machine.cuda.h2d(sdr_cand_ptr,sdr_cand.data(),sdr_cand.size())==CUDA_SUCCESS,
          "allocate and upload SDR fixture");
    GpuPostControlOptions so{}; so.struct_size=sizeof(so);
    so.abi_version=GPU_POST_CONTROL_ABI_VERSION; so.width=width; so.height=height;
    so.pixel_bytes=4; so.total_gain=so.lightness_gain=so.chroma_gain=1.0f;
    so.low_frequency_gain=so.high_frequency_gain=1.0f;
    GpuPostCandidate sc{}; sc.kind=GPU_POST_CANDIDATE_CUDA;
    sc.cuda_ptr=sdr_cand_ptr; sc.cuda_pitch=width*4;
    GpuPostControlCounters sctr{}; sctr.struct_size=sizeof(sctr);
    sctr.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    void* st=nullptr; void* sf=nullptr; uint64_t sv=0;
    int s_rc=gpu_post_control_process(bridge,sdr_src_ptr,width*4,&sc,&so,
        &st,&sf,&sv,&sctr);
    bool s_ok=s_rc==0 && read_output(machine.device.Get(),
        static_cast<ID3D12Resource*>(st),static_cast<ID3D12Fence*>(sf),sv,
        width,height,4,&result);
    check(s_ok,"neutral SDR process/readback");
    check(s_ok && result==sdr_cand,"SDR neutral bytes preserved with UNORM storage");
    gpu_post_control_close(bridge); bridge=nullptr;
    machine.cuda.free(sdr_src_ptr); machine.cuda.free(sdr_cand_ptr);

    check(gpu_post_control_open(machine.device.Get(),machine.context,width,height,
        bpp,&bridge)==0,"reopen after SDR bridge teardown");
    std::vector<Pixel> invalid_source=source;
    invalid_source[7].b=std::numeric_limits<float>::quiet_NaN();
    check(upload_case(invalid_source,candidate),"upload invalid source fixture");
    GpuPostControlCounters bad{}; bad.struct_size=sizeof(bad);
    bad.abi_version=GPU_POST_CONTROL_ABI_VERSION;
    void* bad_tex=nullptr; void* bad_fence=nullptr; uint64_t bad_value=0;
    const int invalid_rc=gpu_post_control_process(bridge,src_ptr,width*bpp,&c,&o,
        &bad_tex,&bad_fence,&bad_value,&bad);
    check(invalid_rc!=0 && bad_tex==nullptr,
          "invalid source fails explicitly without returning output");
    gpu_post_control_close(bridge); bridge=nullptr;
    check(gpu_post_control_open(machine.device.Get(),machine.context,width,height,
        bpp,&bridge)==0,"close and reopen after failed invalid-source frame");
    gpu_post_control_close(bridge);
    machine.cuda.free(src_ptr); machine.cuda.free(cand_ptr);
    machine.close();
    return failures?1:0;
}
