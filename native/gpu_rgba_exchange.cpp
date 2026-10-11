// Original GPU-only CUDA / D3D12 texture exchange for native video effects.
#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cstdint>
#include <cstdio>
#include <memory>
#include "nvof_d3d12_interop.h"
using Microsoft::WRL::ComPtr;
namespace {
thread_local char error[256]{};
struct Exchange {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12Resource> texture;
    CudaD3D12ColorBridge* bridge = nullptr;
    ID3D12Resource* buffer = nullptr;
    ID3D12Fence* cuda_fence = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    uint64_t serial = 0, pointer = 0, copies = 0;
    uint32_t pitch = 0;
    DXGI_FORMAT format = DXGI_FORMAT_R8G8B8A8_UNORM;
    bool poisoned = false;
    HANDLE event = nullptr;
    ~Exchange() { if (bridge) cuda_d3d12_color_destroy(bridge); if (event) CloseHandle(event); }
};
int fail(const char* what, HRESULT hr) {
    std::snprintf(error, sizeof(error), "%s: 0x%08lx", what, static_cast<unsigned long>(hr)); return -1;
}
int begin(Exchange* e, ID3D12Fence* f, uint64_t v) {
    if (e->poisoned) return fail("exchange poisoned after unfinished GPU work", E_FAIL);
    HRESULT hr = f && v ? e->queue->Wait(f,v) : S_OK;
    if (SUCCEEDED(hr)) hr = e->allocator->Reset();
    if (SUCCEEDED(hr)) hr = e->list->Reset(e->allocator.Get(),nullptr);
    return FAILED(hr) ? fail("exchange begin",hr) : 0;
}
int finish(Exchange* e) {
    HRESULT hr = e->list->Close(); if (FAILED(hr)) return fail("exchange list",hr);
    ID3D12CommandList* lists[] = {e->list.Get()}; e->queue->ExecuteCommandLists(1,lists);
    e->poisoned = true;
    hr = e->queue->Signal(e->fence.Get(),++e->serial);
    if (SUCCEEDED(hr)) hr=e->fence->SetEventOnCompletion(e->serial,e->event);
    if (FAILED(hr)) return fail("exchange signal",hr);
    if (WaitForSingleObject(e->event,30000)!=WAIT_OBJECT_0) return fail("exchange timeout",HRESULT_FROM_WIN32(ERROR_TIMEOUT));
    e->poisoned = false; ++e->copies; return 0;
}
}
#define API extern "C" __declspec(dllexport)
API const char* gpu_rgba_exchange_last_error() {return error;}
API void* gpu_rgba_exchange_fence(Exchange* e) {return e ? e->fence.Get() : nullptr;}
API int gpu_effect_device_open(void** out) {
    if (!out) return fail("device output", E_INVALIDARG);
    *out = nullptr;
    return FAILED(D3D12CreateDevice(nullptr,D3D_FEATURE_LEVEL_12_0,
        __uuidof(ID3D12Device),out)) ? fail("effect device", E_FAIL) : 0;
}
API void gpu_effect_device_close(ID3D12Device* device) { if (device) device->Release(); }
API int gpu_rgba_exchange_pointer(Exchange* e,uint64_t* ptr,uint32_t* pitch) {
    if (!e||!ptr||!pitch||e->poisoned) return fail("exchange pointer", E_INVALIDARG);
    *ptr=e->pointer;*pitch=e->pitch;return 0;
}
API void gpu_rgba_exchange_close(Exchange* e) {
    // Pending GPU commands keep their resources alive after an uncertain timeout.
    // Normal completion and device removal release everything immediately.
    if(e && e->poisoned &&
       SUCCEEDED(e->device->GetDeviceRemovedReason())) return;
    delete e;
}
API int gpu_rgba_exchange_open2(ID3D12Device* device,void* context,uint32_t w,uint32_t h,uint32_t pixel_bytes,Exchange** out) {
    if(!out) return fail("null output",E_INVALIDARG); *out=nullptr;
    if(!device||!context||!w||!h||(pixel_bytes!=4 && pixel_bytes!=8)) return fail("exchange arguments",E_INVALIDARG);
    auto e=std::make_unique<Exchange>(); e->device=device;
    D3D12_RESOURCE_DESC desc{}; desc.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width=w;desc.Height=h;desc.DepthOrArraySize=1;desc.MipLevels=1;
    e->format = pixel_bytes==8 ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.Format=e->format;desc.SampleDesc.Count=1;
    device->GetCopyableFootprints(&desc,0,1,0,&e->footprint,nullptr,nullptr,nullptr);
    e->pitch=e->footprint.Footprint.RowPitch;
    if(cuda_d3d12_color_create2(device,context,w,h,e->pitch,pixel_bytes,0,&e->bridge))
        return fail(nvof_d3d12_last_error(),E_FAIL);
    void* buffer=nullptr;void* fence=nullptr;
    if(cuda_d3d12_color_get_resources(e->bridge,&buffer,&fence)||
       cuda_d3d12_color_get_device_pointer(e->bridge,&e->pointer,&e->pitch))
        return fail(nvof_d3d12_last_error(),E_FAIL);
    e->buffer=static_cast<ID3D12Resource*>(buffer);e->cuda_fence=static_cast<ID3D12Fence*>(fence);
    D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr=device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,
        D3D12_RESOURCE_STATE_COPY_SOURCE,nullptr,IID_PPV_ARGS(&e->texture));
    D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    if(SUCCEEDED(hr)) hr=device->CreateCommandQueue(&q,IID_PPV_ARGS(&e->queue));
    if(SUCCEEDED(hr)) hr=device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&e->allocator));
    if(SUCCEEDED(hr)) hr=device->CreateCommandList(0,q.Type,e->allocator.Get(),nullptr,IID_PPV_ARGS(&e->list));
    if(SUCCEEDED(hr)) hr=e->list->Close();
    if(SUCCEEDED(hr)) hr=device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&e->fence));
    if(FAILED(hr)) return fail("exchange resources",hr);
    e->event=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if(!e->event)return fail("exchange event",HRESULT_FROM_WIN32(GetLastError()));
    *out=e.release();return 0;
}
API int gpu_rgba_exchange_open(ID3D12Device* device,void* context,uint32_t w,uint32_t h,Exchange** out) {
    return gpu_rgba_exchange_open2(device,context,w,h,4,out);
}
API int gpu_rgba_exchange_import(Exchange* e,ID3D12Resource* texture,ID3D12Fence* f,uint64_t v,uint64_t* ptr,uint32_t* pitch) {
    if(!e||!texture||!ptr||!pitch)return fail("import arguments",E_INVALIDARG);
    auto d=texture->GetDesc();ComPtr<ID3D12Device> owner;
    if(d.Width!=e->footprint.Footprint.Width||d.Height!=e->footprint.Footprint.Height||
       d.Format!=e->format||FAILED(texture->GetDevice(IID_PPV_ARGS(&owner)))||owner.Get()!=e->device.Get())
        return fail("import extent/format/device",E_INVALIDARG);
    if(begin(e,f,v))return -1;
    D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=texture;
    src.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;dst.pResource=e->buffer;
    dst.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;dst.PlacedFootprint=e->footprint;
    // Shared buffers use COMMON implicit promotion/decay on each completed list.
    e->list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    if(finish(e))return -1;*ptr=e->pointer;*pitch=e->pitch;return 0;
}
API int gpu_rgba_exchange_export(Exchange* e,uint64_t ptr,size_t pitch,void** texture,uint64_t* value) {
    if(!e||!ptr||!texture||!value)return fail("export arguments",E_INVALIDARG);
    uint64_t cv=0;
    // Always publish CUDA writes through the shared fence, including effects
    // that write directly into this exchange's allocation. CUDA completion
    // alone does not order the D3D copy queue against the CUDA producer.
    if (cuda_d3d12_color_copy(e->bridge,ptr,pitch,&cv))
        return fail(nvof_d3d12_last_error(),E_FAIL);
    if(begin(e,e->cuda_fence,cv))return -1;
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource=e->texture.Get();b.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_SOURCE;
    b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_DEST;b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    e->list->ResourceBarrier(1,&b);
    D3D12_TEXTURE_COPY_LOCATION src{},dst{};src.pResource=e->buffer;
    src.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;src.PlacedFootprint=e->footprint;
    dst.pResource=e->texture.Get();dst.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    e->list->CopyTextureRegion(&dst,0,0,0,&src,nullptr);
    b.Transition.StateBefore=D3D12_RESOURCE_STATE_COPY_DEST;b.Transition.StateAfter=D3D12_RESOURCE_STATE_COPY_SOURCE;
    e->list->ResourceBarrier(1,&b);if(finish(e))return -1;
    *texture=e->texture.Get();*value=e->serial;return 0;
}
