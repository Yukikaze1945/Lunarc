// Original FP16/scRGB child-window presenter. No picture readback or encoding.
#define NOMINMAX
#include <windows.h>
#include <windowsx.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <cstdio>
#include <stdexcept>
#include "nvof_d3d12_interop.h"
#include "gpu_dred_diagnostics.h"
using Microsoft::WRL::ComPtr;
namespace {
thread_local std::string error;
constexpr UINT UPDATE=WM_APP+31, FRAME=WM_APP+32;
constexpr DXGI_FORMAT FORMAT=DXGI_FORMAT_R16G16B16A16_FLOAT;
struct Plane {
    CudaD3D12ColorBridge* bridge=nullptr;
    ID3D12Resource* buffer=nullptr; ID3D12Fence* fence=nullptr;
    UINT w=0,h=0,pitch=0,bytes=0; UINT64 value=0; bool hdr=false;
    ~Plane(){if(bridge)cuda_d3d12_color_destroy(bridge);}
};
struct Geometry {
    UINT aw,ah,ap,ab,bw,bh,bp,bb,ow,oh,mode,ahdr,bhdr,nativeHDR;
    float split,zoom,panX,panY,sdrWhite;
};
struct Stats { UINT size,hdr,w,h; UINT64 frames,presents,dragPresents,hostImageBytes;
    float split,sdrWhite; UINT64 hwnd; UINT pointerDowns; float downX; };
const char* shader=R"(
cbuffer Geometry:register(b0){uint AW,AH,AP,AB,BW,BH,BP,BB,OW,OH,Mode,AHDR,BHDR,NativeHDR;
float Split,Zoom,PanX,PanY,SDRWhite;};
ByteAddressBuffer A:register(t0); ByteAddressBuffer B:register(t1);
float3 decode(float3 c){return float3(c.r<.081?c.r/4.5:pow((c.r+.099)/1.099,1./.45),
c.g<.081?c.g/4.5:pow((c.g+.099)/1.099,1./.45),c.b<.081?c.b/4.5:pow((c.b+.099)/1.099,1./.45));}
float3 fetch(bool first,uint2 xy){uint bytes;uint2 packed;
// Raw root SRVs have no size bound. A ternary compiles to loads from BOTH
// resources before movc, which faults for unequal extents or RGBA8/FP16 pairs.
// Keep resource selection branched, with each resource's own address/format.
[branch] if(first){bytes=AB;uint address=xy.y*AP+xy.x*AB;
    [branch] if(AB==4)packed=uint2(A.Load(address),0);else packed=A.Load2(address);
}else{bytes=BB;uint address=xy.y*BP+xy.x*BB;
    [branch] if(BB==4)packed=uint2(B.Load(address),0);else packed=B.Load2(address);
}
if(bytes==4)return decode(float3(packed.x&255,(packed.x>>8)&255,(packed.x>>16)&255)/255.);
return float3(f16tof32(packed.x&65535),f16tof32(packed.x>>16),f16tof32(packed.y&65535));}
float3 sampleImage(bool first,float2 uv){uint2 extent=first?uint2(AW,AH):uint2(BW,BH);
float2 p=clamp(uv*extent-.5,0.,float2(extent-1));uint2 q=(uint2)floor(p),s=min(q+1,extent-1);float2 f=frac(p);
float3 rgb=lerp(lerp(fetch(first,q),fetch(first,uint2(s.x,q.y)),f.x),
lerp(fetch(first,uint2(q.x,s.y)),fetch(first,s),f.x),f.y);
bool hdr=first?AHDR!=0:BHDR!=0;
if(NativeHDR!=0){if(!hdr)rgb*=SDRWhite;return rgb;}
if(hdr)rgb/=2.5375+max(dot(rgb,float3(.2126,.7152,.0722)),0.);
return saturate(rgb);}
struct Vertex {float4 pos:SV_Position;};
Vertex vs(uint n:SV_VertexID){Vertex o;o.pos=float4(n==2?3.:-1.,n==1?3.:-1.,0.,1.);return o;}
float4 ps(Vertex input):SV_Target {
float2 p=input.pos.xy;float panel=Mode==3?OW*.5:OW;
bool first=Mode==0||(Mode==2&&p.x<Split*OW)||(Mode==3&&p.x<panel);
float local=Mode==3?fmod(p.x,panel):p.x;
float2 extent=Mode==0?float2(AW,AH):float2(BW,BH);
float scale=min(panel/extent.x,OH/extent.y)*Zoom;
float2 uv=(float2(local,p.y)-float2(panel,OH)*.5)/(extent*scale)+float2(PanX,PanY);
if(any(uv<0.)||any(uv>1.))return float4(0.,0.,0.,1.);
if(Mode==2&&abs(p.x-Split*OW)<1.25)return float4(float3(.82,.67,.25)*(NativeHDR!=0?SDRWhite:1.),1.);
return float4(sampleImage(first,uv),1.);
})";
struct Presenter {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<IDXGISwapChain3> swap;
    ComPtr<ID3D12DescriptorHeap> rtv;
    ComPtr<ID3D12Resource> back[2];
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    std::unique_ptr<Plane> a,b;
    std::mutex mutex;
    std::thread thread;
    HANDLE ready=CreateEventW(nullptr,TRUE,FALSE,nullptr),done=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    HWND parent=nullptr,window=nullptr;
    int x=0,y=0,w=1,h=1,mode=2;
    bool visible=true,loaded=false,hdr=false,drag=false,panDrag=false,poisoned=false,frozen=false,layoutDirty=true;
    POINT last{}; HMONITOR monitor=nullptr;
    UINT pointerDowns=0;float downX=0;
    float split=.5f,zoom=1.f,panX=.5f,panY=.5f,sdrWhite=1.f;
    float requestedPanX=.5f,requestedPanY=.5f;
    UINT64 serial=0,frames=0,presents=0,dragPresents=0;
    std::string failure;
    ~Presenter(){if(ready)CloseHandle(ready);if(done)CloseHandle(done);}
    void check(HRESULT hr,const char* operation){if(FAILED(hr)){char s[2048];
        if(device&&FAILED(device->GetDeviceRemovedReason()))describe_gpu_removal(device.Get(),operation,hr,s,sizeof(s));
        else std::snprintf(s,sizeof(s),"%s: 0x%08lx",operation,static_cast<unsigned long>(hr));
        throw std::runtime_error(s);}}
    void wait(){check(queue->Signal(fence.Get(),++serial),"presenter signal");
        check(fence->SetEventOnCompletion(serial,done),"presenter event");
        if(WaitForSingleObject(done,30000)!=WAIT_OBJECT_0)throw std::runtime_error("presenter GPU timeout");
        check(device->GetDeviceRemovedReason(),"presenter GPU completion");poisoned=false;}
    void colour(){
        auto m=MonitorFromWindow(window,MONITOR_DEFAULTTONEAREST);monitor=m;
        ComPtr<IDXGIFactory1> factory;check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)),"display factory");
        hdr=false;sdrWhite=1.f;
        for(UINT i=0;;++i){ComPtr<IDXGIAdapter1> adapter;if(factory->EnumAdapters1(i,&adapter)==DXGI_ERROR_NOT_FOUND)break;
            for(UINT j=0;;++j){ComPtr<IDXGIOutput> output;if(adapter->EnumOutputs(j,&output)==DXGI_ERROR_NOT_FOUND)break;
                DXGI_OUTPUT_DESC d{};if(FAILED(output->GetDesc(&d))||d.Monitor!=m)continue;
                ComPtr<IDXGIOutput6> output6;DXGI_OUTPUT_DESC1 d1{};
                if(SUCCEEDED(output.As(&output6))&&SUCCEEDED(output6->GetDesc1(&d1)))
                    hdr=d1.ColorSpace==DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
            }
        }
        if(hdr){
            sdrWhite=203.f/80.f;
            UINT pc=0,mc=0;auto flags=QDC_ONLY_ACTIVE_PATHS|QDC_VIRTUAL_MODE_AWARE;
            if(GetDisplayConfigBufferSizes(flags,&pc,&mc)==ERROR_SUCCESS){
                std::vector<DISPLAYCONFIG_PATH_INFO> paths(pc);std::vector<DISPLAYCONFIG_MODE_INFO> modes(mc);
                if(QueryDisplayConfig(flags,&pc,paths.data(),&mc,modes.data(),nullptr)==ERROR_SUCCESS){
                    MONITORINFOEXW mi{};mi.cbSize=sizeof(mi);GetMonitorInfoW(m,&mi);
                    for(UINT i=0;i<pc;++i){DISPLAYCONFIG_SOURCE_DEVICE_NAME name{};
                        name.header={DISPLAYCONFIG_DEVICE_INFO_GET_SOURCE_NAME,sizeof(name),paths[i].sourceInfo.adapterId,paths[i].sourceInfo.id};
                        if(DisplayConfigGetDeviceInfo(&name.header)||wcscmp(name.viewGdiDeviceName,mi.szDevice))continue;
                        DISPLAYCONFIG_SDR_WHITE_LEVEL white{};
                        white.header={DISPLAYCONFIG_DEVICE_INFO_GET_SDR_WHITE_LEVEL,sizeof(white),paths[i].targetInfo.adapterId,paths[i].targetInfo.id};
                        if(!DisplayConfigGetDeviceInfo(&white.header)&&white.SDRWhiteLevel)sdrWhite=white.SDRWhiteLevel/1000.f;
                    }
                }
            }
        }
        UINT support=0;check(swap->CheckColorSpaceSupport(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709,&support),"scRGB support");
        if(!(support&DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))throw std::runtime_error("scRGB presentation unsupported");
        check(swap->SetColorSpace1(DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709),"scRGB colour space");
    }
    void buffers(){for(auto& item:back)item.Reset();
        for(UINT i=0;i<2;++i){check(swap->GetBuffer(i,IID_PPV_ARGS(&back[i])),"swap buffer");
            auto handle=rtv->GetCPUDescriptorHandleForHeapStart();handle.ptr+=i*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            device->CreateRenderTargetView(back[i].Get(),nullptr,handle);}}
    void init(){
        D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
        check(device->CreateCommandQueue(&q,IID_PPV_ARGS(&queue)),"present queue");
        check(device->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&allocator)),"present allocator");
        check(device->CreateCommandList(0,q.Type,allocator.Get(),nullptr,IID_PPV_ARGS(&list)),"present list");check(list->Close(),"present list close");
        check(device->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence)),"present fence");
        ComPtr<IDXGIFactory2> factory;check(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory)),"swap factory");
        DXGI_SWAP_CHAIN_DESC1 desc{};desc.Width=w;desc.Height=h;desc.Format=FORMAT;desc.SampleDesc.Count=1;
        desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=2;desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;
        ComPtr<IDXGISwapChain1> base;check(factory->CreateSwapChainForHwnd(queue.Get(),window,&desc,nullptr,nullptr,&base),"FP16 child swap chain");
        check(base.As(&swap),"swap interface");swap->SetMaximumFrameLatency(1);factory->MakeWindowAssociation(window,DXGI_MWA_NO_ALT_ENTER);
        D3D12_DESCRIPTOR_HEAP_DESC heap{};heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;heap.NumDescriptors=2;
        check(device->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&rtv)),"RTV heap");buffers();colour();
        D3D12_ROOT_PARAMETER params[3]{};params[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        params[0].Constants.Num32BitValues=sizeof(Geometry)/4;
        params[1].ParameterType=params[2].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;params[2].Descriptor.ShaderRegister=1;
        D3D12_ROOT_SIGNATURE_DESC rs{};rs.NumParameters=3;rs.pParameters=params;
        ComPtr<ID3DBlob> serialized,errors,vs,ps;
        check(D3D12SerializeRootSignature(&rs,D3D_ROOT_SIGNATURE_VERSION_1,&serialized,&errors),"present signature");
        check(device->CreateRootSignature(0,serialized->GetBufferPointer(),serialized->GetBufferSize(),IID_PPV_ARGS(&root)),"present root");
        for(auto stage:{0,1}){auto& code=stage?ps:vs;
            auto hr=D3DCompile(shader,std::strlen(shader),"native_fp16_presenter",nullptr,nullptr,stage?"ps":"vs",stage?"ps_5_1":"vs_5_1",D3DCOMPILE_OPTIMIZATION_LEVEL3,0,&code,&errors);
            if(FAILED(hr))throw std::runtime_error(errors?static_cast<char*>(errors->GetBufferPointer()):"present shader compile failed");}
        D3D12_GRAPHICS_PIPELINE_STATE_DESC p{};p.pRootSignature=root.Get();p.VS={vs->GetBufferPointer(),vs->GetBufferSize()};p.PS={ps->GetBufferPointer(),ps->GetBufferSize()};
        p.SampleMask=UINT_MAX;p.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;p.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;p.RasterizerState.DepthClipEnable=TRUE;
        p.BlendState.RenderTarget[0].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;
        p.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;p.NumRenderTargets=1;p.RTVFormats[0]=FORMAT;p.SampleDesc.Count=1;
        check(device->CreateGraphicsPipelineState(&p,IID_PPV_ARGS(&pipeline)),"present pipeline");
    }
    void resize(){
        if(!swap||frozen)return;
        DXGI_SWAP_CHAIN_DESC1 desc{};check(swap->GetDesc1(&desc),"query FP16 panel size");
        if(desc.Width!=UINT(w)||desc.Height!=UINT(h)){
            for(auto& item:back)item.Reset();
            check(swap->ResizeBuffers(2,w,h,FORMAT,0),"resize FP16 panel");buffers();
        }
    }
    void draw(bool dragging=false){
        if(frozen||!failure.empty()||!swap||!visible||!IsWindowVisible(parent)||w<1||h<1)return;
        check(allocator->Reset(),"draw allocator");check(list->Reset(allocator.Get(),pipeline.Get()),"draw list");
        UINT index=swap->GetCurrentBackBufferIndex();auto handle=rtv->GetCPUDescriptorHandleForHeapStart();
        handle.ptr+=index*device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_RESOURCE_BARRIER barrier{};barrier.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition={back[index].Get(),D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PRESENT,D3D12_RESOURCE_STATE_RENDER_TARGET};
        list->ResourceBarrier(1,&barrier);list->OMSetRenderTargets(1,&handle,FALSE,nullptr);
        const float black[]={0,0,0,1};list->ClearRenderTargetView(handle,black,0,nullptr);
        if(loaded){
            for(auto plane:{a.get(),b.get()}){check(queue->Wait(plane->fence,plane->value),"colour fence");
                D3D12_RESOURCE_BARRIER input{};input.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                input.Transition={plane->buffer,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_COMMON,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE};list->ResourceBarrier(1,&input);}
            Geometry g{a->w,a->h,a->pitch,a->bytes,b->w,b->h,b->pitch,b->bytes,UINT(w),UINT(h),UINT(mode),UINT(a->hdr),UINT(b->hdr),UINT(hdr),split,zoom,panX,panY,sdrWhite};
            list->SetGraphicsRootSignature(root.Get());list->SetGraphicsRoot32BitConstants(0,sizeof(g)/4,&g,0);
            list->SetGraphicsRootShaderResourceView(1,a->buffer->GetGPUVirtualAddress());list->SetGraphicsRootShaderResourceView(2,b->buffer->GetGPUVirtualAddress());
            D3D12_VIEWPORT viewport{0,0,float(w),float(h),0,1};D3D12_RECT rect{0,0,w,h};list->RSSetViewports(1,&viewport);list->RSSetScissorRects(1,&rect);
            list->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);list->DrawInstanced(3,1,0,0);
            for(auto plane:{a.get(),b.get()}){D3D12_RESOURCE_BARRIER input{};input.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                input.Transition={plane->buffer,D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,D3D12_RESOURCE_STATE_COMMON};list->ResourceBarrier(1,&input);}
        }
        std::swap(barrier.Transition.StateBefore,barrier.Transition.StateAfter);list->ResourceBarrier(1,&barrier);check(list->Close(),"draw close");
        ID3D12CommandList* lists[]={list.Get()};poisoned=true;queue->ExecuteCommandLists(1,lists);wait();
        check(swap->Present(0,0),"FP16 present");++presents;if(dragging)++dragPresents;
    }
    void upload(std::unique_ptr<Plane>& plane,void* context,UINT64 ptr,UINT pitch,UINT pw,UINT ph,UINT bytes,bool isHDR){
        if(!ptr||!context||!pw||!ph||pw>16384||ph>16384||(bytes!=4&&bytes!=8)||pitch<pw*bytes)throw std::runtime_error("invalid presenter plane");
        if(!plane||plane->w!=pw||plane->h!=ph||plane->bytes!=bytes){
            auto next=std::make_unique<Plane>();next->w=pw;next->h=ph;next->bytes=bytes;next->pitch=(pw*bytes+255)&~255u;
            if(cuda_d3d12_color_create2(device.Get(),context,pw,ph,next->pitch,bytes,0,&next->bridge))throw std::runtime_error(nvof_d3d12_last_error());
            void* buffer=nullptr;void* producer=nullptr;
            if(cuda_d3d12_color_get_resources(next->bridge,&buffer,&producer))throw std::runtime_error(nvof_d3d12_last_error());
            next->buffer=static_cast<ID3D12Resource*>(buffer);next->fence=static_cast<ID3D12Fence*>(producer);plane=std::move(next);
        }
        if(cuda_d3d12_color_copy(plane->bridge,ptr,pitch,&plane->value))throw std::runtime_error(nvof_d3d12_last_error());plane->hdr=isHDR;
    }
};
LRESULT CALLBACK proc(HWND window,UINT message,WPARAM wp,LPARAM lp){
    auto p=reinterpret_cast<Presenter*>(GetWindowLongPtrW(window,GWLP_USERDATA));
    if(message==WM_NCCREATE){p=static_cast<Presenter*>(reinterpret_cast<CREATESTRUCTW*>(lp)->lpCreateParams);SetWindowLongPtrW(window,GWLP_USERDATA,reinterpret_cast<LONG_PTR>(p));}
    if(!p)return DefWindowProcW(window,message,wp,lp);
    try{
        if(message==WM_CLOSE){DestroyWindow(window);return 0;}
        if(message==WM_DESTROY){PostQuitMessage(0);return 0;}
        if(message==WM_ERASEBKGND)return 1;
        if(message==WM_PAINT){PAINTSTRUCT paint{};BeginPaint(window,&paint);EndPaint(window,&paint);std::lock_guard<std::mutex> lock(p->mutex);p->draw();return 0;}
        if(message==UPDATE){std::lock_guard<std::mutex> lock(p->mutex);
            SetWindowPos(window,HWND_TOP,p->x,p->y,p->w,p->h,SWP_NOACTIVATE|(p->visible&&p->loaded?0:SWP_HIDEWINDOW));
            p->layoutDirty=false;
            if(p->swap&&!p->frozen){p->resize();p->colour();p->draw();}
            if(p->loaded&&p->visible)ShowWindow(window,SW_SHOWNOACTIVATE);return 0;}
        if(message==FRAME){std::lock_guard<std::mutex> lock(p->mutex);
            // Sent FRAME messages may overtake a posted layout UPDATE. Apply
            // its latest size before drawing or revealing the first picture.
            if(p->layoutDirty){SetWindowPos(window,HWND_TOP,p->x,p->y,p->w,p->h,SWP_NOACTIVATE|(p->visible?0:SWP_HIDEWINDOW));p->layoutDirty=false;}
            p->resize();if(MonitorFromWindow(window,MONITOR_DEFAULTTONEAREST)!=p->monitor)p->colour();p->draw();
            if(p->loaded&&p->visible&&!(GetWindowLongPtrW(window,GWL_STYLE)&WS_VISIBLE))
                SetWindowPos(window,HWND_TOP,p->x,p->y,p->w,p->h,SWP_NOACTIVATE|SWP_SHOWWINDOW);
            return 0;}
        if(message==WM_TIMER){std::lock_guard<std::mutex> lock(p->mutex);if(p->frozen)return 0;bool old=p->hdr;float white=p->sdrWhite;p->colour();if(old!=p->hdr||white!=p->sdrWhite)p->draw();return 0;}
        if(message==WM_LBUTTONDOWN||message==WM_RBUTTONDOWN){SetCapture(window);std::lock_guard<std::mutex> lock(p->mutex);
            ++p->pointerDowns;p->downX=float(GET_X_LPARAM(lp));
            p->drag=message==WM_LBUTTONDOWN&&p->mode==2&&std::abs(GET_X_LPARAM(lp)-p->split*p->w)<18.f;
            p->panDrag=message==WM_RBUTTONDOWN&&p->zoom>1;p->last={GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};return 0;}
        if(message==WM_SETCURSOR){POINT point{};GetCursorPos(&point);ScreenToClient(window,&point);std::lock_guard<std::mutex> lock(p->mutex);
            SetCursor(LoadCursorW(nullptr,MAKEINTRESOURCEW(p->mode==2&&std::abs(point.x-p->split*p->w)<18.f?32644:32512)));return TRUE;}
        if(message==WM_LBUTTONUP||message==WM_RBUTTONUP||message==WM_CAPTURECHANGED){ReleaseCapture();std::lock_guard<std::mutex> lock(p->mutex);p->drag=p->panDrag=false;return 0;}
        if(message==WM_MOUSEMOVE){std::lock_guard<std::mutex> lock(p->mutex);POINT point{GET_X_LPARAM(lp),GET_Y_LPARAM(lp)};
            if(p->drag){p->split=std::clamp(float(point.x)/p->w,.0f,1.f);p->draw(true);}
            else if(p->panDrag){p->panX=std::clamp(p->panX-float(point.x-p->last.x)/(p->w*p->zoom),0.f,1.f);
                p->panY=std::clamp(p->panY-float(point.y-p->last.y)/(p->h*p->zoom),0.f,1.f);p->draw(true);}p->last=point;return 0;}
    }catch(const std::exception& e){std::lock_guard<std::mutex> lock(p->mutex);p->failure=e.what();}
    return DefWindowProcW(window,message,wp,lp);
}
void run(Presenter* p){
    SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
    static std::once_flag registered;std::call_once(registered,[]{WNDCLASSW cls{};cls.lpfnWndProc=proc;cls.hInstance=GetModuleHandleW(nullptr);
        cls.lpszClassName=L"LunarcFP16Preview";cls.hCursor=LoadCursorW(nullptr,MAKEINTRESOURCEW(32644));RegisterClassW(&cls);});
    p->window=CreateWindowExW(WS_EX_NOPARENTNOTIFY,L"LunarcFP16Preview",L"Lunarc FP16 preview",WS_CHILD|WS_CLIPSIBLINGS,
        p->x,p->y,p->w,p->h,p->parent,nullptr,GetModuleHandleW(nullptr),p);
    try{if(!p->window)throw std::runtime_error("create native preview window failed");p->init();SetTimer(p->window,1,1000,nullptr);}
    catch(const std::exception& e){p->failure=e.what();SetEvent(p->ready);if(p->window)DestroyWindow(p->window);return;}
    SetEvent(p->ready); // Reveal only after the first complete frame is presented.
    MSG msg{};while(GetMessageW(&msg,nullptr,0,0)>0){TranslateMessage(&msg);DispatchMessageW(&msg);}
}
int fail(const std::exception& e){error=e.what();return -1;}
}
#define API extern "C" __declspec(dllexport)
API const char* gpu_native_presenter_error(){return error.c_str();}
API int gpu_native_presenter_open(ID3D12Device* device,UINT64 parent,Presenter** out){
    if(!out||!device||!IsWindow(reinterpret_cast<HWND>(parent))){error="invalid native preview parent";return -1;}*out=nullptr;
    auto p=std::make_unique<Presenter>();p->device=device;p->parent=reinterpret_cast<HWND>(parent);
    p->thread=std::thread(run,p.get());WaitForSingleObject(p->ready,INFINITE);
    if(!p->failure.empty()){error=p->failure;p->thread.join();return -1;}*out=p.release();return 0;}
API int gpu_native_presenter_config(Presenter* p,int x,int y,int w,int h,int visible,int mode,float zoom,float panX,float panY){
    if(!p||w<1||h<1||w>16384||h>16384||mode<0||mode>3||!std::isfinite(zoom)||zoom<.1f||zoom>32.f){error="invalid native preview layout";return -1;}
    {std::lock_guard<std::mutex> lock(p->mutex);
        p->layoutDirty=p->layoutDirty||p->x!=x||p->y!=y||p->w!=w||p->h!=h||p->visible!=(visible!=0);
        p->x=x;p->y=y;p->w=w;p->h=h;p->visible=visible!=0;p->mode=mode;
        if(p->zoom!=zoom||p->requestedPanX!=panX||p->requestedPanY!=panY){p->panX=std::clamp(panX,0.f,1.f);p->panY=std::clamp(panY,0.f,1.f);}
        p->zoom=zoom;p->requestedPanX=panX;p->requestedPanY=panY;}
    PostMessageW(p->window,UPDATE,0,0);return 0;}
API int gpu_native_presenter_frame(Presenter* p,void* context,UINT64 ap,UINT ar,UINT aw,UINT ah,UINT ab,UINT ahdr,
    UINT64 bp,UINT br,UINT bw,UINT bh,UINT bb,UINT bhdr){if(!p){error="native preview closed";return -1;}
    try{{std::lock_guard<std::mutex> lock(p->mutex);if(!p->failure.empty())throw std::runtime_error(p->failure);
        if(p->frozen)throw std::runtime_error("frozen preview cannot accept a new CUDA context");
        p->upload(p->a,context,ap,ar,aw,ah,ab,ahdr!=0);p->upload(p->b,context,bp,br,bw,bh,bb,bhdr!=0);p->loaded=true;++p->frames;
        }
        DWORD_PTR ignored=0;
        if(!SendMessageTimeoutW(p->window,FRAME,0,0,SMTO_ABORTIFHUNG,30000,&ignored))throw std::runtime_error("native preview presentation timed out");
        std::lock_guard<std::mutex> lock(p->mutex);if(!p->failure.empty())throw std::runtime_error(p->failure);
        return 0;}catch(const std::exception& e){return fail(e);}}
API int gpu_native_presenter_freeze(Presenter* p){if(!p)return -1;
    try{std::lock_guard<std::mutex> lock(p->mutex);
        if(!p->failure.empty())throw std::runtime_error(p->failure);
        // The swapchain already owns the last displayed panel. Retire all CUDA
        // imports while their borrowed context is alive, and keep only that
        // panel instead of retaining two full-resolution source/result buffers.
        p->frozen=true;p->a.reset();p->b.reset();return 0;
    }catch(const std::exception& e){return fail(e);}}
API int gpu_native_presenter_stats(Presenter* p,Stats* out){if(!p||!out||out->size!=sizeof(Stats))return -1;
    std::lock_guard<std::mutex> lock(p->mutex);if(!p->failure.empty()){error=p->failure;return -1;}
    *out={sizeof(Stats),UINT(p->hdr),UINT(p->w),UINT(p->h),p->frames,p->presents,p->dragPresents,0,p->split,p->sdrWhite,reinterpret_cast<UINT64>(p->window),p->pointerDowns,p->downX};return 0;}
API void gpu_native_presenter_close(Presenter* p){if(!p)return;PostMessageW(p->window,WM_CLOSE,0,0);if(p->thread.joinable())p->thread.join();
    if(p->poisoned&&SUCCEEDED(p->device->GetDeviceRemovedReason()))return;delete p;}
