#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <string>
#include <mutex>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <new>
#include <cwctype>
#include "include/cuda.h"
#include "rtx_vsr_bridge.h"

// ABI-only subset of NVIDIA's MIT-licensed public NvCVImage/NvVFX headers.
// Declaring it locally avoids emitting the SDK header's optional C++ wrapper
// methods, which would require static NvCVImage import libraries.
typedef int NvCV_Status;
enum { NVCV_SUCCESS = 0, NVCV_RGBA = 6, NVCV_RGB10A2 = 13, NVCV_U8 = 1, NVCV_F16 = 4, NVCV_P32 = 11, NVCV_INTERLEAVED = 0, NVCV_CUDA = 1 };
struct NvCVImage {
    unsigned int width, height;
    signed int pitch;
    int pixelFormat, componentType;
    unsigned char pixelBytes, componentBytes, numComponents, planar, gpuMem, colorspace, reserved[2];
    void* pixels;
    void* deletePtr;
    void (*deleteProc)(void*);
    unsigned long long bufferBytes;
};
struct NvVFX_Object;
typedef NvVFX_Object* NvVFX_Handle;
typedef const char* NvVFX_EffectSelector;
typedef const char* NvVFX_ParameterSelector;

using CtxPush = CUresult (CUDAAPI*)(CUcontext);
using CtxPop = CUresult (CUDAAPI*)(CUcontext*);
using CreateEffect = NvCV_Status (__cdecl*)(NvVFX_EffectSelector, NvVFX_Handle*);
using DestroyEffect = void (__cdecl*)(NvVFX_Handle);
using SetU32 = NvCV_Status (__cdecl*)(NvVFX_Handle, NvVFX_ParameterSelector, unsigned int);
using SetF32 = NvCV_Status (__cdecl*)(NvVFX_Handle, NvVFX_ParameterSelector, float);
using SetImage = NvCV_Status (__cdecl*)(NvVFX_Handle, NvVFX_ParameterSelector, NvCVImage*);
using SetStream = NvCV_Status (__cdecl*)(NvVFX_Handle, NvVFX_ParameterSelector, CUstream);
using LoadEffect = NvCV_Status (__cdecl*)(NvVFX_Handle);
using RunEffect = NvCV_Status (__cdecl*)(NvVFX_Handle, int);
using ImageInit = NvCV_Status (__cdecl*)(NvCVImage*, unsigned, unsigned, int, void*, int, int, int, int);

static std::mutex g_mutex;
static thread_local std::string g_error;
static HMODULE g_cuda = nullptr;
static HMODULE g_effects = nullptr;
static HMODULE g_image = nullptr;
static std::wstring g_runtime_path;
static uint32_t g_runtime_users = 0;
static std::vector<HMODULE> g_feature_modules;
static std::vector<DLL_DIRECTORY_COOKIE> g_directory_cookies;
static CtxPush g_push = nullptr;
static CtxPop g_pop = nullptr;
static CreateEffect g_create = nullptr;
static DestroyEffect g_destroy = nullptr;
static SetU32 g_set_u32 = nullptr;
static SetF32 g_set_f32 = nullptr;
static SetImage g_set_image = nullptr;
static SetStream g_set_stream = nullptr;
static LoadEffect g_load = nullptr;
static RunEffect g_run = nullptr;
static ImageInit g_image_init = nullptr;

struct RtxVsrHandle {
    CUcontext context{};
    NvVFX_Handle effect{};
    uint32_t iw{}, ih{}, ow{}, oh{}, quality{};
    uint32_t component_type = NVCV_U8;
    bool loaded{};
    RtxVsrCounters counters{};
};

static int fail(const char* msg, int code = -1) { g_error = msg; return code; }
static bool bind(HMODULE m, const char* n, FARPROC* p) { *p = m ? GetProcAddress(m, n) : nullptr; return *p != nullptr; }
static std::wstring normalized_path(const wchar_t* root) {
    if (!root || !*root) return {};
    wchar_t full[MAX_PATH * 4]{};
    DWORD n = GetFullPathNameW(root, static_cast<DWORD>(_countof(full)), full, nullptr);
    if (!n || n >= _countof(full)) return {};
    std::wstring value(full, n);
    for (auto& ch : value) {
        if (ch == L'/') ch = L'\\';
        ch = static_cast<wchar_t>(towlower(ch));
    }
    while (value.size() > 3 && value.back() == L'\\') value.pop_back();
    return value;
}

static void unload_runtime() {
    for (auto it = g_feature_modules.rbegin(); it != g_feature_modules.rend(); ++it) FreeLibrary(*it);
    g_feature_modules.clear();
    if (g_image) FreeLibrary(g_image); g_image = nullptr;
    if (g_effects) FreeLibrary(g_effects); g_effects = nullptr;
    if (g_cuda) FreeLibrary(g_cuda); g_cuda = nullptr;
    for (auto it = g_directory_cookies.rbegin(); it != g_directory_cookies.rend(); ++it) RemoveDllDirectory(*it);
    g_directory_cookies.clear();
    g_push = nullptr; g_pop = nullptr; g_create = nullptr; g_destroy = nullptr;
    g_set_u32 = nullptr; g_set_f32 = nullptr; g_set_image = nullptr; g_set_stream = nullptr;
    g_load = nullptr; g_run = nullptr; g_image_init = nullptr;
}

static bool load_runtime(const wchar_t* root) {
    if (!root || !*root) { fail("VFX runtime path is required; install NVIDIA Maxine VFX SDK Core and VideoSuperRes feature files, then pass their runtime directory"); return false; }
    std::wstring base(root);
    while (!base.empty() && (base.back() == L'\\' || base.back() == L'/')) base.pop_back();
    base = normalized_path(base.c_str());
    if (base.empty()) { fail("could not resolve VFX runtime directory"); return false; }
    const wchar_t* dirs[] = { L"", L"\\bin", L"\\lib", L"\\models", L"\\effects",
        L"\\features\\nvvfxvideosuperres\\bin", L"\\features\\nvvfxvideoframegeneration\\bin" };
    for (auto suffix : dirs) {
        auto cookie = AddDllDirectory((base + suffix).c_str());
        if (cookie) g_directory_cookies.push_back(cookie);
    }
    auto load = [&](const wchar_t* name) -> HMODULE {
        for (auto suffix : dirs) {
            const auto path = base + suffix + L"\\" + name;
            if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) continue;
            auto module = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
            if (module) return module;
        }
        return nullptr;
    };
    g_cuda = LoadLibraryW(L"nvcuda.dll");
    if (!g_cuda) { fail("NVIDIA CUDA driver nvcuda.dll could not be loaded"); return false; }
    g_effects = load(L"NVVideoEffects.dll");
    g_image = load(L"NVCVImage.dll");
    if (!g_effects || !g_image) { fail("NVIDIA VFX runtime missing NVVideoEffects.dll or NVCVImage.dll in configured runtime directory"); return false; }
    for (const wchar_t* f : { L"nvngxruntime.dll" }) {
        HMODULE h = load(f);
        if (!h) { fail("NVIDIA VFX runtime dependency missing (nvngxruntime.dll)"); return false; }
        g_feature_modules.push_back(h);
    }
    FARPROC p{};
    if (!bind(g_cuda, "cuCtxPushCurrent_v2", &p)) { fail("CUDA driver is missing cuCtxPushCurrent_v2"); return false; } g_push = reinterpret_cast<CtxPush>(p);
    if (!bind(g_cuda, "cuCtxPopCurrent_v2", &p)) { fail("CUDA driver is missing cuCtxPopCurrent_v2"); return false; } g_pop = reinterpret_cast<CtxPop>(p);
#define BIND(module, symbol, target) do { if (!bind(module, symbol, &p)) { fail("VFX runtime is missing " symbol); return false; } target = reinterpret_cast<decltype(target)>(p); } while (0)
    BIND(g_effects, "NvVFX_CreateEffect", g_create);
    BIND(g_effects, "NvVFX_DestroyEffect", g_destroy);
    BIND(g_effects, "NvVFX_SetU32", g_set_u32);
    BIND(g_effects, "NvVFX_SetF32", g_set_f32);
    BIND(g_effects, "NvVFX_SetImage", g_set_image);
    BIND(g_effects, "NvVFX_SetCudaStream", g_set_stream);
    BIND(g_effects, "NvVFX_Load", g_load);
    BIND(g_effects, "NvVFX_Run", g_run);
    BIND(g_image, "NvCVImage_Init", g_image_init);
#undef BIND
    g_runtime_path = base;
    return true;
}

extern "C" RTX_VSR_API uint32_t rtx_vsr_abi_version(void) { return 1; }
extern "C" RTX_VSR_API const char* rtx_vsr_last_error(void) { return g_error.c_str(); }

static bool load_vsr_feature(const std::wstring& root) {
    for (const wchar_t* suffix : {L"", L"\\bin", L"\\effects", L"\\features\\nvvfxvideosuperres\\bin"}) {
        auto module = LoadLibraryExW((root + suffix + L"\\nvVFXVideoSuperRes.dll").c_str(), nullptr,
            LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_USER_DIRS | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        if (module) { g_feature_modules.push_back(module); return true; }
    }
    fail("NVIDIA VideoSuperRes feature library is missing"); return false;
}

extern "C" RTX_VSR_API int32_t rtx_vsr_open2(void* context, uint32_t iw, uint32_t ih,
    uint32_t ow, uint32_t oh, uint32_t quality, uint32_t component_type, const wchar_t* runtime, RtxVsrHandle** out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!out) return fail("out_handle is null"); *out = nullptr;
    if(component_type != NVCV_U8 && component_type != NVCV_F16 && component_type != NVCV_P32) return fail("unsupported VSR component type");
    if (!context || iw < 64 || ih < 64 || ow < 64 || oh < 64 || iw > 16384 || ih > 16384 || ow > 16384 || oh > 16384 || quality < 1 || quality > 4)
        return fail("invalid CUDA context, image extent, or VideoSuperRes quality (expected 1..4)");
    if (ow < iw || oh < ih || ow > iw * 4 || oh > ih * 4)
        return fail("VideoSuperRes output extent must upscale each axis by 1..4x");
    const std::wstring requested_path = normalized_path(runtime);
    if (g_effects && requested_path != g_runtime_path) {
        if (g_runtime_users) return fail("a different VFX runtime directory is active; close existing VSR handles before changing runtime path", -2);
        unload_runtime();
    }
    if (!g_effects && !load_runtime(runtime)) { unload_runtime(); return -2; }
    if (!load_vsr_feature(requested_path)) { if (!g_runtime_users) unload_runtime(); return -2; }
    CUcontext old{};
    if (g_push(static_cast<CUcontext>(context)) != CUDA_SUCCESS) return fail("could not make caller CUDA context current");
    auto* h = new (std::nothrow) RtxVsrHandle{};
    if (!h) { g_pop(&old); return fail("out of memory creating VSR handle"); }
    h->context = static_cast<CUcontext>(context); h->iw = iw; h->ih = ih; h->ow = ow; h->oh = oh; h->quality = quality;
    h->component_type = component_type;
    auto status = g_create("VideoSuperRes", &h->effect);
    if (status == NVCV_SUCCESS) status = g_set_u32(h->effect, "QualityLevel", quality);
    if (status == NVCV_SUCCESS && component_type == NVCV_P32)
        status = g_set_u32(h->effect, "ImageEncodingMode", 1);
    if (status != NVCV_SUCCESS) {
        if (h->effect) g_destroy(h->effect);
        g_pop(&old); delete h;
        return fail("NvVFX_CreateEffect/QualityLevel failed", static_cast<int>(status));
    }
    g_pop(&old);
    h->counters.input_width = iw; h->counters.input_height = ih;
    h->counters.output_width = ow; h->counters.output_height = oh; h->counters.quality_level = quality;
    *out = h; ++g_runtime_users; g_error.clear(); return 0;
}

extern "C" RTX_VSR_API int32_t rtx_vsr_open(void* context, uint32_t iw, uint32_t ih,
    uint32_t ow, uint32_t oh, uint32_t quality, const wchar_t* runtime, RtxVsrHandle** out) {
    return rtx_vsr_open2(context,iw,ih,ow,oh,quality,NVCV_U8,runtime,out);
}

extern "C" RTX_VSR_API int32_t rtx_vsr_process(RtxVsrHandle* h, uint64_t input, uint32_t ipitch, uint64_t output, uint32_t opitch, void* cuda_stream) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!h || !input || !output || ipitch < h->iw * (h->component_type==NVCV_F16 ? 8u : 4u) || opitch < h->ow * (h->component_type==NVCV_F16 ? 8u : 4u) ||
        ipitch > 0x7fffffffu || opitch > 0x7fffffffu) return fail("invalid CUDA image pointer or pitch");
    CUcontext old{};
    if (g_push(h->context) != CUDA_SUCCESS) return fail("could not activate caller CUDA context");
    NvCVImage src{}, dst{};
    const char* stage = "NvCVImage_Init(input)";
    const int pixel_format = h->component_type == NVCV_P32 ? NVCV_RGB10A2 : NVCV_RGBA;
    auto status = g_image_init(&src, h->iw, h->ih, static_cast<int>(ipitch), reinterpret_cast<void*>(static_cast<uintptr_t>(input)), pixel_format, static_cast<int>(h->component_type), NVCV_INTERLEAVED, NVCV_CUDA);
    if (status == NVCV_SUCCESS) { stage="NvCVImage_Init(output)"; status = g_image_init(&dst, h->ow, h->oh, static_cast<int>(opitch), reinterpret_cast<void*>(static_cast<uintptr_t>(output)), pixel_format, static_cast<int>(h->component_type), NVCV_INTERLEAVED, NVCV_CUDA); }
    if (status == NVCV_SUCCESS) { stage="NvVFX_SetImage(input)"; status = g_set_image(h->effect, "SrcImage0", &src); }
    if (status == NVCV_SUCCESS) { stage="NvVFX_SetImage(output)"; status = g_set_image(h->effect, "DstImage0", &dst); }
    if (status == NVCV_SUCCESS) { stage="NvVFX_SetCudaStream"; status = g_set_stream(h->effect, "CudaStream", static_cast<CUstream>(cuda_stream)); }
    ++h->counters.process_calls;
    if (status == NVCV_SUCCESS && !h->loaded) { stage="NvVFX_Load"; status = g_load(h->effect); h->loaded = status == NVCV_SUCCESS; h->counters.model_loaded = h->loaded ? 1u : 0u; }
    if (status == NVCV_SUCCESS) { stage="NvVFX_Run"; status = g_run(h->effect, 0); }
    g_pop(&old);
    if (status != NVCV_SUCCESS) return fail(stage, static_cast<int>(status));
    ++h->counters.successful_runs; g_error.clear(); return 0;
}

extern "C" RTX_VSR_API int32_t rtx_vsr_get_counters(RtxVsrHandle* h, RtxVsrCounters* out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!h || !out) return fail("invalid handle or output counters"); *out = h->counters; return 0;
}
extern "C" RTX_VSR_API void rtx_vsr_close(RtxVsrHandle* h) {
    if (!h) return;
    std::lock_guard<std::mutex> lock(g_mutex);
    CUcontext old{};
    if (!g_push || !g_pop || g_push(h->context) != CUDA_SUCCESS) {
        fail("could not activate owning CUDA context; VSR handle retained for safe cleanup");
        return;
    }
    if (h->effect && g_destroy) g_destroy(h->effect);
    g_pop(&old);
    delete h;
    if (g_runtime_users) --g_runtime_users;
    if (!g_runtime_users) unload_runtime();
    g_error.clear();
}

// Offline VFG writes an ordinary CUDA output image; no swap-chain interception.
extern "C" RTX_VSR_API int32_t vfx_framegen_open2(void* context,uint32_t w,uint32_t h,
    uint32_t multiplier,uint32_t mode,uint32_t packed10,uint32_t scene_detection,
    const wchar_t* runtime,RtxVsrHandle** out) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if(!out)return fail("null VFG output");*out=nullptr;
    if(!context||w<64||h<64||w>16384||h>16384||
       (multiplier!=0 && (multiplier<2||multiplier>8))||mode>2||packed10>1||scene_detection>1)
        return fail("invalid VFG context/extent/multiplier/mode");
    const auto path=normalized_path(runtime);
    if(g_effects && path!=g_runtime_path) {
        if(g_runtime_users)return fail("a different VFX runtime is active",-2);
        unload_runtime();
    }
    if(!g_effects && !load_runtime(runtime)){unload_runtime();return -2;}
    const auto feature=path+L"\\features\\nvvfxvideoframegeneration\\bin\\nvVFXVideoFrameGeneration.dll";
    auto module=LoadLibraryExW(feature.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_USER_DIRS|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!module)return fail("VideoFrameGeneration 1.3 feature library is missing",-2);
    g_feature_modules.push_back(module);
    CUcontext old{};if(g_push(static_cast<CUcontext>(context))!=CUDA_SUCCESS)return fail("VFG context activation failed");
    auto handle=std::make_unique<RtxVsrHandle>();handle->context=static_cast<CUcontext>(context);
    handle->iw=handle->ow=w;handle->ih=handle->oh=h;handle->quality=multiplier;
    handle->component_type=packed10 ? NVCV_P32:NVCV_U8;
    auto status=g_create("VideoFrameGeneration",&handle->effect);
    const char* stage="NvVFX_CreateEffect(VideoFrameGeneration)";
    auto set=[&](const char* name,unsigned value) {if(status==NVCV_SUCCESS){stage=name;status=g_set_u32(handle->effect,name,value);}};
    set("InputWidth",w);set("InputHeight",h);set("Mode",mode);set("FrameMultiplier",multiplier);
    set("AutomaticShotChangeDetectionEnabled",scene_detection);
    if(status==NVCV_SUCCESS){stage="CudaStream";status=g_set_stream(handle->effect,"CudaStream",nullptr);}
    if(status==NVCV_SUCCESS){stage="NvVFX_Load(VideoFrameGeneration)";status=g_load(handle->effect);}
    if(status!=NVCV_SUCCESS){if(handle->effect)g_destroy(handle->effect);g_pop(&old);return fail(stage,status);}
    g_pop(&old);handle->loaded=true;handle->counters.model_loaded=1;
    handle->counters.input_width=handle->counters.output_width=w;handle->counters.input_height=handle->counters.output_height=h;
    handle->counters.quality_level=mode;*out=handle.release();++g_runtime_users;g_error.clear();return 0;
}
extern "C" RTX_VSR_API int32_t vfx_framegen_open(void* context,uint32_t w,uint32_t h,
    uint32_t multiplier,uint32_t mode,uint32_t packed10,const wchar_t* runtime,RtxVsrHandle** out) {
    return vfx_framegen_open2(context,w,h,multiplier,mode,packed10,1,runtime,out);
}

static int32_t framegen_process(RtxVsrHandle* h,uint64_t previous,uint32_t ppitch,
    uint64_t current,uint32_t cpitch,uint64_t output,uint32_t opitch,uint32_t index,
    float timestep,uint32_t reset,bool dynamic) {
    if(!h||!previous||!current||!output||ppitch<h->iw*4||cpitch<h->iw*4||opitch<h->iw*4||
       ppitch>0x7fffffffu||cpitch>0x7fffffffu||opitch>0x7fffffffu||reset>1||
       (dynamic ? (h->quality!=0||!std::isfinite(timestep)||timestep<=0||timestep>=1)
                : (index<1||index>=h->quality)))
        return fail("invalid VFG image/index");
    CUcontext old{};if(g_push(h->context)!=CUDA_SUCCESS)return fail("VFG context activation failed");
    NvCVImage images[3]{};NvCV_Status status=NVCV_SUCCESS;const char* stage="NvCVImage_Init(VFG)";
    uint64_t pointers[]={previous,current,output};uint32_t pitches[]={ppitch,cpitch,opitch};
    const char* names[]={"SrcImage0","SrcImage1","DstImage0"};
    for(unsigned i=0;i<3 && status==NVCV_SUCCESS;i++) {
        status=g_image_init(&images[i],h->iw,h->ih,static_cast<int>(pitches[i]),
            reinterpret_cast<void*>(static_cast<uintptr_t>(pointers[i])),
            h->component_type==NVCV_P32 ? NVCV_RGB10A2:NVCV_RGBA,static_cast<int>(h->component_type),NVCV_INTERLEAVED,NVCV_CUDA);
        if(status==NVCV_SUCCESS){stage=names[i];status=g_set_image(h->effect,names[i],&images[i]);}
    }
    if(status==NVCV_SUCCESS){
        if(dynamic){stage="Timestep";status=g_set_f32(h->effect,"Timestep",timestep);}
        else{stage="FrameIndex";status=g_set_u32(h->effect,"FrameIndex",index);}
    }
    if(status==NVCV_SUCCESS){stage="ShotChange";status=g_set_u32(h->effect,"ShotChange",reset);}
    ++h->counters.process_calls;
    if(status==NVCV_SUCCESS){stage="NvVFX_Run(VideoFrameGeneration)";status=g_run(h->effect,0);}
    g_pop(&old);if(status!=NVCV_SUCCESS)return fail(stage,status);
    ++h->counters.successful_runs;g_error.clear();return 0;
}

extern "C" RTX_VSR_API int32_t vfx_framegen_process(RtxVsrHandle* h,uint64_t previous,uint32_t ppitch,
    uint64_t current,uint32_t cpitch,uint64_t output,uint32_t opitch,uint32_t index,uint32_t reset) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return framegen_process(h,previous,ppitch,current,cpitch,output,opitch,index,0,reset,false);
}
extern "C" RTX_VSR_API int32_t vfx_framegen_process_time(RtxVsrHandle* h,uint64_t previous,uint32_t ppitch,
    uint64_t current,uint32_t cpitch,uint64_t output,uint32_t opitch,float timestep,uint32_t reset) {
    std::lock_guard<std::mutex> lock(g_mutex);
    return framegen_process(h,previous,ppitch,current,cpitch,output,opitch,0,timestep,reset,true);
}
