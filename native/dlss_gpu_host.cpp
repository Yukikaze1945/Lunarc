#define DLSS_GPU_HOST_EXPORTS
#include "dlss_gpu_host.h"
#include "nvof_d3d12_interop.h"
#include "gpu_dred_diagnostics.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>
#include <d3dcompiler.h>
#include <nvsdk_ngx.h>
#include <wrl/client.h>

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <new>
#include <unordered_set>

using Microsoft::WRL::ComPtr;

namespace {
// Diagnostic-only build of this DLL under a distinct filename: it opens the
// wide colour format when no open info says otherwise, which is the whole of
// what that artifact asked for. The packaged host never defines this macro, so
// a caller that says nothing gets the RGBA8 it has always got.
#ifdef DLSS_GPU_HOST_FP16_PROBE
constexpr uint32_t kDefaultColorPixels = DLSS_GPU_HOST_COLOR_RGBA16F;
#else
constexpr uint32_t kDefaultColorPixels = DLSS_GPU_HOST_COLOR_RGBA8;
#endif
constexpr DXGI_FORMAT kDefaultColorFormat =
    kDefaultColorPixels == DLSS_GPU_HOST_COLOR_RGBA16F
        ? DXGI_FORMAT_R16G16B16A16_FLOAT : DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t kDefaultColorBytes =
    kDefaultColorPixels == DLSS_GPU_HOST_COLOR_RGBA16F ? 8u : 4u;

bool color_geometry(uint32_t format, DXGI_FORMAT* dxgi, uint32_t* bytes) {
    switch (format) {
        case DLSS_GPU_HOST_COLOR_RGBA8:
            *dxgi = DXGI_FORMAT_R8G8B8A8_UNORM;
            *bytes = 4;
            return true;
        case DLSS_GPU_HOST_COLOR_RGBA16F:
            *dxgi = DXGI_FORMAT_R16G16B16A16_FLOAT;
            *bytes = 8;
            return true;
        default:
            return false;
    }
}

/* The eight-byte row is a property of the interop DLL, not of this one, so it
 * is asked for at runtime instead of being bound at link time: an
 * nvof_d3d12_interop.dll that predates the additions still loads this host and
 * still runs every RGBA8 session exactly as it did, and only a session opened
 * wide needs the new entry points. It is looked up by the name the loader
 * already attached, so a second copy is never loaded and a search path is never
 * trusted; if the module predates the additions the wide format simply stays
 * unready and an FP16 open refuses. */
constexpr wchar_t kRowBridgeModule[] = L"nvof_d3d12_interop.dll";

using ColorCreate2 = int32_t(*)(void*, void*, uint32_t, uint32_t, uint32_t,
    uint32_t, int32_t, CudaD3D12ColorBridge**);
using ColorRowCapabilities = uint32_t(*)();
using ColorGetLayout = int32_t(*)(CudaD3D12ColorBridge*, CudaD3D12ColorLayout*);

struct RowBridge {
    ColorCreate2 create2 = nullptr;
    ColorGetLayout get_layout = nullptr;
    uint32_t capabilities = 0;
    bool wide_ready = false;
};

const RowBridge& row_bridge() {
    static const RowBridge bridge = [] {
        RowBridge out;
        /* The DLL this host imports its colour copies from is already loaded by
         * the time any of this runs, so a name lookup cannot attach a second
         * copy; a missing symbol simply leaves the wide format unready. */
        HMODULE module = GetModuleHandleW(kRowBridgeModule);
        if (!module) return out;
        out.create2 = reinterpret_cast<ColorCreate2>(
            GetProcAddress(module, "cuda_d3d12_color_create2"));
        out.get_layout = reinterpret_cast<ColorGetLayout>(
            GetProcAddress(module, "cuda_d3d12_color_get_layout"));
        const ColorRowCapabilities capabilities =
            reinterpret_cast<ColorRowCapabilities>(GetProcAddress(module,
                "cuda_d3d12_color_row_capabilities"));
        out.capabilities = capabilities ? capabilities() : 0u;
        out.wide_ready = out.create2 && out.get_layout &&
            (out.capabilities & CUDA_D3D12_COLOR_ROW_RGBA16F);
        return out;
    }();
    return bridge;
}

using Init = NVSDK_NGX_Result(NVSDK_CONV*)(unsigned long long, const wchar_t*,
    ID3D12Device*, NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
using Create = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
    NVSDK_NGX_Feature, NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12GraphicsCommandList*,
    const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback);
using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using ModuleName = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);

thread_local char last_error[512] = "";
std::atomic<bool> active{false};
ModuleName original_module_name = nullptr;
HMODULE host_module = nullptr;

struct SharedNgxCore {
    HMODULE snippet = nullptr;
    void** import_slot = nullptr;
    void* old_import = nullptr;
    bool hook_installed = false;
    bool core_initialized = false;
    bool snippet_initialized = false;
    bool owns_active = false;
    Shutdown shutdown = nullptr;
    ComPtr<ID3D12Device> device;
    std::mutex ngx_mutex;
    std::unordered_set<NVSDK_NGX_Handle*> features;
    uint64_t next_session_id = 1;
    std::atomic<bool> auxiliary_dlaa_active{false};
    ~SharedNgxCore();
};

void error(const char* action, unsigned long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", action, code);
}

void trace_cleanup(const char* action, NVSDK_NGX_Result result) {
#ifdef DLSS_GPU_HOST_CLEANUP_PROBE
    std::fprintf(stderr, "NGX cleanup %s: 0x%08lx success=%d\n", action,
        static_cast<unsigned long>(result), NVSDK_NGX_SUCCEED(result) ? 1 : 0);
    std::fflush(stderr);
#else
    (void)action;
    (void)result;
#endif
}

DWORD WINAPI compatible_module_name(HMODULE requested, LPWSTR destination,
                                     DWORD capacity) {
    if (requested != host_module) return original_module_name(
        requested, destination, capacity);
    constexpr wchar_t value[] = L"nvngx.dll";
    constexpr DWORD size = 9;
    if (!destination || !capacity) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    const DWORD copied = capacity > size ? size : capacity - 1;
    std::memcpy(destination, value, copied * sizeof(wchar_t));
    destination[copied] = 0;
    if (capacity <= size) SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return capacity > size ? size : capacity;
}

void** find_module_name_slot(HMODULE library) {
    auto* base = reinterpret_cast<unsigned char*>(library);
    const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return nullptr;
    const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const DWORD image_size = nt->OptionalHeader.SizeOfImage;
    const auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!directory.VirtualAddress || directory.VirtualAddress >= image_size ||
        directory.Size > image_size - directory.VirtualAddress) return nullptr;
    const auto* imports = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(
        base + directory.VirtualAddress);
    const DWORD count = directory.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR);
    for (DWORD i = 0; i < count && imports[i].Name; ++i) {
        const auto& descriptor = imports[i];
        if (!descriptor.OriginalFirstThunk || !descriptor.FirstThunk ||
            descriptor.OriginalFirstThunk >= image_size ||
            descriptor.FirstThunk >= image_size) continue;
        const auto* names = reinterpret_cast<const IMAGE_THUNK_DATA64*>(
            base + descriptor.OriginalFirstThunk);
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(base + descriptor.FirstThunk);
        const DWORD n = static_cast<DWORD>((image_size - descriptor.OriginalFirstThunk) /
            sizeof(IMAGE_THUNK_DATA64));
        const DWORD m = static_cast<DWORD>((image_size - descriptor.FirstThunk) /
            sizeof(IMAGE_THUNK_DATA64));
        for (DWORD j = 0; j < (n < m ? n : m) && names[j].u1.AddressOfData; ++j) {
            if (IMAGE_SNAP_BY_ORDINAL64(names[j].u1.Ordinal)) continue;
            const auto rva = names[j].u1.AddressOfData;
            if (rva >= image_size || image_size - rva < sizeof(IMAGE_IMPORT_BY_NAME))
                continue;
            const auto* imported = reinterpret_cast<const IMAGE_IMPORT_BY_NAME*>(base + rva);
            if (std::strcmp(reinterpret_cast<const char*>(imported->Name),
                            "GetModuleFileNameW") == 0)
                return reinterpret_cast<void**>(&slots[j].u1.Function);
        }
    }
    return nullptr;
}

bool patch_slot(void** slot, void* replacement, void** previous) {
    DWORD old_protection = 0;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &old_protection))
        return false;
    *previous = InterlockedExchangePointer(
        reinterpret_cast<void* volatile*>(slot), replacement);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(*slot), old_protection, &ignored);
    return true;
}

SharedNgxCore::~SharedNgxCore() {
    // No feature may survive the last core owner. NGX APIs are not thread safe.
    if (snippet_initialized && shutdown) trace_cleanup("snippet Shutdown1", shutdown(device.Get()));
    bool restored = true;
    if (hook_installed) {
        void* replaced = nullptr;
        restored = patch_slot(import_slot, old_import, &replaced);
        original_module_name = nullptr;
        host_module = nullptr;
    }
    if (core_initialized) trace_cleanup("core Shutdown1", NVSDK_NGX_D3D12_Shutdown1(device.Get()));
    if (snippet && restored) FreeLibrary(snippet);
    if (owns_active) active.store(false);
}

NVSDK_NGX_Result safe_core_init(const wchar_t* directory, ID3D12Device* device,
                                DWORD* exception) {
    __try {
        return NVSDK_NGX_D3D12_Init_with_ProjectID(
            "86a83b2d-d0a9-4be6-aa59-39c5e931bdc0",
            NVSDK_NGX_ENGINE_TYPE_CUSTOM, "dlss5standaloneV2", directory,
            device, nullptr, NVSDK_NGX_Version_API);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_snippet_init(Init fn, const wchar_t* directory,
                                   ID3D12Device* device, DWORD* exception) {
    __try {
        return fn(0x0876232Cull, directory, device, NVSDK_NGX_Version_API,
                  nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_create(Create fn, ID3D12GraphicsCommandList* list,
                             NVSDK_NGX_Parameter* parameters,
                             NVSDK_NGX_Handle** feature, DWORD* exception,
                             NVSDK_NGX_Feature feature_id = static_cast<NVSDK_NGX_Feature>(18)) {
    __try {
        return fn(list, feature_id, parameters,
                  feature);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result safe_evaluate(Evaluate fn, ID3D12GraphicsCommandList* list,
                               const NVSDK_NGX_Handle* feature,
                               const NVSDK_NGX_Parameter* parameters,
                               DWORD* exception) {
    __try {
        return fn(list, feature, parameters, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

NVSDK_NGX_Result NVSDK_CONV scaling_ratio(NVSDK_NGX_Parameter* parameters) {
    if (!parameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    parameters->Set("DLSSNR.ScalingRatio", 1.0f);
    return NVSDK_NGX_Result_Success;
}

bool configure_create(NVSDK_NGX_Parameter* p, uint32_t width, uint32_t height,
                      int preset, DWORD* exception, bool dlaa = false, bool hdr = false) {
    __try {
        if (dlaa) {
            // DLAA uses SuperSampling render hints, separate from Feature 18 NR presets.
            // Zero preserves driver/default behavior; old model hints may be remapped by NGX.
            if (preset != 0)
                p->Set(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                       static_cast<unsigned int>(preset));
            p->Set(NVSDK_NGX_Parameter_Width, width);
            p->Set(NVSDK_NGX_Parameter_Height, height);
            p->Set(NVSDK_NGX_Parameter_OutWidth, width);
            p->Set(NVSDK_NGX_Parameter_OutHeight, height);
            p->Set(NVSDK_NGX_Parameter_PerfQualityValue,
                   static_cast<int>(NVSDK_NGX_PerfQuality_Value_DLAA));
            p->Set(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                   static_cast<int>(NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                    NVSDK_NGX_DLSS_Feature_Flags_DepthInverted |
                                    NVSDK_NGX_DLSS_Feature_Flags_AutoExposure |
                                    (hdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0)));
            p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
            p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
            return true;
        }
        const char* widths[] = {"DLSSNR.Width", "DLSSNR.InputWidth",
                                "DLSSNR.OutputWidth", "DLSSNR.Output.Width"};
        const char* heights[] = {"DLSSNR.Height", "DLSSNR.InputHeight",
                                 "DLSSNR.OutputHeight", "DLSSNR.Output.Height"};
        for (const auto* key : widths) p->Set(key, width);
        for (const auto* key : heights) p->Set(key, height);
        p->Set("DLSSNR.Upscaling", 0u);
        p->Set("DLSSNR.Scale", 1.0f);
        p->Set("DLSSNR.ScalingRatio", 1.0f);
        p->Set("DLSSNRComputeScalingRatioCallback",
               reinterpret_cast<void*>(&scaling_ratio));
        p->Set("DLSSNR.Hint.Render.Preset", preset);
        p->Set(NVSDK_NGX_Parameter_Width, width);
        p->Set(NVSDK_NGX_Parameter_Height, height);
        p->Set(NVSDK_NGX_Parameter_PerfQualityValue,
               static_cast<int>(NVSDK_NGX_PerfQuality_Value_Balanced));
        p->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
        p->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return false;
    }
}

void set_rect(NVSDK_NGX_Parameter* p, const char* name,
              uint32_t width, uint32_t height) {
    char key[64];
    std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", name);
    p->Set(key, 0u);
    std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", name);
    p->Set(key, 0u);
    std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", name);
    p->Set(key, width);
    std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", name);
    p->Set(key, height);
}

} // namespace

struct DlssGpuHost {
    HMODULE auxiliary_snippet = nullptr;
    Shutdown auxiliary_shutdown = nullptr;
    void** auxiliary_import_slot = nullptr;
    void* auxiliary_old_import = nullptr;
    bool auxiliary_initialized = false;
    bool auxiliary_reserved = false;
    uint32_t width = 0, height = 0;
    bool dlaa = false;
    /* Colour is a property of the session, chosen at open and never adapted:
     * every row the session stages, copies, binds or reads back is this wide,
     * so an FP16 session and an RGBA8 one can be alive in the same process
     * without either one forcing the other's textures. Both open paths write
     * the three together off one color_geometry call. */
    uint32_t color_pixels = kDefaultColorPixels;
    uint32_t bytes_per_pixel = kDefaultColorBytes;
    DXGI_FORMAT color_format = kDefaultColorFormat;
    DlssGpuHostOptions options{};
    HMODULE snippet = nullptr;
    void** import_slot = nullptr;
    void* old_import = nullptr;
    bool hook_installed = false;
    bool core_initialized = false;
    bool snippet_initialized = false;
    Init init = nullptr;
    Shutdown shutdown = nullptr;
    Create create = nullptr;
    Evaluate evaluate = nullptr;
    Release release = nullptr;
    NVSDK_NGX_Parameter* parameters = nullptr;
    NVSDK_NGX_Handle* feature = nullptr;
    std::shared_ptr<SharedNgxCore> shared_core;
    uint64_t session_id = 0;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE fence_event = nullptr;
    uint64_t fence_value = 0;
    ComPtr<ID3D12Resource> color;
    ComPtr<ID3D12Resource> depth;
    ComPtr<ID3D12Resource> motion;
    ComPtr<ID3D12Resource> output;
    ComPtr<ID3D12Resource> color_upload;
#ifdef DLSS_GPU_HOST_FP16_PROBE
    ComPtr<ID3D12Resource> color_probe_readback;
#endif
    ComPtr<ID3D12Resource> output_readback;
    ComPtr<ID3D12Resource> motion_upload;
    ComPtr<ID3D12Resource> motion_stage;
    ComPtr<ID3D12Resource> motion_readback;
    ComPtr<ID3D12RootSignature> motion_root;
    ComPtr<ID3D12PipelineState> motion_pipeline;
    ComPtr<ID3D12DescriptorHeap> motion_descriptors;
    bool motion_conversion_ready = false;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT color_layout{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT depth_layout{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT motion_layout{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT output_layout{};
    NvofD3D12FlowBridge* flow_bridge = nullptr;
    void* flow_context = nullptr;
    CudaD3D12ColorBridge* color_bridge = nullptr;
    void* color_context = nullptr;
    uint64_t host_color_stage_count = 0, host_color_stage_bytes = 0;
    uint64_t device_color_copy_count = 0, device_color_copy_bytes = 0;
    uint64_t output_readback_count = 0, output_readback_bytes = 0;
    /* What the live colour transport actually is, read off the bridge itself,
     * and what the last device-colour copy actually moved. */
    uint32_t bridge_row_pitch = 0, bridge_bytes_per_row = 0;
    uint64_t last_copy_src_pitch = 0, last_copy_row_bytes = 0;
    uint64_t last_copy_fence_value = 0;
    bool processed = false;
    bool reset_next_evaluate = false;
    bool failed = false;
#ifdef DLSS_GPU_HOST_FP16_PROBE
    float probe_pre_exposure = 1.0f;
    float probe_exposure_scale = 1.0f;
#endif
};

namespace {

std::atomic<uint64_t> dlaa_creates{0}, dlaa_evaluates{0}, dlaa_releases{0};

bool configure_evaluate(DlssGpuHost* h, ID3D12Resource* motion,
                        bool reset, DWORD* exception) {
    __try {
        auto* p = h->parameters;
        if (h->dlaa) {
            p->Set(NVSDK_NGX_Parameter_Color, h->color.Get());
            p->Set(NVSDK_NGX_Parameter_Output, h->output.Get());
            p->Set(NVSDK_NGX_Parameter_MotionVectors, motion);
            p->Set(NVSDK_NGX_Parameter_Depth, h->depth.Get());
            p->Set(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
            p->Set(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
            p->Set(NVSDK_NGX_Parameter_MV_Scale_X, h->options.motion_scale_x);
            p->Set(NVSDK_NGX_Parameter_MV_Scale_Y, h->options.motion_scale_y);
            p->Set(NVSDK_NGX_Parameter_Reset, reset ? 1 : 0);
            p->Set(NVSDK_NGX_Parameter_Sharpness, 0.0f);
            p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width, h->width);
            p->Set(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height, h->height);
            p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
            p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
            return true;
        }
        p->Set("DLSSNR.Color", h->color.Get());
        p->Set("DLSSNR.Output", h->output.Get());
        p->Set("DLSSNR.MVec", motion);
        p->Set("DLSSNR.Depth", h->depth.Get());
        set_rect(p, "Color", h->width, h->height);
        set_rect(p, "Output", h->width, h->height);
        set_rect(p, "MVec", h->width, h->height);
        set_rect(p, "Depth", h->width, h->height);
        p->Set("DLSSNR.MVecScaleX", h->options.motion_scale_x);
        p->Set("DLSSNR.MVecScaleY", h->options.motion_scale_y);
        p->Set("DLSSNR.DepthInverted", 1);
        p->Set("DLSS.Indicator.Invert.X.Axis", 0);
        p->Set("DLSS.Indicator.Invert.Y.Axis", 0);
        p->Set("DLSSNR.Enabled", 1);
        p->Set("DLSSNR.Reset", reset ? 1 : 0);
        p->Set("DLSSNR.Style", h->options.style);
        p->Set("DLSSNR.Intensity", h->options.intensity);
        p->Set("DLSSNR.LocalToneStrength", h->options.local_tone);
        p->Set("DLSSNR.LocalStructureStrength", h->options.local_structure);
        p->Set("DLSSNR.SkinStructureStrength", h->options.skin_structure);
        p->Set("DLSSNR.UseAutoMask", h->options.auto_mask);
        p->Set("DLSSNR.UICorrection", h->options.ui_correction);
#ifdef DLSS_GPU_HOST_FP16_PROBE
        // Generic DLSS-SR keys, tested here only to check whether this local
        // Feature 18 build responds. They are not a documented NR contract.
        p->Set(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, h->probe_pre_exposure);
        p->Set(NVSDK_NGX_Parameter_DLSS_Exposure_Scale,
               h->probe_exposure_scale);
#endif
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception = 1;
        return false;
    }
}

bool wait_for_gpu(DlssGpuHost* h, uint64_t value) {
    if (h->fence->GetCompletedValue() >= value) return true;
    const HRESULT hr = h->fence->SetEventOnCompletion(value, h->fence_event);
    if (FAILED(hr)) { error("SetEventOnCompletion", hr); return false; }
    if (WaitForSingleObject(h->fence_event, 15000) != WAIT_OBJECT_0) {
        error("GPU fence timeout", ERROR_TIMEOUT);
        return false;
    }
    return true;
}

HRESULT make_texture(DlssGpuHost* h, DXGI_FORMAT format,
                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                     ComPtr<ID3D12Resource>* output) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = h->width;
    desc.Height = h->height;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return h->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, state, nullptr, IID_PPV_ARGS(output->ReleaseAndGetAddressOf()));
}

HRESULT make_buffer(DlssGpuHost* h, UINT64 size, D3D12_HEAP_TYPE type,
                    D3D12_RESOURCE_STATES state,
                    ComPtr<ID3D12Resource>* output) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = size;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return h->device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
        &desc, state, nullptr, IID_PPV_ARGS(output->ReleaseAndGetAddressOf()));
}

HRESULT prepare_textures(DlssGpuHost* h) {
    HRESULT hr = make_texture(h, h->color_format,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &h->color);
    if (SUCCEEDED(hr)) hr = make_texture(h, DXGI_FORMAT_R32_FLOAT,
        D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST, &h->depth);
    // Extent, format and flags of the MVec resource the NVOF conversion hands
    // DLSSNR, so an unguided frame binds what a guided one does.
    if (SUCCEEDED(hr)) hr = make_texture(h, DXGI_FORMAT_R16G16_FLOAT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &h->motion);
    if (SUCCEEDED(hr)) hr = make_texture(h, h->color_format,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &h->output);
    if (FAILED(hr)) return hr;
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    auto desc = h->color->GetDesc();
    h->device->GetCopyableFootprints(&desc, 0, 1, 0,
        &h->color_layout, &rows, &row_bytes, &total);
    // GPU transfer never uses CPU colour/flow staging or host readback. Keep
    // their footprints, but allocate those buffers only at the matching API.
#ifdef DLSS_GPU_HOST_FP16_PROBE
    hr = make_buffer(h, total, D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_STATE_COPY_DEST, &h->color_probe_readback);
    if (FAILED(hr)) return hr;
#endif
    desc = h->motion->GetDesc();
    h->device->GetCopyableFootprints(&desc, 0, 1, 0,
        &h->motion_layout, &rows, &row_bytes, &total);
    desc = h->output->GetDesc();
    h->device->GetCopyableFootprints(&desc, 0, 1, 0,
        &h->output_layout, &rows, &row_bytes, &total);
    return S_OK;
}

HRESULT ensure_buffer(DlssGpuHost* h, ComPtr<ID3D12Resource>& buffer,
    uint64_t bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_STATES state) {
    return buffer ? S_OK : make_buffer(h, bytes, heap, state, &buffer);
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after);
void copy_texture(ID3D12GraphicsCommandList* list, ID3D12Resource* source,
                  ID3D12Resource* destination,
                  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout,
                  bool to_texture);

// The staging buffer is handed back rather than dropped: a command list that
// has not executed yet holds no reference of its own, and freeing the memory it
// was told to read removes the device.
HRESULT zero_plane(DlssGpuHost* h, ID3D12Resource* plane,
                   D3D12_RESOURCE_STATES initial,
                   D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout,
                   ComPtr<ID3D12Resource>& staging) {
    auto desc = plane->GetDesc();
    UINT rows = 0;
    UINT64 row_bytes = 0, total = 0;
    h->device->GetCopyableFootprints(&desc, 0, 1, 0,
        &layout, &rows, &row_bytes, &total);
    HRESULT hr = make_buffer(h, total, D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_STATE_GENERIC_READ, &staging);
    if (FAILED(hr)) return hr;
    void* mapped = nullptr;
    hr = staging->Map(0, nullptr, &mapped);
    if (FAILED(hr)) return hr;
    std::memset(mapped, 0, static_cast<size_t>(total));
    staging->Unmap(0, nullptr);
    if (initial != D3D12_RESOURCE_STATE_COPY_DEST)
        transition(h->list.Get(), plane, initial,
                   D3D12_RESOURCE_STATE_COPY_DEST);
    copy_texture(h->list.Get(), staging.Get(), plane, layout, true);
    transition(h->list.Get(), plane, D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    return S_OK;
}

// Both immutable planes are written once, here, and never again: nothing on the
// frame path has to be told which of them is which.
HRESULT initialize_zero_planes(DlssGpuHost* h) {
    HRESULT hr = h->allocator->Reset();
    if (SUCCEEDED(hr)) hr = h->list->Reset(h->allocator.Get(), nullptr);
    if (FAILED(hr)) return hr;
    ComPtr<ID3D12Resource> depth_staging, motion_staging;
    hr = zero_plane(h, h->depth.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                    h->depth_layout, depth_staging);
    if (SUCCEEDED(hr))
        hr = zero_plane(h, h->motion.Get(),
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        h->motion_layout, motion_staging);
    if (FAILED(hr)) return hr;
    hr = h->list->Close();
    if (FAILED(hr)) return hr;
    ID3D12CommandList* commands[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr)) return hr;
    return wait_for_gpu(h, h->fence_value) ? S_OK : E_FAIL;
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

void copy_texture(ID3D12GraphicsCommandList* list, ID3D12Resource* source,
                  ID3D12Resource* destination,
                  const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& layout,
                  bool to_texture) {
    D3D12_TEXTURE_COPY_LOCATION source_location{};
    D3D12_TEXTURE_COPY_LOCATION destination_location{};
    source_location.pResource = source;
    destination_location.pResource = destination;
    source_location.Type = to_texture ? D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT :
        D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    destination_location.Type = to_texture ? D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX :
        D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    if (to_texture) source_location.PlacedFootprint = layout;
    else destination_location.PlacedFootprint = layout;
    list->CopyTextureRegion(&destination_location, 0, 0, 0,
                            &source_location, nullptr);
}

// The host-motion float32x2 plane is narrowed to the bound R16G16_FLOAT motion
// texture by this compute pass, byte for byte the same instruction the guided
// NVOF conversion runs, so a host frame and a flow frame hand DLSS identical
// motion for identical numbers. Doing the cast on the card keeps numpy's
// double-rounding (which drifts by 1 ULP and the temporal filter amplifies it)
// out of the loop.
bool prepare_motion_conversion(DlssGpuHost* h) {
    if (h->motion_conversion_ready) return true;
    static constexpr char shader[] = R"(
ByteAddressBuffer Flow : register(t0);
RWTexture2D<float2> Motion : register(u0);
cbuffer Extent : register(b0) { uint Width; uint Height; };
[numthreads(8, 8, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= Width || id.y >= Height) return;
    uint offset = (id.y * Width + id.x) * 8;
    Motion[id.xy] = float2(asfloat(Flow.Load(offset)),
                           asfloat(Flow.Load(offset + 4)));
}
)";
    ID3DBlob* bytecode = nullptr;
    ID3DBlob* compile_error = nullptr;
    HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, "host_motion", nullptr,
        nullptr, "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
        &bytecode, &compile_error);
    if (FAILED(hr)) {
        error("compile motion conversion", static_cast<unsigned long>(hr));
        if (compile_error) {
            std::snprintf(last_error, sizeof(last_error), "D3DCompile: %s",
                static_cast<const char*>(compile_error->GetBufferPointer()));
            compile_error->Release();
        }
        return false;
    }
    if (compile_error) compile_error->Release();
    D3D12_DESCRIPTOR_RANGE ranges[2]{};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[0].NumDescriptors = 1;
    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[1].NumDescriptors = 1;
    D3D12_ROOT_PARAMETER params[3]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[0].DescriptorTable.NumDescriptorRanges = 1;
    params[0].DescriptorTable.pDescriptorRanges = &ranges[0];
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    params[1].DescriptorTable.NumDescriptorRanges = 1;
    params[1].DescriptorTable.pDescriptorRanges = &ranges[1];
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[2].Constants.Num32BitValues = 2;
    D3D12_ROOT_SIGNATURE_DESC root_desc{};
    root_desc.NumParameters = 3;
    root_desc.pParameters = params;
    ID3DBlob* root_blob = nullptr;
    hr = D3D12SerializeRootSignature(&root_desc, D3D_ROOT_SIGNATURE_VERSION_1,
                                     &root_blob, nullptr);
    if (SUCCEEDED(hr)) hr = h->device->CreateRootSignature(0,
        root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
        IID_PPV_ARGS(&h->motion_root));
    if (root_blob) root_blob->Release();
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc{};
        pipeline_desc.pRootSignature = h->motion_root.Get();
        pipeline_desc.CS = {bytecode->GetBufferPointer(), bytecode->GetBufferSize()};
        hr = h->device->CreateComputePipelineState(&pipeline_desc,
            IID_PPV_ARGS(&h->motion_pipeline));
    }
    bytecode->Release();
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = 2;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = h->device->CreateDescriptorHeap(&heap_desc,
            IID_PPV_ARGS(&h->motion_descriptors));
    }
    if (FAILED(hr)) {
        error("create motion conversion", static_cast<unsigned long>(hr));
        return false;
    }
    auto cpu = h->motion_descriptors->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = h->width * h->height * 2;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    h->device->CreateShaderResourceView(h->motion_stage.Get(), &srv, cpu);
    cpu.ptr += h->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    h->device->CreateUnorderedAccessView(h->motion.Get(), nullptr, &uav, cpu);
    h->motion_conversion_ready = true;
    return true;
}

} // namespace

extern "C" DLSS_GPU_HOST_API uint32_t dlss_gpu_host_abi_version(void) {
    return DLSS_GPU_HOST_ABI_VERSION;
}

extern "C" DLSS_GPU_HOST_API const char* dlss_gpu_host_last_error(void) {
    return last_error;
}

extern "C" DLSS_GPU_HOST_API uint64_t dlss_gpu_host_staging_bytes(DlssGpuHost* h) {
    if (!h) return 0;
    uint64_t bytes = 0;
    for (auto* buffer : {h->color_upload.Get(), h->motion_upload.Get(),
                        h->motion_stage.Get(), h->motion_readback.Get(),
                        h->output_readback.Get()})
        if (buffer) bytes += buffer->GetDesc().Width;
    return bytes;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_memory_query(void* raw_device, DlssGpuMemoryInfo* out) {
    if (!raw_device || !out || out->struct_size != sizeof(*out)) return int32_t(E_INVALIDARG);
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) return int32_t(hr);
    auto luid = static_cast<ID3D12Device*>(raw_device)->GetAdapterLuid();
    ComPtr<IDXGIAdapter3> adapter;
    hr = factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter));
    if (FAILED(hr)) return int32_t(hr);
    DXGI_QUERY_VIDEO_MEMORY_INFO local{}, nonlocal{};
    hr = adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &local);
    if (SUCCEEDED(hr)) hr = adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal);
    if (FAILED(hr)) return int32_t(hr);
    out->adapter_luid = (uint64_t(uint32_t(luid.HighPart)) << 32) | luid.LowPart;
    out->local_usage = local.CurrentUsage; out->local_budget = local.Budget;
    out->nonlocal_usage = nonlocal.CurrentUsage; out->nonlocal_budget = nonlocal.Budget;
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_resource_info(DlssGpuHost* h, DlssGpuResourceInfo* out) {
    if (!h || !out || out->struct_size != sizeof(*out)) return int32_t(E_INVALIDARG);
    auto allocation = [h](ID3D12Resource* resource) -> uint64_t {
        if (!resource) return 0;
        const auto desc = resource->GetDesc();
        return h->device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
    };
    out->color_bytes = allocation(h->color.Get());
    out->depth_bytes = allocation(h->depth.Get());
    out->motion_bytes = allocation(h->motion.Get());
    out->output_bytes = allocation(h->output.Get());
    out->staging_bytes = dlss_gpu_host_staging_bytes(h);
    return 0;
}

extern "C" DLSS_GPU_HOST_API void dlss_gpu_host_close(DlssGpuHost* h) {
    if (!h) return;
    auto shared = h->shared_core;
    if (h->color_bridge) cuda_d3d12_color_destroy(h->color_bridge);
    if (h->flow_bridge) nvof_d3d12_destroy(h->flow_bridge);
    if (shared) {
        std::lock_guard<std::mutex> lock(shared->ngx_mutex);
        if (h->feature && h->release) {
            trace_cleanup(h->dlaa ? "DLAA ReleaseFeature" : "NR ReleaseFeature", h->release(h->feature));
            if (h->dlaa) ++dlaa_releases;
            shared->features.erase(h->feature);
        }
        if (h->parameters) trace_cleanup("DestroyParameters", NVSDK_NGX_D3D12_DestroyParameters(h->parameters));
        if (h->auxiliary_initialized && h->auxiliary_shutdown)
            trace_cleanup("auxiliary DLAA Shutdown1", h->auxiliary_shutdown(h->device.Get()));
        bool auxiliary_restored = true;
        if (h->auxiliary_import_slot) {
            void* replaced = nullptr;
            auxiliary_restored = patch_slot(h->auxiliary_import_slot, h->auxiliary_old_import, &replaced);
        }
        if (h->auxiliary_snippet && auxiliary_restored) FreeLibrary(h->auxiliary_snippet);
        if (h->auxiliary_reserved) shared->auxiliary_dlaa_active.store(false);
    } else {
        // A failed first open has not transferred NGX ownership to a shared
        // core yet. Preserve the original partial-open teardown ordering.
        if (h->feature && h->release) {
            trace_cleanup(h->dlaa ? "DLAA ReleaseFeature" : "NR ReleaseFeature", h->release(h->feature));
            if (h->dlaa) ++dlaa_releases;
        }
        if (h->snippet_initialized && h->shutdown) trace_cleanup("partial snippet Shutdown1", h->shutdown(h->device.Get()));
        bool restored = true;
        if (h->hook_installed) {
            void* replaced = nullptr;
            restored = patch_slot(h->import_slot, h->old_import, &replaced);
            original_module_name = nullptr;
            host_module = nullptr;
        }
        if (h->parameters) trace_cleanup("partial DestroyParameters", NVSDK_NGX_D3D12_DestroyParameters(h->parameters));
        if (h->core_initialized) trace_cleanup("partial core Shutdown1", NVSDK_NGX_D3D12_Shutdown1(h->device.Get()));
        if (h->snippet && restored) FreeLibrary(h->snippet);
    }
    if (h->fence_event) CloseHandle(h->fence_event);
    delete h;
    if (!shared) active.store(false);
}

static int32_t open_impl(uint32_t width, uint32_t height,
    const wchar_t* signed_snippet_path, const DlssGpuHostOptions* options,
    const DlssGpuHostOpenInfo* info, DlssGpuHost** out_host, bool dlaa = false) {
    last_error[0] = 0;
    /* The colour is settled before anything global is taken: a format this
     * build cannot transport, or an open info that describes neither, refuses
     * the call instead of opening the RGBA8 session nobody asked for. */
    uint32_t color_pixels = kDefaultColorPixels;
    if (info) {
        if (info->struct_size != sizeof(*info) || info->flags) {
            error("invalid open info", 0);
            return 22;
        }
        color_pixels = info->color_format;
    }
    DXGI_FORMAT color_dxgi = kDefaultColorFormat;
    uint32_t color_bytes = kDefaultColorBytes;
    if (!color_geometry(color_pixels, &color_dxgi, &color_bytes)) {
        error("unsupported colour format", 0);
        return 22;
    }
    if (color_bytes != 4 && !row_bridge().wide_ready) {
        std::snprintf(last_error, sizeof(last_error),
            "%s", "HDR colour needs an nvof_d3d12_interop.dll that transports "
            "an eight-byte row (cuda_d3d12_color_create2)");
        return 23;
    }
    if (!width || !height || !signed_snippet_path || !options || !out_host ||
        options->struct_size != sizeof(DlssGpuHostOptions) ||
        uint64_t(width) * height > UINT32_MAX / color_bytes) {
        error("invalid open arguments", 0);
        return 1;
    }
    *out_host = nullptr;
    if (active.exchange(true)) { error("host already active", 0); return 2; }
    auto* h = new (std::nothrow) DlssGpuHost;
    if (!h) { active.store(false); error("allocate host", 0); return 3; }
    h->width = width;
    h->height = height;
    h->dlaa = dlaa;
    h->color_pixels = color_pixels;
    h->bytes_per_pixel = color_bytes;
    h->color_format = color_dxgi;
    h->options = *options;
    std::unique_ptr<DlssGpuHost, decltype(&dlss_gpu_host_close)> pending(h, dlss_gpu_host_close);
    const auto abort_open = [&](int code) {
        pending.reset();
        return code;
    };
    const std::filesystem::path path = std::filesystem::absolute(signed_snippet_path);
    const std::wstring directory = path.parent_path().wstring();
    h->snippet = LoadLibraryExW(path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h->snippet) { error("LoadLibrary signed snippet", GetLastError()); return abort_open(4); }
    h->init = reinterpret_cast<Init>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_Init_Ext"));
    h->shutdown = reinterpret_cast<Shutdown>(GetProcAddress(h->snippet,
        "NVSDK_NGX_D3D12_Shutdown1"));
    h->create = reinterpret_cast<Create>(GetProcAddress(h->snippet,
        "NVSDK_NGX_D3D12_CreateFeature"));
    h->evaluate = reinterpret_cast<Evaluate>(GetProcAddress(h->snippet,
        "NVSDK_NGX_D3D12_EvaluateFeature"));
    h->release = reinterpret_cast<Release>(GetProcAddress(h->snippet,
        "NVSDK_NGX_D3D12_ReleaseFeature"));
    if (!h->init || !h->shutdown || !h->create || !h->evaluate || !h->release) {
        error("signed snippet exports", 0); return abort_open(5);
    }
    enable_gpu_dred_diagnostics();
    HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
                                    IID_PPV_ARGS(h->device.GetAddressOf()));
    if (FAILED(hr)) { error("D3D12CreateDevice", hr); return abort_open(6); }
    DWORD exception = 0;
    NVSDK_NGX_Result ngx = safe_core_init(directory.c_str(), h->device.Get(),
                                           &exception);
    if (!NVSDK_NGX_SUCCEED(ngx) || exception) {
        error("NGX Core init", static_cast<unsigned long>(ngx)); return abort_open(7);
    }
    h->core_initialized = true;
    ngx = NVSDK_NGX_D3D12_AllocateParameters(&h->parameters);
    if (!NVSDK_NGX_SUCCEED(ngx) || !h->parameters) {
        error("NGX AllocateParameters", static_cast<unsigned long>(ngx));
        return abort_open(8);
    }
    h->import_slot = find_module_name_slot(h->snippet);
    if (!h->import_slot) { error("GetModuleFileNameW import", 0); return abort_open(9); }
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&compatible_module_name), &host_module)) {
        error("resolve host module", GetLastError()); return abort_open(10);
    }
    if (!patch_slot(h->import_slot, reinterpret_cast<void*>(&compatible_module_name),
                    &h->old_import) || !h->old_import) {
        error("install snippet caller adapter", GetLastError());
        return abort_open(11);
    }
    original_module_name = reinterpret_cast<ModuleName>(h->old_import);
    h->hook_installed = true;
    exception = 0;
    ngx = safe_snippet_init(h->init, directory.c_str(), h->device.Get(),
                            &exception);
    if (!NVSDK_NGX_SUCCEED(ngx) || exception) {
        error("signed snippet init", static_cast<unsigned long>(ngx));
        return abort_open(12);
    }
    h->snippet_initialized = true;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = h->device->CreateCommandQueue(&queue_desc,
                                        IID_PPV_ARGS(h->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(h->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, h->allocator.Get(), nullptr,
        IID_PPV_ARGS(h->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(h->fence.GetAddressOf()));
    if (FAILED(hr)) { error("D3D12 command resources", hr); return abort_open(13); }
    h->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!h->fence_event) { error("CreateEvent", GetLastError()); return abort_open(14); }
    exception = 0;
    if (!configure_create(h->parameters, width, height, options->preset,
                          &exception, dlaa, color_bytes == 8)) {
        error(dlaa ? "configure native DLAA" : "configure Feature 18", exception); return abort_open(15);
    }
    ngx = safe_create(h->create, h->list.Get(), h->parameters,
                      &h->feature, &exception,
                      dlaa ? NVSDK_NGX_Feature_SuperSampling : static_cast<NVSDK_NGX_Feature>(18));
    if (!NVSDK_NGX_SUCCEED(ngx) || !h->feature || exception) {
        error(dlaa ? "create native DLAA" : "create Feature 18", static_cast<unsigned long>(ngx));
        return abort_open(16);
    }
    if (dlaa) ++dlaa_creates;
    hr = h->list->Close();
    if (FAILED(hr)) { error("close feature command list", hr); return abort_open(17); }
    ID3D12CommandList* commands[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr) || !wait_for_gpu(h, h->fence_value)) {
        if (FAILED(hr)) error("signal feature fence", hr);
        return abort_open(18);
    }
    hr = prepare_textures(h);
    if (FAILED(hr)) { error("create frame textures", hr); return abort_open(19); }
    hr = initialize_zero_planes(h);
    if (FAILED(hr)) { error("initialize zero planes", hr); return abort_open(20); }
    try {
        auto shared = std::make_shared<SharedNgxCore>();
        shared->features.insert(h->feature);
        shared->snippet = h->snippet;
        shared->import_slot = h->import_slot;
        shared->old_import = h->old_import;
        shared->hook_installed = h->hook_installed;
        shared->core_initialized = h->core_initialized;
        shared->snippet_initialized = h->snippet_initialized;
        shared->shutdown = h->shutdown;
        shared->device = h->device;
        h->shared_core = std::move(shared);
        h->shared_core->owns_active = true;
        h->session_id = h->shared_core->next_session_id++;
        // The shared core now owns NGX init, the snippet and the import hook.
        h->hook_installed = h->core_initialized = h->snippet_initialized = false;
    } catch (const std::bad_alloc&) {
        error("allocate shared NGX core", 0);
        return abort_open(21);
    }
    *out_host = pending.release();
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open(
    uint32_t width, uint32_t height, const wchar_t* signed_snippet_path,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host) {
    return open_impl(width, height, signed_snippet_path, options, nullptr,
                     out_host);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_ex(
    uint32_t width, uint32_t height, const wchar_t* signed_snippet_path,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host) {
    return open_impl(width, height, signed_snippet_path, options, info,
                     out_host);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa(
    uint32_t width, uint32_t height, const wchar_t* signed_dlss_path,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host) {
    return open_impl(width, height, signed_dlss_path, options, nullptr, out_host, true);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_ex(
    uint32_t width, uint32_t height, const wchar_t* path,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host) {
    return open_impl(width, height, path, options, info, out_host, true);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_dlaa_counters(uint64_t* out) {
    if (!out) return 1;
    out[0] = dlaa_creates.load(); out[1] = dlaa_evaluates.load();
    out[2] = dlaa_releases.load(); return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_query_capabilities(
    DlssGpuHostCapabilities* out) {
    last_error[0] = 0;
    if (!out || out->struct_size != sizeof(*out)) {
        error("invalid capabilities arguments", 0);
        return 1;
    }
    const RowBridge& bridge = row_bridge();
    *out = DlssGpuHostCapabilities{sizeof(*out),
        DLSS_GPU_HOST_COLOR_FORMAT_BIT(DLSS_GPU_HOST_COLOR_RGBA8) |
        DLSS_GPU_HOST_COLOR_FORMAT_BIT(DLSS_GPU_HOST_COLOR_RGBA16F),
        bridge.capabilities, bridge.wide_ready ? 1u : 0u};
    return 0;
}

extern "C" DLSS_GPU_HOST_API void* dlss_gpu_host_d3d12_device(DlssGpuHost* h) {
    return h && !h->failed ? h->device.Get() : nullptr;
}

extern "C" DLSS_GPU_HOST_API uint64_t dlss_gpu_host_session_id(DlssGpuHost* h) {
    return h ? h->session_id : 0;
}

static int32_t open_layer_impl(DlssGpuHost* anchor, uint32_t width,
    uint32_t height, const DlssGpuHostOptions* options,
    const DlssGpuHostOpenInfo* info, DlssGpuHost** out_host,
    const wchar_t* dlaa_path = nullptr) {
    last_error[0] = 0;
    uint32_t color_pixels = kDefaultColorPixels;
    if (info) {
        if (info->struct_size != sizeof(*info) || info->flags) {
            error("invalid layer open info", 0);
            return 22;
        }
        color_pixels = info->color_format;
    }
    DXGI_FORMAT color_dxgi = kDefaultColorFormat;
    uint32_t color_bytes = kDefaultColorBytes;
    if (!color_geometry(color_pixels, &color_dxgi, &color_bytes)) {
        error("unsupported colour format", 0);
        return 22;
    }
    if (color_bytes != 4 && !row_bridge().wide_ready) {
        std::snprintf(last_error, sizeof(last_error),
            "%s", "HDR colour needs an nvof_d3d12_interop.dll that transports "
            "an eight-byte row (cuda_d3d12_color_create2)");
        return 23;
    }
    if (!anchor || !anchor->shared_core || !width || !height || !options ||
        !out_host || options->struct_size != sizeof(DlssGpuHostOptions) ||
        uint64_t(width) * height > UINT32_MAX / color_bytes) {
        error("invalid layer open arguments", 0);
        return 1;
    }
    if (anchor->dlaa) {
        error("DLAA cannot be used as a Feature 18 anchor", 0);
        return 1;
    }
    *out_host = nullptr;
    auto* h = new (std::nothrow) DlssGpuHost;
    if (!h) { error("allocate layer", 0); return 2; }
    h->shared_core = anchor->shared_core;
    h->device = h->shared_core->device;
    h->snippet = anchor->snippet;
    h->create = anchor->create;
    h->evaluate = anchor->evaluate;
    h->release = anchor->release;
    h->width = width;
    h->height = height;
    h->color_pixels = color_pixels;
    h->bytes_per_pixel = color_bytes;
    h->color_format = color_dxgi;
    h->options = *options;
    h->dlaa = dlaa_path != nullptr;
    std::unique_ptr<DlssGpuHost, decltype(&dlss_gpu_host_close)> pending(h, dlss_gpu_host_close);
    const auto abort_open = [&](int code) { pending.reset(); return code; };
    if (dlaa_path) {
        if (h->shared_core->auxiliary_dlaa_active.exchange(true)) {
            error("only one auxiliary DLAA feature per shared NR core is supported", 0);
            return abort_open(24);
        }
        h->auxiliary_reserved = true;
        h->auxiliary_snippet = LoadLibraryExW(dlaa_path, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (!h->auxiliary_snippet) { error("load signed DLAA module", GetLastError()); return abort_open(24); }
        h->snippet = h->auxiliary_snippet;
        auto init = reinterpret_cast<Init>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_Init_Ext"));
        h->create = reinterpret_cast<Create>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_CreateFeature"));
        h->evaluate = reinterpret_cast<Evaluate>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_EvaluateFeature"));
        h->release = reinterpret_cast<Release>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_ReleaseFeature"));
        h->auxiliary_shutdown = reinterpret_cast<Shutdown>(GetProcAddress(h->snippet, "NVSDK_NGX_D3D12_Shutdown1"));
        if (!init || !h->create || !h->evaluate || !h->release || !h->auxiliary_shutdown) {
            error("DLAA module API", 0); return abort_open(25);
        }
        h->auxiliary_import_slot = find_module_name_slot(h->snippet);
        if (!h->auxiliary_import_slot || !patch_slot(h->auxiliary_import_slot,
            reinterpret_cast<void*>(&compatible_module_name), &h->auxiliary_old_import)) {
            h->auxiliary_import_slot = nullptr;
            error("DLAA caller adapter", 0); return abort_open(26);
        }
        DWORD exception = 0;
        const std::wstring path(dlaa_path);
        const std::wstring directory = path.substr(0, path.find_last_of(L"\\/"));
        std::lock_guard<std::mutex> lock(h->shared_core->ngx_mutex);
        const auto result = safe_snippet_init(init, directory.c_str(), h->device.Get(), &exception);
        if (!NVSDK_NGX_SUCCEED(result) || exception) {
            error("DLAA layer init", static_cast<unsigned long>(result));
        } else h->auxiliary_initialized = true;
    }
    if (dlaa_path && !h->auxiliary_initialized) return abort_open(27);
    {
        std::lock_guard<std::mutex> lock(h->shared_core->ngx_mutex);
        const NVSDK_NGX_Result ngx = NVSDK_NGX_D3D12_AllocateParameters(&h->parameters);
        if (!NVSDK_NGX_SUCCEED(ngx) || !h->parameters) {
            error("NGX layer AllocateParameters", static_cast<unsigned long>(ngx));
        }
    }
    if (!h->parameters) return abort_open(3);
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = h->device->CreateCommandQueue(&queue_desc,
                                        IID_PPV_ARGS(h->queue.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(h->allocator.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, h->allocator.Get(), nullptr,
        IID_PPV_ARGS(h->list.GetAddressOf()));
    if (SUCCEEDED(hr)) hr = h->device->CreateFence(
        0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(h->fence.GetAddressOf()));
    if (FAILED(hr)) { error("layer D3D12 command resources", hr); return abort_open(4); }
    h->fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!h->fence_event) { error("layer CreateEvent", GetLastError()); return abort_open(5); }
    DWORD exception = 0;
    NVSDK_NGX_Result ngx = NVSDK_NGX_Result_Fail;
    {
        std::lock_guard<std::mutex> lock(h->shared_core->ngx_mutex);
        if (!configure_create(h->parameters, width, height, options->preset,
                              &exception, h->dlaa, color_bytes == 8)) {
            error("configure layer Feature 18", exception);
        } else {
            ngx = safe_create(h->create, h->list.Get(), h->parameters,
                              &h->feature, &exception,
                              h->dlaa ? NVSDK_NGX_Feature_SuperSampling : static_cast<NVSDK_NGX_Feature>(18));
        }
    }
    if (!NVSDK_NGX_SUCCEED(ngx) || !h->feature || exception) {
        error(h->dlaa ? "create DLAA layer" : "create layer Feature 18", static_cast<unsigned long>(ngx));
        return abort_open(6);
    }
    if (h->dlaa) ++dlaa_creates;
    hr = h->list->Close();
    if (FAILED(hr)) { error("close layer feature list", hr); return abort_open(7); }
    ID3D12CommandList* commands[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr) || !wait_for_gpu(h, h->fence_value)) {
        if (FAILED(hr)) error("signal layer feature fence", hr);
        return abort_open(8);
    }
    hr = prepare_textures(h);
    if (FAILED(hr)) { error("create layer textures", hr); return abort_open(9); }
    hr = initialize_zero_planes(h);
    if (FAILED(hr)) { error("initialize layer zero planes", hr); return abort_open(10); }
    bool distinct_feature = false;
    {
        std::lock_guard<std::mutex> lock(h->shared_core->ngx_mutex);
        distinct_feature = h->shared_core->features.insert(h->feature).second;
        if (distinct_feature) h->session_id = h->shared_core->next_session_id++;
    }
    if (!distinct_feature) {
        error("NGX reused an active Feature 18 handle", 0);
        // A duplicate is not ours to release: it remains owned by the
        // pre-existing session. Teardown only the partially opened wrapper.
        h->feature = nullptr;
        return abort_open(11);
    }
    *out_host = pending.release();
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_layer(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host) {
    return open_layer_impl(anchor, width, height, options, nullptr, out_host);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_layer(
    DlssGpuHost* anchor, uint32_t width, uint32_t height, const wchar_t* dll_path,
    const DlssGpuHostOptions* options, DlssGpuHost** out_host) {
    return open_layer_impl(anchor, width, height, options, nullptr, out_host, dll_path);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_dlaa_layer_ex(
    DlssGpuHost* anchor, uint32_t width, uint32_t height, const wchar_t* path,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host) {
    return open_layer_impl(anchor, width, height, options, info, out_host, path);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_open_layer_ex(
    DlssGpuHost* anchor, uint32_t width, uint32_t height,
    const DlssGpuHostOptions* options, const DlssGpuHostOpenInfo* info,
    DlssGpuHost** out_host) {
    return open_layer_impl(anchor, width, height, options, info, out_host);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_update_options(
    DlssGpuHost* h, const DlssGpuHostOptions* options) {
    last_error[0] = 0;
    if (!h || h->failed || !options || options->struct_size != sizeof(*options) ||
        options->preset != h->options.preset) {
        std::snprintf(last_error, sizeof(last_error), "Runtime options require a healthy session and unchanged preset");
        return -1;
    }
    if (!std::isfinite(options->intensity) || !std::isfinite(options->local_tone) ||
        !std::isfinite(options->local_structure) || !std::isfinite(options->skin_structure) ||
        !std::isfinite(options->motion_scale_x) || !std::isfinite(options->motion_scale_y)) {
        std::snprintf(last_error, sizeof(last_error), "Runtime options require finite values");
        return -1;
    }
    h->options = *options;
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_recreate_dlaa(
    DlssGpuHost* h, const DlssGpuHostOptions* options) {
    last_error[0] = 0;
    if (!h || !h->dlaa || h->failed || !h->shared_core || !options ||
        options->struct_size != sizeof(*options)) {
        error("invalid DLAA recreation", 0); return 1;
    }
    if (options->preset == h->options.preset) return 0;
    if (h->fence_value && !wait_for_gpu(h, h->fence_value)) return 2;
    std::lock_guard<std::mutex> lock(h->shared_core->ngx_mutex);
    const auto fail = [&](const char* stage, unsigned long code) {
        h->failed = true; error(stage, code); return 3;
    };
    auto* old_feature = h->feature;
    h->feature = nullptr;
    auto ngx = h->release(old_feature);
    ++dlaa_releases;
    trace_cleanup("DLAA recreate ReleaseFeature", ngx);
    if (!NVSDK_NGX_SUCCEED(ngx)) return fail("release old DLAA model", static_cast<unsigned long>(ngx));
    h->shared_core->features.erase(old_feature);
    auto* old_parameters = h->parameters;
    h->parameters = nullptr;
    ngx = NVSDK_NGX_D3D12_DestroyParameters(old_parameters);
    trace_cleanup("DLAA recreate DestroyParameters", ngx);
    if (!NVSDK_NGX_SUCCEED(ngx)) return fail("destroy old DLAA parameters", static_cast<unsigned long>(ngx));
    ngx = NVSDK_NGX_D3D12_AllocateParameters(&h->parameters);
    if (!NVSDK_NGX_SUCCEED(ngx) || !h->parameters)
        return fail("allocate new DLAA parameters", static_cast<unsigned long>(ngx));
    DWORD exception = 0;
    if (!configure_create(h->parameters, h->width, h->height, options->preset,
                          &exception, true, h->bytes_per_pixel == 8))
        return fail("configure new DLAA model", exception);
    auto hr = h->allocator->Reset();
    if (SUCCEEDED(hr)) hr = h->list->Reset(h->allocator.Get(), nullptr);
    if (FAILED(hr)) return fail("reset DLAA recreation list", hr);
    ngx = safe_create(h->create, h->list.Get(), h->parameters, &h->feature,
                      &exception, NVSDK_NGX_Feature_SuperSampling);
    if (!NVSDK_NGX_SUCCEED(ngx) || exception || !h->feature) {
        h->list->Close(); return fail("create new DLAA model", static_cast<unsigned long>(ngx));
    }
    ++dlaa_creates;
    if (!h->shared_core->features.insert(h->feature).second) {
        // A pre-existing feature belongs to another shared session.
        h->feature = nullptr;
        return fail("NGX reused an active feature during DLAA recreation", 0);
    }
    hr = h->list->Close();
    if (FAILED(hr)) return fail("close DLAA recreation list", hr);
    ID3D12CommandList* lists[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, lists);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr)) return fail("signal DLAA recreation fence", hr);
    if (!wait_for_gpu(h, h->fence_value)) { h->failed = true; return 4; }
    h->options = *options;
    h->session_id = h->shared_core->next_session_id++;
    h->reset_next_evaluate = true;
    // processed describes the retained texture states, not the new feature's
    // history. Its first Evaluate must reset; do not change these states here.
    return 0;
}

static int32_t process_impl(DlssGpuHost* h,
    const uint8_t* rgba, size_t rgba_stride,
    void* color_context, uint64_t rgba_device_ptr, size_t rgba_device_pitch,
    void* flow_context, uint64_t flow_ptr, size_t flow_pitch, int32_t reset,
    uint8_t* output_rgba, size_t output_stride, bool keep_device_output,
    bool use_zero_motion = false, const uint8_t* host_motion = nullptr,
    size_t motion_stride = 0) {
    last_error[0] = 0;
    // A guided frame borrows NVOF flow; a host-motion frame carries its own
    // motion plane and touches no CUDA; a zero-motion frame carries neither.
    const bool guided_flow = !use_zero_motion && !host_motion;
    if (!h || h->failed ||
        (guided_flow && (!flow_context || !flow_ptr)) ||
        (host_motion && (!rgba || rgba_stride < size_t(h->width) * h->bytes_per_pixel ||
            motion_stride < size_t(h->width) * 8)) ||
        (rgba_device_ptr ? (!color_context ||
            rgba_device_pitch < size_t(h->width) * h->bytes_per_pixel) :
            (!rgba || rgba_stride < size_t(h->width) * h->bytes_per_pixel)) ||
        (!keep_device_output && (!output_rgba ||
            output_stride < size_t(h->width) * h->bytes_per_pixel))) {
        error("invalid process arguments", 0);
        return 1;
    }
    const auto abort_frame = [&](int code) { h->failed = true; return code; };
    HRESULT staging_hr = S_OK;
    if (!rgba_device_ptr)
        staging_hr = ensure_buffer(h, h->color_upload,
            uint64_t(h->color_layout.Footprint.RowPitch) * h->height,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (SUCCEEDED(staging_hr) && host_motion)
        staging_hr = ensure_buffer(h, h->motion_upload, uint64_t(h->width) * h->height * 8,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    if (SUCCEEDED(staging_hr) && host_motion)
        staging_hr = ensure_buffer(h, h->motion_stage, uint64_t(h->width) * h->height * 8,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON);
    if (SUCCEEDED(staging_hr) && !keep_device_output)
        staging_hr = ensure_buffer(h, h->output_readback,
            uint64_t(h->output_layout.Footprint.RowPitch) * h->height,
            D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    if (FAILED(staging_hr)) { error("allocate requested CPU staging", staging_hr); return abort_frame(6); }
    uint64_t flow_ready = 0;
    void* raw_buffer = nullptr;
    void* raw_fence = nullptr;
    if (guided_flow) {
        if (!h->flow_bridge) {
            const int created = nvof_d3d12_create(h->device.Get(), flow_context,
                h->width, h->height, &h->flow_bridge);
            if (created) {
                std::snprintf(last_error, sizeof(last_error), "flow bridge: %s",
                              nvof_d3d12_last_error());
                return abort_frame(2);
            }
            h->flow_context = flow_context;
        } else if (h->flow_context != flow_context) {
            error("NVOF CUDA context changed", 0);
            return abort_frame(3);
        }
        if (nvof_d3d12_copy_flow(h->flow_bridge, flow_ptr, flow_pitch,
                                 &flow_ready)) {
            std::snprintf(last_error, sizeof(last_error), "flow copy: %s",
                          nvof_d3d12_last_error());
            return abort_frame(4);
        }
        nvof_d3d12_get_resources(h->flow_bridge, &raw_buffer, &raw_fence);
    }
    // The resident motion texture is bound directly on a zero-motion or a
    // host-motion frame; a guided frame swaps in the flow conversion's texture.
    ID3D12Resource* motion_resource = h->motion.Get();
    if (guided_flow) {
        const HRESULT wait = h->queue->Wait(
            static_cast<ID3D12Fence*>(raw_fence), flow_ready);
        if (FAILED(wait)) { error("queue wait on CUDA flow", wait);
            return abort_frame(5); }
    }
    HRESULT hr = S_OK;
    void* mapped = nullptr;
    ID3D12Resource* color_source = h->color_upload.Get();
    if (rgba_device_ptr) {
        const uint32_t row_bytes = h->width * h->bytes_per_pixel;
        const RowBridge& bridge = row_bridge();
        if (!h->color_bridge) {
            const uint32_t pitch = h->color_layout.Footprint.RowPitch;
            const int created =
                h->color_pixels == DLSS_GPU_HOST_COLOR_RGBA8
                    ? cuda_d3d12_color_create(h->device.Get(), color_context,
                          h->width, h->height, pitch, &h->color_bridge)
                    : bridge.create2(h->device.Get(), color_context, h->width,
                          h->height, pitch, h->bytes_per_pixel, 0,
                          &h->color_bridge);
            if (created) {
                std::snprintf(last_error, sizeof(last_error), "color bridge: %s",
                              nvof_d3d12_last_error());
                return abort_frame(6);
            }
            h->color_context = color_context;
            /* The transport's own layout is read back rather than assumed from
             * what this caller asked for: a bridge whose row is shorter than
             * this session's colour would drop the tail half of every row and
             * still report a successful copy. */
            CudaD3D12ColorLayout layout{sizeof(CudaD3D12ColorLayout)};
            if (bridge.get_layout &&
                bridge.get_layout(h->color_bridge, &layout)) {
                error("color bridge layout", 0);
                return abort_frame(6);
            }
            h->bridge_row_pitch = bridge.get_layout ? layout.row_pitch : pitch;
            h->bridge_bytes_per_row = bridge.get_layout ? layout.bytes_per_row
                                                       : row_bytes;
            if (h->bridge_bytes_per_row != row_bytes ||
                h->bridge_row_pitch < row_bytes) {
                error("color bridge row does not match the session colour", 0);
                return abort_frame(6);
            }
        } else if (h->color_context != color_context) {
            error("color CUDA context changed", 0);
            return abort_frame(6);
        }
        uint64_t color_ready = 0;
        if (cuda_d3d12_color_copy(h->color_bridge, rgba_device_ptr,
                                  rgba_device_pitch, &color_ready)) {
            std::snprintf(last_error, sizeof(last_error), "color copy: %s",
                          nvof_d3d12_last_error());
            return abort_frame(6);
        }
        h->last_copy_src_pitch = rgba_device_pitch;
        h->last_copy_row_bytes = row_bytes;
        h->last_copy_fence_value = color_ready;
        void* raw_color = nullptr;
        void* color_fence = nullptr;
        cuda_d3d12_color_get_resources(h->color_bridge,
                                       &raw_color, &color_fence);
        hr = h->queue->Wait(static_cast<ID3D12Fence*>(color_fence), color_ready);
        if (FAILED(hr)) { error("queue wait on CUDA color", hr);
            return abort_frame(6); }
        color_source = static_cast<ID3D12Resource*>(raw_color);
        h->device_color_copy_count++;
        h->device_color_copy_bytes += uint64_t(h->width) * h->height * h->bytes_per_pixel;
    } else {
        hr = h->color_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) { error("map color upload", hr); return abort_frame(6); }
        for (uint32_t y = 0; y < h->height; ++y)
            std::memcpy(static_cast<uint8_t*>(mapped) +
                            size_t(y) * h->color_layout.Footprint.RowPitch,
                        rgba + size_t(y) * rgba_stride, size_t(h->width) * h->bytes_per_pixel);
        h->color_upload->Unmap(0, nullptr);
        h->host_color_stage_count++;
        h->host_color_stage_bytes += uint64_t(h->width) * h->height * h->bytes_per_pixel;
    }
    if (host_motion) {
        // The caller hands a host float32x2 plane; the compute pass narrows it to
        // the bound half2 motion texture. Staging repacks each row into the flat
        // upload buffer at width*8 so the shader's (y*width+x)*8 index is exact,
        // whatever stride the caller used: no reinterpret, no scale, no zero.
        void* motion_mapped = nullptr;
        hr = h->motion_upload->Map(0, nullptr, &motion_mapped);
        if (FAILED(hr)) { error("map motion upload", hr); return abort_frame(7); }
        const size_t plane_row = size_t(h->width) * 8;
        for (uint32_t y = 0; y < h->height; ++y)
            std::memcpy(static_cast<uint8_t*>(motion_mapped) +
                            size_t(y) * plane_row,
                        host_motion + size_t(y) * motion_stride, plane_row);
        h->motion_upload->Unmap(0, nullptr);
        if (!prepare_motion_conversion(h)) return abort_frame(7);
    }
    hr = h->allocator->Reset();
    if (SUCCEEDED(hr)) hr = h->list->Reset(h->allocator.Get(), nullptr);
    if (FAILED(hr)) { error("reset frame command list", hr); return abort_frame(8); }
    auto* list = h->list.Get();
    if (h->processed) {
        transition(list, h->color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        transition(list, h->output.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
    if (rgba_device_ptr) transition(list, color_source,
        D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_texture(list, color_source, h->color.Get(), h->color_layout, true);
    if (rgba_device_ptr) transition(list, color_source,
        D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON);
    transition(list, h->color.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
#ifdef DLSS_GPU_HOST_FP16_PROBE
    transition(list, h->color.Get(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_texture(list, h->color.Get(), h->color_probe_readback.Get(),
                 h->color_layout, false);
    transition(list, h->color.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
#endif
    if (host_motion) {
        // Copy the packed float32x2 plane into the compute's read buffer, then run
        // the same narrowing pass NVOF uses so the resident motion texture DLSS
        // binds holds the caller's numbers cast with a single rounded RNE. The
        // motion texture returns to the read state the other entries leave it in.
        const uint64_t plane_bytes = uint64_t(h->width) * h->height * 8;
        transition(list, h->motion_stage.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(h->motion_stage.Get(), 0,
                               h->motion_upload.Get(), 0, plane_bytes);
        transition(list, h->motion_stage.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, h->motion.Get(),
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ID3D12DescriptorHeap* heaps[] = {h->motion_descriptors.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(h->motion_root.Get());
        list->SetPipelineState(h->motion_pipeline.Get());
        auto gpu = h->motion_descriptors->GetGPUDescriptorHandleForHeapStart();
        list->SetComputeRootDescriptorTable(0, gpu);
        gpu.ptr += h->device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        list->SetComputeRootDescriptorTable(1, gpu);
        const uint32_t extent[] = {h->width, h->height};
        list->SetComputeRoot32BitConstants(2, 2, extent, 0);
        list->Dispatch((h->width + 7) / 8, (h->height + 7) / 8, 1);
        transition(list, h->motion.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(list, h->motion_stage.Get(),
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                   D3D12_RESOURCE_STATE_COMMON);
    }
    if (guided_flow) {
        void* raw_motion = nullptr;
        if (nvof_d3d12_record_motion(h->flow_bridge, list, &raw_motion)) {
            std::snprintf(last_error, sizeof(last_error), "motion texture: %s",
                          nvof_d3d12_last_error());
            list->Close();
            return abort_frame(9);
        }
        motion_resource = static_cast<ID3D12Resource*>(raw_motion);
    }
    DWORD exception = 0;
    NVSDK_NGX_Result ngx = NVSDK_NGX_Result_Fail;
    {
        // NGX explicitly does not promise thread safety. Distinct Feature 18
        // handles keep temporal state separate, while calls into NGX itself
        // are serialized for every layer sharing this core.
        std::unique_lock<std::mutex> lock;
        if (h->shared_core) lock = std::unique_lock<std::mutex>(
            h->shared_core->ngx_mutex);
        if (!configure_evaluate(h, motion_resource, reset != 0 || h->reset_next_evaluate, &exception)) {
            error("configure evaluate", exception);
        } else {
            ngx = safe_evaluate(h->evaluate, list, h->feature,
                                h->parameters, &exception);
        }
    }
    if (!NVSDK_NGX_SUCCEED(ngx) || exception) {
        if (!exception) error(h->dlaa ? "evaluate native DLAA" : "evaluate Feature 18", static_cast<unsigned long>(ngx));
        list->Close();
        return abort_frame(11);
    }
    transition(list, h->output.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    if (!keep_device_output)
        copy_texture(list, h->output.Get(), h->output_readback.Get(),
                     h->output_layout, false);
    hr = list->Close();
    if (FAILED(hr)) { error("close frame command list", hr); return abort_frame(12); }
    ID3D12CommandList* commands[] = {list};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr) || !wait_for_gpu(h, h->fence_value)) {
        if (FAILED(hr)) error("signal frame fence", hr);
        return abort_frame(13);
    }
    if (!keep_device_output) {
        mapped = nullptr;
        hr = h->output_readback->Map(0, nullptr, &mapped);
        if (FAILED(hr)) { error("map output readback", hr); return abort_frame(14); }
        for (uint32_t y = 0; y < h->height; ++y)
            std::memcpy(output_rgba + size_t(y) * output_stride,
                        static_cast<const uint8_t*>(mapped) +
                            size_t(y) * h->output_layout.Footprint.RowPitch,
                        size_t(h->width) * h->bytes_per_pixel);
        h->output_readback->Unmap(0, nullptr);
        h->output_readback_count++;
        h->output_readback_bytes += uint64_t(h->width) * h->height * h->bytes_per_pixel;
    }
    if (h->dlaa) ++dlaa_evaluates;
    h->processed = true;
    h->reset_next_evaluate = false;
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_process(
    DlssGpuHost* h, const uint8_t* rgba, size_t rgba_stride,
    void* cuda_context, uint64_t flow_ptr, size_t flow_pitch, int32_t reset,
    uint8_t* output_rgba, size_t output_stride) {
    return process_impl(h, rgba, rgba_stride, nullptr, 0, 0,
        cuda_context, flow_ptr, flow_pitch, reset, output_rgba, output_stride,
        false);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_device_color(
    DlssGpuHost* h, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, void* flow_cuda_context, uint64_t flow_ptr,
    size_t flow_pitch, int32_t reset, uint8_t* output_rgba,
    size_t output_stride) {
    if (!rgba_ptr) { error("invalid CUDA color pointer", 0); return 1; }
    return process_impl(h, nullptr, 0, color_cuda_context, rgba_ptr, rgba_pitch,
        flow_cuda_context, flow_ptr, flow_pitch, reset, output_rgba,
        output_stride, false);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_device_color_resource(
    DlssGpuHost* h, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, void* flow_cuda_context, uint64_t flow_ptr,
    size_t flow_pitch, int32_t reset, DlssGpuHostDeviceOutput* output) {
    if (!output || output->struct_size != sizeof(*output) || !rgba_ptr) {
        error("invalid device output arguments", 0);
        return 1;
    }
    const int32_t result = process_impl(h, nullptr, 0, color_cuda_context,
        rgba_ptr, rgba_pitch, flow_cuda_context, flow_ptr, flow_pitch,
        reset, nullptr, 0, true);
    if (result) return result;
    *output = DlssGpuHostDeviceOutput{sizeof(*output), h->width, h->height,
        static_cast<uint32_t>(h->color_format), h->device.Get(),
        h->output.Get(), h->fence.Get(), h->fence_value};
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_zero_motion(
    DlssGpuHost* h, const uint8_t* rgba, size_t rgba_stride, int32_t reset,
    uint8_t* output_rgba, size_t output_stride) {
    return process_impl(h, rgba, rgba_stride, nullptr, 0, 0, nullptr, 0, 0,
        reset, output_rgba, output_stride, false, true);
}

extern "C" DLSS_GPU_HOST_API int32_t
dlss_gpu_host_process_device_color_zero_motion_resource(
    DlssGpuHost* h, void* color_cuda_context, uint64_t rgba_ptr,
    size_t rgba_pitch, int32_t reset, DlssGpuHostDeviceOutput* output) {
    if (!output || output->struct_size != sizeof(*output) || !rgba_ptr) {
        error("invalid zero-motion device output arguments", 0);
        return 1;
    }
    const int32_t result = process_impl(h, nullptr, 0, color_cuda_context,
        rgba_ptr, rgba_pitch, nullptr, 0, 0, reset, nullptr, 0, true, true);
    if (result) return result;
    *output = DlssGpuHostDeviceOutput{sizeof(*output), h->width, h->height,
        static_cast<uint32_t>(h->color_format), h->device.Get(),
        h->output.Get(), h->fence.Get(), h->fence_value};
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_process_host_motion(
    DlssGpuHost* h, const uint8_t* rgba, size_t rgba_stride,
    const uint8_t* motion, size_t motion_stride, int32_t reset,
    uint8_t* output_rgba, size_t output_stride) {
    if (!motion) { error("invalid host motion arguments", 0); return 1; }
    // Host color in, host motion plane in, host color out: the guided frame a
    // dlssnr_host.dll Live.process caller can hand over without any CUDA.
    return process_impl(h, rgba, rgba_stride, nullptr, 0, 0, nullptr, 0, 0,
        reset, output_rgba, output_stride, false, false, motion, motion_stride);
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_readback_output(
    DlssGpuHost* h, uint8_t* output_rgba, size_t output_stride) {
    last_error[0] = 0;
    if (!h || h->failed || !h->processed || !output_rgba ||
        output_stride < size_t(h->width) * h->bytes_per_pixel) {
        error("invalid output readback arguments", 0);
        return 1;
    }
    HRESULT hr = ensure_buffer(h, h->output_readback,
        uint64_t(h->output_layout.Footprint.RowPitch) * h->height,
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    if (FAILED(hr)) { error("allocate output readback", hr); return 6; }
    hr = h->allocator->Reset();
    if (SUCCEEDED(hr)) hr = h->list->Reset(h->allocator.Get(), nullptr);
    if (FAILED(hr)) { error("reset readback command list", hr); h->failed = true; return 2; }
    copy_texture(h->list.Get(), h->output.Get(), h->output_readback.Get(),
                 h->output_layout, false);
    hr = h->list->Close();
    if (FAILED(hr)) { error("close readback command list", hr); h->failed = true; return 3; }
    ID3D12CommandList* commands[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr) || !wait_for_gpu(h, h->fence_value)) {
        if (FAILED(hr)) error("signal readback fence", hr);
        h->failed = true;
        return 4;
    }
    void* mapped = nullptr;
    hr = h->output_readback->Map(0, nullptr, &mapped);
    if (FAILED(hr)) { error("map output readback", hr); h->failed = true; return 5; }
    for (uint32_t y = 0; y < h->height; ++y)
        std::memcpy(output_rgba + size_t(y) * output_stride,
                    static_cast<const uint8_t*>(mapped) +
                        size_t(y) * h->output_layout.Footprint.RowPitch,
                    size_t(h->width) * h->bytes_per_pixel);
    h->output_readback->Unmap(0, nullptr);
    h->output_readback_count++;
    h->output_readback_bytes += uint64_t(h->width) * h->height * h->bytes_per_pixel;
    return 0;
}

#ifdef DLSS_GPU_HOST_FP16_PROBE
// Diagnostic ABI only. Copy was recorded before NGX evaluate in process_impl.
extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_probe_set_generic_exposure(
    DlssGpuHost* h, float pre_exposure, float exposure_scale) {
    if (!h || h->failed || !std::isfinite(pre_exposure) || pre_exposure <= 0 ||
        !std::isfinite(exposure_scale) || exposure_scale <= 0) return 1;
    h->probe_pre_exposure = pre_exposure;
    h->probe_exposure_scale = exposure_scale;
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_probe_readback_input(
    DlssGpuHost* h, uint8_t* dst, size_t dst_stride) {
    last_error[0] = 0;
    if (!h || h->failed || !h->processed || !dst ||
        dst_stride < size_t(h->width) * h->bytes_per_pixel) {
        error("invalid probe input readback arguments", 0);
        return 1;
    }
    void* mapped = nullptr;
    HRESULT hr = h->color_probe_readback->Map(0, nullptr, &mapped);
    if (FAILED(hr)) { error("map probe input readback", hr); return 2; }
    for (uint32_t y = 0; y < h->height; ++y)
        std::memcpy(dst + size_t(y) * dst_stride,
                    static_cast<const uint8_t*>(mapped) +
                        size_t(y) * h->color_layout.Footprint.RowPitch,
                    size_t(h->width) * h->bytes_per_pixel);
    h->color_probe_readback->Unmap(0, nullptr);
    return 0;
}
#endif

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_readback_motion(
    DlssGpuHost* h, uint8_t* dst, size_t dst_stride) {
    last_error[0] = 0;
    if (!h || h->failed || !dst ||
        dst_stride < size_t(h->width) * 4) {
        error("invalid motion readback arguments", 0);
        return 1;
    }
    HRESULT hr = ensure_buffer(h, h->motion_readback,
        uint64_t(h->motion_layout.Footprint.RowPitch) * h->height,
        D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    if (FAILED(hr)) { error("allocate motion readback", hr); return 6; }
    hr = h->allocator->Reset();
    if (SUCCEEDED(hr)) hr = h->list->Reset(h->allocator.Get(), nullptr);
    if (FAILED(hr)) { error("reset motion readback command list", hr);
        h->failed = true; return 2; }
    transition(h->list.Get(), h->motion.Get(),
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    copy_texture(h->list.Get(), h->motion.Get(), h->motion_readback.Get(),
                 h->motion_layout, false);
    transition(h->list.Get(), h->motion.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    hr = h->list->Close();
    if (FAILED(hr)) { error("close motion readback command list", hr);
        h->failed = true; return 3; }
    ID3D12CommandList* commands[] = {h->list.Get()};
    h->queue->ExecuteCommandLists(1, commands);
    hr = h->queue->Signal(h->fence.Get(), ++h->fence_value);
    if (FAILED(hr) || !wait_for_gpu(h, h->fence_value)) {
        if (FAILED(hr)) error("signal motion readback fence", hr);
        h->failed = true;
        return 4;
    }
    void* mapped = nullptr;
    hr = h->motion_readback->Map(0, nullptr, &mapped);
    if (FAILED(hr)) { error("map motion readback", hr); h->failed = true; return 5; }
    for (uint32_t y = 0; y < h->height; ++y)
        std::memcpy(dst + size_t(y) * dst_stride,
                    static_cast<const uint8_t*>(mapped) +
                        size_t(y) * h->motion_layout.Footprint.RowPitch,
                    size_t(h->width) * 4);
    h->motion_readback->Unmap(0, nullptr);
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_get_color_info(
    DlssGpuHost* h, DlssGpuHostColorInfo* out) {
    if (!h || !out || out->struct_size != sizeof(*out)) {
        error("invalid colour info arguments", 0);
        return 1;
    }
    *out = DlssGpuHostColorInfo{sizeof(*out), h->width, h->height,
        static_cast<uint32_t>(h->color_format), h->bytes_per_pixel,
        uint64_t(h->width) * h->bytes_per_pixel,
        h->color_layout.Footprint.RowPitch, h->output_layout.Footprint.RowPitch,
        h->bridge_row_pitch, h->bridge_bytes_per_row, h->last_copy_src_pitch,
        h->last_copy_row_bytes, h->last_copy_fence_value,
        h->device_color_copy_count};
    return 0;
}

extern "C" DLSS_GPU_HOST_API int32_t dlss_gpu_host_get_transfer_counters(
    DlssGpuHost* h, DlssGpuHostTransferCounters* out) {
    if (!h || !out || out->struct_size != sizeof(*out)) return 1;
    *out = DlssGpuHostTransferCounters{sizeof(*out),
        h->host_color_stage_count, h->host_color_stage_bytes,
        h->device_color_copy_count, h->device_color_copy_bytes,
        h->output_readback_count, h->output_readback_bytes};
    return 0;
}
