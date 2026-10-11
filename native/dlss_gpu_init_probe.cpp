#include <windows.h>
#include <d3d12.h>
#include <nvsdk_ngx.h>
#include "nvof_bridge.h"
#include "nvof_d3d12_interop.h"

#include <filesystem>
#include <cstdio>
#include <cstring>
#include <vector>

using InitExt = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
using CreateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, NVSDK_NGX_Feature, NVSDK_NGX_Parameter*,
    NVSDK_NGX_Handle**);
using ReleaseFeature = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using EvaluateFeature = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ModuleName = DWORD(WINAPI*)(HMODULE, LPWSTR, DWORD);
constexpr unsigned probe_width = 1920;
constexpr unsigned probe_height = 1080;

static ModuleName original_module_name = nullptr;
static HMODULE probe_module = nullptr;

static DWORD WINAPI compatible_module_name(HMODULE queried, LPWSTR buffer, DWORD capacity) {
    if (queried != probe_module) {
        return original_module_name(queried, buffer, capacity);
    }
    constexpr wchar_t name[] = L"nvngx.dll";
    constexpr DWORD length = static_cast<DWORD>(sizeof(name) / sizeof(name[0]) - 1);
    if (!buffer || !capacity) {
        SetLastError(ERROR_INSUFFICIENT_BUFFER);
        return 0;
    }
    const DWORD copied = capacity > length ? length : capacity - 1;
    std::memcpy(buffer, name, copied * sizeof(wchar_t));
    buffer[copied] = 0;
    if (capacity <= length) SetLastError(ERROR_INSUFFICIENT_BUFFER);
    return capacity > length ? length : capacity;
}

// The PE import table gives us the one call site owned by the loaded snippet.
// This probe only changes that slot, and restores it before unloading the DLL.
static void** module_name_import(HMODULE module) {
    auto* image = reinterpret_cast<unsigned char*>(module);
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image);
    if (dos->e_magic != IMAGE_DOS_SIGNATURE || dos->e_lfanew <= 0) return nullptr;
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE ||
        nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) return nullptr;
    const DWORD image_size = nt->OptionalHeader.SizeOfImage;
    const auto& imports = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (!imports.VirtualAddress || imports.VirtualAddress >= image_size ||
        imports.Size > image_size - imports.VirtualAddress) return nullptr;
    const DWORD count = imports.Size / sizeof(IMAGE_IMPORT_DESCRIPTOR);
    auto* table = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(image + imports.VirtualAddress);
    for (DWORD i = 0; i < count && table[i].Name; ++i) {
        const auto& entry = table[i];
        if (!entry.OriginalFirstThunk || !entry.FirstThunk ||
            entry.OriginalFirstThunk >= image_size || entry.FirstThunk >= image_size) continue;
        auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(image + entry.OriginalFirstThunk);
        auto* slots = reinterpret_cast<IMAGE_THUNK_DATA64*>(image + entry.FirstThunk);
        const DWORD max_names = (image_size - entry.OriginalFirstThunk) / sizeof(*names);
        const DWORD max_slots = (image_size - entry.FirstThunk) / sizeof(*slots);
        const DWORD limit = max_names < max_slots ? max_names : max_slots;
        for (DWORD j = 0; j < limit && names[j].u1.AddressOfData; ++j) {
            if (IMAGE_SNAP_BY_ORDINAL64(names[j].u1.Ordinal)) continue;
            const auto rva = names[j].u1.AddressOfData;
            if (rva >= image_size || image_size - rva < sizeof(IMAGE_IMPORT_BY_NAME)) continue;
            auto* imported = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(image + rva);
            if (std::strcmp(reinterpret_cast<const char*>(imported->Name),
                            "GetModuleFileNameW") == 0) {
                return reinterpret_cast<void**>(&slots[j].u1.Function);
            }
        }
    }
    return nullptr;
}

static bool change_import(void** slot, void* replacement, void** previous) {
    DWORD protection = 0;
    if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &protection)) return false;
    *previous = InterlockedExchangePointer(reinterpret_cast<void* volatile*>(slot), replacement);
    DWORD ignored = 0;
    VirtualProtect(slot, sizeof(*slot), protection, &ignored);
    return true;
}

static NVSDK_NGX_Result call_init(InitExt fn, const wchar_t* dir,
                                 ID3D12Device* device, DWORD* exception_code) {
    __try {
        return fn(0x0876232Cull, dir, device, NVSDK_NGX_Version_API, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

static NVSDK_NGX_Result call_core_init(const wchar_t* dir,
                                      ID3D12Device* device, DWORD* exception_code) {
    __try {
        return NVSDK_NGX_D3D12_Init_with_ProjectID(
            "86a83b2d-d0a9-4be6-aa59-39c5e931bdc0",
            NVSDK_NGX_ENGINE_TYPE_CUSTOM, "dlss5standaloneV2",
            dir, device, nullptr, NVSDK_NGX_Version_API);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

static NVSDK_NGX_Result call_allocate(NVSDK_NGX_Parameter** parameters,
                                     DWORD* exception_code) {
    __try {
        return NVSDK_NGX_D3D12_AllocateParameters(parameters);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

static NVSDK_NGX_Result NVSDK_CONV scaling_ratio(
    NVSDK_NGX_Parameter* parameters) {
    if (!parameters) return NVSDK_NGX_Result_FAIL_InvalidParameter;
    parameters->Set("DLSSNR.ScalingRatio", 1.0f);
    return NVSDK_NGX_Result_Success;
}

static bool configure_feature(NVSDK_NGX_Parameter* parameters, DWORD* exception_code) {
    constexpr unsigned width = probe_width, height = probe_height;
    __try {
        for (const char* key : {"DLSSNR.Width", "DLSSNR.InputWidth",
                                "DLSSNR.OutputWidth", "DLSSNR.Output.Width"}) {
            parameters->Set(key, width);
        }
        for (const char* key : {"DLSSNR.Height", "DLSSNR.InputHeight",
                                "DLSSNR.OutputHeight", "DLSSNR.Output.Height"}) {
            parameters->Set(key, height);
        }
        parameters->Set("DLSSNR.Upscaling", 0u);
        parameters->Set("DLSSNR.Scale", 1.0f);
        parameters->Set("DLSSNR.ScalingRatio", 1.0f);
        parameters->Set("DLSSNRComputeScalingRatioCallback",
                        reinterpret_cast<void*>(&scaling_ratio));
        parameters->Set("DLSSNR.Hint.Render.Preset", 0);
        parameters->Set(NVSDK_NGX_Parameter_Width, width);
        parameters->Set(NVSDK_NGX_Parameter_Height, height);
        parameters->Set(NVSDK_NGX_Parameter_PerfQualityValue,
                        static_cast<int>(NVSDK_NGX_PerfQuality_Value_Balanced));
        parameters->Set(NVSDK_NGX_Parameter_CreationNodeMask, 1u);
        parameters->Set(NVSDK_NGX_Parameter_VisibilityNodeMask, 1u);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return false;
    }
}

static NVSDK_NGX_Result call_create(CreateFeature fn,
                                   ID3D12GraphicsCommandList* list,
                                   NVSDK_NGX_Parameter* parameters,
                                   NVSDK_NGX_Handle** handle, DWORD* exception_code) {
    __try {
        return fn(list, static_cast<NVSDK_NGX_Feature>(18), parameters, handle);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

static NVSDK_NGX_Result call_evaluate(EvaluateFeature fn,
                                     ID3D12GraphicsCommandList* list,
                                     const NVSDK_NGX_Handle* feature,
                                     const NVSDK_NGX_Parameter* parameters,
                                     DWORD* exception_code) {
    __try {
        return fn(list, feature, parameters, nullptr);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return NVSDK_NGX_Result_FAIL_PlatformError;
    }
}

static bool configure_frame(NVSDK_NGX_Parameter* parameters,
                            ID3D12Resource* const textures[4], DWORD* exception_code) {
    __try {
        const char* resource_names[] = {
            "DLSSNR.Color", "DLSSNR.MVec", "DLSSNR.Depth", "DLSSNR.Output"};
        for (unsigned i = 0; i < 4; ++i) parameters->Set(resource_names[i], textures[i]);
        const char* rect_names[] = {"Color", "MVec", "Depth", "Output"};
        for (const char* name : rect_names) {
            char key[64];
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseX", name);
            parameters->Set(key, 0u);
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectBaseY", name);
            parameters->Set(key, 0u);
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectWidth", name);
            parameters->Set(key, probe_width);
            std::snprintf(key, sizeof(key), "DLSSNR.%sSubrectHeight", name);
            parameters->Set(key, probe_height);
        }
        parameters->Set("DLSSNR.MVecScaleX", 1.0f);
        parameters->Set("DLSSNR.MVecScaleY", 1.0f);
        parameters->Set("DLSSNR.DepthInverted", 1);
        parameters->Set("DLSS.Indicator.Invert.X.Axis", 0);
        parameters->Set("DLSS.Indicator.Invert.Y.Axis", 0);
        parameters->Set("DLSSNR.Enabled", 1);
        parameters->Set("DLSSNR.Reset", 1);
        parameters->Set("DLSSNR.Style", 0);
        parameters->Set("DLSSNR.Intensity", 1.0f);
        parameters->Set("DLSSNR.LocalToneStrength", 1.0f);
        parameters->Set("DLSSNR.LocalStructureStrength", 1.0f);
        parameters->Set("DLSSNR.SkinStructureStrength", 0.0f);
        parameters->Set("DLSSNR.UseAutoMask", 0);
        parameters->Set("DLSSNR.UICorrection", 0);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        *exception_code = 1;
        return false;
    }
}

static bool run_frame(HMODULE module, ID3D12Device* device,
                      ID3D12CommandQueue* queue, ID3D12CommandAllocator* allocator,
                      ID3D12GraphicsCommandList* list, ID3D12Fence* fence,
                      NVSDK_NGX_Handle* feature, NVSDK_NGX_Parameter* parameters,
                      const wchar_t* dump_path) {
    auto evaluate = reinterpret_cast<EvaluateFeature>(
        GetProcAddress(module, "NVSDK_NGX_D3D12_EvaluateFeature"));
    if (!evaluate) return false;
    const DXGI_FORMAT formats[4] = {
        DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R32_FLOAT, DXGI_FORMAT_R8G8B8A8_UNORM};
    ID3D12Resource* textures[4]{};
    ID3D12DescriptorHeap* descriptors = nullptr;
    ID3D12Resource* color_upload = nullptr;
    ID3D12Resource* output_readback = nullptr;
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT color_layout{};
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT output_layout{};
    void* nvof_session = nullptr;
    NvofD3D12FlowBridge* flow_bridge = nullptr;
    uint64_t flow_ready = 0;
    std::vector<uint8_t> previous(size_t(probe_width) * probe_height * 3);
    std::vector<uint8_t> current(previous.size());
    for (unsigned y = 0; y < probe_height; ++y)
        for (unsigned x = 0; x < probe_width; ++x) {
        const size_t p = (size_t(y) * probe_width + x) * 3;
        previous[p] = static_cast<uint8_t>((x * 7 + y) % 256);
        previous[p + 1] = static_cast<uint8_t>((x + y * 5) % 256);
        previous[p + 2] = static_cast<uint8_t>((x * 3 + y * 2) % 256);
        const size_t q = (size_t(y) * probe_width + (x ? x - 1 : 0)) * 3;
        current[p] = previous[q];
        current[p + 1] = previous[q + 1];
        current[p + 2] = previous[q + 2];
    }
    if (nvof_bridge_create(probe_width, probe_height, 4, 10, &nvof_session)) {
        std::printf("nvof_create_failed: %s\n", nvof_bridge_get_last_error());
        return false;
    }
    NvOFBridgeTiming nvof_timing{};
    if (nvof_bridge_execute_bgr_sequential_device(nvof_session,
            current.data(), probe_width * 3, previous.data(),
            probe_width * 3, 1, &nvof_timing)) {
        std::printf("nvof_execute_failed: %s\n", nvof_bridge_get_last_error());
        nvof_bridge_destroy(nvof_session);
        return false;
    }
    uint64_t flow_ptr = 0;
    size_t flow_pitch = 0;
    uint32_t cuda_index = 0, owner_thread = 0;
    void* cuda_context = nullptr;
    nvof_bridge_get_output_deviceptr(nvof_session, &flow_ptr, &flow_pitch);
    nvof_bridge_get_session_info(nvof_session, &cuda_index, &cuda_context,
                                 &owner_thread);
    if (nvof_d3d12_create(device, cuda_context, probe_width, probe_height, &flow_bridge) ||
        nvof_d3d12_copy_flow(flow_bridge, flow_ptr, flow_pitch, &flow_ready)) {
        std::printf("flow_interop_failed: %s\n", nvof_d3d12_last_error());
        nvof_d3d12_destroy(flow_bridge);
        nvof_bridge_destroy(nvof_session);
        return false;
    }
    void* raw_flow_buffer = nullptr;
    void* raw_flow_fence = nullptr;
    nvof_d3d12_get_resources(flow_bridge, &raw_flow_buffer, &raw_flow_fence);
    if (FAILED(queue->Wait(static_cast<ID3D12Fence*>(raw_flow_fence), flow_ready))) {
        nvof_d3d12_destroy(flow_bridge);
        nvof_bridge_destroy(nvof_session);
        return false;
    }
    D3D12_HEAP_PROPERTIES heap_props{};
    heap_props.Type = D3D12_HEAP_TYPE_DEFAULT;
    HRESULT hr = S_OK;
    for (unsigned i = 0; i < 4; ++i) {
        if (i == 1) continue;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = probe_width;
        desc.Height = probe_height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = formats[i];
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        hr = device->CreateCommittedResource(&heap_props, D3D12_HEAP_FLAG_NONE,
            &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&textures[i]));
        if (FAILED(hr)) break;
    }
    if (SUCCEEDED(hr)) {
        UINT rows = 0;
        UINT64 row_size = 0, bytes = 0;
        const auto color_desc = textures[0]->GetDesc();
        device->GetCopyableFootprints(&color_desc, 0, 1, 0,
                                      &color_layout, &rows, &row_size, &bytes);
        D3D12_RESOURCE_DESC buffer_desc{};
        buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_desc.Width = bytes;
        buffer_desc.Height = 1;
        buffer_desc.DepthOrArraySize = 1;
        buffer_desc.MipLevels = 1;
        buffer_desc.SampleDesc.Count = 1;
        buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        D3D12_HEAP_PROPERTIES upload_props{};
        upload_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        hr = device->CreateCommittedResource(&upload_props, D3D12_HEAP_FLAG_NONE,
            &buffer_desc, D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&color_upload));
        if (SUCCEEDED(hr)) {
            uint8_t* mapped = nullptr;
            hr = color_upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped));
            if (SUCCEEDED(hr)) {
                for (unsigned y = 0; y < probe_height; ++y)
                    for (unsigned x = 0; x < probe_width; ++x) {
                    const size_t source = (size_t(y) * probe_width + x) * 3;
                    const size_t destination = size_t(y) * color_layout.Footprint.RowPitch + x * 4;
                    mapped[destination] = current[source + 2];
                    mapped[destination + 1] = current[source + 1];
                    mapped[destination + 2] = current[source];
                    mapped[destination + 3] = 255;
                }
                color_upload->Unmap(0, nullptr);
            }
        }
        if (SUCCEEDED(hr)) {
            const auto output_desc = textures[3]->GetDesc();
            device->GetCopyableFootprints(&output_desc, 0, 1, 0,
                &output_layout, &rows, &row_size, &bytes);
            buffer_desc.Width = bytes;
            D3D12_HEAP_PROPERTIES readback_props{};
            readback_props.Type = D3D12_HEAP_TYPE_READBACK;
            hr = device->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE,
                &buffer_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&output_readback));
        }
    }
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.NumDescriptors = 3;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    if (SUCCEEDED(hr)) hr = device->CreateDescriptorHeap(
        &heap_desc, IID_PPV_ARGS(&descriptors));
    if (SUCCEEDED(hr)) hr = allocator->Reset();
    if (SUCCEEDED(hr)) hr = list->Reset(allocator, nullptr);
    if (SUCCEEDED(hr)) {
        ID3D12DescriptorHeap* heaps[] = {descriptors};
        list->SetDescriptorHeaps(1, heaps);
        auto cpu = descriptors->GetCPUDescriptorHandleForHeapStart();
        auto gpu = descriptors->GetGPUDescriptorHandleForHeapStart();
        const UINT increment = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
        const float zero[4]{};
        for (unsigned i = 0; i < 4; ++i) {
            if (i == 1) continue;
            device->CreateUnorderedAccessView(textures[i], nullptr, nullptr, cpu);
            list->ClearUnorderedAccessViewFloat(gpu, cpu, textures[i], zero, 0, nullptr);
            cpu.ptr += increment;
            gpu.ptr += increment;
        }
        D3D12_RESOURCE_BARRIER barriers[2]{};
        barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[0].Transition.pResource = textures[0];
        barriers[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barriers[1].Transition.pResource = textures[2];
        barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        barriers[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(2, barriers);
        D3D12_TEXTURE_COPY_LOCATION color_source{};
        color_source.pResource = color_upload;
        color_source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        color_source.PlacedFootprint = color_layout;
        D3D12_TEXTURE_COPY_LOCATION color_destination{};
        color_destination.pResource = textures[0];
        color_destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        list->CopyTextureRegion(&color_destination, 0, 0, 0,
                                &color_source, nullptr);
        barriers[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barriers[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        list->ResourceBarrier(1, barriers);
        void* raw_motion = nullptr;
        const int motion_result = nvof_d3d12_record_motion(
            flow_bridge, list, &raw_motion);
        if (motion_result) {
            std::printf("flow_motion_failed: %s\n", nvof_d3d12_last_error());
            hr = E_FAIL;
        } else {
            textures[1] = static_cast<ID3D12Resource*>(raw_motion);
        }
        DWORD exception = 0;
        if (SUCCEEDED(hr) && !configure_frame(parameters, textures, &exception)) {
            std::printf("frame_config_exception=%lu\n", exception);
            hr = E_FAIL;
        } else if (SUCCEEDED(hr)) {
            const NVSDK_NGX_Result result = call_evaluate(
                evaluate, list, feature, parameters, &exception);
            std::printf("evaluate_result=0x%08x exception_caught=%lu\n",
                static_cast<unsigned>(result), exception);
            if (!NVSDK_NGX_SUCCEED(result) || exception) hr = E_FAIL;
        }
        if (SUCCEEDED(hr)) {
            D3D12_RESOURCE_BARRIER output_barrier{};
            output_barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            output_barrier.Transition.pResource = textures[3];
            output_barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            output_barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            output_barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
            list->ResourceBarrier(1, &output_barrier);
            D3D12_TEXTURE_COPY_LOCATION source{};
            source.pResource = textures[3];
            source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
            D3D12_TEXTURE_COPY_LOCATION destination{};
            destination.pResource = output_readback;
            destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
            destination.PlacedFootprint = output_layout;
            list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        }
        if (SUCCEEDED(hr)) hr = list->Close();
        if (SUCCEEDED(hr)) {
            ID3D12CommandList* commands[] = {list};
            queue->ExecuteCommandLists(1, commands);
            hr = queue->Signal(fence, 2);
        }
        if (SUCCEEDED(hr) && fence->GetCompletedValue() < 2) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event) hr = HRESULT_FROM_WIN32(GetLastError());
            else {
                hr = fence->SetEventOnCompletion(2, event);
                if (SUCCEEDED(hr) && WaitForSingleObject(event, 15000) != WAIT_OBJECT_0)
                    hr = HRESULT_FROM_WIN32(ERROR_TIMEOUT);
                CloseHandle(event);
            }
        }
    }
    std::printf("evaluate_queue_result=0x%08x\n", static_cast<unsigned>(hr));
    if (SUCCEEDED(hr) && output_readback) {
        const uint8_t* mapped = nullptr;
        const HRESULT map_result = output_readback->Map(
            0, nullptr, reinterpret_cast<void**>(const_cast<uint8_t**>(&mapped)));
        if (SUCCEEDED(map_result)) {
            uint64_t checksum = 0;
            for (unsigned y = 0; y < probe_height; ++y)
                for (unsigned x = 0; x < probe_width * 4; ++x)
                checksum += mapped[size_t(y) * output_layout.Footprint.RowPitch + x];
            output_readback->Unmap(0, nullptr);
            std::printf("output_byte_sum=%llu\n",
                        static_cast<unsigned long long>(checksum));
            if (dump_path) {
                FILE* dump = nullptr;
                if (_wfopen_s(&dump, dump_path, L"wb") || !dump) hr = E_FAIL;
                else {
                    for (unsigned y = 0; y < probe_height; ++y)
                        if (std::fwrite(mapped + size_t(y) * output_layout.Footprint.RowPitch,
                                        1, probe_width * 4, dump) != probe_width * 4)
                            hr = E_FAIL;
                    std::fclose(dump);
                }
            }
            if (!checksum) hr = E_FAIL;
        } else hr = map_result;
    }
    NvOFBridgeCounters nvof_counters{};
    nvof_bridge_get_counters(nvof_session, &nvof_counters);
    std::printf("nvof_flow_d2h_copies=%llu bytes=%llu\n",
        static_cast<unsigned long long>(nvof_counters.d2h_copy_count),
        static_cast<unsigned long long>(nvof_counters.d2h_bytes));
    if (descriptors) descriptors->Release();
    if (color_upload) color_upload->Release();
    if (output_readback) output_readback->Release();
    for (unsigned i = 0; i < 4; ++i)
        if (i != 1 && textures[i]) textures[i]->Release();
    nvof_d3d12_destroy(flow_bridge);
    nvof_bridge_destroy(nvof_session);
    return SUCCEEDED(hr) && !nvof_counters.d2h_copy_count;
}

static bool probe_feature(HMODULE module, ID3D12Device* device,
                          NVSDK_NGX_Parameter* parameters,
                          const wchar_t* dump_path) {
    auto create = reinterpret_cast<CreateFeature>(
        GetProcAddress(module, "NVSDK_NGX_D3D12_CreateFeature"));
    auto release = reinterpret_cast<ReleaseFeature>(
        GetProcAddress(module, "NVSDK_NGX_D3D12_ReleaseFeature"));
    if (!create || !release) return false;
    DWORD exception = 0;
    if (!configure_feature(parameters, &exception)) {
        std::printf("configure_exception=%lu\n", exception);
        return false;
    }
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    ID3D12CommandQueue* queue = nullptr;
    ID3D12Fence* fence = nullptr;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
    if (SUCCEEDED(hr)) hr = device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
    if (SUCCEEDED(hr)) hr = device->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr,
        IID_PPV_ARGS(&list));
    if (FAILED(hr)) {
        std::printf("command_list_error=0x%08x\n", static_cast<unsigned>(hr));
        if (list) list->Release();
        if (allocator) allocator->Release();
        if (queue) queue->Release();
        return false;
    }
    NVSDK_NGX_Handle* feature = nullptr;
    const NVSDK_NGX_Result result = call_create(
        create, list, parameters, &feature, &exception);
    std::printf("create_result=0x%08x exception_caught=%lu has_feature=%u\n",
        static_cast<unsigned>(result), exception, feature ? 1u : 0u);
    bool success = NVSDK_NGX_SUCCEED(result) && feature;
    if (success) {
        hr = list->Close();
        if (SUCCEEDED(hr)) hr = device->CreateFence(
            0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        if (SUCCEEDED(hr)) {
            ID3D12CommandList* commands[] = {list};
            queue->ExecuteCommandLists(1, commands);
            hr = queue->Signal(fence, 1);
        }
        if (SUCCEEDED(hr) && fence->GetCompletedValue() < 1) {
            HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
            if (!event) {
                hr = HRESULT_FROM_WIN32(GetLastError());
            } else {
                hr = fence->SetEventOnCompletion(1, event);
                if (SUCCEEDED(hr) && WaitForSingleObject(event, 15000) != WAIT_OBJECT_0)
                    hr = HRESULT_FROM_WIN32(ERROR_TIMEOUT);
                CloseHandle(event);
            }
        }
        std::printf("create_queue_result=0x%08x\n", static_cast<unsigned>(hr));
        success = SUCCEEDED(hr);
        if (success) success = run_frame(
            module, device, queue, allocator, list, fence, feature, parameters,
            dump_path);
    }
    if (feature) release(feature);
    if (fence) fence->Release();
    list->Release();
    allocator->Release();
    queue->Release();
    return success;
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2 && argc != 3) {
        std::fputs("usage: dlss_gpu_init_probe.exe <nvngx_dlssnr.dll> [output.rgba]\n", stderr);
        return 2;
    }
    const std::filesystem::path dll_path = std::filesystem::absolute(argv[1]);
    const std::wstring directory = dll_path.parent_path().wstring();
    HMODULE module = LoadLibraryExW(dll_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!module) {
        std::printf("load_error=%lu\n", GetLastError());
        return 3;
    }
    auto init = reinterpret_cast<InitExt>(
        GetProcAddress(module, "NVSDK_NGX_D3D12_Init_Ext"));
    auto shutdown = reinterpret_cast<Shutdown>(
        GetProcAddress(module, "NVSDK_NGX_D3D12_Shutdown1"));
    if (!init || !shutdown) {
        std::puts("signed_snippet_exports=missing");
        FreeLibrary(module);
        return 4;
    }
    ID3D12Device* device = nullptr;
    const HRESULT hr = D3D12CreateDevice(
        nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        std::printf("d3d12_create_error=0x%08x\n", static_cast<unsigned>(hr));
        FreeLibrary(module);
        return 5;
    }
    DWORD core_exception = 0;
    const NVSDK_NGX_Result core_result = call_core_init(
        directory.c_str(), device, &core_exception);
    std::printf("core_init_result=0x%08x exception_caught=%lu\n",
        static_cast<unsigned>(core_result), core_exception);
    if (!NVSDK_NGX_SUCCEED(core_result)) {
        device->Release();
        FreeLibrary(module);
        return 9;
    }
    NVSDK_NGX_Parameter* parameters = nullptr;
    DWORD allocation_exception = 0;
    const NVSDK_NGX_Result allocation_result = call_allocate(
        &parameters, &allocation_exception);
    std::printf("allocate_result=0x%08x exception_caught=%lu has_parameters=%u\n",
        static_cast<unsigned>(allocation_result), allocation_exception,
        parameters ? 1u : 0u);
    if (!NVSDK_NGX_SUCCEED(allocation_result) || !parameters) {
        NVSDK_NGX_D3D12_Shutdown1(device);
        device->Release();
        FreeLibrary(module);
        return 10;
    }
    void** import_slot = module_name_import(module);
    if (!import_slot) {
        std::puts("module_name_import=missing");
        NVSDK_NGX_D3D12_DestroyParameters(parameters);
        NVSDK_NGX_D3D12_Shutdown1(device);
        device->Release();
        FreeLibrary(module);
        return 6;
    }
    probe_module = GetModuleHandleW(nullptr);
    void* old_import = nullptr;
    const bool patched = change_import(
        import_slot, reinterpret_cast<void*>(&compatible_module_name), &old_import);
    if (!patched || !old_import) {
        std::puts("module_name_import=patch_failed");
        if (patched) {
            void* replaced = nullptr;
            change_import(import_slot, old_import, &replaced);
        }
        NVSDK_NGX_D3D12_DestroyParameters(parameters);
        NVSDK_NGX_D3D12_Shutdown1(device);
        device->Release();
        FreeLibrary(module);
        return 7;
    }
    original_module_name = reinterpret_cast<ModuleName>(old_import);
    DWORD exception_code = 0;
    const NVSDK_NGX_Result result = call_init(
        init, directory.c_str(), device, &exception_code);
    std::printf("init_result=0x%08x exception_caught=%lu\n",
        static_cast<unsigned>(result), exception_code);
    bool feature_created = false;
    if (NVSDK_NGX_SUCCEED(result)) {
        feature_created = probe_feature(module, device, parameters,
                                        argc == 3 ? argv[2] : nullptr);
        shutdown(device);
    }
    void* replaced = nullptr;
    const bool restored = change_import(import_slot, old_import, &replaced);
    if (!restored || replaced != reinterpret_cast<void*>(&compatible_module_name)) {
        std::puts("module_name_import=restore_failed");
        // A still-patched DLL must stay loaded while the callback lives.
        device->Release();
        return 8;
    }
    original_module_name = nullptr;
    NVSDK_NGX_D3D12_DestroyParameters(parameters);
    NVSDK_NGX_D3D12_Shutdown1(device);
    device->Release();
    FreeLibrary(module);
    return feature_created ? 0 : 1;
}
