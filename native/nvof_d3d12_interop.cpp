#define NVOF_D3D12_INTEROP_EXPORTS
#include "nvof_d3d12_interop.h"
#include "include/cuda.h"

#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

// CUDA Driver ABI declarations for external D3D12 resources. Kept local so the
// bridge builds without a CUDA Toolkit installation; functions load from nvcuda.
struct ExternalMemoryDesc {
    int type;
    union { int fd; struct { void* handle; const void* name; } win32; } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct ExternalBufferDesc {
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
};
struct ExternalSemaphoreDesc {
    int type;
    union { int fd; struct { void* handle; const void* name; } win32; } handle;
    unsigned int flags;
    unsigned int reserved[16];
};
struct ExternalSignalParams {
    struct {
        struct { unsigned long long value; } fence;
        unsigned long long nvsci_reserved;
        unsigned long long mutex_key;
        unsigned int reserved[12];
    } params;
    unsigned int flags;
    unsigned int reserved[16];
};
using ExternalMemory = void*;
using ExternalSemaphore = void*;

using CtxPush = CUresult(CUDAAPI*)(CUcontext);
using CtxPop = CUresult(CUDAAPI*)(CUcontext*);
using CtxDevice = CUresult(CUDAAPI*)(CUdevice*);
using DeviceLuid = CUresult(CUDAAPI*)(char*, unsigned int*, CUdevice);
using ImportMemory = CUresult(CUDAAPI*)(ExternalMemory*, const ExternalMemoryDesc*);
using MapBuffer = CUresult(CUDAAPI*)(CUdeviceptr*, ExternalMemory, const ExternalBufferDesc*);
using DestroyMemory = CUresult(CUDAAPI*)(ExternalMemory);
using MemFree = CUresult(CUDAAPI*)(CUdeviceptr);
using ImportSemaphore = CUresult(CUDAAPI*)(ExternalSemaphore*, const ExternalSemaphoreDesc*);
using DestroySemaphore = CUresult(CUDAAPI*)(ExternalSemaphore);
using SignalSemaphore = CUresult(CUDAAPI*)(const ExternalSemaphore*,
    const ExternalSignalParams*, unsigned int, CUstream);
using StreamCreate = CUresult(CUDAAPI*)(CUstream*, unsigned int);
using StreamDestroy = CUresult(CUDAAPI*)(CUstream);
using StreamSync = CUresult(CUDAAPI*)(CUstream);
using Memcpy2D = CUresult(CUDAAPI*)(const CUDA_MEMCPY2D*);

struct Driver {
    HMODULE module = nullptr;
    CtxPush push = nullptr;
    CtxPop pop = nullptr;
    CtxDevice ctx_device = nullptr;
    DeviceLuid device_luid = nullptr;
    ImportMemory import_memory = nullptr;
    MapBuffer map_buffer = nullptr;
    DestroyMemory destroy_memory = nullptr;
    MemFree mem_free = nullptr;
    ImportSemaphore import_semaphore = nullptr;
    DestroySemaphore destroy_semaphore = nullptr;
    SignalSemaphore signal_semaphore = nullptr;
    StreamCreate stream_create = nullptr;
    StreamDestroy stream_destroy = nullptr;
    StreamSync stream_sync = nullptr;
    Memcpy2D memcpy_2d = nullptr;
};

thread_local char error_text[256] = "";
void fail(const char* step, long code) {
    std::snprintf(error_text, sizeof(error_text), "%s failed (0x%08lx)", step,
                  static_cast<unsigned long>(code));
}

template<class T> bool load(HMODULE module, const char* name, T* target) {
    *target = reinterpret_cast<T>(GetProcAddress(module, name));
    if (!*target) fail(name, GetLastError());
    return *target != nullptr;
}

bool load_driver(Driver* d) {
    d->module = LoadLibraryW(L"nvcuda.dll");
    if (!d->module) { fail("LoadLibrary(nvcuda)", GetLastError()); return false; }
    return load(d->module, "cuCtxPushCurrent_v2", &d->push) &&
        load(d->module, "cuCtxPopCurrent_v2", &d->pop) &&
        load(d->module, "cuCtxGetDevice", &d->ctx_device) &&
        load(d->module, "cuDeviceGetLuid", &d->device_luid) &&
        load(d->module, "cuImportExternalMemory", &d->import_memory) &&
        load(d->module, "cuExternalMemoryGetMappedBuffer", &d->map_buffer) &&
        load(d->module, "cuDestroyExternalMemory", &d->destroy_memory) &&
        load(d->module, "cuMemFree_v2", &d->mem_free) &&
        load(d->module, "cuImportExternalSemaphore", &d->import_semaphore) &&
        load(d->module, "cuDestroyExternalSemaphore", &d->destroy_semaphore) &&
        load(d->module, "cuSignalExternalSemaphoresAsync", &d->signal_semaphore) &&
        load(d->module, "cuStreamCreate", &d->stream_create) &&
        load(d->module, "cuStreamDestroy_v2", &d->stream_destroy) &&
        load(d->module, "cuStreamSynchronize", &d->stream_sync) &&
        load(d->module, "cuMemcpy2D_v2", &d->memcpy_2d);
}

} // namespace

struct NvofD3D12FlowBridge {
    Driver cuda;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    ExternalMemory memory = nullptr;
    ExternalSemaphore semaphore = nullptr;
    CUdeviceptr mapped = 0;
    ID3D12Device* device = nullptr;
    ID3D12Resource* buffer = nullptr;
    ID3D12Fence* fence = nullptr;
    ID3D12Resource* motion = nullptr;
    ID3D12DescriptorHeap* descriptors = nullptr;
    ID3D12RootSignature* root = nullptr;
    ID3D12PipelineState* pipeline = nullptr;
    bool motion_readable = false;
    HANDLE memory_handle = nullptr;
    HANDLE fence_handle = nullptr;
    uint32_t width = 0, height = 0;
    uint64_t fence_value = 0;
};

struct CudaD3D12ColorBridge {
    Driver cuda;
    CUcontext context = nullptr;
    CUstream stream = nullptr;
    ExternalMemory memory = nullptr;
    ExternalSemaphore semaphore = nullptr;
    CUdeviceptr mapped = 0;
    ID3D12Device* device = nullptr;
    ID3D12Resource* buffer = nullptr;
    ID3D12Fence* fence = nullptr;
    HANDLE memory_handle = nullptr;
    HANDLE fence_handle = nullptr;
    uint32_t width = 0, height = 0, row_pitch = 0;
    /* The valid payload of one row, kept apart from the pitch so a plane whose
     * samples are wider than RGBA8 copies whole rows. An RGBA8 bridge has
     * bytes_per_row == width * 4, which is exactly what the older entry points
     * always moved; every copy, validation and byte count reads this field. */
    uint32_t bytes_per_row = 0;
    uint32_t pixel_bytes = 4;
    uint64_t fence_value = 0;
};

static bool prepare_motion_pipeline(NvofD3D12FlowBridge* b) {
    if (b->pipeline) return true;
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
    ID3DBlob* compiler_error = nullptr;
    HRESULT hr = D3DCompile(shader, sizeof(shader) - 1, "nvof_motion", nullptr,
        nullptr, "main", "cs_5_0", D3DCOMPILE_ENABLE_STRICTNESS, 0,
        &bytecode, &compiler_error);
    if (FAILED(hr)) {
        fail("D3DCompile motion conversion", hr);
        if (compiler_error) {
            std::snprintf(error_text, sizeof(error_text), "D3DCompile: %s",
                static_cast<const char*>(compiler_error->GetBufferPointer()));
        }
        if (compiler_error) compiler_error->Release();
        return false;
    }
    if (compiler_error) compiler_error->Release();
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
    if (SUCCEEDED(hr)) hr = b->device->CreateRootSignature(0,
        root_blob->GetBufferPointer(), root_blob->GetBufferSize(),
        IID_PPV_ARGS(&b->root));
    if (root_blob) root_blob->Release();
    if (SUCCEEDED(hr)) {
        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_desc{};
        pipeline_desc.pRootSignature = b->root;
        pipeline_desc.CS = {bytecode->GetBufferPointer(), bytecode->GetBufferSize()};
        hr = b->device->CreateComputePipelineState(&pipeline_desc,
            IID_PPV_ARGS(&b->pipeline));
    }
    bytecode->Release();
    if (SUCCEEDED(hr)) {
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = 2;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = b->device->CreateDescriptorHeap(&heap_desc,
            IID_PPV_ARGS(&b->descriptors));
    }
    if (FAILED(hr)) { fail("D3D12 motion pipeline", hr); return false; }
    auto cpu = b->descriptors->GetCPUDescriptorHandleForHeapStart();
    D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
    srv.Format = DXGI_FORMAT_R32_TYPELESS;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.NumElements = b->width * b->height * 2;
    srv.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
    b->device->CreateShaderResourceView(b->buffer, &srv, cpu);
    cpu.ptr += b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
    uav.Format = DXGI_FORMAT_R16G16_FLOAT;
    uav.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE2D;
    b->device->CreateUnorderedAccessView(b->motion, nullptr, &uav, cpu);
    return true;
}

extern "C" NVOF_D3D12_API uint32_t nvof_d3d12_abi_version(void) {
    return NVOF_D3D12_INTEROP_ABI_VERSION;
}

extern "C" NVOF_D3D12_API const char* nvof_d3d12_last_error(void) {
    return error_text;
}

extern "C" NVOF_D3D12_API void nvof_d3d12_destroy(NvofD3D12FlowBridge* b) {
    if (!b) return;
    CUcontext previous = nullptr;
    const bool pushed = b->cuda.push && b->context &&
        b->cuda.push(b->context) == CUDA_SUCCESS;
    if (pushed) {
        if (b->stream && b->cuda.stream_sync) b->cuda.stream_sync(b->stream);
        if (b->mapped && b->cuda.mem_free) b->cuda.mem_free(b->mapped);
        if (b->memory && b->cuda.destroy_memory) b->cuda.destroy_memory(b->memory);
        if (b->semaphore && b->cuda.destroy_semaphore)
            b->cuda.destroy_semaphore(b->semaphore);
        if (b->stream && b->cuda.stream_destroy) b->cuda.stream_destroy(b->stream);
        b->cuda.pop(&previous);
    }
    if (b->memory_handle) CloseHandle(b->memory_handle);
    if (b->fence_handle) CloseHandle(b->fence_handle);
    if (b->fence) b->fence->Release();
    if (b->pipeline) b->pipeline->Release();
    if (b->root) b->root->Release();
    if (b->descriptors) b->descriptors->Release();
    if (b->motion) b->motion->Release();
    if (b->buffer) b->buffer->Release();
    if (b->device) b->device->Release();
    if (b->cuda.module) FreeLibrary(b->cuda.module);
    delete b;
}

extern "C" NVOF_D3D12_API int32_t nvof_d3d12_create(
    void* raw_device, void* raw_context, uint32_t width, uint32_t height,
    NvofD3D12FlowBridge** out_bridge) {
    error_text[0] = 0;
    if (!raw_device || !raw_context || !width || !height || !out_bridge ||
        static_cast<uint64_t>(width) * height > UINT64_MAX / 8) return 1;
    *out_bridge = nullptr;
    auto* b = new (std::nothrow) NvofD3D12FlowBridge;
    if (!b) return 2;
    b->context = static_cast<CUcontext>(raw_context);
    b->device = static_cast<ID3D12Device*>(raw_device);
    b->device->AddRef();
    b->width = width;
    b->height = height;
    if (!load_driver(&b->cuda)) { nvof_d3d12_destroy(b); return 3; }
    CUcontext previous = nullptr;
    CUresult cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) { fail("cuCtxPushCurrent", cu); nvof_d3d12_destroy(b); return 4; }
    CUdevice cuda_device = 0;
    char cuda_luid[8]{};
    unsigned int node_mask = 0;
    cu = b->cuda.ctx_device(&cuda_device);
    if (cu == CUDA_SUCCESS) cu = b->cuda.device_luid(cuda_luid, &node_mask, cuda_device);
    const LUID dx_luid = b->device->GetAdapterLuid();
    const bool same_adapter = cu == CUDA_SUCCESS &&
        std::memcmp(cuda_luid, &dx_luid, sizeof(cuda_luid)) == 0;
    if (!same_adapter) fail("CUDA/D3D12 adapter LUID mismatch", cu);
    b->cuda.pop(&previous);
    if (!same_adapter) { nvof_d3d12_destroy(b); return 5; }

    const uint64_t bytes = static_cast<uint64_t>(width) * height * 8;
    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    HRESULT hr = b->device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_SHARED,
        &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b->buffer));
    if (SUCCEEDED(hr)) hr = b->device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&b->fence));
    if (SUCCEEDED(hr)) hr = b->device->CreateSharedHandle(
        b->buffer, nullptr, GENERIC_ALL, nullptr, &b->memory_handle);
    if (SUCCEEDED(hr)) hr = b->device->CreateSharedHandle(
        b->fence, nullptr, GENERIC_ALL, nullptr, &b->fence_handle);
    if (FAILED(hr)) { fail("D3D12 shared resource", hr); nvof_d3d12_destroy(b); return 6; }

    D3D12_RESOURCE_DESC motion_desc{};
    motion_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    motion_desc.Width = width;
    motion_desc.Height = height;
    motion_desc.DepthOrArraySize = 1;
    motion_desc.MipLevels = 1;
    motion_desc.Format = DXGI_FORMAT_R16G16_FLOAT;
    motion_desc.SampleDesc.Count = 1;
    motion_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    hr = b->device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE,
        &motion_desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
        IID_PPV_ARGS(&b->motion));
    if (FAILED(hr)) { fail("D3D12 motion texture", hr); nvof_d3d12_destroy(b); return 9; }

    const auto allocation = b->device->GetResourceAllocationInfo(0, 1, &desc);
    cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) { fail("cuCtxPushCurrent", cu); nvof_d3d12_destroy(b); return 7; }
    ExternalMemoryDesc memory_desc{};
    memory_desc.type = 5; // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
    memory_desc.handle.win32.handle = b->memory_handle;
    memory_desc.size = allocation.SizeInBytes;
    memory_desc.flags = 1; // CUDA_EXTERNAL_MEMORY_DEDICATED
    cu = b->cuda.import_memory(&b->memory, &memory_desc);
    if (cu == CUDA_SUCCESS) {
        ExternalBufferDesc mapped_desc{};
        mapped_desc.size = allocation.SizeInBytes;
        cu = b->cuda.map_buffer(&b->mapped, b->memory, &mapped_desc);
    }
    if (cu == CUDA_SUCCESS) {
        ExternalSemaphoreDesc semaphore_desc{};
        semaphore_desc.type = 4; // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        semaphore_desc.handle.win32.handle = b->fence_handle;
        cu = b->cuda.import_semaphore(&b->semaphore, &semaphore_desc);
    }
    if (cu == CUDA_SUCCESS) cu = b->cuda.stream_create(&b->stream, 0);
    if (cu != CUDA_SUCCESS) fail("CUDA external resource import", cu);
    b->cuda.pop(&previous);
    if (cu != CUDA_SUCCESS) { nvof_d3d12_destroy(b); return 8; }
    *out_bridge = b;
    return 0;
}

extern "C" NVOF_D3D12_API int32_t nvof_d3d12_copy_flow(
    NvofD3D12FlowBridge* b, uint64_t flow_ptr, size_t flow_pitch,
    uint64_t* out_fence_value) {
    error_text[0] = 0;
    if (!b || !flow_ptr || !out_fence_value || flow_pitch < size_t(b->width) * 8)
        return 1;
    CUcontext previous = nullptr;
    CUresult cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) { fail("cuCtxPushCurrent", cu); return 2; }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = flow_ptr;
    copy.srcPitch = flow_pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = b->mapped;
    copy.dstPitch = size_t(b->width) * 8;
    copy.WidthInBytes = size_t(b->width) * 8;
    copy.Height = b->height;
    cu = b->cuda.memcpy_2d(&copy);
    if (cu == CUDA_SUCCESS) {
        ExternalSignalParams params{};
        params.params.fence.value = ++b->fence_value;
        cu = b->cuda.signal_semaphore(&b->semaphore, &params, 1, b->stream);
    }
    if (cu == CUDA_SUCCESS) cu = b->cuda.stream_sync(b->stream);
    if (cu != CUDA_SUCCESS) fail("CUDA flow copy/fence signal", cu);
    b->cuda.pop(&previous);
    if (cu != CUDA_SUCCESS) return 3;
    *out_fence_value = b->fence_value;
    return 0;
}

extern "C" NVOF_D3D12_API int32_t nvof_d3d12_get_resources(
    NvofD3D12FlowBridge* b, void** out_buffer, void** out_fence) {
    if (!b || !out_buffer || !out_fence) return 1;
    *out_buffer = b->buffer;
    *out_fence = b->fence;
    return 0;
}

extern "C" NVOF_D3D12_API void cuda_d3d12_color_destroy(
    CudaD3D12ColorBridge* b) {
    if (!b) return;
    CUcontext previous = nullptr;
    const bool pushed = b->cuda.push && b->cuda.pop && b->context &&
        b->cuda.push(b->context) == CUDA_SUCCESS;
    if (pushed) {
        if (b->stream && b->cuda.stream_sync) b->cuda.stream_sync(b->stream);
        if (b->mapped && b->cuda.mem_free) b->cuda.mem_free(b->mapped);
        if (b->memory && b->cuda.destroy_memory) b->cuda.destroy_memory(b->memory);
        if (b->semaphore && b->cuda.destroy_semaphore)
            b->cuda.destroy_semaphore(b->semaphore);
        if (b->stream && b->cuda.stream_destroy) b->cuda.stream_destroy(b->stream);
        b->cuda.pop(&previous);
    }
    if (b->memory_handle) CloseHandle(b->memory_handle);
    if (b->fence_handle) CloseHandle(b->fence_handle);
    if (b->fence) b->fence->Release();
    if (b->buffer) b->buffer->Release();
    if (b->device) b->device->Release();
    if (b->cuda.module) FreeLibrary(b->cuda.module);
    delete b;
}

static int32_t create_color_bridge(
    void* raw_device, void* raw_context, uint32_t width, uint32_t height,
    uint32_t row_pitch, uint32_t pixel_bytes, bool allow_uav,
    CudaD3D12ColorBridge** out_bridge) {
    error_text[0] = 0;
    if (out_bridge) *out_bridge = nullptr;
    /* The payload of one row is derived from the pixel width the caller named,
     * and every limit is checked in 64-bit: bytes_per_row feeds a memcpy width,
     * plane_bytes a buffer extent and row_pitch a placed footprint, so an
     * overflow here would become a short copy that still reports success. */
    if (pixel_bytes != 4 && pixel_bytes != 8) {
        fail("color row format", pixel_bytes);
        return 1;
    }
    if (!raw_device || !raw_context || !out_bridge || !width || !height ||
        width > UINT32_MAX / pixel_bytes ||
        row_pitch < width * pixel_bytes || (row_pitch & 255) ||
        uint64_t(row_pitch) * height > UINT64_MAX / 2)
        return 1;
    auto* b = new (std::nothrow) CudaD3D12ColorBridge;
    if (!b) return 2;
    b->context = static_cast<CUcontext>(raw_context);
    b->device = static_cast<ID3D12Device*>(raw_device);
    b->device->AddRef();
    b->width = width;
    b->height = height;
    b->row_pitch = row_pitch;
    b->pixel_bytes = pixel_bytes;
    b->bytes_per_row = width * pixel_bytes;
    if (!load_driver(&b->cuda)) { cuda_d3d12_color_destroy(b); return 3; }
    CUcontext previous = nullptr;
    CUresult cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) {
        fail("color cuCtxPushCurrent", cu);
        cuda_d3d12_color_destroy(b);
        return 4;
    }
    CUdevice cuda_device = 0;
    char cuda_luid[8]{};
    unsigned int node_mask = 0;
    cu = b->cuda.ctx_device(&cuda_device);
    if (cu == CUDA_SUCCESS) cu = b->cuda.device_luid(cuda_luid, &node_mask, cuda_device);
    const LUID dx_luid = b->device->GetAdapterLuid();
    const bool same_adapter = cu == CUDA_SUCCESS &&
        std::memcmp(cuda_luid, &dx_luid, sizeof(cuda_luid)) == 0;
    if (!same_adapter) fail("color CUDA/D3D12 adapter LUID mismatch", cu);
    b->cuda.pop(&previous);
    if (!same_adapter) { cuda_d3d12_color_destroy(b); return 5; }

    D3D12_HEAP_PROPERTIES props{};
    props.Type = D3D12_HEAP_TYPE_DEFAULT;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = uint64_t(row_pitch) * height;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    if (allow_uav) desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    HRESULT hr = b->device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_SHARED,
        &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b->buffer));
    if (SUCCEEDED(hr)) hr = b->device->CreateFence(
        0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&b->fence));
    if (SUCCEEDED(hr)) hr = b->device->CreateSharedHandle(
        b->buffer, nullptr, GENERIC_ALL, nullptr, &b->memory_handle);
    if (SUCCEEDED(hr)) hr = b->device->CreateSharedHandle(
        b->fence, nullptr, GENERIC_ALL, nullptr, &b->fence_handle);
    if (FAILED(hr)) { fail("color D3D12 shared resource", hr);
        cuda_d3d12_color_destroy(b); return 6; }

    const auto allocation = b->device->GetResourceAllocationInfo(0, 1, &desc);
    cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) { fail("color cuCtxPushCurrent", cu);
        cuda_d3d12_color_destroy(b); return 7; }
    ExternalMemoryDesc memory_desc{};
    memory_desc.type = 5; // CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
    memory_desc.handle.win32.handle = b->memory_handle;
    memory_desc.size = allocation.SizeInBytes;
    memory_desc.flags = 1; // CUDA_EXTERNAL_MEMORY_DEDICATED
    cu = b->cuda.import_memory(&b->memory, &memory_desc);
    if (cu == CUDA_SUCCESS) {
        ExternalBufferDesc mapped_desc{};
        mapped_desc.size = allocation.SizeInBytes;
        cu = b->cuda.map_buffer(&b->mapped, b->memory, &mapped_desc);
    }
    if (cu == CUDA_SUCCESS) {
        ExternalSemaphoreDesc semaphore_desc{};
        semaphore_desc.type = 4; // CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE
        semaphore_desc.handle.win32.handle = b->fence_handle;
        cu = b->cuda.import_semaphore(&b->semaphore, &semaphore_desc);
    }
    if (cu == CUDA_SUCCESS) cu = b->cuda.stream_create(&b->stream, 0);
    if (cu != CUDA_SUCCESS) fail("color CUDA external resource import", cu);
    b->cuda.pop(&previous);
    if (cu != CUDA_SUCCESS) { cuda_d3d12_color_destroy(b); return 8; }
    *out_bridge = b;
    return 0;
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_create(
    void* device, void* context, uint32_t width, uint32_t height,
    uint32_t pitch, CudaD3D12ColorBridge** out) {
    return create_color_bridge(device, context, width, height, pitch, 4, false,
                               out);
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_create_output(
    void* device, void* context, uint32_t width, uint32_t height,
    uint32_t pitch, CudaD3D12ColorBridge** out) {
    return create_color_bridge(device, context, width, height, pitch, 4, true,
                               out);
}

extern "C" NVOF_D3D12_API uint32_t cuda_d3d12_color_row_capabilities(void) {
    return CUDA_D3D12_COLOR_ROW_RGBA8 | CUDA_D3D12_COLOR_ROW_RGBA16F;
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_create2(
    void* device, void* context, uint32_t width, uint32_t height,
    uint32_t row_pitch, uint32_t pixel_bytes, int32_t allow_uav,
    CudaD3D12ColorBridge** out) {
    return create_color_bridge(device, context, width, height, row_pitch,
                               pixel_bytes, allow_uav != 0, out);
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_get_layout(
    CudaD3D12ColorBridge* b, CudaD3D12ColorLayout* out_layout) {
    if (!b || !out_layout || out_layout->struct_size != sizeof(*out_layout))
        return 1;
    *out_layout = CudaD3D12ColorLayout{out_layout->struct_size, b->width,
        b->height, b->row_pitch, b->bytes_per_row, b->pixel_bytes,
        uint64_t(b->row_pitch) * b->height};
    return 0;
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_get_device_pointer(
    CudaD3D12ColorBridge* b, uint64_t* out_ptr, uint32_t* out_pitch) {
    if (!b || !out_ptr || !out_pitch || !b->mapped) return 1;
    *out_ptr = b->mapped;
    *out_pitch = b->row_pitch;
    return 0;
}

/* Every row the bridge moves is the bridge's own payload width: the source pitch
 * has to be able to supply it, the destination pitch has to be able to hold it,
 * and the copy takes it from the first byte of the row to the last so a wide
 * sample's tail half arrives rather than being dropped at the old width * 4. */
extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_copy(
    CudaD3D12ColorBridge* b, uint64_t rgba_ptr, size_t rgba_pitch,
    uint64_t* out_fence_value) {
    error_text[0] = 0;
    if (!b || !rgba_ptr || !out_fence_value ||
        rgba_pitch < size_t(b->bytes_per_row))
        return 1;
    CUcontext previous = nullptr;
    CUresult cu = b->cuda.push(b->context);
    if (cu != CUDA_SUCCESS) { fail("color cuCtxPushCurrent", cu); return 2; }
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.srcDevice = rgba_ptr;
    copy.srcPitch = rgba_pitch;
    copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy.dstDevice = b->mapped;
    copy.dstPitch = b->row_pitch;
    copy.WidthInBytes = b->bytes_per_row;
    copy.Height = b->height;
    cu = b->cuda.memcpy_2d(&copy);
    if (cu == CUDA_SUCCESS) {
        ExternalSignalParams params{};
        params.params.fence.value = ++b->fence_value;
        cu = b->cuda.signal_semaphore(&b->semaphore, &params, 1, b->stream);
    }
    if (cu == CUDA_SUCCESS) cu = b->cuda.stream_sync(b->stream);
    if (cu != CUDA_SUCCESS) fail("CUDA color copy/fence signal", cu);
    b->cuda.pop(&previous);
    if (cu != CUDA_SUCCESS) return 3;
    *out_fence_value = b->fence_value;
    return 0;
}

extern "C" NVOF_D3D12_API int32_t cuda_d3d12_color_get_resources(
    CudaD3D12ColorBridge* b, void** out_buffer, void** out_fence) {
    if (!b || !out_buffer || !out_fence) return 1;
    *out_buffer = b->buffer;
    *out_fence = b->fence;
    return 0;
}

extern "C" NVOF_D3D12_API int32_t nvof_d3d12_record_motion(
    NvofD3D12FlowBridge* b, void* raw_list, void** out_texture) {
    error_text[0] = 0;
    if (!b || !raw_list || !out_texture || !b->fence_value) return 1;
    if (!prepare_motion_pipeline(b)) return 2;
    auto* list = static_cast<ID3D12GraphicsCommandList*>(raw_list);
    *out_texture = b->motion;
    D3D12_RESOURCE_BARRIER before[2]{};
    before[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    before[0].Transition.pResource = b->buffer;
    before[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    before[0].Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
    before[0].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    unsigned count = 1;
    if (b->motion_readable) {
        before[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        before[1].Transition.pResource = b->motion;
        before[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        before[1].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        before[1].Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        count = 2;
    }
    list->ResourceBarrier(count, before);
    ID3D12DescriptorHeap* heaps[] = {b->descriptors};
    list->SetDescriptorHeaps(1, heaps);
    list->SetComputeRootSignature(b->root);
    list->SetPipelineState(b->pipeline);
    auto gpu = b->descriptors->GetGPUDescriptorHandleForHeapStart();
    list->SetComputeRootDescriptorTable(0, gpu);
    gpu.ptr += b->device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    list->SetComputeRootDescriptorTable(1, gpu);
    const uint32_t extent[] = {b->width, b->height};
    list->SetComputeRoot32BitConstants(2, 2, extent, 0);
    list->Dispatch((b->width + 7) / 8, (b->height + 7) / 8, 1);
    D3D12_RESOURCE_BARRIER after[2]{};
    after[0].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    after[0].Transition.pResource = b->buffer;
    after[0].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    after[0].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    after[0].Transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
    after[1].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    after[1].Transition.pResource = b->motion;
    after[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    after[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    after[1].Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    list->ResourceBarrier(2, after);
    b->motion_readable = true;
    return 0;
}
