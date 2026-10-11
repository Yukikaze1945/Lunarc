#define NV12_RGBA_BRIDGE_EXPORTS
#include "nv12_rgba_bridge.h"
#include "include/cuda.h"

#include <windows.h>
#include <cstdio>
#include <new>

namespace {
thread_local char last_error[256] = "";

// Limited-range NV12. BT.601 coefficients match OpenCV's
// COLOR_YUV2RGBA_NV12 integer conversion. One thread writes one RGBA8 pixel.
const char kKernel[] = R"ptx(
.version 6.5
.target sm_50
.address_size 64
.visible .entry nv12_to_rgba(
    .param .u64 y_ptr, .param .u32 y_pitch,
    .param .u64 uv_ptr, .param .u32 uv_pitch,
    .param .u64 rgba_ptr, .param .u32 rgba_pitch,
    .param .u32 width, .param .u32 height,
    .param .s32 cy, .param .s32 crv, .param .s32 cgu,
    .param .s32 cgv, .param .s32 cbu)
{
    .reg .pred %p<3>;
    .reg .b32 %r<45>;
    .reg .b64 %rd<12>;
    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %tid.y;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    ld.param.u32 %r8, [width];
    ld.param.u32 %r9, [height];
    setp.ge.u32 %p0, %r3, %r8;
    setp.ge.u32 %p1, %r7, %r9;
    or.pred %p2, %p0, %p1;
    @%p2 bra done;
    ld.param.u64 %rd0, [y_ptr];
    ld.param.u32 %r10, [y_pitch];
    mad.wide.u32 %rd1, %r7, %r10, %rd0;
    cvt.u64.u32 %rd7, %r3;
    add.u64 %rd1, %rd1, %rd7;
    ld.global.u8 %r11, [%rd1];
    ld.param.u64 %rd2, [uv_ptr];
    ld.param.u32 %r12, [uv_pitch];
    shr.u32 %r13, %r7, 1;
    mad.wide.u32 %rd3, %r13, %r12, %rd2;
    and.b32 %r14, %r3, 0xfffffffe;
    cvt.u64.u32 %rd8, %r14;
    add.u64 %rd3, %rd3, %rd8;
    ld.global.u8 %r15, [%rd3];
    add.u64 %rd4, %rd3, 1;
    ld.global.u8 %r16, [%rd4];
    add.s32 %r11, %r11, -16;
    max.s32 %r11, %r11, 0;
    add.s32 %r15, %r15, -128;
    add.s32 %r16, %r16, -128;
    ld.param.s32 %r24, [cy];
    ld.param.s32 %r25, [crv];
    ld.param.s32 %r26, [cgu];
    ld.param.s32 %r27, [cgv];
    ld.param.s32 %r28, [cbu];
    mul.lo.s32 %r17, %r11, %r24;
    mad.lo.s32 %r18, %r16, %r25, %r17;
    add.s32 %r18, %r18, 524288;
    shr.s32 %r18, %r18, 20;
    mad.lo.s32 %r19, %r15, %r26, %r17;
    mad.lo.s32 %r19, %r16, %r27, %r19;
    add.s32 %r19, %r19, 524288;
    shr.s32 %r19, %r19, 20;
    mad.lo.s32 %r20, %r15, %r28, %r17;
    add.s32 %r20, %r20, 524288;
    shr.s32 %r20, %r20, 20;
    max.s32 %r18, %r18, 0;
    min.s32 %r18, %r18, 255;
    max.s32 %r19, %r19, 0;
    min.s32 %r19, %r19, 255;
    max.s32 %r20, %r20, 0;
    min.s32 %r20, %r20, 255;
    shl.b32 %r19, %r19, 8;
    shl.b32 %r20, %r20, 16;
    or.b32 %r21, %r18, %r19;
    or.b32 %r21, %r21, %r20;
    or.b32 %r21, %r21, 0xff000000;
    ld.param.u64 %rd5, [rgba_ptr];
    ld.param.u32 %r22, [rgba_pitch];
    mad.wide.u32 %rd6, %r7, %r22, %rd5;
    shl.b32 %r23, %r3, 2;
    cvt.u64.u32 %rd9, %r23;
    add.u64 %rd6, %rd6, %rd9;
    st.global.u32 [%rd6], %r21;
done:
    ret;
}
)ptx";

using Init = CUresult (CUDAAPI*)(unsigned);
using Push = CUresult (CUDAAPI*)(CUcontext);
using Pop = CUresult (CUDAAPI*)(CUcontext*);
using Load = CUresult (CUDAAPI*)(CUmodule*, const void*);
using Unload = CUresult (CUDAAPI*)(CUmodule);
using Function = CUresult (CUDAAPI*)(CUfunction*, CUmodule, const char*);
using Allocate = CUresult (CUDAAPI*)(CUdeviceptr*, size_t);
using Free = CUresult (CUDAAPI*)(CUdeviceptr);
using Launch = CUresult (CUDAAPI*)(CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
using Sync = CUresult (CUDAAPI*)();

struct Driver {
    HMODULE library = nullptr;
    Init init = nullptr;
    Push push = nullptr;
    Pop pop = nullptr;
    Load load = nullptr;
    Unload unload = nullptr;
    Function function = nullptr;
    Allocate allocate = nullptr;
    Free free = nullptr;
    Launch launch = nullptr;
    Sync sync = nullptr;
};

template<class T> bool symbol(HMODULE module, const char* name, T* out) {
    *out = reinterpret_cast<T>(GetProcAddress(module, name));
    return *out != nullptr;
}

bool load_driver(Driver* d) {
    d->library = LoadLibraryW(L"nvcuda.dll");
    if (!d->library) return false;
    return symbol(d->library, "cuInit", &d->init) &&
        symbol(d->library, "cuCtxPushCurrent_v2", &d->push) &&
        symbol(d->library, "cuCtxPopCurrent_v2", &d->pop) &&
        symbol(d->library, "cuModuleLoadData", &d->load) &&
        symbol(d->library, "cuModuleUnload", &d->unload) &&
        symbol(d->library, "cuModuleGetFunction", &d->function) &&
        symbol(d->library, "cuMemAlloc_v2", &d->allocate) &&
        symbol(d->library, "cuMemFree_v2", &d->free) &&
        symbol(d->library, "cuLaunchKernel", &d->launch) &&
        symbol(d->library, "cuCtxSynchronize", &d->sync);
}
}

struct Nv12RgbaBridge {
    Driver driver;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    CUdeviceptr output = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t pitch = 0;
    int32_t coefficients[5] = {};
    Nv12RgbaTransferCounters counters{sizeof(Nv12RgbaTransferCounters)};
};

extern "C" NV12_RGBA_API uint32_t nv12_rgba_abi_version(void) {
    return NV12_RGBA_BRIDGE_ABI_VERSION;
}

extern "C" NV12_RGBA_API const char* nv12_rgba_last_error(void) {
    return last_error;
}

extern "C" NV12_RGBA_API void nv12_rgba_close(Nv12RgbaBridge* b) {
    if (!b) return;
    if (b->driver.push && b->driver.pop && b->context &&
        b->driver.push(b->context) == CUDA_SUCCESS) {
        if (b->output && b->driver.free) b->driver.free(b->output);
        if (b->module && b->driver.unload) b->driver.unload(b->module);
        CUcontext previous = nullptr;
        b->driver.pop(&previous);
    }
    if (b->driver.library) FreeLibrary(b->driver.library);
    delete b;
}

extern "C" NV12_RGBA_API int32_t nv12_rgba_open(
    void* cuda_context, uint32_t width, uint32_t height,
    Nv12RgbaBridge** out_bridge) {
    return nv12_rgba_open_matrix(cuda_context, width, height,
        NV12_RGBA_COLOR_MATRIX_BT601, out_bridge);
}

extern "C" NV12_RGBA_API int32_t nv12_rgba_open_matrix(
    void* cuda_context, uint32_t width, uint32_t height, int32_t color_matrix,
    Nv12RgbaBridge** out_bridge) {
    last_error[0] = 0;
    if (out_bridge) *out_bridge = nullptr;
    if (!out_bridge || !cuda_context || !width || !height ||
        (width & 1) || (height & 1) || width > UINT32_MAX / 4 ||
        uint64_t(width) * height * 4 > SIZE_MAX) {
        std::snprintf(last_error, sizeof(last_error), "invalid CUDA context or NV12 size");
        return -1;
    }
    if (color_matrix != NV12_RGBA_COLOR_MATRIX_BT601 &&
        color_matrix != NV12_RGBA_COLOR_MATRIX_BT709) {
        std::snprintf(last_error, sizeof(last_error), "unsupported NV12 color matrix: %d", color_matrix);
        return -1;
    }
    auto* b = new (std::nothrow) Nv12RgbaBridge;
    if (!b) return -1;
    b->context = static_cast<CUcontext>(cuda_context);
    b->width = width;
    b->height = height;
    b->pitch = width * 4;
    // Fixed point scale 2^20; round each pixel only after summing channels.
    // BT.709 limited range: 255/219, 255*1.5748/224,
    // -255*0.187324/224, -255*0.468124/224, 255*1.8556/224.
    const int32_t bt601[] = {1220542, 1673527, -409993, -852492, 2116026};
    const int32_t bt709[] = {1220945, 1879825, -223607, -558796, 2215014};
    const int32_t* selected = color_matrix == NV12_RGBA_COLOR_MATRIX_BT709 ? bt709 : bt601;
    for (int i = 0; i < 5; ++i) b->coefficients[i] = selected[i];
    if (!load_driver(&b->driver) || b->driver.init(0) != CUDA_SUCCESS) {
        std::snprintf(last_error, sizeof(last_error), "load CUDA driver");
        nv12_rgba_close(b);
        return -1;
    }
    CUresult rc = b->driver.push(b->context);
    if (rc != CUDA_SUCCESS) {
        std::snprintf(last_error, sizeof(last_error), "push CUDA context: %d", rc);
        b->context = nullptr;
        nv12_rgba_close(b);
        return -1;
    }
    rc = b->driver.load(&b->module, kKernel);
    if (rc == CUDA_SUCCESS) rc = b->driver.function(&b->kernel, b->module, "nv12_to_rgba");
    if (rc == CUDA_SUCCESS) rc = b->driver.allocate(&b->output,
        size_t(b->pitch) * height);
    CUcontext previous = nullptr;
    b->driver.pop(&previous);
    if (rc != CUDA_SUCCESS) {
        std::snprintf(last_error, sizeof(last_error), "initialize NV12 CUDA kernel: %d", rc);
        nv12_rgba_close(b);
        return -1;
    }
    *out_bridge = b;
    return 0;
}

extern "C" NV12_RGBA_API int32_t nv12_rgba_convert(Nv12RgbaBridge* b,
    uint64_t y_device_ptr, uint32_t y_pitch_bytes,
    uint64_t uv_device_ptr, uint32_t uv_pitch_bytes,
    uint64_t* out_device_ptr, uint32_t* out_pitch_bytes) {
    last_error[0] = 0;
    if (!b || !y_device_ptr || !uv_device_ptr ||
        y_pitch_bytes < b->width || uv_pitch_bytes < b->width ||
        !out_device_ptr || !out_pitch_bytes) {
        std::snprintf(last_error, sizeof(last_error), "invalid NV12 CUDA frame");
        return -1;
    }
    auto& d = b->driver;
    CUresult rc = d.push(b->context);
    if (rc != CUDA_SUCCESS) {
        std::snprintf(last_error, sizeof(last_error), "push CUDA context: %d", rc);
        return -1;
    }
    void* args[] = {&y_device_ptr, &y_pitch_bytes, &uv_device_ptr,
        &uv_pitch_bytes, &b->output, &b->pitch, &b->width, &b->height,
        &b->coefficients[0], &b->coefficients[1], &b->coefficients[2],
        &b->coefficients[3], &b->coefficients[4]};
    rc = d.launch(b->kernel, (b->width + 15) / 16, (b->height + 15) / 16, 1,
        16, 16, 1, 0, nullptr, args, nullptr);
    if (rc == CUDA_SUCCESS) rc = d.sync();
    CUcontext previous = nullptr;
    d.pop(&previous);
    if (rc != CUDA_SUCCESS) {
        std::snprintf(last_error, sizeof(last_error), "convert NV12 to RGBA: %d", rc);
        return -1;
    }
    // The two counts below describe the conversion just made, so they are
    // written after it succeeded and never before. The kernel writes one RGBA8
    // pixel per thread over the full frame: exactly width*height*4 bytes.
    b->counters.convert_count += 1;
    b->counters.output_device_write_bytes +=
        static_cast<uint64_t>(b->width) * b->height * 4;
    *out_device_ptr = b->output;
    *out_pitch_bytes = b->pitch;
    return 0;
}

extern "C" NV12_RGBA_API int32_t nv12_rgba_get_transfer_counters(
    Nv12RgbaBridge* b, Nv12RgbaTransferCounters* counters) {
    if (!b || !counters ||
        counters->struct_size != sizeof(Nv12RgbaTransferCounters)) return -1;
    *counters = b->counters;
    return 0;
}
