/*
 * nvof_bridge.cpp — High-performance Native Bridge implementation for NVIDIA Optical Flow
 *
 * Implements:
 * 1. ABGR8 input buffer format (OpenCV BGR24 -> ABGR8 explicit conversion)
 * 2. Fully GPU-accelerated post-processing: S10.5 -> float32 conversion and 2x2/4x4 bilinear upscale on CUDA kernel
 * 3. One GPU float flow tensor allocated at session create and reused by every frame, so
 *    the per-frame path performs no allocations. Host callers download it; the opt-in
 *    D3D12 backend consumes the device result through explicit CUDA interop.
 * 4. CUcontext ownership, thread serialization (CriticalSection), and explicit context push/pop
 * 5. Temporal-hints enable/reset flag for continuous export vs seeking
 * 6. An additive CUDA device-input path accepts decoded RGBA through a pitched
 *    cross-context copy. The host-pointer DLSSNR ABI still cannot take device pointers.
 */

#define NVOF_BRIDGE_EXPORTS
#include "nvof_bridge.h"

#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "include/cuda.h"
#include "include/nvOpticalFlowCommon.h"
#include "include/nvOpticalFlowCuda.h"

static __declspec(thread) char g_last_error[512] = {0};

static void set_error(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vsnprintf(g_last_error, sizeof(g_last_error) - 1, fmt, args);
    g_last_error[sizeof(g_last_error) - 1] = '\0';
    va_end(args);
}

/* CUDA Driver API Function Pointers */
typedef CUresult (CUDAAPI *PFN_cuInit)(unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuDeviceGet)(CUdevice *device, int ordinal);
typedef CUresult (CUDAAPI *PFN_cuDeviceGetName)(char *name, int len, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuDriverGetVersion)(int *driverVersion);
typedef CUresult (CUDAAPI *PFN_cuCtxCreate_v2)(CUcontext *pctx, unsigned int flags, CUdevice dev);
typedef CUresult (CUDAAPI *PFN_cuCtxDestroy_v2)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxPushCurrent_v2)(CUcontext ctx);
typedef CUresult (CUDAAPI *PFN_cuCtxPopCurrent_v2)(CUcontext *pctx);
typedef CUresult (CUDAAPI *PFN_cuCtxGetDevice)(CUdevice* device);
typedef CUresult (CUDAAPI *PFN_cuMemcpy2D_v2)(const CUDA_MEMCPY2D *pCopy);
typedef CUresult (CUDAAPI *PFN_cuMemcpy3DPeer)(const CUDA_MEMCPY3D_PEER* copy);
typedef CUresult (CUDAAPI *PFN_cuCtxSynchronize)(void);
typedef CUresult (CUDAAPI *PFN_cuModuleLoadData)(CUmodule *module, const void *image);
typedef CUresult (CUDAAPI *PFN_cuModuleUnload)(CUmodule module);
typedef CUresult (CUDAAPI *PFN_cuModuleGetFunction)(CUfunction *hfunc, CUmodule hmod, const char *name);
typedef CUresult (CUDAAPI *PFN_cuLaunchKernel)(CUfunction f,
                                               unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
                                               unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
                                               unsigned int sharedMemBytes, CUstream hStream,
                                               void **kernelParams, void **extra);
typedef CUresult (CUDAAPI *PFN_cuMemAlloc_v2)(CUdeviceptr *dptr, size_t bytesize);
typedef CUresult (CUDAAPI *PFN_cuMemFree_v2)(CUdeviceptr dptr);
typedef CUresult (CUDAAPI *PFN_cuStreamCreate)(CUstream *phStream, unsigned int Flags);
typedef CUresult (CUDAAPI *PFN_cuStreamDestroy_v2)(CUstream hStream);
typedef CUresult (CUDAAPI *PFN_cuStreamSynchronize)(CUstream hStream);

static HMODULE g_hCuda = NULL;
static HMODULE g_hNvof = NULL;

static PFN_cuInit               pfn_cuInit = NULL;
static PFN_cuDeviceGet          pfn_cuDeviceGet = NULL;
static PFN_cuDeviceGetName      pfn_cuDeviceGetName = NULL;
static PFN_cuDriverGetVersion   pfn_cuDriverGetVersion = NULL;
static PFN_cuCtxCreate_v2       pfn_cuCtxCreate = NULL;
static PFN_cuCtxDestroy_v2      pfn_cuCtxDestroy = NULL;
static PFN_cuCtxPushCurrent_v2  pfn_cuCtxPushCurrent = NULL;
static PFN_cuCtxPopCurrent_v2   pfn_cuCtxPopCurrent = NULL;
static PFN_cuCtxGetDevice       pfn_cuCtxGetDevice = NULL;
static PFN_cuMemcpy2D_v2        pfn_cuMemcpy2D = NULL;
static PFN_cuMemcpy3DPeer       pfn_cuMemcpy3DPeer = NULL;
static PFN_cuCtxSynchronize     pfn_cuCtxSynchronize = NULL;
static PFN_cuModuleLoadData     pfn_cuModuleLoadData = NULL;
static PFN_cuModuleUnload       pfn_cuModuleUnload = NULL;
static PFN_cuModuleGetFunction  pfn_cuModuleGetFunction = NULL;
static PFN_cuLaunchKernel       pfn_cuLaunchKernel = NULL;
static PFN_cuMemAlloc_v2        pfn_cuMemAlloc = NULL;
static PFN_cuMemFree_v2         pfn_cuMemFree = NULL;
static PFN_cuStreamCreate       pfn_cuStreamCreate = NULL;
static PFN_cuStreamDestroy_v2   pfn_cuStreamDestroy = NULL;
static PFN_cuStreamSynchronize  pfn_cuStreamSynchronize = NULL;

typedef NV_OF_STATUS (NVOFAPI *PFN_NvOFGetMaxSupportedApiVersion)(uint32_t* version);
typedef NV_OF_STATUS (NVOFAPI *PFN_NvOFAPICreateInstanceCuda)(uint32_t apiVer, NV_OF_CUDA_API_FUNCTION_LIST *functionList);

static PFN_NvOFGetMaxSupportedApiVersion pfn_NvOFGetMaxSupportedApiVersion = NULL;
static PFN_NvOFAPICreateInstanceCuda     pfn_NvOFAPICreateInstanceCuda = NULL;

/* Embedded CUDA PTX Kernel for GPU Flow Post-Processing (S10.5 -> float32 and 2x2/4x4 bilinear upscaling) */
static const char g_ptx_postprocess[] =
".version 6.5\n"
".target sm_50\n"
".address_size 64\n"
"\n"
".visible .entry nvof_postprocess_kernel(\n"
"    .param .u64 d_in_short2,\n"
"    .param .u64 in_pitch_bytes,\n"
"    .param .u32 in_w,\n"
"    .param .u32 in_h,\n"
"    .param .u64 d_out_float2,\n"
"    .param .u64 out_pitch_bytes,\n"
"    .param .u32 out_w,\n"
"    .param .u32 out_h,\n"
"    .param .u32 grid_size\n"
")\n"
"{\n"
"    .reg .pred %p<6>;\n"
"    .reg .b32 %r<35>;\n"
"    .reg .b64 %rd<35>;\n"
"    .reg .f32 %f<35>;\n"
"\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %ntid.x;\n"
"    mov.u32 %r2, %tid.x;\n"
"    mad.lo.s32 %r3, %r0, %r1, %r2;\n"
"\n"
"    mov.u32 %r4, %ctaid.y;\n"
"    mov.u32 %r5, %ntid.y;\n"
"    mov.u32 %r6, %tid.y;\n"
"    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
"\n"
"    ld.param.u32 %r8, [out_w];\n"
"    ld.param.u32 %r9, [out_h];\n"
"\n"
"    setp.ge.u32 %p0, %r3, %r8;\n"
"    setp.ge.u32 %p1, %r7, %r9;\n"
"    or.pred %p2, %p0, %p1;\n"
"    @%p2 ret;\n"
"\n"
"    ld.param.u64 %rd1, [d_in_short2];\n"
"    ld.param.u64 %rd2, [in_pitch_bytes];\n"
"    ld.param.u32 %r10, [in_w];\n"
"    ld.param.u32 %r11, [in_h];\n"
"    ld.param.u64 %rd3, [d_out_float2];\n"
"    ld.param.u64 %rd4, [out_pitch_bytes];\n"
"    ld.param.u32 %r12, [grid_size];\n"
"\n"
"    cvt.u64.u32 %rd5, %r7;\n"
"    mul.lo.u64 %rd7, %rd5, %rd4;\n"
"    add.u64 %rd8, %rd3, %rd7;\n"
"    cvt.u64.u32 %rd9, %r3;\n"
"    shl.b64 %rd10, %rd9, 3;\n"
"    add.u64 %rd11, %rd8, %rd10;\n"
"\n"
"    mov.f32 %f1, 0f3d000000;\n"
"\n"
"    setp.eq.u32 %p3, %r12, 1;\n"
"    @!%p3 bra BB_UPSCALE;\n"
"\n"
"    mul.lo.u64 %rd12, %rd5, %rd2;\n"
"    add.u64 %rd13, %rd1, %rd12;\n"
"    shl.b64 %rd14, %rd9, 2;\n"
"    add.u64 %rd15, %rd13, %rd14;\n"
"\n"
"    ld.global.v2.s16 {%r13, %r14}, [%rd15];\n"
"    cvt.rn.f32.s32 %f2, %r13;\n"
"    cvt.rn.f32.s32 %f3, %r14;\n"
"    mul.f32 %f4, %f2, %f1;\n"
"    mul.f32 %f5, %f3, %f1;\n"
"    st.global.v2.f32 [%rd11], {%f4, %f5};\n"
"    ret;\n"
"\n"
"BB_UPSCALE:\n"
"    cvt.rn.f32.u32 %f6, %r3;\n"
"    cvt.rn.f32.u32 %f7, %r7;\n"
"    cvt.rn.f32.u32 %f8, %r12;\n"
"    rcp.rn.f32 %f9, %f8;\n"
"    mul.f32 %f10, %f6, %f9;\n"
"    mul.f32 %f11, %f7, %f9;\n"
"\n"
"    cvt.rzi.s32.f32 %r15, %f10;\n"
"    cvt.rzi.s32.f32 %r16, %f11;\n"
"\n"
"    cvt.rn.f32.s32 %f12, %r15;\n"
"    sub.f32 %f13, %f10, %f12;\n"
"\n"
"    cvt.rn.f32.s32 %f14, %r16;\n"
"    sub.f32 %f15, %f11, %f14;\n"
"\n"
"    add.s32 %r17, %r15, 1;\n"
"    add.s32 %r18, %r10, -1;\n"
"    min.s32 %r19, %r17, %r18;\n"
"\n"
"    add.s32 %r20, %r16, 1;\n"
"    add.s32 %r21, %r11, -1;\n"
"    min.s32 %r22, %r20, %r21;\n"
"\n"
"    cvt.u64.s32 %rd16, %r16;\n"
"    mul.lo.u64 %rd17, %rd16, %rd2;\n"
"    add.u64 %rd18, %rd1, %rd17;\n"
"\n"
"    cvt.u64.s32 %rd19, %r22;\n"
"    mul.lo.u64 %rd20, %rd19, %rd2;\n"
"    add.u64 %rd21, %rd1, %rd20;\n"
"\n"
"    cvt.u64.s32 %rd22, %r15;\n"
"    shl.b64 %rd23, %rd22, 2;\n"
"    add.u64 %rd24, %rd18, %rd23;\n"
"    ld.global.v2.s16 {%r23, %r24}, [%rd24];\n"
"\n"
"    cvt.u64.s32 %rd25, %r19;\n"
"    shl.b64 %rd26, %rd25, 2;\n"
"    add.u64 %rd27, %rd18, %rd26;\n"
"    ld.global.v2.s16 {%r25, %r26}, [%rd27];\n"
"\n"
"    add.u64 %rd28, %rd21, %rd23;\n"
"    ld.global.v2.s16 {%r27, %r28}, [%rd28];\n"
"\n"
"    add.u64 %rd29, %rd21, %rd26;\n"
"    ld.global.v2.s16 {%r29, %r30}, [%rd29];\n"
"\n"
"    cvt.rn.f32.s32 %f16, %r23;\n"
"    cvt.rn.f32.s32 %f17, %r25;\n"
"    cvt.rn.f32.s32 %f18, %r27;\n"
"    cvt.rn.f32.s32 %f19, %r29;\n"
"\n"
"    mov.f32 %f20, 0f3f800000;\n"
"    sub.f32 %f21, %f20, %f13;\n"
"    sub.f32 %f22, %f20, %f15;\n"
"\n"
"    mul.f32 %f23, %f21, %f16;\n"
"    fma.rn.f32 %f24, %f13, %f17, %f23;\n"
"\n"
"    mul.f32 %f25, %f21, %f18;\n"
"    fma.rn.f32 %f26, %f13, %f19, %f25;\n"
"\n"
"    mul.f32 %f27, %f22, %f24;\n"
"    fma.rn.f32 %f28, %f15, %f26, %f27;\n"
"\n"
"    cvt.rn.f32.s32 %f16, %r24;\n"
"    cvt.rn.f32.s32 %f17, %r26;\n"
"    cvt.rn.f32.s32 %f18, %r28;\n"
"    cvt.rn.f32.s32 %f19, %r30;\n"
"\n"
"    mul.f32 %f23, %f21, %f16;\n"
"    fma.rn.f32 %f24, %f13, %f17, %f23;\n"
"\n"
"    mul.f32 %f25, %f21, %f18;\n"
"    fma.rn.f32 %f26, %f13, %f19, %f25;\n"
"\n"
"    mul.f32 %f27, %f22, %f24;\n"
"    fma.rn.f32 %f29, %f15, %f26, %f27;\n"
"\n"
"    mul.f32 %f4, %f28, %f1;\n"
"    mul.f32 %f5, %f29, %f1;\n"
"\n"
"    st.global.v2.f32 [%rd11], {%f4, %f5};\n"
"    ret;\n"
"}\n";

/* Embedded PTX for the BGR24 -> ABGR8 byte map (ABI 3).
 *
 * One thread per pixel: three byte loads from the staging row, then one 32-bit store whose
 * top byte is the constant 255. Nothing else - no colour matrix, no gamma, no normalisation,
 * no range conversion - because the surface has to hold exactly the bytes cv2.cvtColor(frame,
 * COLOR_BGR2RGBA) would have put there for the RGBA path to upload. In a little-endian 32-bit
 * word the stored byte order is R,G,B,A, so R lands in bits 0..7, G in 8..15, B in 16..23 and
 * 0xFF in 24..31: word = (B<<16) | (G<<8) | R | 0xFF000000.
 *
 * The loads are byte-wide because a staging row packs three bytes per pixel, so pixel x starts
 * at byte 3*x, which is only word-aligned every fourth pixel. The single store is word-wide
 * because the ABGR8 surface starts pixel x at byte 4*x; create() checks that surface pitch and
 * base address are multiples of four before this kernel can be reached. */
static const char g_ptx_bgr_map[] =
".version 6.5\n"
".target sm_50\n"
".address_size 64\n"
"\n"
".visible .entry nvof_bgr_to_abgr_kernel(\n"
"    .param .u64 d_bgr,\n"
"    .param .u64 bgr_pitch,\n"
"    .param .u64 d_abgr,\n"
"    .param .u64 abgr_pitch,\n"
"    .param .u32 w,\n"
"    .param .u32 h\n"
")\n"
"{\n"
"    .reg .pred %p<4>;\n"
"    .reg .b32 %r<16>;\n"
"    .reg .b64 %rd<16>;\n"
"\n"
"    mov.u32 %r0, %ctaid.x;\n"
"    mov.u32 %r1, %ntid.x;\n"
"    mov.u32 %r2, %tid.x;\n"
"    mad.lo.s32 %r3, %r0, %r1, %r2;\n"
"\n"
"    mov.u32 %r4, %ctaid.y;\n"
"    mov.u32 %r5, %ntid.y;\n"
"    mov.u32 %r6, %tid.y;\n"
"    mad.lo.s32 %r7, %r4, %r5, %r6;\n"
"\n"
"    ld.param.u32 %r8, [w];\n"
"    ld.param.u32 %r9, [h];\n"
"\n"
"    setp.ge.u32 %p0, %r3, %r8;\n"
"    setp.ge.u32 %p1, %r7, %r9;\n"
"    or.pred %p2, %p0, %p1;\n"
"    @%p2 ret;\n"
"\n"
"    ld.param.u64 %rd0, [d_bgr];\n"
"    ld.param.u64 %rd1, [bgr_pitch];\n"
"    ld.param.u64 %rd2, [d_abgr];\n"
"    ld.param.u64 %rd3, [abgr_pitch];\n"
"\n"
"    cvt.u64.u32 %rd4, %r7;\n"
"    mul.lo.u64 %rd5, %rd4, %rd1;\n"
"    add.u64 %rd6, %rd0, %rd5;\n"
"    cvt.u64.u32 %rd7, %r3;\n"
"    mov.u64 %rd8, 3;\n"
"    mul.lo.u64 %rd9, %rd7, %rd8;\n"
"    add.u64 %rd10, %rd6, %rd9;\n"
"\n"
"    ld.global.u8 %r10, [%rd10];\n"
"    ld.global.u8 %r11, [%rd10+1];\n"
"    ld.global.u8 %r12, [%rd10+2];\n"
"\n"
"    mul.lo.u64 %rd11, %rd4, %rd3;\n"
"    add.u64 %rd12, %rd2, %rd11;\n"
"    mov.u64 %rd13, 4;\n"
"    mul.lo.u64 %rd14, %rd7, %rd13;\n"
"    add.u64 %rd15, %rd12, %rd14;\n"
"\n"
"    shl.b32 %r10, %r10, 16;\n"
"    shl.b32 %r11, %r11, 8;\n"
"    or.b32 %r13, %r10, %r11;\n"
"    or.b32 %r13, %r13, %r12;\n"
"    or.b32 %r13, %r13, 4278190080;\n"
"    st.global.b32 [%rd15], %r13;\n"
"    ret;\n"
"}\n";

static int load_libraries(void) {
    if (g_hCuda && g_hNvof) return 1;

    if (!g_hCuda) {
        g_hCuda = LoadLibraryA("nvcuda.dll");
        if (!g_hCuda) {
            set_error("Failed to load nvcuda.dll (NVIDIA CUDA driver not installed or unavailable)");
            return 0;
        }
        pfn_cuInit = (PFN_cuInit)GetProcAddress(g_hCuda, "cuInit");
        pfn_cuDeviceGet = (PFN_cuDeviceGet)GetProcAddress(g_hCuda, "cuDeviceGet");
        pfn_cuDeviceGetName = (PFN_cuDeviceGetName)GetProcAddress(g_hCuda, "cuDeviceGetName");
        pfn_cuDriverGetVersion = (PFN_cuDriverGetVersion)GetProcAddress(g_hCuda, "cuDriverGetVersion");
        pfn_cuCtxCreate = (PFN_cuCtxCreate_v2)GetProcAddress(g_hCuda, "cuCtxCreate_v2");
        pfn_cuCtxDestroy = (PFN_cuCtxDestroy_v2)GetProcAddress(g_hCuda, "cuCtxDestroy_v2");
        pfn_cuCtxPushCurrent = (PFN_cuCtxPushCurrent_v2)GetProcAddress(g_hCuda, "cuCtxPushCurrent_v2");
        pfn_cuCtxPopCurrent = (PFN_cuCtxPopCurrent_v2)GetProcAddress(g_hCuda, "cuCtxPopCurrent_v2");
        pfn_cuCtxGetDevice = (PFN_cuCtxGetDevice)GetProcAddress(g_hCuda, "cuCtxGetDevice");
        pfn_cuMemcpy2D = (PFN_cuMemcpy2D_v2)GetProcAddress(g_hCuda, "cuMemcpy2D_v2");
        pfn_cuMemcpy3DPeer = (PFN_cuMemcpy3DPeer)GetProcAddress(g_hCuda, "cuMemcpy3DPeer");
        pfn_cuCtxSynchronize = (PFN_cuCtxSynchronize)GetProcAddress(g_hCuda, "cuCtxSynchronize");

        pfn_cuModuleLoadData = (PFN_cuModuleLoadData)GetProcAddress(g_hCuda, "cuModuleLoadData");
        pfn_cuModuleUnload = (PFN_cuModuleUnload)GetProcAddress(g_hCuda, "cuModuleUnload");
        pfn_cuModuleGetFunction = (PFN_cuModuleGetFunction)GetProcAddress(g_hCuda, "cuModuleGetFunction");
        pfn_cuLaunchKernel = (PFN_cuLaunchKernel)GetProcAddress(g_hCuda, "cuLaunchKernel");
        pfn_cuMemAlloc = (PFN_cuMemAlloc_v2)GetProcAddress(g_hCuda, "cuMemAlloc_v2");
        pfn_cuMemFree = (PFN_cuMemFree_v2)GetProcAddress(g_hCuda, "cuMemFree_v2");
        pfn_cuStreamCreate = (PFN_cuStreamCreate)GetProcAddress(g_hCuda, "cuStreamCreate");
        pfn_cuStreamDestroy = (PFN_cuStreamDestroy_v2)GetProcAddress(g_hCuda, "cuStreamDestroy_v2");
        pfn_cuStreamSynchronize = (PFN_cuStreamSynchronize)GetProcAddress(g_hCuda, "cuStreamSynchronize");

        if (!pfn_cuInit || !pfn_cuDeviceGet || !pfn_cuCtxCreate || !pfn_cuMemcpy2D ||
            !pfn_cuModuleLoadData || !pfn_cuLaunchKernel || !pfn_cuMemAlloc || !pfn_cuMemFree) {
            set_error("nvcuda.dll is missing required driver entry points");
            return 0;
        }
    }

    if (!g_hNvof) {
        g_hNvof = LoadLibraryA("nvofapi64.dll");
        if (!g_hNvof) {
            set_error("Failed to load nvofapi64.dll (NVIDIA Optical Flow driver library missing)");
            return 0;
        }
        pfn_NvOFGetMaxSupportedApiVersion = (PFN_NvOFGetMaxSupportedApiVersion)GetProcAddress(g_hNvof, "NvOFGetMaxSupportedApiVersion");
        pfn_NvOFAPICreateInstanceCuda = (PFN_NvOFAPICreateInstanceCuda)GetProcAddress(g_hNvof, "NvOFAPICreateInstanceCuda");

        if (!pfn_NvOFAPICreateInstanceCuda) {
            set_error("nvofapi64.dll is missing NvOFAPICreateInstanceCuda");
            return 0;
        }
    }
    return 1;
}

typedef struct NvOFBridgeSession {
    uint32_t width;
    uint32_t height;
    uint32_t grid_size;
    uint32_t out_w;
    uint32_t out_h;

    uint32_t device_index;
    CUcontext ctx;
    DWORD thread_owner_id;

    NvOFHandle hOf;
    NV_OF_CUDA_API_FUNCTION_LIST ofFn;

    NvOFGPUBufferHandle hIn;
    NvOFGPUBufferHandle hRef;
    NvOFGPUBufferHandle hOut;

    CUdeviceptr dIn;
    CUdeviceptr dRef;
    CUdeviceptr dOut;

    size_t pitch_in;
    size_t pitch_out;

    /* GPU Post-Processing (S10.5 -> float32 and GPU 2x2/4x4 Bilinear Upscale) */
    CUmodule modPostProcess;
    CUfunction fnPostProcess;
    CUstream stream;
    CUdeviceptr dFlowFloat;
    size_t pitch_flow_float;

    /* ABI 3, BGR input path: the byte-map kernel and the device rows it reads. Two staging
     * buffers rather than one because a reset frame has a previous and a current surface to
     * fill, and with a single buffer the second upload would have to wait for the first
     * convert to finish - a synchronize that belongs to neither the upload span nor the
     * convert span, and so would make both of them lie. */
    CUmodule modBgrMap;
    CUfunction fnBgrMap;
    CUdeviceptr dBgrCur;
    CUdeviceptr dBgrPrev;
    size_t bgr_staging_pitch;

    NvOFBridgeTiming last_timing;
    NvOFBridgeCounters counters;
    uint64_t device_input_copy_count;
    uint64_t device_input_bytes;
    CRITICAL_SECTION cs;
    int is_destroyed;
    int has_previous_gpu_frame;
} NvOFBridgeSession;

NVOF_API uint32_t nvof_bridge_get_abi_version(void) {
    return NVOF_BRIDGE_ABI_VERSION;
}

/* Called before every execute and once per session. calloc leaves zeros, and a zero in a
 * timing field reads as "this stage measured no time", so each stage starts unmeasured and
 * only the span that is actually timed overwrites it. */
static void reset_timing(NvOFBridgeTiming* t)
{
    t->upload_ms = NVOF_TIMING_UNMEASURED;
    t->submit_cpu_ms = NVOF_TIMING_UNMEASURED;
    t->dependency_wait_ms = NVOF_TIMING_UNMEASURED;
    t->download_ms = NVOF_TIMING_UNMEASURED;
    t->postprocess_wait_inclusive_ms = NVOF_TIMING_UNMEASURED;
    t->total_native_ms = NVOF_TIMING_UNMEASURED;
    /* Stays unmeasured on the RGBA-input entry points: they upload pixels a CPU already
     * reordered, so there is no byte-map launch to time. */
    t->input_convert_wait_ms = NVOF_TIMING_UNMEASURED;
}

NVOF_API const char* nvof_bridge_get_last_error(void) {
    return g_last_error;
}

NVOF_API int32_t nvof_bridge_probe(NvOFBridgeProbeInfo* out_info) {
    if (!out_info) {
        set_error("Null pointer provided to nvof_bridge_probe");
        return -1;
    }
    memset(out_info, 0, sizeof(NvOFBridgeProbeInfo));

    if (!load_libraries()) {
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    CUresult cu_res = pfn_cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuInit(0) failed with error code %d", cu_res);
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    CUdevice dev = 0;
    cu_res = pfn_cuDeviceGet(&dev, 0);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuDeviceGet failed (no CUDA device found)");
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    pfn_cuDeviceGetName(out_info->gpu_name, sizeof(out_info->gpu_name) - 1, dev);
    if (pfn_cuDriverGetVersion) {
        int drv_ver = 0;
        pfn_cuDriverGetVersion(&drv_ver);
        out_info->driver_version = (uint32_t)drv_ver;
    }

    if (pfn_NvOFGetMaxSupportedApiVersion) {
        pfn_NvOFGetMaxSupportedApiVersion(&out_info->max_api_version);
    }

    CUcontext temp_ctx = NULL;
    cu_res = pfn_cuCtxCreate(&temp_ctx, 0, dev);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuCtxCreate failed with code %d", cu_res);
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    NV_OF_CUDA_API_FUNCTION_LIST fnList;
    memset(&fnList, 0, sizeof(fnList));
    NV_OF_STATUS of_res = pfn_NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &fnList);
    if (of_res != NV_OF_SUCCESS) {
        pfn_cuCtxDestroy(temp_ctx);
        set_error("NvOFAPICreateInstanceCuda(0x%04x) failed with status %d", NV_OF_API_VERSION, of_res);
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    NvOFHandle temp_hOf = NULL;
    of_res = fnList.nvCreateOpticalFlowCuda(temp_ctx, &temp_hOf);
    if (of_res != NV_OF_SUCCESS) {
        pfn_cuCtxDestroy(temp_ctx);
        set_error("nvCreateOpticalFlowCuda failed with status %d", of_res);
        strncpy(out_info->error_message, g_last_error, sizeof(out_info->error_message) - 1);
        return -1;
    }

    /* Query grid sizes from NvOFGetCaps */
    uint32_t grid_count = 0;
    fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, NULL, &grid_count);
    if (grid_count > 0 && grid_count <= 16) {
        uint32_t grid_vals[16] = {0};
        fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_SUPPORTED_OUTPUT_GRID_SIZES, grid_vals, &grid_count);
        for (uint32_t i = 0; i < grid_count; ++i) {
            out_info->supported_grid_mask |= (1u << grid_vals[i]);
        }
    }

    /* Query min/max width and height */
    uint32_t size = 1;
    fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_WIDTH_MIN, &out_info->min_width, &size);
    size = 1;
    fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_HEIGHT_MIN, &out_info->min_height, &size);
    size = 1;
    fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_WIDTH_MAX, &out_info->max_width, &size);
    size = 1;
    fnList.nvOFGetCaps(temp_hOf, NV_OF_CAPS_HEIGHT_MAX, &out_info->max_height, &size);

    fnList.nvOFDestroy(temp_hOf);
    pfn_cuCtxDestroy(temp_ctx);

    out_info->is_available = 1;
    return 0;
}

NVOF_API int32_t nvof_bridge_create(
    uint32_t width,
    uint32_t height,
    uint32_t grid_size,
    uint32_t perf_level,
    void** out_session
) {
    if (!out_session) {
        set_error("Null out_session pointer");
        return -1;
    }
    *out_session = NULL;

    if (!load_libraries()) return -1;

    if (grid_size != 1 && grid_size != 2 && grid_size != 4) {
        set_error("Invalid grid_size %u. Supported grid sizes are 1, 2, 4", grid_size);
        return -1;
    }

    CUresult cu_res = pfn_cuInit(0);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuInit failed: %d", cu_res);
        return -1;
    }

    CUdevice dev = 0;
    cu_res = pfn_cuDeviceGet(&dev, 0);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuDeviceGet failed: %d", cu_res);
        return -1;
    }

    CUcontext ctx = NULL;
    cu_res = pfn_cuCtxCreate(&ctx, 0, dev);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuCtxCreate failed: %d", cu_res);
        return -1;
    }

    NvOFBridgeSession* sess = (NvOFBridgeSession*)calloc(1, sizeof(NvOFBridgeSession));
    if (!sess) {
        pfn_cuCtxDestroy(ctx);
        set_error("Out of host memory allocating NvOFBridgeSession");
        return -1;
    }

    sess->width = width;
    sess->height = height;
    sess->grid_size = grid_size;
    sess->out_w = (width + grid_size - 1) / grid_size;
    sess->out_h = (height + grid_size - 1) / grid_size;
    sess->device_index = (uint32_t)dev;
    sess->ctx = ctx;
    sess->thread_owner_id = GetCurrentThreadId();
    reset_timing(&sess->last_timing);
    InitializeCriticalSection(&sess->cs);

    NV_OF_STATUS of_res = pfn_NvOFAPICreateInstanceCuda(NV_OF_API_VERSION, &sess->ofFn);
    if (of_res != NV_OF_SUCCESS) {
        set_error("NvOFAPICreateInstanceCuda failed: status %d", of_res);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    of_res = sess->ofFn.nvCreateOpticalFlowCuda(ctx, &sess->hOf);
    if (of_res != NV_OF_SUCCESS) {
        set_error("nvCreateOpticalFlowCuda failed: status %d", of_res);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    NV_OF_PERF_LEVEL p_lvl = NV_OF_PERF_LEVEL_FAST;
    if (perf_level == 5) p_lvl = NV_OF_PERF_LEVEL_SLOW;
    else if (perf_level == 10) p_lvl = NV_OF_PERF_LEVEL_MEDIUM;
    else p_lvl = NV_OF_PERF_LEVEL_FAST;

    NV_OF_INIT_PARAMS init_params;
    memset(&init_params, 0, sizeof(init_params));
    init_params.width = width;
    init_params.height = height;
    init_params.outGridSize = (NV_OF_OUTPUT_VECTOR_GRID_SIZE)grid_size;
    init_params.hintGridSize = (NV_OF_HINT_VECTOR_GRID_SIZE)grid_size;
    init_params.mode = NV_OF_MODE_OPTICALFLOW;
    init_params.perfLevel = p_lvl;
    init_params.enableExternalHints = NV_OF_FALSE;
    init_params.enableOutputCost = NV_OF_FALSE;
    init_params.hPrivData = NULL;
    init_params.disparityRange = NV_OF_STEREO_DISPARITY_RANGE_UNDEFINED;
    init_params.enableRoi = NV_OF_FALSE;

    of_res = sess->ofFn.nvOFInit(sess->hOf, &init_params);
    if (of_res != NV_OF_SUCCESS) {
        char err_buf[256] = {0};
        uint32_t err_size = sizeof(err_buf);
        sess->ofFn.nvOFGetLastError(sess->hOf, err_buf, &err_size);
        set_error("nvOFInit failed (status %d): %s", of_res, err_buf);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    const char* perf_str = (p_lvl == NV_OF_PERF_LEVEL_SLOW) ? "SLOW" :
                           (p_lvl == NV_OF_PERF_LEVEL_MEDIUM) ? "MEDIUM" : "FAST";
    printf("[NVOF init] input=%ux%u grid=%u output=%ux%u preset=%s (NV_OF_INIT_PARAMS.outGridSize=%u, perfLevel=%u)\n",
           width, height, grid_size, sess->out_w, sess->out_h, perf_str,
           (uint32_t)init_params.outGridSize, (uint32_t)init_params.perfLevel);
    fflush(stdout);

    /* Allocate GPU buffers with ABGR8 input format (4 bytes/pixel) */
    NV_OF_BUFFER_DESCRIPTOR desc_in;
    memset(&desc_in, 0, sizeof(desc_in));
    desc_in.width = width;
    desc_in.height = height;
    desc_in.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;
    desc_in.bufferFormat = NV_OF_BUFFER_FORMAT_ABGR8;

    of_res = sess->ofFn.nvOFCreateGPUBufferCuda(sess->hOf, &desc_in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &sess->hIn);
    if (of_res != NV_OF_SUCCESS) {
        set_error("nvOFCreateGPUBufferCuda(hIn ABGR8) failed: %d", of_res);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    of_res = sess->ofFn.nvOFCreateGPUBufferCuda(sess->hOf, &desc_in, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &sess->hRef);
    if (of_res != NV_OF_SUCCESS) {
        set_error("nvOFCreateGPUBufferCuda(hRef ABGR8) failed: %d", of_res);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    NV_OF_BUFFER_DESCRIPTOR desc_out;
    memset(&desc_out, 0, sizeof(desc_out));
    desc_out.width = sess->out_w;
    desc_out.height = sess->out_h;
    desc_out.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;
    desc_out.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2;

    of_res = sess->ofFn.nvOFCreateGPUBufferCuda(sess->hOf, &desc_out, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &sess->hOut);
    if (of_res != NV_OF_SUCCESS) {
        set_error("nvOFCreateGPUBufferCuda(hOut) failed: %d", of_res);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hRef);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return of_res;
    }

    sess->dIn = sess->ofFn.nvOFGPUBufferGetCUdeviceptr(sess->hIn);
    sess->dRef = sess->ofFn.nvOFGPUBufferGetCUdeviceptr(sess->hRef);
    sess->dOut = sess->ofFn.nvOFGPUBufferGetCUdeviceptr(sess->hOut);

    NV_OF_CUDA_BUFFER_STRIDE_INFO stride_info;
    sess->ofFn.nvOFGPUBufferGetStrideInfo(sess->hIn, &stride_info);
    sess->pitch_in = stride_info.strideInfo[0].strideXInBytes;

    sess->ofFn.nvOFGPUBufferGetStrideInfo(sess->hOut, &stride_info);
    sess->pitch_out = stride_info.strideInfo[0].strideXInBytes;

    /* Initialize GPU Post-Processing Module & Kernel (PTX) */
    cu_res = pfn_cuModuleLoadData(&sess->modPostProcess, g_ptx_postprocess);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuModuleLoadData for GPU post-processing kernel failed: %d", cu_res);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hRef);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hOut);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return -1;
    }

    cu_res = pfn_cuModuleGetFunction(&sess->fnPostProcess, sess->modPostProcess, "nvof_postprocess_kernel");
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuModuleGetFunction(nvof_postprocess_kernel) failed: %d", cu_res);
        pfn_cuModuleUnload(sess->modPostProcess);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hRef);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hOut);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return -1;
    }

    /* Allocate GPU full-resolution float output buffer: width * height * 2 * sizeof(float) */
    sess->pitch_flow_float = sess->width * 2 * sizeof(float);
    size_t flow_float_bytes = sess->pitch_flow_float * sess->height;
    cu_res = pfn_cuMemAlloc(&sess->dFlowFloat, flow_float_bytes);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuMemAlloc for dFlowFloat (%zu bytes) failed: %d", flow_float_bytes, cu_res);
        pfn_cuModuleUnload(sess->modPostProcess);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hRef);
        sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hOut);
        sess->ofFn.nvOFDestroy(sess->hOf);
        pfn_cuCtxDestroy(ctx);
        free(sess);
        return -1;
    }

    if (pfn_cuStreamCreate) {
        pfn_cuStreamCreate(&sess->stream, 0);
    }

    /* ABI 3: everything the BGR input path needs, built once, here.
     *
     * The byte-map kernel stores one 32-bit word per pixel into the ABGR8 surface, which is
     * only defined if every row of that surface starts on a 4-byte boundary. The check runs
     * at create rather than per frame, so a driver that handed back an oddly pitched surface
     * fails the session with a message instead of faulting mid-export. The staging side needs
     * no such check because its reads are byte-wide - a BGR24 row starts at 3*x, which is
     * word-aligned only every fourth pixel. */
    if (sess->pitch_in % 4 != 0 || sess->dIn % 4 != 0 || sess->dRef % 4 != 0) {
        set_error("ABGR8 input surface cannot host the byte map: dIn=0x%llx dRef=0x%llx pitch=%zu",
                  (unsigned long long)sess->dIn, (unsigned long long)sess->dRef, sess->pitch_in);
        nvof_bridge_destroy(sess);
        return -1;
    }

    cu_res = pfn_cuModuleLoadData(&sess->modBgrMap, g_ptx_bgr_map);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuModuleLoadData for the BGR byte-map kernel failed: %d", cu_res);
        nvof_bridge_destroy(sess);
        return -1;
    }
    cu_res = pfn_cuModuleGetFunction(&sess->fnBgrMap, sess->modBgrMap, "nvof_bgr_to_abgr_kernel");
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuModuleGetFunction(nvof_bgr_to_abgr_kernel) failed: %d", cu_res);
        nvof_bridge_destroy(sess);
        return -1;
    }

    /* A staging row carries exactly the bytes that crossed PCIe - width*3, no padding - and
     * rows are spaced by a pitch padded up to a 64-byte multiple, because that is the
     * alignment a pitched copy into linear device memory is written against. The kernel reads
     * the pitch as a parameter and the copy only ever moves row_bytes, so the padding is
     * never read and never counted: h2d_bytes stays width*height*3 for any width, including
     * one that is not a multiple of 16. Two buffers, and no allocation anywhere else -
     * bgr_staging_alloc_count is the evidence that a frame did not cuMemAlloc its own. */
    sess->bgr_staging_pitch = ((size_t)sess->width * 3 + 63) & ~(size_t)63;
    const size_t staging_bytes = sess->bgr_staging_pitch * sess->height;
    cu_res = pfn_cuMemAlloc(&sess->dBgrCur, staging_bytes);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuMemAlloc for BGR staging (%zu bytes) failed: %d", staging_bytes, cu_res);
        nvof_bridge_destroy(sess);
        return -1;
    }
    sess->counters.bgr_staging_alloc_count++;
    cu_res = pfn_cuMemAlloc(&sess->dBgrPrev, staging_bytes);
    if (cu_res != CUDA_SUCCESS) {
        set_error("cuMemAlloc for reference BGR staging (%zu bytes) failed: %d",
                  staging_bytes, cu_res);
        nvof_bridge_destroy(sess);
        return -1;
    }
    sess->counters.bgr_staging_alloc_count++;

    sess->has_previous_gpu_frame = 0;

    *out_session = (void*)sess;
    return 0;
}

NVOF_API int32_t nvof_bridge_execute_timed(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
) {
    if (!session || !cur_abgr || !prev_abgr || !out_flow) {
        set_error("Invalid argument to nvof_bridge_execute_timed");
        return -1;
    }

    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session is already destroyed");
        return -1;
    }

    sess->thread_owner_id = GetCurrentThreadId();
    reset_timing(&sess->last_timing);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double inv_freq_ms = (freq.QuadPart > 0) ? (1000.0 / (double)freq.QuadPart) : 0.0;

    /* Explicit context activation for calling thread */
    LARGE_INTEGER t_start, t_end, t_wall_0;
    QueryPerformanceCounter(&t_wall_0);   /* total_native_ms spans the whole execute */
    CUcontext prev_ctx = NULL;
    if (pfn_cuCtxPushCurrent) {
        pfn_cuCtxPushCurrent(sess->ctx);
    }

    /* 1. Upload cur_abgr (inputFrame, cur, t) and prev_abgr (referenceFrame, prev, t-1) */
    QueryPerformanceCounter(&t_start);

    CUDA_MEMCPY2D copy_p;
    memset(&copy_p, 0, sizeof(copy_p));
    copy_p.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_p.srcPitch = sess->width * 4;
    copy_p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_p.dstPitch = sess->pitch_in;
    copy_p.WidthInBytes = sess->width * 4;
    copy_p.Height = sess->height;

    copy_p.srcHost = cur_abgr;
    copy_p.dstDevice = sess->dIn;
    CUresult cu_res = pfn_cuMemcpy2D(&copy_p);
    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuMemcpy2D (cur_abgr) failed: %d", cu_res);
        return -1;
    }

    copy_p.srcHost = prev_abgr;
    copy_p.dstDevice = sess->dRef;
    cu_res = pfn_cuMemcpy2D(&copy_p);
    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuMemcpy2D (prev_abgr) failed: %d", cu_res);
        return -1;
    }

    QueryPerformanceCounter(&t_end);
    sess->last_timing.upload_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 2. Execute hardware optical flow and synchronize */
    NV_OF_EXECUTE_INPUT_PARAMS exec_in;
    memset(&exec_in, 0, sizeof(exec_in));
    exec_in.inputFrame = sess->hIn;         /* cur (t) */
    exec_in.referenceFrame = sess->hRef;     /* prev (t-1) */
    exec_in.disableTemporalHints = reset_temporal_hints ? NV_OF_TRUE : NV_OF_FALSE;

    NV_OF_EXECUTE_OUTPUT_PARAMS exec_out;
    memset(&exec_out, 0, sizeof(exec_out));
    exec_out.outputBuffer = sess->hOut;

    QueryPerformanceCounter(&t_start);
    NV_OF_STATUS of_res = sess->ofFn.nvOFExecute(sess->hOf, &exec_in, &exec_out);
    QueryPerformanceCounter(&t_end);
    /* Submit only. nvOFExecute returns once the work is queued, so this number is CPU API
     * time and says nothing about how long the hardware took. */
    sess->last_timing.submit_cpu_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;
    if (of_res != NV_OF_SUCCESS) {
        char err_buf[256] = {0};
        uint32_t err_size = sizeof(err_buf);
        sess->ofFn.nvOFGetLastError(sess->hOf, err_buf, &err_size);
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("nvOFExecute failed (%d): %s", of_res, err_buf);
        return of_res;
    }

    /* The hardware completion wait, on its own, so it cannot be read as submit cost. */
    QueryPerformanceCounter(&t_start);
    if (pfn_cuCtxSynchronize) {
        pfn_cuCtxSynchronize();
    }
    QueryPerformanceCounter(&t_end);
    sess->last_timing.dependency_wait_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 3. GPU Post-Processing: Convert S10.5 -> float32 and 2x2/4x4 bilinear upscaling on CUDA kernel */
    QueryPerformanceCounter(&t_start);

    void* kernel_params[9] = {
        &sess->dOut,
        &sess->pitch_out,
        &sess->out_w,
        &sess->out_h,
        &sess->dFlowFloat,
        &sess->pitch_flow_float,
        &sess->width,
        &sess->height,
        &sess->grid_size
    };

    unsigned int block_x = 16, block_y = 16;
    unsigned int grid_x = (sess->width + block_x - 1) / block_x;
    unsigned int grid_y = (sess->height + block_y - 1) / block_y;

    cu_res = pfn_cuLaunchKernel(
        sess->fnPostProcess,
        grid_x, grid_y, 1,
        block_x, block_y, 1,
        0, sess->stream, kernel_params, NULL
    );

    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuLaunchKernel (nvof_postprocess_kernel) failed: %d", cu_res);
        return -1;
    }

    if (pfn_cuStreamSynchronize && sess->stream) {
        cu_res = pfn_cuStreamSynchronize(sess->stream);
    } else if (pfn_cuCtxSynchronize) {
        cu_res = pfn_cuCtxSynchronize();
    }
    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("NVOF postprocess synchronize failed: %d", cu_res);
        return -1;
    }

    QueryPerformanceCounter(&t_end);
    /* One fused kernel performs both the S10.5 -> float32 conversion and the upscale, so
     * neither has a span of its own; what is measurable here is launch plus the wait for
     * the stream, which is what this field reports. */
    sess->last_timing.postprocess_wait_inclusive_ms =
        (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 4. Download final full-resolution float32 flow tensor directly from GPU to host */
    QueryPerformanceCounter(&t_start);
    const uint32_t out_stride = (flow_stride_bytes == 0) ? (sess->width * 2 * sizeof(float)) : flow_stride_bytes;

    memset(&copy_p, 0, sizeof(copy_p));
    copy_p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_p.srcDevice = sess->dFlowFloat;
    copy_p.srcPitch = sess->pitch_flow_float;
    copy_p.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy_p.dstHost = out_flow;
    copy_p.dstPitch = out_stride;
    copy_p.WidthInBytes = sess->width * 2 * sizeof(float);
    copy_p.Height = sess->height;

    cu_res = pfn_cuMemcpy2D(&copy_p);
    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuMemcpy2D (dFlowFloat to host) failed: %d", cu_res);
        return -1;
    }

    if (pfn_cuCtxPopCurrent) {
        pfn_cuCtxPopCurrent(&prev_ctx);
    }
    QueryPerformanceCounter(&t_end);
    sess->last_timing.download_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    LARGE_INTEGER t_wall_1;
    QueryPerformanceCounter(&t_wall_1);
    /* Wall clock over the whole call, measured rather than added up: the stage spans leave
     * out the context push/pop, the session critical section and the argument checks. */
    sess->last_timing.total_native_ms = (double)(t_wall_1.QuadPart - t_wall_0.QuadPart) * inv_freq_ms;

    sess->counters.execute_count++;
    sess->counters.h2d_copy_count += 2;
    sess->counters.h2d_bytes += (uint64_t)sess->width * 4 * sess->height * 2;
    sess->counters.d2h_copy_count += 1;
    sess->counters.d2h_bytes += (uint64_t)sess->width * 2 * sizeof(float) * sess->height;
    if (reset_temporal_hints) {
        sess->counters.reset_count++;
    }

    if (out_timing) {
        *out_timing = sess->last_timing;
    }

    LeaveCriticalSection(&sess->cs);
    return 0;
}

/* Everything that happens after the two NVOF input surfaces hold this frame's pixels: the
 * hardware execute, the flow postprocess, the flow D2H, and the counters and timings that
 * belong to that tail. Both sequential entry points end here, so the only thing the BGR path
 * is allowed to have changed is how bytes reached the surfaces - a flow difference can come
 * from the input and nowhere else, which is what the legacy-versus-BGR parity gate asserts.
 *
 * The session critical section is held on entry and released on every path out of here, and
 * the context the caller pushed is popped here. */
static int32_t execute_after_inputs(NvOFBridgeSession* sess,
                                    float* out_flow,
                                    uint32_t flow_stride_bytes,
                                    int32_t reset_temporal_hints,
                                    double inv_freq_ms,
                                    LARGE_INTEGER t_wall_0,
                                    CUcontext prev_ctx,
                                    bool download_flow,
                                    NvOFBridgeTiming* out_timing)
{
    LARGE_INTEGER t_start, t_end;
    CUDA_MEMCPY2D copy_p;
    CUresult cu_res;

    /* 2. Execute hardware optical flow and synchronize */
    NV_OF_EXECUTE_INPUT_PARAMS exec_in;
    memset(&exec_in, 0, sizeof(exec_in));
    exec_in.inputFrame = sess->hIn;         /* cur (t) */
    exec_in.referenceFrame = sess->hRef;     /* prev (t-1) */
    exec_in.disableTemporalHints = reset_temporal_hints ? NV_OF_TRUE : NV_OF_FALSE;

    NV_OF_EXECUTE_OUTPUT_PARAMS exec_out;
    memset(&exec_out, 0, sizeof(exec_out));
    exec_out.outputBuffer = sess->hOut;

    QueryPerformanceCounter(&t_start);
    NV_OF_STATUS of_res = sess->ofFn.nvOFExecute(sess->hOf, &exec_in, &exec_out);
    if (of_res != NV_OF_SUCCESS) {
        char err_buf[256] = {0};
        uint32_t err_size = sizeof(err_buf);
        sess->ofFn.nvOFGetLastError(sess->hOf, err_buf, &err_size);
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("nvOFExecute sequential failed (%d): %s", of_res, err_buf);
        return of_res;
    }
    QueryPerformanceCounter(&t_end);
    sess->last_timing.submit_cpu_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 3. GPU Post-Processing: Convert S10.5 -> float32 and 2x2/4x4 bilinear upscaling on CUDA kernel */
    QueryPerformanceCounter(&t_start);

    void* kernel_params[9] = {
        &sess->dOut,
        &sess->pitch_out,
        &sess->out_w,
        &sess->out_h,
        &sess->dFlowFloat,
        &sess->pitch_flow_float,
        &sess->width,
        &sess->height,
        &sess->grid_size
    };

    unsigned int block_x = 16, block_y = 16;
    unsigned int grid_x = (sess->width + block_x - 1) / block_x;
    unsigned int grid_y = (sess->height + block_y - 1) / block_y;

    cu_res = pfn_cuLaunchKernel(
        sess->fnPostProcess,
        grid_x, grid_y, 1,
        block_x, block_y, 1,
        0, sess->stream, kernel_params, NULL
    );

    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuLaunchKernel sequential (nvof_postprocess_kernel) failed: %d", cu_res);
        return -1;
    }

    if (pfn_cuStreamSynchronize && sess->stream) {
        cu_res = pfn_cuStreamSynchronize(sess->stream);
    } else if (pfn_cuCtxSynchronize) {
        cu_res = pfn_cuCtxSynchronize();
    }
    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("NVOF postprocess synchronize failed: %d", cu_res);
        return -1;
    }

    QueryPerformanceCounter(&t_end);
    /* One fused kernel performs both the S10.5 -> float32 conversion and the upscale, so
     * neither has a span of its own; what is measurable here is launch plus the wait for
     * the stream, which is what this field reports. */
    sess->last_timing.postprocess_wait_inclusive_ms =
        (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 4. The device result is already complete after the stream synchronize above.
     * Host callers retain the existing D2H behavior; GPU callers skip it. */
    if (download_flow) {
        QueryPerformanceCounter(&t_start);
        const uint32_t out_stride = (flow_stride_bytes == 0) ?
            (sess->width * 2 * sizeof(float)) : flow_stride_bytes;

        memset(&copy_p, 0, sizeof(copy_p));
        copy_p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy_p.srcDevice = sess->dFlowFloat;
        copy_p.srcPitch = sess->pitch_flow_float;
        copy_p.dstMemoryType = CU_MEMORYTYPE_HOST;
        copy_p.dstHost = out_flow;
        copy_p.dstPitch = out_stride;
        copy_p.WidthInBytes = sess->width * 2 * sizeof(float);
        copy_p.Height = sess->height;

        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D sequential (dFlowFloat to host) failed: %d", cu_res);
            return -1;
        }
        QueryPerformanceCounter(&t_end);
        sess->last_timing.download_ms =
            (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;
    }

    if (pfn_cuCtxPopCurrent) {
        pfn_cuCtxPopCurrent(&prev_ctx);
    }
    LARGE_INTEGER t_wall_1;
    QueryPerformanceCounter(&t_wall_1);
    /* Wall clock over the whole call, measured rather than added up: the stage spans leave
     * out the context push/pop, the session critical section and the argument checks. */
    sess->last_timing.total_native_ms = (double)(t_wall_1.QuadPart - t_wall_0.QuadPart) * inv_freq_ms;

    sess->counters.execute_count++;
    if (download_flow) {
        sess->counters.d2h_copy_count += 1;
        sess->counters.d2h_bytes += (uint64_t)sess->width * 2 * sizeof(float) * sess->height;
    }

    if (out_timing) {
        *out_timing = sess->last_timing;
    }

    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_execute_sequential(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
) {
    if (!session || !cur_abgr || !out_flow) {
        set_error("Invalid argument to nvof_bridge_execute_sequential");
        return -1;
    }

    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session is already destroyed");
        return -1;
    }

    sess->thread_owner_id = GetCurrentThreadId();
    reset_timing(&sess->last_timing);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double inv_freq_ms = (freq.QuadPart > 0) ? (1000.0 / (double)freq.QuadPart) : 0.0;

    /* Explicit context activation for calling thread */
    LARGE_INTEGER t_start, t_end, t_wall_0;
    QueryPerformanceCounter(&t_wall_0);   /* total_native_ms spans the whole execute */
    CUcontext prev_ctx = NULL;
    if (pfn_cuCtxPushCurrent) {
        pfn_cuCtxPushCurrent(sess->ctx);
    }

    QueryPerformanceCounter(&t_start);

    CUDA_MEMCPY2D copy_p;
    memset(&copy_p, 0, sizeof(copy_p));
    copy_p.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_p.srcPitch = sess->width * 4;
    copy_p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_p.dstPitch = sess->pitch_in;
    copy_p.WidthInBytes = sess->width * 4;
    copy_p.Height = sess->height;

    CUresult cu_res = CUDA_SUCCESS;

    if (reset_temporal_hints || !sess->has_previous_gpu_frame) {
        /* Initial or discontinuous frame: upload both reference and current */
        copy_p.srcHost = prev_abgr ? prev_abgr : cur_abgr;
        copy_p.dstDevice = sess->dRef;
        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D (prev_abgr initial) failed: %d", cu_res);
            return -1;
        }

        copy_p.srcHost = cur_abgr;
        copy_p.dstDevice = sess->dIn;
        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D (cur_abgr initial) failed: %d", cu_res);
            return -1;
        }
        sess->has_previous_gpu_frame = 1;
        sess->counters.h2d_copy_count += 2;
        sess->counters.h2d_bytes += (uint64_t)sess->width * 4 * sess->height * 2;
        sess->counters.reset_count++;
    } else {
        /* Steady-state continuous sequential stream:
         * The previous frame t-1 is ALREADY on GPU in dIn!
         * Swap handles so old dIn becomes new dRef (reference frame for t),
         * and old dRef becomes new dIn (input frame for t).
         */
        NvOFGPUBufferHandle tmp_h = sess->hIn;
        sess->hIn = sess->hRef;
        sess->hRef = tmp_h;

        CUdeviceptr tmp_d = sess->dIn;
        sess->dIn = sess->dRef;
        sess->dRef = tmp_d;

        /* Upload ONLY cur_abgr (1 H2D upload per frame) */
        copy_p.srcHost = cur_abgr;
        copy_p.dstDevice = sess->dIn;
        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D (cur_abgr sequential) failed: %d", cu_res);
            return -1;
        }
        sess->counters.h2d_copy_count += 1;
        sess->counters.h2d_bytes += (uint64_t)sess->width * 4 * sess->height;
    }

    QueryPerformanceCounter(&t_end);
    sess->last_timing.upload_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    return execute_after_inputs(sess, out_flow, flow_stride_bytes, reset_temporal_hints,
                                inv_freq_ms, t_wall_0, prev_ctx, true, out_timing);
}

/* One launch of the byte map: staging rows of BGR24 in, one ABGR8 surface out. The launch is
 * counted where it succeeds rather than where it is asked for, so the number reports kernels
 * that ran on this session's GPU. */
static CUresult launch_bgr_to_abgr(NvOFBridgeSession* sess,
                                   CUdeviceptr d_bgr,
                                   CUdeviceptr d_abgr)
{
    void* kernel_params[6] = {
        &d_bgr,
        &sess->bgr_staging_pitch,
        &d_abgr,
        &sess->pitch_in,
        &sess->width,
        &sess->height
    };

    const unsigned int block_x = 16, block_y = 16;
    const unsigned int grid_x = (sess->width + block_x - 1) / block_x;
    const unsigned int grid_y = (sess->height + block_y - 1) / block_y;

    const CUresult cu_res = pfn_cuLaunchKernel(
        sess->fnBgrMap,
        grid_x, grid_y, 1,
        block_x, block_y, 1,
        0, sess->stream, kernel_params, NULL
    );
    if (cu_res == CUDA_SUCCESS) {
        sess->counters.input_convert_launch_count++;
    }
    return cu_res;
}

/* ABI 3: the same execute from BGR24 host frames.
 *
 * Only one thing changes: how the two NVOF input surfaces get their pixels. The RGBA path is
 * handed a frame a CPU has already reordered and uploads 4 bytes per pixel for a surface
 * whose fourth byte is a constant. This one uploads the 3-byte rows as they lie into
 * persistent device staging and lets the byte-map kernel fill the surfaces, so the CPU never
 * produces an NVOF copy and the PCIe payload is width*height*3. Everything after that is
 * execute_after_inputs, shared with the RGBA path, so the two can only differ by their input.
 *
 * The upload/convert order is why create() reserves two staging buffers. A reset frame fills
 * both surfaces; with a single buffer the second upload would have to wait for the first
 * convert to finish, and that wait belongs to neither upload_ms nor
 * input_convert_wait_ms - it would make both of them wrong. */
static int32_t execute_bgr_sequential_impl(
    void* session,
    const uint8_t* cur_bgr,
    uint32_t cur_stride,
    const uint8_t* prev_bgr,
    uint32_t prev_stride,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing,
    bool download_flow
) {
    if (!session || !cur_bgr || (download_flow && !out_flow)) {
        set_error("Invalid argument to nvof_bridge_execute_bgr_sequential");
        return -1;
    }

    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    const uint32_t row_bytes = sess->width * 3;
    if (cur_stride == 0) {
        cur_stride = row_bytes;
    }
    if (prev_stride == 0) {
        prev_stride = row_bytes;
    }
    if (cur_stride < row_bytes || (prev_bgr && prev_stride < row_bytes)) {
        set_error("BGR24 row too long for the given stride: need %u bytes, got cur %u prev %u",
                  row_bytes, cur_stride, prev_stride);
        return -1;
    }

    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session is already destroyed");
        return -1;
    }
    if (!sess->fnBgrMap || !sess->dBgrCur || !sess->dBgrPrev) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session has no BGR staging or byte-map kernel");
        return -1;
    }

    sess->thread_owner_id = GetCurrentThreadId();
    reset_timing(&sess->last_timing);

    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    double inv_freq_ms = (freq.QuadPart > 0) ? (1000.0 / (double)freq.QuadPart) : 0.0;

    LARGE_INTEGER t_start, t_end, t_wall_0;
    QueryPerformanceCounter(&t_wall_0);
    CUcontext prev_ctx = NULL;
    if (pfn_cuCtxPushCurrent) {
        pfn_cuCtxPushCurrent(sess->ctx);
    }

    const int do_reset = (reset_temporal_hints || !sess->has_previous_gpu_frame) ? 1 : 0;
    const uint64_t payload_bytes = (uint64_t)row_bytes * sess->height;
    CUresult cu_res = CUDA_SUCCESS;

    /* 1. Host -> device staging: one cuMemcpy2D per frame in steady state, two on a reset,
     * which is the same history behaviour the sequential RGBA path has. */
    QueryPerformanceCounter(&t_start);

    CUDA_MEMCPY2D copy_p;
    memset(&copy_p, 0, sizeof(copy_p));
    copy_p.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy_p.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_p.dstPitch = sess->bgr_staging_pitch;
    copy_p.WidthInBytes = row_bytes;
    copy_p.Height = sess->height;

    if (do_reset) {
        copy_p.srcHost = prev_bgr ? prev_bgr : cur_bgr;
        copy_p.srcPitch = prev_bgr ? prev_stride : cur_stride;
        copy_p.dstDevice = sess->dBgrPrev;
        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res == CUDA_SUCCESS) {
            copy_p.srcHost = cur_bgr;
            copy_p.srcPitch = cur_stride;
            copy_p.dstDevice = sess->dBgrCur;
            cu_res = pfn_cuMemcpy2D(&copy_p);
        }
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D (BGR24 initial pair) failed: %d", cu_res);
            return -1;
        }
        sess->has_previous_gpu_frame = 1;
        sess->counters.h2d_copy_count += 2;
        sess->counters.h2d_bytes += payload_bytes * 2;
        sess->counters.reset_count++;
    } else {
        /* The previous frame is already on the GPU in dIn: swap the handles and the
         * addresses so it becomes this frame's reference, then upload only cur. */
        NvOFGPUBufferHandle tmp_h = sess->hIn;
        sess->hIn = sess->hRef;
        sess->hRef = tmp_h;

        CUdeviceptr tmp_d = sess->dIn;
        sess->dIn = sess->dRef;
        sess->dRef = tmp_d;

        copy_p.srcHost = cur_bgr;
        copy_p.srcPitch = cur_stride;
        copy_p.dstDevice = sess->dBgrCur;
        cu_res = pfn_cuMemcpy2D(&copy_p);
        if (cu_res != CUDA_SUCCESS) {
            if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
            LeaveCriticalSection(&sess->cs);
            set_error("cuMemcpy2D (cur_bgr sequential) failed: %d", cu_res);
            return -1;
        }
        sess->counters.h2d_copy_count += 1;
        sess->counters.h2d_bytes += payload_bytes;
    }

    QueryPerformanceCounter(&t_end);
    sess->last_timing.upload_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    /* 2. Staging -> ABGR8 surfaces, then the wait that makes them safe for the hardware to
     * read. This span holds the launches and that synchronize and nothing else: not the copy
     * above, not the flow postprocess below. */
    QueryPerformanceCounter(&t_start);

    if (do_reset) {
        cu_res = launch_bgr_to_abgr(sess, sess->dBgrPrev, sess->dRef);
        if (cu_res == CUDA_SUCCESS) {
            cu_res = launch_bgr_to_abgr(sess, sess->dBgrCur, sess->dIn);
        }
    } else {
        cu_res = launch_bgr_to_abgr(sess, sess->dBgrCur, sess->dIn);
    }

    if (cu_res == CUDA_SUCCESS) {
        if (pfn_cuStreamSynchronize && sess->stream) {
            pfn_cuStreamSynchronize(sess->stream);
        } else if (pfn_cuCtxSynchronize) {
            pfn_cuCtxSynchronize();
        }
    } else {
        /* A surface this frame did not finish filling must not become next frame's reference,
         * so the session goes back to the discontinuous shape rather than carrying half a
         * frame through the history. */
        sess->has_previous_gpu_frame = 0;
    }

    QueryPerformanceCounter(&t_end);
    sess->last_timing.input_convert_wait_ms =
        (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;

    if (cu_res != CUDA_SUCCESS) {
        if (pfn_cuCtxPopCurrent) pfn_cuCtxPopCurrent(&prev_ctx);
        LeaveCriticalSection(&sess->cs);
        set_error("cuLaunchKernel (nvof_bgr_to_abgr_kernel) failed: %d", cu_res);
        return -1;
    }

    sess->counters.bgr_input_call_count++;

    return execute_after_inputs(sess, out_flow, flow_stride_bytes, reset_temporal_hints,
                                inv_freq_ms, t_wall_0, prev_ctx, download_flow, out_timing);
}

NVOF_API int32_t nvof_bridge_execute_bgr_sequential(
    void* session, const uint8_t* cur_bgr, uint32_t cur_stride,
    const uint8_t* prev_bgr, uint32_t prev_stride, float* out_flow,
    uint32_t flow_stride_bytes, int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing
) {
    return execute_bgr_sequential_impl(
        session, cur_bgr, cur_stride, prev_bgr, prev_stride, out_flow,
        flow_stride_bytes, reset_temporal_hints, out_timing, true);
}

NVOF_API int32_t nvof_bridge_execute_bgr_sequential_device(
    void* session, const uint8_t* cur_bgr, uint32_t cur_stride,
    const uint8_t* prev_bgr, uint32_t prev_stride,
    int32_t reset_temporal_hints, NvOFBridgeTiming* out_timing
) {
    return execute_bgr_sequential_impl(
        session, cur_bgr, cur_stride, prev_bgr, prev_stride, nullptr,
        0, reset_temporal_hints, out_timing, false);
}

NVOF_API int32_t nvof_bridge_execute_rgba_device_sequential(
    void* session, uint64_t cur_device_ptr, uint32_t cur_pitch_bytes,
    void* source_cuda_context, int32_t reset_temporal_hints,
    NvOFBridgeTiming* out_timing) {
    if (!session || !cur_device_ptr || !source_cuda_context ||
        !pfn_cuMemcpy3DPeer || !pfn_cuCtxGetDevice) {
        set_error("RGBA device input or CUDA peer-copy support is unavailable");
        return -1;
    }
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    const uint32_t row_bytes = sess->width * 4;
    if (cur_pitch_bytes < row_bytes || sess->pitch_in < row_bytes) {
        set_error("RGBA device row pitch is too small");
        return -1;
    }
    const CUcontext source = static_cast<CUcontext>(source_cuda_context);
    CUresult rc = pfn_cuCtxPushCurrent(source);
    if (rc != CUDA_SUCCESS) {
        set_error("push source CUDA context failed: %d", rc);
        return -1;
    }
    CUdevice source_device = -1;
    rc = pfn_cuCtxGetDevice(&source_device);
    CUcontext previous = NULL;
    pfn_cuCtxPopCurrent(&previous);
    if (rc != CUDA_SUCCESS || source_device != CUdevice(sess->device_index)) {
        set_error("RGBA source and NVOF CUDA devices differ");
        return -1;
    }
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session is already destroyed");
        return -1;
    }
    sess->thread_owner_id = GetCurrentThreadId();
    reset_timing(&sess->last_timing);
    LARGE_INTEGER freq, start;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&start);
    const double inv_freq_ms = freq.QuadPart > 0 ? 1000.0 / double(freq.QuadPart) : 0.0;
    previous = NULL;
    rc = pfn_cuCtxPushCurrent(sess->ctx);
    if (rc != CUDA_SUCCESS) {
        LeaveCriticalSection(&sess->cs);
        set_error("push NVOF CUDA context failed: %d", rc);
        return -1;
    }
    const int do_reset = reset_temporal_hints || !sess->has_previous_gpu_frame;
    if (!do_reset) {
        NvOFGPUBufferHandle old_input = sess->hIn;
        sess->hIn = sess->hRef;
        sess->hRef = old_input;
        CUdeviceptr old_pointer = sess->dIn;
        sess->dIn = sess->dRef;
        sess->dRef = old_pointer;
    }
    if (source == sess->ctx) {
        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = cur_device_ptr;
        copy.srcPitch = cur_pitch_bytes;
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = sess->dIn;
        copy.dstPitch = sess->pitch_in;
        copy.WidthInBytes = row_bytes;
        copy.Height = sess->height;
        rc = pfn_cuMemcpy2D(&copy);
        if (rc == CUDA_SUCCESS) {
            sess->device_input_copy_count++;
            sess->device_input_bytes += uint64_t(row_bytes) * sess->height;
        }
    } else {
        // CUDA contexts cannot share raw CUdeviceptr values. One pitched peer
        // copy names both contexts and copies the entire working frame.
        CUDA_MEMCPY3D_PEER copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = cur_device_ptr;
        copy.srcContext = source;
        copy.srcPitch = cur_pitch_bytes;
        copy.srcHeight = sess->height;
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = sess->dIn;
        copy.dstContext = sess->ctx;
        copy.dstPitch = sess->pitch_in;
        copy.dstHeight = sess->height;
        copy.WidthInBytes = row_bytes;
        copy.Height = sess->height;
        copy.Depth = 1;
        rc = pfn_cuMemcpy3DPeer(&copy);
        if (rc == CUDA_SUCCESS) {
            sess->device_input_copy_count++;
            sess->device_input_bytes += uint64_t(row_bytes) * sess->height;
        }
    }
    if (rc == CUDA_SUCCESS && do_reset) {
        CUDA_MEMCPY2D copy = {};
        copy.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.srcDevice = sess->dIn;
        copy.srcPitch = sess->pitch_in;
        copy.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        copy.dstDevice = sess->dRef;
        copy.dstPitch = sess->pitch_in;
        copy.WidthInBytes = row_bytes;
        copy.Height = sess->height;
        rc = pfn_cuMemcpy2D(&copy);
        if (rc == CUDA_SUCCESS) {
            sess->device_input_copy_count++;
            sess->device_input_bytes += uint64_t(row_bytes) * sess->height;
        }
    }
    if (rc == CUDA_SUCCESS) rc = pfn_cuCtxSynchronize();
    if (rc != CUDA_SUCCESS) {
        sess->has_previous_gpu_frame = 0;
        pfn_cuCtxPopCurrent(&previous);
        LeaveCriticalSection(&sess->cs);
        set_error("copy RGBA device input failed: %d", rc);
        return -1;
    }
    sess->has_previous_gpu_frame = 1;
    if (do_reset) sess->counters.reset_count++;
    return execute_after_inputs(sess, nullptr, 0, reset_temporal_hints,
        inv_freq_ms, start, previous, false, out_timing);
}

NVOF_API int32_t nvof_bridge_get_device_input_counters(
    void* session, uint64_t* out_copy_count, uint64_t* out_bytes) {
    if (!session || !out_copy_count || !out_bytes) return -1;
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    *out_copy_count = sess->device_input_copy_count;
    *out_bytes = sess->device_input_bytes;
    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_execute(
    void* session,
    const uint8_t* cur_abgr,
    const uint8_t* prev_abgr,
    float* out_flow,
    uint32_t flow_stride_bytes,
    int32_t reset_temporal_hints
) {
    return nvof_bridge_execute_timed(
        session, cur_abgr, prev_abgr, out_flow, flow_stride_bytes, reset_temporal_hints, NULL
    );
}

NVOF_API int32_t nvof_bridge_get_last_timing(
    void* session,
    NvOFBridgeTiming* out_timing
) {
    if (!session || !out_timing) return -1;
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    *out_timing = sess->last_timing;
    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_get_counters(
    void* session,
    NvOFBridgeCounters* out_counters
) {
    if (!session || !out_counters) return -1;
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        return -1;
    }
    *out_counters = sess->counters;
    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_get_session_info(
    void* session,
    uint32_t* out_device_idx,
    void** out_ctx,
    uint32_t* out_thread_owner_id
) {
    if (!session) return -1;
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        return -1;
    }
    if (out_device_idx) *out_device_idx = sess->device_index;
    if (out_ctx) *out_ctx = (void*)sess->ctx;
    if (out_thread_owner_id) *out_thread_owner_id = (uint32_t)sess->thread_owner_id;
    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_get_output_deviceptr(
    void* session,
    uint64_t* out_dptr,
    size_t* out_pitch_bytes
) {
    if (!session || !out_dptr) return -1;
    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        return -1;
    }
    *out_dptr = (uint64_t)sess->dFlowFloat;
    if (out_pitch_bytes) *out_pitch_bytes = sess->pitch_flow_float;
    LeaveCriticalSection(&sess->cs);
    return 0;
}

NVOF_API int32_t nvof_bridge_destroy(void* session) {
    if (!session) return 0;

    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        return 0;
    }
    sess->is_destroyed = 1;

    if (pfn_cuMemFree && sess->dFlowFloat) {
        pfn_cuMemFree(sess->dFlowFloat);
        sess->dFlowFloat = 0;
    }
    if (pfn_cuMemFree) {
        if (sess->dBgrCur) {
            pfn_cuMemFree(sess->dBgrCur);
            sess->dBgrCur = 0;
        }
        if (sess->dBgrPrev) {
            pfn_cuMemFree(sess->dBgrPrev);
            sess->dBgrPrev = 0;
        }
    }
    if (pfn_cuStreamDestroy && sess->stream) {
        pfn_cuStreamDestroy(sess->stream);
        sess->stream = NULL;
    }
    if (pfn_cuModuleUnload && sess->modPostProcess) {
        pfn_cuModuleUnload(sess->modPostProcess);
        sess->modPostProcess = NULL;
    }
    if (pfn_cuModuleUnload && sess->modBgrMap) {
        pfn_cuModuleUnload(sess->modBgrMap);
        sess->modBgrMap = NULL;
    }

    if (sess->ofFn.nvOFDestroyGPUBufferCuda) {
        if (sess->hIn) sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hIn);
        if (sess->hRef) sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hRef);
        if (sess->hOut) sess->ofFn.nvOFDestroyGPUBufferCuda(sess->hOut);
    }
    if (sess->ofFn.nvOFDestroy && sess->hOf) {
        sess->ofFn.nvOFDestroy(sess->hOf);
    }
    if (pfn_cuCtxDestroy && sess->ctx) {
        pfn_cuCtxDestroy(sess->ctx);
    }

    LeaveCriticalSection(&sess->cs);
    DeleteCriticalSection(&sess->cs);
    free(sess);
    return 0;
}

#ifdef NVOF_TEST_DEBUG_EXPORTS
/* Test build only, and only present in a DLL compiled with NVOF_TEST_DEBUG_EXPORTS: copy one
 * of the session's ABGR8 input surfaces back to the host so a test can compare the bytes the
 * byte-map kernel wrote against cv2.cvtColor(frame, COLOR_BGR2RGBA) byte for byte.
 *
 * That is the difference between a kernel whose source looks like a permutation and one whose
 * output is measured to be a permutation: the surface is what NVOF reads, so reading it back
 * is the only place the claim can actually be checked. A production build does not export
 * this, and no pipeline path ever reads an input surface back to the host.
 *
 * extern "C" because nothing declares it: every other entry point in this file gets C linkage
 * from its declaration in nvof_bridge.h, and a definition the file invents on its own would
 * otherwise be name-mangled and invisible to a caller looking it up by name - which is not an
 * error a test can distinguish from a build that forgot the macro. */
extern "C" {
NVOF_API int32_t nvof_bridge_test_read_input_surface(
    void* session, int32_t which, uint8_t* dst, uint32_t dst_stride)
{
    if (!session || !dst || (which != 0 && which != 1)) {
        set_error("Invalid argument to nvof_bridge_test_read_input_surface");
        return -1;
    }

    NvOFBridgeSession* sess = (NvOFBridgeSession*)session;
    EnterCriticalSection(&sess->cs);
    if (sess->is_destroyed) {
        LeaveCriticalSection(&sess->cs);
        set_error("Session is already destroyed");
        return -1;
    }

    const uint32_t row_bytes = sess->width * 4;
    const uint32_t stride = dst_stride ? dst_stride : (uint32_t)sess->pitch_in;
    if (stride < row_bytes) {
        LeaveCriticalSection(&sess->cs);
        set_error("dst stride %u too small for a %u byte ABGR row", stride, row_bytes);
        return -1;
    }

    CUDA_MEMCPY2D copy_p;
    memset(&copy_p, 0, sizeof(copy_p));
    copy_p.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    copy_p.srcDevice = which ? sess->dRef : sess->dIn;
    copy_p.srcPitch = sess->pitch_in;
    copy_p.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy_p.dstHost = dst;
    copy_p.dstPitch = stride;
    copy_p.WidthInBytes = row_bytes;
    copy_p.Height = sess->height;

    CUcontext prev_ctx = NULL;
    if (pfn_cuCtxPushCurrent) {
        pfn_cuCtxPushCurrent(sess->ctx);
    }
    const CUresult cu_res = pfn_cuMemcpy2D(&copy_p);
    if (pfn_cuCtxPopCurrent) {
        pfn_cuCtxPopCurrent(&prev_ctx);
    }
    LeaveCriticalSection(&sess->cs);

    if (cu_res != CUDA_SUCCESS) {
        set_error("cuMemcpy2D (input surface to host) failed: %d", cu_res);
        return -1;
    }
    return 0;
}
}  /* extern "C" */
#endif
