#define NVENC_BRIDGE_EXPORTS
#include "nvenc_bridge.h"
#include "nvof_d3d12_interop.h"
#include "include/cuda.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/mastering_display_metadata.h>
#include <libavutil/common.h>
#include <libavcodec/defs.h>
}

#include <windows.h>
#include <d3d12.h>
#include <wrl/client.h>
#include <cerrno>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <new>
#include <string>
#include <vector>

#include "nvenc_av1_metadata.h"

using Microsoft::WRL::ComPtr;

namespace {
thread_local char last_error[512] = "";

void fail(const char* step, long code) {
    std::snprintf(last_error, sizeof(last_error), "%s: 0x%08lx", step,
                  static_cast<unsigned long>(code));
}

void set_av_error(const char* step, int code) {
    char detail[AV_ERROR_MAX_STRING_SIZE] = "";
    av_strerror(code, detail, sizeof(detail));
    std::snprintf(last_error, sizeof(last_error), "%s: %s (%d)", step, detail, code);
}

uint32_t align256(uint32_t value) { return (value + 255u) & ~255u; }

// 16-bit fixed-point limited-range coefficients. The smoke recomputes the same
// integers on the host, so the kernel and its reference cannot drift silently.
struct Matrix { int32_t y[3]; int32_t u[3]; int32_t v[3]; };
constexpr Matrix k709 = {{11960, 40201, 4064},
                         {-6596, -22188, 28784},
                         {28784, -26148, -2639}};
constexpr Matrix k601 = {{16823, 33039, 6418},
                         {-9714, -19070, 28784},
                         {28784, -24104, -4681}};

// One thread converts a 2x2 RGBA8 block into four NV12 luma bytes and one
// interleaved UV pair. Hand-written PTX keeps the build free of a CUDA Toolkit.
const char kKernel[] = R"ptx(
.version 6.5
.target sm_50
.address_size 64
.visible .entry rgba_to_nv12(
    .param .u64 p_rgba, .param .u32 p_rgba_pitch,
    .param .u64 p_y, .param .u32 p_y_pitch,
    .param .u64 p_uv, .param .u32 p_uv_pitch,
    .param .u32 p_width, .param .u32 p_height,
    .param .u32 p_cyr, .param .u32 p_cyg, .param .u32 p_cyb,
    .param .u32 p_cur, .param .u32 p_cug, .param .u32 p_cub,
    .param .u32 p_cvr, .param .u32 p_cvg, .param .u32 p_cvb)
{
    .reg .pred %p<3>;
    .reg .b16 %h<4>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<20>;

    mov.u32 %r0, %ctaid.x;
    mov.u32 %r1, %ntid.x;
    mov.u32 %r2, %tid.x;
    mad.lo.u32 %r3, %r0, %r1, %r2;
    mov.u32 %r4, %ctaid.y;
    mov.u32 %r5, %ntid.y;
    mov.u32 %r6, %tid.y;
    mad.lo.u32 %r7, %r4, %r5, %r6;
    ld.param.u32 %r8, [p_width];
    ld.param.u32 %r9, [p_height];
    mul.lo.u32 %r10, %r3, 2;
    mul.lo.u32 %r11, %r7, 2;
    setp.ge.u32 %p0, %r10, %r8;
    setp.ge.u32 %p1, %r11, %r9;
    or.pred %p2, %p0, %p1;
    @%p2 bra done;

    ld.param.u32 %r50, [p_cyr];
    ld.param.u32 %r51, [p_cyg];
    ld.param.u32 %r52, [p_cyb];
    ld.param.u32 %r53, [p_cur];
    ld.param.u32 %r54, [p_cug];
    ld.param.u32 %r55, [p_cub];
    ld.param.u32 %r56, [p_cvr];
    ld.param.u32 %r57, [p_cvg];
    ld.param.u32 %r58, [p_cvb];

    mov.u32 %r63, 4;
    mov.u32 %r62, 2;
    ld.param.u32 %r12, [p_rgba_pitch];
    mul.wide.u32 %rd1, %r11, %r12;
    mul.wide.u32 %rd2, %r10, %r63;
    ld.param.u64 %rd0, [p_rgba];
    add.u64 %rd1, %rd1, %rd0;
    add.u64 %rd3, %rd1, %rd2;
    cvt.u64.u32 %rd4, %r12;
    add.u64 %rd5, %rd3, %rd4;

    ld.global.u32 %r20, [%rd3];
    ld.global.u32 %r21, [%rd3+4];
    ld.global.u32 %r22, [%rd5];
    ld.global.u32 %r23, [%rd5+4];

    mov.u32 %r30, 0;
    mov.u32 %r31, 0;
    mov.u32 %r32, 0;

    and.b32 %r24, %r20, 255;
    shr.u32 %r25, %r20, 8;
    and.b32 %r25, %r25, 255;
    shr.u32 %r26, %r20, 16;
    and.b32 %r26, %r26, 255;
    add.u32 %r30, %r30, %r24;
    add.u32 %r31, %r31, %r25;
    add.u32 %r32, %r32, %r26;
    mul.lo.s32 %r27, %r50, %r24;
    mad.lo.s32 %r27, %r51, %r25, %r27;
    mad.lo.s32 %r27, %r52, %r26, %r27;
    add.s32 %r27, %r27, 32768;
    shr.s32 %r27, %r27, 16;
    add.s32 %r27, %r27, 16;
    max.s32 %r40, %r27, 0;
    min.s32 %r40, %r40, 255;

    and.b32 %r24, %r21, 255;
    shr.u32 %r25, %r21, 8;
    and.b32 %r25, %r25, 255;
    shr.u32 %r26, %r21, 16;
    and.b32 %r26, %r26, 255;
    add.u32 %r30, %r30, %r24;
    add.u32 %r31, %r31, %r25;
    add.u32 %r32, %r32, %r26;
    mul.lo.s32 %r27, %r50, %r24;
    mad.lo.s32 %r27, %r51, %r25, %r27;
    mad.lo.s32 %r27, %r52, %r26, %r27;
    add.s32 %r27, %r27, 32768;
    shr.s32 %r27, %r27, 16;
    add.s32 %r27, %r27, 16;
    max.s32 %r41, %r27, 0;
    min.s32 %r41, %r41, 255;

    and.b32 %r24, %r22, 255;
    shr.u32 %r25, %r22, 8;
    and.b32 %r25, %r25, 255;
    shr.u32 %r26, %r22, 16;
    and.b32 %r26, %r26, 255;
    add.u32 %r30, %r30, %r24;
    add.u32 %r31, %r31, %r25;
    add.u32 %r32, %r32, %r26;
    mul.lo.s32 %r27, %r50, %r24;
    mad.lo.s32 %r27, %r51, %r25, %r27;
    mad.lo.s32 %r27, %r52, %r26, %r27;
    add.s32 %r27, %r27, 32768;
    shr.s32 %r27, %r27, 16;
    add.s32 %r27, %r27, 16;
    max.s32 %r42, %r27, 0;
    min.s32 %r42, %r42, 255;

    and.b32 %r24, %r23, 255;
    shr.u32 %r25, %r23, 8;
    and.b32 %r25, %r25, 255;
    shr.u32 %r26, %r23, 16;
    and.b32 %r26, %r26, 255;
    add.u32 %r30, %r30, %r24;
    add.u32 %r31, %r31, %r25;
    add.u32 %r32, %r32, %r26;
    mul.lo.s32 %r27, %r50, %r24;
    mad.lo.s32 %r27, %r51, %r25, %r27;
    mad.lo.s32 %r27, %r52, %r26, %r27;
    add.s32 %r27, %r27, 32768;
    shr.s32 %r27, %r27, 16;
    add.s32 %r27, %r27, 16;
    max.s32 %r43, %r27, 0;
    min.s32 %r43, %r43, 255;

    add.u32 %r30, %r30, %r62;
    shr.u32 %r30, %r30, 2;
    add.u32 %r31, %r31, %r62;
    shr.u32 %r31, %r31, 2;
    add.u32 %r32, %r32, %r62;
    shr.u32 %r32, %r32, 2;
    mul.lo.s32 %r27, %r53, %r30;
    mad.lo.s32 %r27, %r54, %r31, %r27;
    mad.lo.s32 %r27, %r55, %r32, %r27;
    add.s32 %r27, %r27, 32768;
    shr.s32 %r27, %r27, 16;
    add.s32 %r27, %r27, 128;
    max.s32 %r27, %r27, 0;
    min.s32 %r27, %r27, 255;
    mul.lo.s32 %r28, %r56, %r30;
    mad.lo.s32 %r28, %r57, %r31, %r28;
    mad.lo.s32 %r28, %r58, %r32, %r28;
    add.s32 %r28, %r28, 32768;
    shr.s32 %r28, %r28, 16;
    add.s32 %r28, %r28, 128;
    max.s32 %r28, %r28, 0;
    min.s32 %r28, %r28, 255;
    shl.b32 %r28, %r28, 8;
    or.b32 %r44, %r27, %r28;

    ld.param.u64 %rd10, [p_y];
    ld.param.u32 %r13, [p_y_pitch];
    mul.wide.u32 %rd11, %r11, %r13;
    add.u64 %rd11, %rd11, %rd10;
    cvt.u64.u32 %rd17, %r10;
    add.u64 %rd11, %rd11, %rd17;
    cvt.u64.u32 %rd12, %r13;
    add.u64 %rd13, %rd11, %rd12;
    shl.b32 %r27, %r41, 8;
    or.b32 %r27, %r27, %r40;
    cvt.u16.u32 %h0, %r27;
    st.global.u16 [%rd11], %h0;
    shl.b32 %r27, %r43, 8;
    or.b32 %r27, %r27, %r42;
    cvt.u16.u32 %h1, %r27;
    st.global.u16 [%rd13], %h1;

    ld.param.u64 %rd14, [p_uv];
    ld.param.u32 %r14, [p_uv_pitch];
    mul.wide.u32 %rd15, %r7, %r14;
    add.u64 %rd15, %rd15, %rd14;
    mul.wide.u32 %rd16, %r3, %r62;
    add.u64 %rd15, %rd15, %rd16;
    cvt.u16.u32 %h2, %r44;
    st.global.u16 [%rd15], %h2;
done:
    ret;
}
)ptx";

using CtxPush = CUresult (CUDAAPI*)(CUcontext);
using CtxPop = CUresult (CUDAAPI*)(CUcontext*);
using CtxDevice = CUresult (CUDAAPI*)(CUdevice*);
using DeviceLuid = CUresult (CUDAAPI*)(char*, unsigned int*, CUdevice);
using ModuleLoad = CUresult (CUDAAPI*)(CUmodule*, const void*);
using ModuleUnload = CUresult (CUDAAPI*)(CUmodule);
using GetFunction = CUresult (CUDAAPI*)(CUfunction*, CUmodule, const char*);
using Launch = CUresult (CUDAAPI*)(CUfunction, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, CUstream, void**, void**);
using CtxSync = CUresult (CUDAAPI*)();
// Pitched device-to-device plane copy, used only by the HDR10 entry. It is
// loaded separately so a driver without the symbol still opens SDR bridges.
using Memcpy2D = CUresult (CUDAAPI*)(const CUDA_MEMCPY2D*);

struct Driver {
    HMODULE module = nullptr;
    CtxPush push = nullptr;
    CtxPop pop = nullptr;
    CtxDevice ctx_device = nullptr;
    DeviceLuid device_luid = nullptr;
    ModuleLoad load = nullptr;
    ModuleUnload unload = nullptr;
    GetFunction function = nullptr;
    Launch launch = nullptr;
    CtxSync sync = nullptr;
    Memcpy2D memcpy_2d = nullptr;
};

template<class T> bool symbol(HMODULE module, const char* name, T* out) {
    *out = reinterpret_cast<T>(GetProcAddress(module, name));
    return *out != nullptr;
}

bool load_driver(Driver* d) {
    d->module = LoadLibraryW(L"nvcuda.dll");
    if (!d->module) { fail("LoadLibrary(nvcuda)", GetLastError()); return false; }
    symbol(d->module, "cuMemcpy2D_v2", &d->memcpy_2d);
    // The pitched copy is required only by the HDR10 entry; the SDR paths must
    // still open if a driver does not export it.
    return symbol(d->module, "cuCtxPushCurrent_v2", &d->push) &&
        symbol(d->module, "cuCtxPopCurrent_v2", &d->pop) &&
        symbol(d->module, "cuCtxGetDevice", &d->ctx_device) &&
        symbol(d->module, "cuDeviceGetLuid", &d->device_luid) &&
        symbol(d->module, "cuModuleLoadData", &d->load) &&
        symbol(d->module, "cuModuleUnload", &d->unload) &&
        symbol(d->module, "cuModuleGetFunction", &d->function) &&
        symbol(d->module, "cuLaunchKernel", &d->launch) &&
        symbol(d->module, "cuCtxSynchronize", &d->sync);
}

void barrier(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
             D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = resource;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    list->ResourceBarrier(1, &b);
}
}

struct NvencBridge {
    uint32_t width = 0, height = 0;
    uint32_t stage_pitch = 0;
    AVRational fps{0, 1};
    AVRational time_base{1, 1};
    Matrix matrix = k709;
    enum AVColorSpace colorspace = AVCOL_SPC_BT709;
    enum AVColorPrimaries primaries = AVCOL_PRI_BT709;
    enum AVColorTransferCharacteristic transfer = AVCOL_TRC_BT709;
    std::string codec_name;
    std::string output_path;
    NvencBridgeOptions options_copy{};

    Driver cuda;
    CUcontext context = nullptr;
    CUmodule module = nullptr;
    CUfunction kernel = nullptr;
    bool kernel_loaded = false;

    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    HANDLE event = nullptr;
    uint64_t fence_value = 0;
    CudaD3D12ColorBridge* staging = nullptr;
    ID3D12Resource* staging_buffer = nullptr;
    uint64_t staging_mapped = 0;

    AVBufferRef* hw_device = nullptr;
    AVBufferRef* hw_frames = nullptr;
    AVCodecContext* encoder = nullptr;
    AVFormatContext* format = nullptr;
    AVStream* stream = nullptr;
    AVFrame* frame = nullptr;
    AVPacket* packet = nullptr;

    bool header_written = false;
    bool failed = false;
    bool finished = false;
    bool wrote_trailer = false;
    NvencTransferCounters counters{0, 0, 0, 0, 0, 0};
    uint64_t last_y = 0, last_uv = 0;
    uint32_t last_y_pitch = 0, last_uv_pitch = 0;

    /* P010 entry state. Ordinary SDR paths keep their staging buffer, the
     * RGB->NV12 kernel and NV12 pool. hdr_mode adds HDR-only metadata and SEI. */
    bool p010_mode = false;
    bool hdr_mode = false;
    int32_t p010_color_range = AVCOL_RANGE_MPEG;
    int32_t p010_color_primaries = AVCOL_PRI_BT709;
    int32_t p010_color_trc = AVCOL_TRC_BT709;
    int32_t p010_color_space = AVCOL_SPC_BT709;
    NvencHdrMetadata hdr_metadata{0};
    NvencHdrCounters hdr_counters{0, 0, 0, 0, 0, 0, 0, 0};
    NvencHdrContract hdr_contract{0};
    NvencP010MetadataStatus metadata_status{0};
    /* In-bitstream metadata writer state. hdr_sei_wanted is only true when the
     * source declared something this writer can serialise; hdr_sei_blocked
     * records the honest refusal when it cannot. Both are decided once at the
     * open, because the declaration is static for the whole sequence. */
    bool hdr_sei_wanted = false;
    bool hdr_sei_blocked = false;
    NvencHdrSeiStatus hdr_sei{0};
    std::vector<uint8_t> hdr_sei_nal;
    std::vector<uint8_t> hdr_sei_rbsp;
    std::vector<uint8_t> hdr_sei_escaped;
    std::vector<uint8_t> hdr_sei_packet;
};

namespace {

// Defined with the HDR10 section below. Rewrites b->packet with an inserted
// prefix SEI NAL unit; only ever reached from an HDR bridge that was given
// metadata it could serialise, so the SDR packets stay exactly as received.
int hdr_sei_apply(NvencBridge* b);
int hdr_av1_apply(NvencBridge* b);

bool wait_gpu(NvencBridge* b) {    if (b->fence->GetCompletedValue() >= b->fence_value) return true;
    HRESULT hr = b->fence->SetEventOnCompletion(b->fence_value, b->event);
    if (FAILED(hr)) { fail("SetEventOnCompletion", hr); return false; }
    if (WaitForSingleObject(b->event, 30000) != WAIT_OBJECT_0) {
        fail("NVENC staging fence timeout", ERROR_TIMEOUT); return false;
    }
    return true;
}

// Borrowed CUcontext must be current while FFmpeg allocates or touches CUDA
// objects, so every such call is wrapped.
struct ContextScope {
    NvencBridge* b;
    CUcontext previous = nullptr;
    bool ok = false;
    explicit ContextScope(NvencBridge* bridge) : b(bridge) {
        if (!bridge->context || !bridge->cuda.push) return;
        ok = b->cuda.push(b->context) == CUDA_SUCCESS;
        if (!ok) fail("cuCtxPushCurrent", -1);
    }
    ~ContextScope() { if (ok) b->cuda.pop(&previous); }
};

int write_packet(NvencBridge* b) {
    while (true) {
        const int rc = avcodec_receive_packet(b->encoder, b->packet);
        if (rc == AVERROR(EAGAIN) || rc == AVERROR_EOF) return 0;
        if (rc < 0) { set_av_error("receive packet", rc); return rc; }
        /* Receive-packet-before-mux boundary. An SDR bridge, and an HDR bridge
         * whose source declared no metadata, never enter this branch. */
        if (b->hdr_mode && b->hdr_sei_wanted &&
            (b->codec_name == "av1_nvenc" ? hdr_av1_apply(b) : hdr_sei_apply(b)) < 0) return -1;
        av_packet_rescale_ts(b->packet, b->encoder->time_base,
                             b->stream->time_base);
        b->packet->stream_index = b->stream->index;
        const int written = av_interleaved_write_frame(b->format, b->packet);
        if (written < 0) { set_av_error("write packet", written); return written; }
        ++b->counters.packets_written;
    }
}

// Runs the shared RGB->NV12 kernel from a device RGBA source into the NV12
// planes of an already allocated CUDA frame, then waits for completion.
int convert(NvencBridge* b, uint64_t rgba_ptr, uint32_t rgba_pitch,
            uint64_t y_ptr, uint32_t y_pitch, uint64_t uv_ptr,
            uint32_t uv_pitch) {
    void* args[] = {&rgba_ptr, &rgba_pitch, &y_ptr, &y_pitch, &uv_ptr,
        &uv_pitch, &b->width, &b->height,
        const_cast<int32_t*>(&b->matrix.y[0]), const_cast<int32_t*>(&b->matrix.y[1]),
        const_cast<int32_t*>(&b->matrix.y[2]), const_cast<int32_t*>(&b->matrix.u[0]),
        const_cast<int32_t*>(&b->matrix.u[1]), const_cast<int32_t*>(&b->matrix.u[2]),
        const_cast<int32_t*>(&b->matrix.v[0]), const_cast<int32_t*>(&b->matrix.v[1]),
        const_cast<int32_t*>(&b->matrix.v[2])};
    const CUresult rc = b->cuda.launch(b->kernel,
        (b->width / 2 + 15) / 16, (b->height / 2 + 15) / 16, 1, 16, 16, 1,
        0, nullptr, args, nullptr);
    if (rc != CUDA_SUCCESS) { fail("cuLaunchKernel rgba_to_nv12", rc); return -1; }
    const CUresult synced = b->cuda.sync();
    if (synced != CUDA_SUCCESS) { fail("cuCtxSynchronize", synced); return -1; }
    return 0;
}

int encode_frame(NvencBridge* b, uint64_t rgba_ptr, uint32_t rgba_pitch,
                 int64_t pts) {
    ContextScope scope(b);
    if (!scope.ok) return -1;
    av_frame_unref(b->frame);
    b->frame->hw_frames_ctx = av_buffer_ref(b->hw_frames);
    if (!b->frame->hw_frames_ctx) { fail("ref frames", AVERROR(ENOMEM)); return -1; }
    b->frame->format = AV_PIX_FMT_CUDA;
    b->frame->width = static_cast<int>(b->width);
    b->frame->height = static_cast<int>(b->height);
    int rc = av_hwframe_get_buffer(b->hw_frames, b->frame, 0);
    if (rc < 0) { set_av_error("get CUDA frame", rc); return -1; }
    if (!b->frame->data[0] || !b->frame->data[1]) {
        fail("CUDA frame has no planes", 0); return -1;
    }
    const uint64_t y_ptr = reinterpret_cast<uintptr_t>(b->frame->data[0]);
    const uint64_t uv_ptr = reinterpret_cast<uintptr_t>(b->frame->data[1]);
    const uint32_t y_pitch = static_cast<uint32_t>(b->frame->linesize[0]);
    const uint32_t uv_pitch = static_cast<uint32_t>(b->frame->linesize[1]);
    if (convert(b, rgba_ptr, rgba_pitch, y_ptr, y_pitch, uv_ptr, uv_pitch) < 0)
        return -1;
    b->last_y = y_ptr;
    b->last_uv = uv_ptr;
    b->last_y_pitch = y_pitch;
    b->last_uv_pitch = uv_pitch;
    b->counters.conversion_output_bytes +=
        static_cast<uint64_t>(b->width) * b->height * 3 / 2;
    b->frame->pts = pts;
    b->frame->color_range = AVCOL_RANGE_MPEG;
    b->frame->colorspace = b->colorspace;
    b->frame->color_primaries = b->primaries;
    b->frame->color_trc = b->transfer;
    rc = avcodec_send_frame(b->encoder, b->frame);
    if (rc < 0) { set_av_error("send frame", rc); return -1; }
    ++b->counters.frames_submitted;
    return write_packet(b);
}

// D3D12 texture -> this bridge's CUDA-importable linear staging buffer. The
// caller's texture is only read; it must already be COPY_SOURCE.
int stage_texture(NvencBridge* b, ID3D12Resource* texture, ID3D12Fence* fence,
                  uint64_t fence_value) {
    const D3D12_RESOURCE_DESC desc = texture->GetDesc();
    if (desc.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D ||
        desc.Format != DXGI_FORMAT_R8G8B8A8_UNORM ||
        desc.Width != b->width || desc.Height != b->height ||
        desc.DepthOrArraySize != 1 || desc.MipLevels != 1 ||
        desc.SampleDesc.Count != 1) {
        fail("encoder input must be a matching RGBA8 2D texture", E_INVALIDARG);
        return -1;
    }
    ComPtr<ID3D12Device> texture_device;
    HRESULT hr = texture->GetDevice(IID_PPV_ARGS(texture_device.GetAddressOf()));
    if (FAILED(hr) || texture_device.Get() != b->device.Get()) {
        fail("encoder texture belongs to another D3D12 device", E_INVALIDARG);
        return -1;
    }
    if (fence && fence_value) {
        ComPtr<ID3D12Device> fence_device;
        hr = fence->GetDevice(IID_PPV_ARGS(fence_device.GetAddressOf()));
        if (FAILED(hr) || fence_device.Get() != b->device.Get()) {
            fail("producer fence belongs to another D3D12 device", E_INVALIDARG);
            return -1;
        }
    }
    if (fence && fence_value) {
        const HRESULT waited = b->queue->Wait(fence, fence_value);
        if (FAILED(waited)) { fail("queue wait producer fence", waited); return -1; }
    }
    hr = b->allocator->Reset();
    if (SUCCEEDED(hr)) hr = b->list->Reset(b->allocator.Get(), nullptr);
    if (FAILED(hr)) { fail("reset staging command list", hr); return -1; }
    auto* list = b->list.Get();
    barrier(list, b->staging_buffer, D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_COPY_DEST);
    D3D12_TEXTURE_COPY_LOCATION dst{}, src{};
    dst.pResource = b->staging_buffer;
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint.Offset = 0;
    dst.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    dst.PlacedFootprint.Footprint.Width = b->width;
    dst.PlacedFootprint.Footprint.Height = b->height;
    dst.PlacedFootprint.Footprint.Depth = 1;
    dst.PlacedFootprint.Footprint.RowPitch = b->stage_pitch;
    src.pResource = texture;
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    src.SubresourceIndex = 0;
    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    barrier(list, b->staging_buffer, D3D12_RESOURCE_STATE_COPY_DEST,
            D3D12_RESOURCE_STATE_COMMON);
    hr = list->Close();
    if (FAILED(hr)) { fail("close staging command list", hr); return -1; }
    ID3D12CommandList* commands[] = {list};
    b->queue->ExecuteCommandLists(1, commands);
    hr = b->queue->Signal(b->fence.Get(), ++b->fence_value);
    if (FAILED(hr)) { fail("signal staging fence", hr); return -1; }
    if (!wait_gpu(b)) return -1;
    b->counters.staging_gpu_copy_bytes +=
        static_cast<uint64_t>(b->width) * b->height * 4;
    return 0;
}

int configure_encoder(NvencBridge* b, const char* codec,
                      const NvencBridgeOptions* options) {
    const AVCodec* found = avcodec_find_encoder_by_name(codec);
    if (!found) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder %s not available in this FFmpeg build", codec);
        return -1;
    }
    b->encoder = avcodec_alloc_context3(found);
    if (!b->encoder) { fail("alloc encoder", AVERROR(ENOMEM)); return -1; }
    b->encoder->width = static_cast<int>(b->width);
    b->encoder->height = static_cast<int>(b->height);
    b->encoder->pix_fmt = AV_PIX_FMT_CUDA;
    b->encoder->sw_pix_fmt = AV_PIX_FMT_NV12;
    b->encoder->time_base = b->time_base;
    b->encoder->framerate = b->fps;
    b->encoder->sample_aspect_ratio = {1, 1};
    b->encoder->color_range = AVCOL_RANGE_MPEG;
    b->encoder->colorspace = b->colorspace;
    b->encoder->color_primaries = b->primaries;
    b->encoder->color_trc = b->transfer;
    b->encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    b->encoder->hw_frames_ctx = av_buffer_ref(b->hw_frames);
    if (!b->encoder->hw_frames_ctx) { fail("ref frames", AVERROR(ENOMEM)); return -1; }

    struct Option { const char* name; const char* value; };
    // NVENC keeps preset/tune/rc/multipass/cq in the codec private data, which
    // is only reachable through the AVCodecContext child class.
    const Option text[] = {
        {"preset", options->preset}, {"tune", options->tune},
        {"multipass", options->multipass}, {"rc", options->rate_control}};
    for (const auto& item : text) {
        if (!item.value || !*item.value) continue;
        const int rc = av_opt_set(b->encoder, item.name, item.value,
                                  AV_OPT_SEARCH_CHILDREN);
        if (rc < 0) {
            const std::string step = b->codec_name + ": " + item.name + "=" + item.value;
            set_av_error(step.c_str(), rc); return -1;
        }
    }
    if (options->constant_quality > 0) {
        const int rc = av_opt_set_int(b->encoder, "cq", options->constant_quality,
                                      AV_OPT_SEARCH_CHILDREN);
        if (rc < 0) { set_av_error("cq", rc); return -1; }
    }
    if (options->bitrate > 0) b->encoder->bit_rate = options->bitrate;
    if (options->gop > 0) b->encoder->gop_size = options->gop;

    // Anything the struct cannot model still has to reach the codec, or a
    // setting the user chose would quietly disappear from the encode.
    if (options->extra_options && *options->extra_options) {
        std::string remaining(options->extra_options);
        size_t start = 0;
        while (start < remaining.size()) {
            const size_t end = remaining.find('\n', start);
            const std::string pair = remaining.substr(
                start, end == std::string::npos ? std::string::npos : end - start);
            start = (end == std::string::npos) ? remaining.size() : end + 1;
            const size_t equals = pair.find('=');
            if (equals == std::string::npos || equals == 0) continue;
            const std::string name = pair.substr(0, equals);
            const std::string value = pair.substr(equals + 1);
            int rc = av_opt_set(b->encoder, name.c_str(), value.c_str(),
                                AV_OPT_SEARCH_CHILDREN);
            if (rc == AVERROR_OPTION_NOT_FOUND)
                rc = av_opt_set(b->encoder, name.c_str(), value.c_str(), 0);
            if (rc == AVERROR_OPTION_NOT_FOUND) {
                std::snprintf(last_error, sizeof(last_error),
                              "encoder %s has no option %s", codec, name.c_str());
                return -1;
            }
            if (rc < 0) {
                const std::string step = name + "=" + value;
                std::snprintf(last_error, sizeof(last_error), "%s", step.c_str());
                set_av_error(step.c_str(), rc);
                return -1;
            }
        }
    }

    const int rc = avcodec_open2(b->encoder, found, nullptr);
    if (rc < 0) { set_av_error("open NVENC encoder", rc); return -1; }
    if (b->encoder->pix_fmt != AV_PIX_FMT_CUDA) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder fell back to pixel format %s",
                      av_get_pix_fmt_name(b->encoder->pix_fmt));
        return -1;
    }
    return 0;
}

int open_muxer(NvencBridge* b, const char* path) {
    int rc = avformat_alloc_output_context2(&b->format, nullptr, nullptr, path);
    if (rc < 0 || !b->format) { set_av_error("alloc output context", rc); return -1; }
    b->stream = avformat_new_stream(b->format, nullptr);
    if (!b->stream) { fail("new stream", AVERROR(ENOMEM)); return -1; }
    b->stream->avg_frame_rate = b->fps;
    b->stream->r_frame_rate = b->fps;
    rc = avcodec_parameters_from_context(b->stream->codecpar, b->encoder);
    if (rc < 0) { set_av_error("copy codec parameters", rc); return -1; }
    if (!(b->format->oformat->flags & AVFMT_NOFILE)) {
        rc = avio_open(&b->format->pb, path, AVIO_FLAG_WRITE);
        if (rc < 0) { set_av_error("open output file", rc); return -1; }
    }
    rc = avformat_write_header(b->format, nullptr);
    if (rc < 0) { set_av_error("write header", rc); return -1; }
    b->header_written = true;
    return 0;
}

bool same_adapter(NvencBridge* b) {
    ContextScope scope(b);
    if (!scope.ok) return false;
    CUdevice cuda_device = 0;
    char cuda_luid[8]{};
    unsigned int node_mask = 0;
    CUresult rc = b->cuda.ctx_device(&cuda_device);
    if (rc == CUDA_SUCCESS) rc = b->cuda.device_luid(cuda_luid, &node_mask, cuda_device);
    if (rc != CUDA_SUCCESS) { fail("cuDeviceGetLuid", rc); return false; }
    const LUID dx_luid = b->device->GetAdapterLuid();
    if (std::memcmp(cuda_luid, &dx_luid, sizeof(cuda_luid)) != 0) {
        std::snprintf(last_error, sizeof(last_error),
                      "CUDA context and D3D12 device are different adapters");
        return false;
    }
    return true;
}

int create_command_graph(NvencBridge* b) {
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    HRESULT hr = b->device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&b->queue));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandAllocator(
        D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&b->allocator));
    if (SUCCEEDED(hr)) hr = b->device->CreateCommandList(0,
        D3D12_COMMAND_LIST_TYPE_DIRECT, b->allocator.Get(), nullptr,
        IID_PPV_ARGS(&b->list));
    if (SUCCEEDED(hr)) hr = b->list->Close();
    if (SUCCEEDED(hr)) hr = b->device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&b->fence));
    if (FAILED(hr)) { fail("create staging command graph", hr); return -1; }
    b->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!b->event) { fail("CreateEvent", GetLastError()); return -1; }
    return 0;
}

int create_staging(NvencBridge* b) {
    uint32_t pitch = 0;
    if (cuda_d3d12_color_create(b->device.Get(), b->context, b->width,
        b->height, b->stage_pitch, &b->staging)) {
        std::snprintf(last_error, sizeof(last_error),
            "color staging create failed: %s", nvof_d3d12_last_error());
        return -1;
    }
    void* buffer = nullptr;
    void* unused_fence = nullptr;
    if (cuda_d3d12_color_get_resources(b->staging, &buffer, &unused_fence)) {
        std::snprintf(last_error, sizeof(last_error),
            "color staging resources failed: %s", nvof_d3d12_last_error());
        return -1;
    }
    b->staging_buffer = static_cast<ID3D12Resource*>(buffer);
    if (cuda_d3d12_color_get_device_pointer(b->staging, &b->staging_mapped,
        &pitch)) {
        std::snprintf(last_error, sizeof(last_error),
            "color staging pointer failed: %s", nvof_d3d12_last_error());
        return -1;
    }
    if (pitch != b->stage_pitch) {
        std::snprintf(last_error, sizeof(last_error),
            "staging pitch mismatch (%u != %u)", pitch, b->stage_pitch);
        return -1;
    }
    return 0;
}

// FFmpeg is bound to the caller's CUcontext so the surfaces it allocates and
// the staging import live in one address space.
int create_ffmpeg(NvencBridge* b) {
    ContextScope scope(b);
    if (!scope.ok) return -1;
    int rc = av_hwdevice_ctx_create(&b->hw_device, AV_HWDEVICE_TYPE_CUDA,
        nullptr, nullptr, AV_CUDA_USE_CURRENT_CONTEXT);
    if (rc < 0) { set_av_error("create CUDA device context", rc); return -1; }
    b->hw_frames = av_hwframe_ctx_alloc(b->hw_device);
    if (!b->hw_frames) { fail("alloc frames", AVERROR(ENOMEM)); return -1; }
    auto* frames = reinterpret_cast<AVHWFramesContext*>(b->hw_frames->data);
    frames->format = AV_PIX_FMT_CUDA;
    frames->sw_format = AV_PIX_FMT_NV12;
    frames->width = static_cast<int>(b->width);
    frames->height = static_cast<int>(b->height);
    // The CUDA pool has to be created up front: av_frame_get_buffer refuses a
    // hardware frame whose pool is null. Sixteen 4K NV12 surfaces cover the
    // widest NVENC reference queue plus the frame being filled.
    frames->initial_pool_size = 16;
    rc = av_hwframe_ctx_init(b->hw_frames);
    if (rc < 0) { set_av_error("init CUDA frames", rc); return -1; }
    if (configure_encoder(b, b->codec_name.c_str(), &b->options_copy) < 0)
        return -1;
    if (open_muxer(b, b->output_path.c_str()) < 0) return -1;
    b->frame = av_frame_alloc();
    b->packet = av_packet_alloc();
    if (!b->frame || !b->packet) { fail("alloc frame or packet", AVERROR(ENOMEM)); return -1; }
    return 0;
}

int load_kernel(NvencBridge* b) {
    ContextScope scope(b);
    if (!scope.ok) return -1;
    CUresult rc = b->cuda.load(&b->module, kKernel);
    if (rc != CUDA_SUCCESS) { fail("cuModuleLoadData rgba_to_nv12", rc); return -1; }
    rc = b->cuda.function(&b->kernel, b->module, "rgba_to_nv12");
    if (rc != CUDA_SUCCESS) { fail("cuModuleGetFunction rgba_to_nv12", rc); return -1; }
    b->kernel_loaded = true;
    return 0;
}

void release_all(NvencBridge* b) {
    if (!b) return;
    {
        ContextScope scope(b);
        if (b->frame) {
            av_frame_unref(b->frame);
            av_frame_free(&b->frame);
        }
        if (b->packet) av_packet_free(&b->packet);
        if (b->encoder) avcodec_free_context(&b->encoder);
        if (b->hw_frames) av_buffer_unref(&b->hw_frames);
        if (b->hw_device) av_buffer_unref(&b->hw_device);
        if (b->module && b->kernel_loaded) b->cuda.unload(b->module);
        b->module = nullptr;
        if (b->staging) { cuda_d3d12_color_destroy(b->staging); b->staging = nullptr; }
    }
    // The staging import and the queue belong to the borrowed device, which is
    // released only after every handle referring to it is gone.
    b->staging_buffer = nullptr;
    if (b->format) {
        if (b->header_written && !b->wrote_trailer) {
            av_write_trailer(b->format);
            b->wrote_trailer = true;
        }
        if (b->format->pb) avio_closep(&b->format->pb);
        avformat_free_context(b->format);
        b->format = nullptr;
    }
    b->list.Reset();
    b->allocator.Reset();
    b->fence.Reset();
    b->queue.Reset();
    b->device.Reset();
    if (b->event) CloseHandle(b->event);
    if (b->cuda.module) FreeLibrary(b->cuda.module);
}

}

// ---------------------------------------------------------------------------
// HDR10 entry. Everything here is reached only through nvenc_hdr_open; the SDR
// paths above keep their staging, their kernel and their NV12 pool.
// ---------------------------------------------------------------------------
namespace {

constexpr const char* kHdrCodec = "hevc_nvenc";
constexpr const char* kHdrProfileName = "main10";
constexpr int32_t kHdrProfile = AV_PROFILE_HEVC_MAIN_10;
constexpr int32_t kHdrSwFormat = AV_PIX_FMT_P010LE;
constexpr int32_t kHdrColorRange = AVCOL_RANGE_MPEG;
constexpr int32_t kHdrColorPrimaries = AVCOL_PRI_BT2020;
constexpr int32_t kHdrColorTrc = AVCOL_TRC_SMPTE2084;
constexpr int32_t kHdrColorSpace = AVCOL_SPC_BT2020_NCL;
constexpr int32_t kHdrBitsPerSample = 10;
constexpr int32_t kSdr10ColorRange = AVCOL_RANGE_MPEG;
constexpr int32_t kSdr10ColorPrimaries = AVCOL_PRI_BT709;
constexpr int32_t kSdr10ColorTrc = AVCOL_TRC_BT709;
constexpr int32_t kSdr10ColorSpace = AVCOL_SPC_BT709;
struct P010ColorContract {
    const char* label;
    const char* codec;
    const char* profile_name;
    int32_t sw_format;
    int32_t color_range;
    int32_t color_primaries;
    int32_t color_trc;
    int32_t color_space;
    int32_t bits_per_sample;
};
constexpr P010ColorContract kHdrP010Contract{
    "HDR10", kHdrCodec, kHdrProfileName, kHdrSwFormat, kHdrColorRange,
    kHdrColorPrimaries, kHdrColorTrc, kHdrColorSpace, kHdrBitsPerSample};
constexpr P010ColorContract kSdr10P010Contract{
    "SDR10", kHdrCodec, kHdrProfileName, kHdrSwFormat, kSdr10ColorRange,
    kSdr10ColorPrimaries, kSdr10ColorTrc, kSdr10ColorSpace, kHdrBitsPerSample};
// The encoder-owned pool depth, the same value the SDR path uses: NVENC keeps
// a reference queue of surfaces, and one more is being filled.
constexpr int kHdrPoolSize = 16;

// A named request is honoured only by matching the HDR10 contract; 0 means the
// caller took it. Anything else is refused at the open rather than reinterpreted.
bool hdr_name_ok(const char* value, const char* required) {
    return !value || !*value || std::strcmp(value, required) == 0;
}

bool hdr_number_ok(int32_t requested, int32_t required, const char* what) {
    if (requested == 0 || requested == required) return true;
    std::snprintf(last_error, sizeof(last_error),
                  "%s %d is not the HDR10 value %d: this entry encodes "
                  "P010 HEVC Main10 with BT.2020/PQ/limited only",
                  what, requested, required);
    return false;
}

// Options that would move the submitted picture out of the HDR10 contract.
// Refusing them is the point: an option the caller asked for is never quietly
// overridden, and never quietly dropped either.
bool hdr_locked_option_absent(const char* extra) {
    static const char* locked[] = {"profile", "pix_fmt", "sw_pix_fmt", "rc",
                                   "colorspace", "color_range", "alpha_bits"};
    if (!extra || !*extra) return true;
    const std::string text(extra);
    size_t start = 0;
    while (start < text.size()) {
        const size_t end = text.find('\n', start);
        const std::string pair =
            text.substr(start, end == std::string::npos ? std::string::npos
                                                        : end - start);
        start = (end == std::string::npos) ? text.size() : end + 1;
        const size_t equals = pair.find('=');
        if (equals == std::string::npos || equals == 0) continue;
        const std::string name = pair.substr(0, equals);
        for (const char* key : locked) {
            if (name == key) {
                std::snprintf(last_error, sizeof(last_error),
                              "extra option %s is fixed by the HDR10 entry; "
                              "refusing rather than overriding the request",
                              name.c_str());
                return false;
            }
        }
    }
    return true;
}

// ST 2086 and CTA-861.3 values travel as the source's own rationals and its own
// integers: numerator and denominator are copied, so no float edits them.
void fill_mastering(const NvencHdrMetadata& in, AVMasteringDisplayMetadata* out) {
    for (int i = 0; i < 3; ++i) {
        out->display_primaries[i][0] =
            AVRational{in.primary_x_num[i], in.primary_x_den[i]};
        out->display_primaries[i][1] =
            AVRational{in.primary_y_num[i], in.primary_y_den[i]};
    }
    out->white_point[0] = AVRational{in.white_x_num, in.white_x_den};
    out->white_point[1] = AVRational{in.white_y_num, in.white_y_den};
    out->min_luminance = AVRational{in.min_luminance_num, in.min_luminance_den};
    out->max_luminance = AVRational{in.max_luminance_num, in.max_luminance_den};
    out->has_primaries = in.mastering_has_primaries;
    out->has_luminance = in.mastering_has_luminance;
}

void fill_content_light(const NvencHdrMetadata& in, AVContentLightMetadata* out) {
    out->MaxCLL = in.max_cll;
    out->MaxFALL = in.max_fall;
}

bool hdr_wants_mastering(const NvencBridge* b) {
    return b->hdr_metadata.has_mastering != 0;
}

bool hdr_wants_content_light(const NvencBridge* b) {
    return b->hdr_metadata.has_content_light != 0;
}

// Copy a bridge struct into a side data array that FFmpeg then owns. An entry
// of the same type is left alone: the source declared it once, not twice.
bool add_coded_side_data(AVPacketSideData** array, int* count,
                         enum AVPacketSideDataType type, const void* payload,
                         size_t size) {
    if (av_packet_side_data_get(*array, *count, type)) return true;
    void* copy = av_malloc(size);
    if (!copy) { fail("alloc coded side data", AVERROR(ENOMEM)); return false; }
    std::memcpy(copy, payload, size);
    if (!av_packet_side_data_add(array, count, type, copy, size, 0)) {
        av_free(copy);
        fail("add coded side data", AVERROR(ENOMEM));
        return false;
    }
    return true;
}

int attach_frame_hdr_metadata(NvencBridge* b, AVFrame* frame) {
    if (hdr_wants_mastering(b)) {
        AVMasteringDisplayMetadata mastering{};
        fill_mastering(b->hdr_metadata, &mastering);
        AVFrameSideData* sd = av_frame_new_side_data(frame,
            AV_FRAME_DATA_MASTERING_DISPLAY_METADATA, sizeof(mastering));
        if (!sd) { fail("frame mastering side data", AVERROR(ENOMEM)); return -1; }
        std::memcpy(sd->data, &mastering, sizeof(mastering));
        ++b->hdr_counters.mastering_side_data;
    }
    if (hdr_wants_content_light(b)) {
        AVContentLightMetadata light{};
        fill_content_light(b->hdr_metadata, &light);
        AVFrameSideData* sd = av_frame_new_side_data(frame,
            AV_FRAME_DATA_CONTENT_LIGHT_LEVEL, sizeof(light));
        if (!sd) { fail("frame content light side data", AVERROR(ENOMEM)); return -1; }
        std::memcpy(sd->data, &light, sizeof(light));
        ++b->hdr_counters.content_light_side_data;
    }
    return 0;
}

/* ------------------------------------------------------------------ *
 * In-bitstream HDR10 metadata writer: compressed bytes in, compressed
 * bytes out. Nothing below reads, allocates or copies an image.
 *
 * Syntax and semantics are ITU-T H.265 Annex D (D.2.27 mastering display
 * colour volume, D.2.28 content light level information), the NAL header of
 * 7.3.2 and the emulation-prevention process of Annex B.4. The primary
 * order and the two grids are not guessed: the fixture this project
 * decodes carries c = 0 as 13250/34500 and the installed decoder labels
 * that pair green_x/green_y, while AVMasteringDisplayMetadata hands the
 * caller red first. Wire index c therefore maps to struct index {1, 2, 0}.
 * ------------------------------------------------------------------ */
constexpr int32_t kSeiNalPrefix = 39;
constexpr int32_t kSeiTypeMastering = 137;
constexpr int32_t kSeiTypeContentLight = 144;
constexpr uint64_t kSeiChromaDen = 50000;      // D.2.27 units of 1/50000
constexpr uint64_t kSeiLumaDen = 10000;        // units of 0.0001 cd/m2
constexpr uint32_t kSeiMasteringBytes = 24;    // 8 * u16 + 2 * u32
constexpr uint32_t kSeiContentLightBytes = 4;  // 2 * u16
// Bounds a conforming decoder accepts. Outside them the installed FFmpeg
// clears has_primaries / has_luminance, so an SEI carrying such values
// would be present and still read back as no metadata at all: that is a
// contract miss dressed up as a pass, so this writer refuses instead.
constexpr uint64_t kSeiChromaXLow = 5, kSeiChromaXHigh = 37000;
constexpr uint64_t kSeiChromaYLow = 5, kSeiChromaYHigh = 42000;
constexpr uint64_t kSeiLumaMaxLow = 50000, kSeiLumaMaxHigh = 100000000;
// A minimum of 0 really ships (blu-ray discs declare 0.000 cd/m2), and the
// installed decoder only clears has_luminance above 50000 or at/above the max,
// so 0 is accepted here and the min < max check below carries the meaning.
constexpr uint64_t kSeiLumaMinLow = 0, kSeiLumaMinHigh = 50000;
constexpr int kSeiWireToStructIndex[3] = {1, 2, 0};  // green, blue, red

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t fnv_extend(uint64_t hash, const uint8_t* data, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= kFnvPrime;
    }
    return hash;
}

void sei_note(NvencHdrSeiInfo* info, int code, const char* fmt, ...) {
    if (!info) return;
    info->error = code;
    va_list args;
    va_start(args, fmt);
    std::vsnprintf(info->detail, sizeof(info->detail), fmt, args);
    va_end(args);
}

// Container metadata (notably Matroska Float fields) need not be an exact
// multiple of the HEVC wire grid. Validate the original value, then quantize
// to the nearest wire unit, ties upward, using integer arithmetic. Values
// already on the grid remain bit exact; no invalid value is clamped into range.
bool sei_to_wire(int32_t num, int32_t den, uint64_t scale, uint64_t low,
                 uint64_t high, const char* what, uint64_t* out,
                 NvencHdrSeiInfo* info) {
    if (den <= 0) {
        sei_note(info, NVENC_HDR_SEI_BAD_RATIONAL,
                 "%s has denominator %d: a rational with no positive scale "
                 "cannot be placed on the ST 2086 grid", what, den);
        return false;
    }
    if (num < 0) {
        sei_note(info, NVENC_HDR_SEI_BAD_RATIONAL,
                 "%s is negative (%d/%d): ST 2086 declares no negative value",
                 what, num, den);
        return false;
    }
    const uint64_t numerator = static_cast<uint64_t>(num);
    if (scale != 0 && numerator > 0xFFFFFFFFFFFFFFFFull / scale) {
        sei_note(info, NVENC_HDR_SEI_BAD_RATIONAL,
                 "%s %d/%d overflows the conversion to %llu wire units", what,
                 num, den, static_cast<unsigned long long>(scale));
        return false;
    }
    const uint64_t scaled = numerator * scale;
    const uint64_t unsigned_den = static_cast<uint64_t>(den);
    if (scaled < low * unsigned_den || scaled > high * unsigned_den) {
        sei_note(info, NVENC_HDR_SEI_OUT_OF_RANGE,
                 "%s %d/%d is outside %llu..%llu units on the 1/%llu wire grid",
                 what, num, den,
                 static_cast<unsigned long long>(low),
                 static_cast<unsigned long long>(high),
                 static_cast<unsigned long long>(scale));
        return false;
    }
    const uint64_t wire = scaled / unsigned_den +
        ((scaled % unsigned_den) >= (unsigned_den + 1) / 2 ? 1 : 0);
    *out = wire;
    return true;
}

void sei_put16(std::vector<uint8_t>& out, uint64_t value) {
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
}

void sei_put32(std::vector<uint8_t>& out, uint64_t value) {
    out.push_back(static_cast<uint8_t>((value >> 24) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 16) & 0xFFu));
    out.push_back(static_cast<uint8_t>((value >> 8) & 0xFFu));
    out.push_back(static_cast<uint8_t>(value & 0xFFu));
}

uint64_t sei_get16(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 8) | p[1];
}

uint64_t sei_get32(const uint8_t* p) {
    return (static_cast<uint64_t>(p[0]) << 24) | (static_cast<uint64_t>(p[1]) << 16) |
           (static_cast<uint64_t>(p[2]) << 8) | p[3];
}

// Annex B.1.3: after two zero bytes a third byte of 0x00..0x03 has to be
// preceded by an emulation_prevention_three_byte.
void sei_escape(const std::vector<uint8_t>& rbsp, std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(rbsp.size() + rbsp.size() / 2 + 1);
    int zeros = 0;
    for (const uint8_t byte : rbsp) {
        if (zeros >= 2 && byte <= 0x03) {
            out.push_back(0x03);
            zeros = 0;
        }
        out.push_back(byte);
        zeros = (byte == 0) ? zeros + 1 : 0;
    }
}

void sei_unescape(const uint8_t* data, size_t size, std::vector<uint8_t>& rbsp) {
    rbsp.clear();
    rbsp.reserve(size);
    int zeros = 0;
    for (size_t i = 0; i < size; ++i) {
        const uint8_t byte = data[i];
        if (zeros >= 2 && byte == 0x03) {
            zeros = 0;   // this 0x03 is the escape, never data
            continue;
        }
        rbsp.push_back(byte);
        zeros = (byte == 0) ? zeros + 1 : 0;
    }
}

bool sei_mastering_complete(const NvencHdrMetadata& md) {
    return md.has_mastering != 0 && md.mastering_has_primaries != 0 &&
           md.mastering_has_luminance != 0;
}

bool sei_mastering_partial(const NvencHdrMetadata& md) {
    if (md.has_mastering == 0) return false;
    return (md.mastering_has_primaries != 0) !=
           (md.mastering_has_luminance != 0);
}

// The ten mastering integers in wire order: for c = 0 green, 1 blue, 2 red the
// pair x then y, then white point x and y, then max luminance, then min. Eight
// of them are u(16) and the last two are u(32).
bool sei_mastering_wire(const NvencHdrMetadata& md, uint64_t out[10],
                        NvencHdrSeiInfo* info) {
    for (int c = 0; c < 3; ++c) {
        const int i = kSeiWireToStructIndex[c];
        char name[48];
        std::snprintf(name, sizeof(name), "display primary %d x", c);
        if (!sei_to_wire(md.primary_x_num[i], md.primary_x_den[i],
                         kSeiChromaDen, kSeiChromaXLow, kSeiChromaXHigh, name,
                         &out[c * 2], info))
            return false;
        std::snprintf(name, sizeof(name), "display primary %d y", c);
        if (!sei_to_wire(md.primary_y_num[i], md.primary_y_den[i],
                         kSeiChromaDen, kSeiChromaYLow, kSeiChromaYHigh, name,
                         &out[c * 2 + 1], info))
            return false;
    }
    if (!sei_to_wire(md.white_x_num, md.white_x_den, kSeiChromaDen,
                     kSeiChromaXLow, kSeiChromaXHigh, "white point x", &out[6],
                     info))
        return false;
    if (!sei_to_wire(md.white_y_num, md.white_y_den, kSeiChromaDen,
                     kSeiChromaYLow, kSeiChromaYHigh, "white point y", &out[7],
                     info))
        return false;
    if (!sei_to_wire(md.max_luminance_num, md.max_luminance_den, kSeiLumaDen,
                     kSeiLumaMaxLow, kSeiLumaMaxHigh,
                     "max display mastering luminance", &out[8], info))
        return false;
    if (!sei_to_wire(md.min_luminance_num, md.min_luminance_den, kSeiLumaDen,
                     kSeiLumaMinLow, kSeiLumaMinHigh,
                     "min display mastering luminance", &out[9], info))
        return false;
    if (out[9] >= out[8]) {
        sei_note(info, NVENC_HDR_SEI_OUT_OF_RANGE,
                 "min display mastering luminance (%llu) is not below max "
                 "(%llu): the declaration is self-inconsistent",
                 static_cast<unsigned long long>(out[9]),
                 static_cast<unsigned long long>(out[8]));
        return false;
    }
    return true;
}

bool sei_content_light_wire(const NvencHdrMetadata& md, uint64_t out[2],
                            NvencHdrSeiInfo* info) {
    if (md.max_cll > 0xFFFFu || md.max_fall > 0xFFFFu) {
        sei_note(info, NVENC_HDR_SEI_OUT_OF_RANGE,
                 "MaxCLL %u / MaxFALL %u do not fit the 16-bit CTA-861.3 "
                 "fields", md.max_cll, md.max_fall);
        return false;
    }
    out[0] = md.max_cll;
    out[1] = md.max_fall;
    return true;
}

// RBSP of one prefix SEI NAL unit: the declared messages, then
// rbsp_trailing_bits(). The mastering payload is always written before the
// content-light one; both apply to the whole sequence, and Annex D fixes no
// relative order between them.
bool sei_build_rbsp(const NvencHdrMetadata& md, bool want_mastering,
                    bool want_light, std::vector<uint8_t>& rbsp,
                    NvencHdrSeiInfo* info) {
    rbsp.clear();
    if (want_mastering) {
        uint64_t v[10];
        if (!sei_mastering_wire(md, v, info)) return false;
        rbsp.push_back(static_cast<uint8_t>(kSeiTypeMastering));
        rbsp.push_back(static_cast<uint8_t>(kSeiMasteringBytes));
        for (int c = 0; c < 3; ++c) {
            sei_put16(rbsp, v[c * 2]);
            sei_put16(rbsp, v[c * 2 + 1]);
        }
        sei_put16(rbsp, v[6]);
        sei_put16(rbsp, v[7]);
        sei_put32(rbsp, v[8]);   // max first, then min: D.2.27 order
        sei_put32(rbsp, v[9]);
    }
    if (want_light) {
        uint64_t v[2];
        if (!sei_content_light_wire(md, v, info)) return false;
        rbsp.push_back(static_cast<uint8_t>(kSeiTypeContentLight));
        rbsp.push_back(static_cast<uint8_t>(kSeiContentLightBytes));
        sei_put16(rbsp, v[0]);
        sei_put16(rbsp, v[1]);
    }
    rbsp.push_back(0x80);   // rbsp_stop_one_bit, then alignment zeros
    return true;
}

bool sei_build_nal(const NvencHdrMetadata& md, bool want_mastering,
                   bool want_light, std::vector<uint8_t>& nal,
                   std::vector<uint8_t>& rbsp, std::vector<uint8_t>& escaped,
                   NvencHdrSeiInfo* info) {
    if (!sei_build_rbsp(md, want_mastering, want_light, rbsp, info))
        return false;
    sei_escape(rbsp, escaped);
    nal.clear();
    // nal_unit_type 39, nuh_layer_id 0, nuh_temporal_id_plus1 1: the same
    // header the encoder itself puts on its own prefix SEI NAL unit.
    nal.push_back(static_cast<uint8_t>(0x40 | (kSeiNalPrefix << 1)));
    nal.push_back(0x01);
    nal.insert(nal.end(), escaped.begin(), escaped.end());
    if (info) info->nal_bytes = static_cast<uint32_t>(nal.size());
    return true;
}

struct SeiNalEntry {
    uint32_t start;   // first byte of its start code / length field
    uint32_t offset;  // first byte of the NAL unit itself
    uint32_t size;    // NAL bytes, excluding framing and trailing zeros
    uint32_t framing; // framing bytes immediately before this NAL
    int type;
    int layer_id;
    int temporal_id_plus1;
};

bool sei_parse_annexb(const uint8_t* data, uint32_t size,
                      std::vector<SeiNalEntry>& nals, NvencHdrSeiInfo* info) {
    if (size < 4) return false;
    // Locate every 00 00 01. Inside a NAL such a triple cannot occur, because
    // the encoder escapes a 0x01 that follows two zero bytes, so this scan is
    // unambiguous for a conforming packet.
    std::vector<uint32_t> codes;
    for (uint32_t i = 0; i + 2 < size; ++i) {
        if (data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1) {
            codes.push_back(i);
            i += 2;
        }
    }
    if (codes.empty() || codes.front() > 1) return false;
    const bool starts_long = (codes.front() == 1 && data[0] == 0);
    if (info)
        info->framing = starts_long ? NVENC_HDR_SEI_FRAMING_ANNEXB4
                                    : NVENC_HDR_SEI_FRAMING_ANNEXB3;
    for (size_t k = 0; k < codes.size(); ++k) {
        const uint32_t code = codes[k];
        const bool long_code = (code >= 1) && (data[code - 1] == 0);
        // The 0x01 that closes the start code is the third byte of the triple,
        // so the NAL unit itself begins at code + 3 whether the code was three
        // or four bytes long; the fourth byte of a long code sits before it.
        const uint32_t width = long_code ? 4u : 3u;
        const uint32_t body = code + 3u;
        if (body + 2 > size) return false;
        uint32_t next = (k + 1 < codes.size()) ? codes[k + 1] : size;
        // trailing_zero_8bits before a start code are not part of the NAL
        while (next > body && data[next - 1] == 0) --next;
        if (next < body || next - body < 2) return false;
        SeiNalEntry entry{};
        entry.start = long_code ? code - 1 : code;
        entry.offset = body;
        entry.size = next - body;
        entry.framing = width;
        entry.type = (data[body] >> 1) & 0x3F;
        entry.layer_id = ((data[body] & 0x01) << 5) | ((data[body + 1] >> 3) & 0x1F);
        entry.temporal_id_plus1 = data[body + 1] & 0x07;
        nals.push_back(entry);
    }
    return !nals.empty();
}

bool sei_parse_lp4(const uint8_t* data, uint32_t size,
                   std::vector<SeiNalEntry>& nals) {
    uint32_t cursor = 0;
    while (cursor < size) {
        if (size - cursor < 6) return false;   // 4 framing bytes + 2 NAL header
        const uint32_t length = (static_cast<uint32_t>(data[cursor]) << 24) |
                                (static_cast<uint32_t>(data[cursor + 1]) << 16) |
                                (static_cast<uint32_t>(data[cursor + 2]) << 8) |
                                static_cast<uint32_t>(data[cursor + 3]);
        if (length < 2) return false;
        if (length > size - cursor - 4) return false;
        const uint32_t body = cursor + 4;
        SeiNalEntry entry{};
        entry.start = cursor;
        entry.offset = body;
        entry.size = length;
        entry.framing = 4;
        entry.type = (data[body] >> 1) & 0x3F;
        entry.layer_id = ((data[body] & 0x01) << 5) | ((data[body + 1] >> 3) & 0x1F);
        entry.temporal_id_plus1 = data[body + 1] & 0x07;
        nals.push_back(entry);
        cursor = body + length;
    }
    return !nals.empty();
}

// One sei_message() header: ff-prefixed payloadType then ff-prefixed
// payloadSize, checked against the bytes that are actually left.
bool sei_read_message_header(const std::vector<uint8_t>& rbsp, size_t* cursor,
                             uint32_t* payload_type, uint32_t* payload_size) {
    uint64_t value = 0;
    for (int guard = 0; *cursor < rbsp.size(); ++guard) {
        if (rbsp[*cursor] != 0xFF) break;
        if (guard > 8 || value > 0xFFFFFFull) return false;
        value += 255;
        ++*cursor;
    }
    if (*cursor >= rbsp.size()) return false;
    *payload_type = static_cast<uint32_t>(value + rbsp[*cursor]);
    ++*cursor;

    value = 0;
    for (int guard = 0; *cursor < rbsp.size(); ++guard) {
        if (rbsp[*cursor] != 0xFF) break;
        if (guard > 8192 || value > 0xFFFFFFull) return false;
        value += 255;
        ++*cursor;
    }
    if (*cursor >= rbsp.size()) return false;
    *payload_size = static_cast<uint32_t>(value + rbsp[*cursor]);
    ++*cursor;
    return *payload_size <= rbsp.size() - *cursor;
}

struct SeiExisting {
    bool mastering = false;
    bool light = false;
    uint64_t mastering_values[10] = {0};
    uint32_t max_cll = 0;
    uint32_t max_fall = 0;
};

// What the access unit already declares, so that an encoder which ever does
// emit these payloads is not doubled and a contradicting declaration is
// never merged into one.
void sei_collect_existing(const uint8_t* data, const std::vector<SeiNalEntry>& nals,
                          std::vector<uint8_t>& rbsp, SeiExisting& found) {
    for (const auto& nal : nals) {
        if (nal.type != kSeiNalPrefix || nal.size < 3) continue;
        sei_unescape(data + nal.offset + 2, nal.size - 2, rbsp);
        size_t cursor = 0;
        while (cursor < rbsp.size()) {
            uint32_t type = 0, size = 0;
            if (!sei_read_message_header(rbsp, &cursor, &type, &size)) break;
            const uint8_t* p = rbsp.data() + cursor;
            if (type == static_cast<uint32_t>(kSeiTypeMastering) &&
                size == kSeiMasteringBytes && !found.mastering) {
                for (int c = 0; c < 3; ++c) {
                    found.mastering_values[c * 2] = sei_get16(p + c * 4);
                    found.mastering_values[c * 2 + 1] = sei_get16(p + c * 4 + 2);
                }
                found.mastering_values[6] = sei_get16(p + 12);
                found.mastering_values[7] = sei_get16(p + 14);
                found.mastering_values[8] = sei_get32(p + 16);
                found.mastering_values[9] = sei_get32(p + 20);
                found.mastering = true;
            } else if (type == static_cast<uint32_t>(kSeiTypeContentLight) &&
                       size == kSeiContentLightBytes && !found.light) {
                found.max_cll = static_cast<uint32_t>(sei_get16(p));
                found.max_fall = static_cast<uint32_t>(sei_get16(p + 2));
                found.light = true;
            }
            cursor += size;
            // rbsp_trailing_bits() ends the message list
            if (cursor >= rbsp.size()) break;
            if (rbsp[cursor] == 0x80 && cursor + 1 == rbsp.size()) break;
        }
    }
}

// The core, shared by the bridge and the host-only public entries. On
// success out_packet holds the rewritten packet, or is empty when the access
// unit needed nothing. *out_size is reported in the public wrapper.
int sei_apply_bytes(const NvencHdrMetadata& md, const uint8_t* in,
                    uint32_t in_size, std::vector<uint8_t>& out_packet,
                    std::vector<uint8_t>& nal_scratch,
                    std::vector<uint8_t>& rbsp_scratch,
                    std::vector<uint8_t>& escaped_scratch,
                    NvencHdrSeiInfo* info, NvencHdrSeiStatus* status) {
    if (!info) return -1;   // every caller supplies the readout
    info->error = NVENC_HDR_SEI_OK;
    info->detail[0] = '\0';
    info->vcl_index = -1;
    info->framing = NVENC_HDR_SEI_FRAMING_UNKNOWN;
    info->nal_count = 0;
    info->mastering_present = 0;
    info->content_light_present = 0;
    info->mastering_equal = 0;
    info->content_light_equal = 0;
    info->mastering_inserted = 0;
    info->content_light_inserted = 0;
    info->nal_bytes = 0;
    info->bytes_inserted = 0;
    out_packet.clear();
    if (!in || in_size == 0) {
        sei_note(info, NVENC_HDR_SEI_MALFORMED,
                 "packet is %u bytes: there are no NAL units to parse",
                 in_size);
        return -1;
    }
    const bool want_mastering = sei_mastering_complete(md);
    const bool want_light = md.has_content_light != 0;
    if (!want_mastering && !want_light) {
        sei_note(info, NVENC_HDR_SEI_NOTHING_TO_DO,
                 "the source declared no complete mastering display and no "
                 "content light level: nothing to author");
        return 0;
    }
    std::vector<SeiNalEntry> nals;
    if (!sei_parse_annexb(in, in_size, nals, info)) {
        nals.clear();
        if (!sei_parse_lp4(in, in_size, nals)) {
            sei_note(info, NVENC_HDR_SEI_FRAMING_UNSUPPORTED,
                     "packet of %u bytes is neither Annex-B nor 4-byte "
                     "length-prefixed HEVC: refusing to guess a framing",
                     in_size);
            return -1;
        }
        info->framing = NVENC_HDR_SEI_FRAMING_LP4;
    }
    info->nal_count = static_cast<int>(nals.size());
    for (const auto& nal : nals) {
        if (nal.layer_id != 0) {
            sei_note(info, NVENC_HDR_SEI_FRAMING_UNSUPPORTED,
                     "NAL type %d carries nuh_layer_id %d: this writer only "
                     "authors base-layer metadata", nal.type, nal.layer_id);
            return -1;
        }
    }

    SeiExisting found;
    sei_collect_existing(in, nals, rbsp_scratch, found);
    info->mastering_present = found.mastering ? 1 : 0;
    info->content_light_present = found.light ? 1 : 0;

    uint64_t vcl_bytes = 0;
    uint64_t all_bytes = 0;
    int vcl_index = -1;
    for (size_t k = 0; k < nals.size(); ++k) {
        const auto& nal = nals[k];
        all_bytes += nal.size;
        if (status) {
            // The session hashes are folds over the same bytes in packet order,
            // so a caller can recompute them over the whole elementary stream
            // and get one number for the session rather than one per packet.
            status->existing_nal_hash =
                fnv_extend(status->existing_nal_hash, in + nal.offset, nal.size);
        }
        if (nal.type <= 31) {   // 0..31 are the VCL NAL unit types
            vcl_bytes += nal.size;
            if (status) {
                status->vcl_hash =
                    fnv_extend(status->vcl_hash, in + nal.offset, nal.size);
            }
            if (vcl_index < 0) vcl_index = static_cast<int>(k);
        }
    }
    info->vcl_index = vcl_index;
    auto record = [&]() {
        if (!status) return;
        status->vcl_bytes += vcl_bytes;
        status->existing_nal_bytes += all_bytes;
    };

    bool add_mastering = want_mastering;
    bool add_light = want_light;
    if (found.mastering && want_mastering) {
        info->mastering_present = 1;
        uint64_t expected[10];
        if (sei_mastering_wire(md, expected, info)) {
            int first_difference = -1;
            for (int i = 0; i < 10; ++i) {
                if (expected[i] != found.mastering_values[i]) {
                    first_difference = i;
                    break;
                }
            }
            info->mastering_equal = first_difference < 0 ? 1 : 0;
            if (first_difference >= 0) {
                sei_note(info, NVENC_HDR_SEI_CONFLICT,
                         "the access unit already declares mastering integer %d "
                         "as %llu where the source declares %llu: refusing to "
                         "author a second, contradicting declaration",
                         first_difference,
                         static_cast<unsigned long long>(
                             found.mastering_values[first_difference]),
                         static_cast<unsigned long long>(
                             expected[first_difference]));
                return -1;
            }
            add_mastering = false;
            if (status) ++status->mastering_already_present;
        } else {
            return -1;
        }
    }
    if (found.light && want_light) {
        info->content_light_present = 1;
        uint64_t expected[2];
        if (!sei_content_light_wire(md, expected, info)) return -1;
        const bool equal = expected[0] == found.max_cll &&
                           expected[1] == found.max_fall;
        info->content_light_equal = equal ? 1 : 0;
        if (!equal) {
            sei_note(info, NVENC_HDR_SEI_CONFLICT,
                     "the access unit already declares MaxCLL %u / MaxFALL %u, "
                     "which differ from the requested %u / %u: refusing to "
                     "author a second, contradicting declaration",
                     found.max_cll, found.max_fall, md.max_cll, md.max_fall);
            return -1;
        }
        add_light = false;
        if (status) ++status->content_light_already_present;
    }
    if (!add_mastering && !add_light) {
        record();
        return 0;
    }
    if (vcl_index < 0) {
        if (status) ++status->packets_without_vcl;
        sei_note(info, NVENC_HDR_SEI_NO_VCL,
                 "packet of %u bytes carries %d NAL units and none of them a "
                 "slice: there is no access unit to attach metadata to",
                 in_size, static_cast<int>(nals.size()));
        return -1;
    }
    if (!sei_build_nal(md, add_mastering, add_light, nal_scratch, rbsp_scratch,
                       escaped_scratch, info))
        return -1;

    // Insert immediately before the first VCL NAL unit. A prefix SEI has to
    // precede every NAL unit except another prefix SEI, a parameter set or an
    // SLES, so the AUD, the parameter sets, any existing prefix SEI and every
    // slice keep their place and their bytes.
    const SeiNalEntry& vcl = nals[static_cast<size_t>(vcl_index)];
    const uint32_t insert_at = vcl.start;
    const uint32_t framing_width = vcl.framing;
    const uint64_t total = static_cast<uint64_t>(in_size) + framing_width +
                           nal_scratch.size();
    if (total > 0x7FFFFFF0ull) {
        sei_note(info, NVENC_HDR_SEI_CAPACITY,
                 "packet would grow to %llu bytes, past what an AVPacket size "
                 "can carry", (unsigned long long)total);
        return -1;
    }
    out_packet.clear();
    out_packet.reserve(static_cast<size_t>(total));
    out_packet.insert(out_packet.end(), in, in + insert_at);
    if (info->framing == NVENC_HDR_SEI_FRAMING_LP4) {
        const uint32_t length = static_cast<uint32_t>(nal_scratch.size());
        out_packet.push_back(static_cast<uint8_t>((length >> 24) & 0xFFu));
        out_packet.push_back(static_cast<uint8_t>((length >> 16) & 0xFFu));
        out_packet.push_back(static_cast<uint8_t>((length >> 8) & 0xFFu));
        out_packet.push_back(static_cast<uint8_t>(length & 0xFFu));
    } else {
        for (uint32_t i = 0; i < framing_width; ++i)
            out_packet.push_back(i + 1 == framing_width ? 0x01 : 0x00);
    }
    out_packet.insert(out_packet.end(), nal_scratch.begin(), nal_scratch.end());
    out_packet.insert(out_packet.end(), in + insert_at, in + in_size);

    const uint64_t inserted = total - in_size;
    info->bytes_inserted = static_cast<uint32_t>(inserted);
    info->mastering_inserted = add_mastering ? 1 : 0;
    info->content_light_inserted = add_light ? 1 : 0;
    record();
    if (status) {
        ++status->nals_inserted;
        status->bytes_inserted += inserted;
        if (add_mastering) ++status->mastering_inserted;
        if (add_light) ++status->content_light_inserted;
    }
    return 0;
}

// Decide once, at the open, what this session authors. Returns 0 when the
// session may proceed - nothing declared, a serialisable declaration, or a
// partial mastering declaration whose content light level can still be
// authored - and -1 only when the source declared a complete mastering
// display whose numbers cannot be placed on the ST 2086 grid: that is refused
// rather than rounded, dropped or left to the container alone. A refusal
// happens before any encoder or muxer object exists, so no file is written.
int hdr_sei_prepare(NvencBridge* b) {
    NvencHdrSeiStatus& st = b->hdr_sei;
    st = NvencHdrSeiStatus{0};
    st.struct_size = sizeof(NvencHdrSeiStatus);
    st.writer_available = 1;
    // A fold has to start at the FNV offset basis, not at zero, or the number a
    // caller recomputes over the elementary stream would not be this number.
    st.vcl_hash = kFnvOffset;
    st.existing_nal_hash = kFnvOffset;
    const NvencHdrMetadata& md = b->hdr_metadata;
    const bool anything = md.has_mastering != 0 || md.has_content_light != 0;
    st.metadata_requested = anything ? 1 : 0;
    if (!anything) {
        st.error = NVENC_HDR_SEI_NOTHING_TO_DO;
        std::snprintf(st.detail, sizeof(st.detail),
                      "the source declared no mastering display and no "
                      "content light level: no SEI is authored and none is "
                      "required");
        return 0;
    }
    char partial[256] = "";
    if (sei_mastering_partial(md)) {
        st.error = NVENC_HDR_SEI_PARTIAL_MASTERING;
        std::snprintf(partial, sizeof(partial),
                      "the source declared mastering display metadata with "
                      "primaries %d and luminance %d: the ST 2086 payload has "
                      "no way to carry only one of the two, so no mastering "
                      "SEI is authored and the required-SEI contract is NOT "
                      "met. ",
                      md.mastering_has_primaries, md.mastering_has_luminance);
        std::snprintf(st.detail, sizeof(st.detail), "%s", partial);
        b->hdr_sei_blocked = true;
        if (md.has_content_light == 0) return 0;
    }
    NvencHdrSeiInfo info{0};
    info.struct_size = sizeof(info);
    const bool complete = sei_mastering_complete(md);
    if (!sei_build_nal(md, complete, md.has_content_light != 0, b->hdr_sei_nal,
                       b->hdr_sei_rbsp, b->hdr_sei_escaped, &info)) {
        st.error = info.error;
        std::snprintf(st.detail, sizeof(st.detail), "%s", info.detail);
        b->hdr_sei_blocked = true;
        // A complete declaration the writer cannot place on the standard's
        // grid is a contract the session was asked to honour and cannot: the
        // open is refused rather than quietly downgraded.
        return complete ? -1 : 0;
    }
    st.mastering_serialized = complete ? 1 : 0;
    st.content_light_serialized = md.has_content_light ? 1 : 0;
    // A partial mastering declaration stays reported as PARTIAL_MASTERING even
    // when the content-light half was authored: the session did not meet the
    // required-SEI contract in full, and an OK here would say otherwise.
    st.error = (st.mastering_serialized || st.content_light_serialized)
                   ? (complete ? NVENC_HDR_SEI_OK
                               : NVENC_HDR_SEI_PARTIAL_MASTERING)
                   : NVENC_HDR_SEI_NOTHING_TO_DO;
    std::snprintf(st.detail, sizeof(st.detail),
                  "%sprefix SEI NAL of %u bytes prepared at the open "
                  "(mastering %d, content light %d; nearest wire units, "
                  "exact-grid values unchanged); inserted before the first "
                  "VCL NAL of every access unit", partial, info.nal_bytes,
                  complete ? 1 : 0, md.has_content_light ? 1 : 0);
    b->hdr_sei_wanted = (complete || md.has_content_light != 0);
    return 0;
}

int hdr_sei_apply(NvencBridge* b) {
    NvencHdrSeiStatus& st = b->hdr_sei;
    ++st.packets_seen;
    if (st.packets_seen == 1) {
        const uint32_t take = (b->packet->size < 24)
                                  ? static_cast<uint32_t>(b->packet->size)
                                  : 24u;
        for (uint32_t i = 0; i < take; ++i)
            std::snprintf(st.packet_sample + i * 2, 3, "%02x",
                          b->packet->data[i]);
        st.packet_sample[take * 2] = '\0';
    }
    if (!b->packet->buf) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder packet is not reference-counted: refusing to "
                      "rewrite it in place");
        b->failed = true;
        return -1;
    }
    NvencHdrSeiInfo info{0};
    info.struct_size = sizeof(info);
    const int rc = sei_apply_bytes(b->hdr_metadata, b->packet->data,
                                  static_cast<uint32_t>(b->packet->size),
                                  b->hdr_sei_packet, b->hdr_sei_nal,
                                  b->hdr_sei_rbsp, b->hdr_sei_escaped, &info,
                                  &st);
    if (rc < 0) {
        st.error = info.error;
        std::snprintf(st.detail, sizeof(st.detail), "%s", info.detail);
        b->failed = true;
        return -1;
    }
    if (st.framing == 0) st.framing = info.framing;
    if (b->hdr_sei_packet.empty()) return 0;   // nothing to add to this AU
    const size_t size = b->hdr_sei_packet.size();
    AVBufferRef* ref = av_buffer_alloc(
        static_cast<int>(size + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!ref) {
        fail("alloc SEI packet buffer", AVERROR(ENOMEM));
        b->failed = true;
        return -1;
    }
    std::memcpy(ref->data, b->hdr_sei_packet.data(), size);
    std::memset(ref->data + size, 0, AV_INPUT_BUFFER_PADDING_SIZE);
    av_buffer_unref(&b->packet->buf);
    b->packet->buf = ref;
    b->packet->data = ref->data;
    b->packet->size = static_cast<int>(size);
    b->hdr_contract.metadata_written_to |= 16;   // in-bitstream SEI authored
    return 0;
}

// The profile is this build's own answer, not an assumed number: the encoder is
// asked to resolve the name "main10", and the number it lands on is checked
// against AV_PROFILE_HEVC_MAIN_10. The installed library spells that number
// "Main 10", which is a different string from the option word, so the check is
// on the number and on the library naming it at all - not on a spelling that
// would refuse a working encoder. NVENC switches on the generic
// AVCodecContext field, so a name resolution that left a different value there
// is corrected before the open.
int hdr_apply_profile(AVCodecContext* encoder, const P010ColorContract& contract) {
    const int profile = encoder->codec_id == AV_CODEC_ID_AV1 ? AV_PROFILE_AV1_MAIN : kHdrProfile;
    const char* profile_name = contract.profile_name;
    int rc = encoder->codec_id == AV_CODEC_ID_AV1
        ? av_opt_set_int(encoder, "profile", profile, AV_OPT_SEARCH_CHILDREN)
        : av_opt_set(encoder, "profile", profile_name, AV_OPT_SEARCH_CHILDREN);
    if (encoder->profile != profile) {
        const int plain = encoder->codec_id == AV_CODEC_ID_AV1
            ? av_opt_set_int(encoder, "profile", profile, 0)
            : av_opt_set(encoder, "profile", profile_name, 0);
        if (rc < 0) rc = plain;
    }
    if (rc < 0) { set_av_error("profile", rc); return -1; }
    if (encoder->profile != profile) encoder->profile = profile;
    const char* named = avcodec_profile_name(encoder->codec_id, encoder->profile);
    if (!named || !*named) {
        std::snprintf(last_error, sizeof(last_error),
                      "this FFmpeg build does not name profile %d: refusing "
                      "rather than encoding an unverified profile",
                      encoder->profile);
        return -1;
    }
    return 0;
}

int configure_hdr_encoder(NvencBridge* b, const NvencHdrOptions* options,
                          const P010ColorContract& contract) {
    const AVCodec* found = avcodec_find_encoder_by_name(contract.codec);
    if (!found) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder %s not available in this FFmpeg build", contract.codec);
        return -1;
    }
    b->encoder = avcodec_alloc_context3(found);
    if (!b->encoder) { fail("alloc HDR encoder", AVERROR(ENOMEM)); return -1; }
    b->encoder->width = static_cast<int>(b->width);
    b->encoder->height = static_cast<int>(b->height);
    b->encoder->time_base = b->time_base;
    b->encoder->framerate = b->fps;
    b->encoder->sample_aspect_ratio = {1, 1};
    b->encoder->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    b->encoder->hw_frames_ctx = av_buffer_ref(b->hw_frames);
    if (!b->encoder->hw_frames_ctx) { fail("ref HDR frames", AVERROR(ENOMEM)); return -1; }

    // Rate control first, so nothing the caller chose is overwritten, and the
    // format and colour fields last, because this entry does not negotiate them.
    const struct Option { const char* name; const char* value; } text[] = {
        {"preset", options->preset}, {"tune", options->tune},
        {"multipass", options->multipass}, {"rc", options->rate_control}};
    for (const auto& item : text) {
        if (!item.value || !*item.value) continue;
        const int rc = av_opt_set(b->encoder, item.name, item.value,
                                  AV_OPT_SEARCH_CHILDREN);
        if (rc < 0) {
            const std::string step = std::string(contract.codec) + ": " + item.name + "=" + item.value;
            set_av_error(step.c_str(), rc); return -1;
        }
    }
    if (options->constant_quality > 0) {
        const int rc = av_opt_set_int(b->encoder, "cq", options->constant_quality,
                                      AV_OPT_SEARCH_CHILDREN);
        if (rc < 0) { set_av_error("cq", rc); return -1; }
    }
    if (options->bitrate > 0) b->encoder->bit_rate = options->bitrate;
    if (options->gop > 0) b->encoder->gop_size = options->gop;
    if (options->extra_options && *options->extra_options) {
        std::string remaining(options->extra_options);
        size_t start = 0;
        while (start < remaining.size()) {
            const size_t end = remaining.find('\n', start);
            const std::string pair = remaining.substr(
                start, end == std::string::npos ? std::string::npos : end - start);
            start = (end == std::string::npos) ? remaining.size() : end + 1;
            const size_t equals = pair.find('=');
            if (equals == std::string::npos || equals == 0) continue;
            const std::string name = pair.substr(0, equals);
            const std::string value = pair.substr(equals + 1);
            int rc = av_opt_set(b->encoder, name.c_str(), value.c_str(),
                                AV_OPT_SEARCH_CHILDREN);
            if (rc == AVERROR_OPTION_NOT_FOUND)
                rc = av_opt_set(b->encoder, name.c_str(), value.c_str(), 0);
            if (rc == AVERROR_OPTION_NOT_FOUND) {
                std::snprintf(last_error, sizeof(last_error),
                              "encoder %s has no option %s", contract.codec,
                              name.c_str());
                return -1;
            }
            if (rc < 0) {
                const std::string step = name + "=" + value;
                std::snprintf(last_error, sizeof(last_error), "%s", step.c_str());
                set_av_error(step.c_str(), rc);
                return -1;
            }
        }
    }

    b->encoder->pix_fmt = AV_PIX_FMT_CUDA;
    b->encoder->sw_pix_fmt = static_cast<enum AVPixelFormat>(contract.sw_format);
    if (hdr_apply_profile(b->encoder, contract) < 0) return -1;
    const int32_t profile_value = b->encoder->profile;
    b->encoder->color_range = static_cast<enum AVColorRange>(contract.color_range);
    b->encoder->color_primaries =
        static_cast<enum AVColorPrimaries>(contract.color_primaries);
    b->encoder->color_trc = static_cast<enum AVColorTransferCharacteristic>(
        contract.color_trc);
    b->encoder->colorspace = static_cast<enum AVColorSpace>(contract.color_space);

    // Declared on the encoder before the parameters are copied to the stream,
    // so a muxer that writes the mastering element from codec parameters sees
    // it whether or not the per-frame path reaches it.
    if (b->hdr_mode && (hdr_wants_mastering(b) || hdr_wants_content_light(b))) {
        if (hdr_wants_mastering(b)) {
            AVMasteringDisplayMetadata mastering{};
            fill_mastering(b->hdr_metadata, &mastering);
            if (!add_coded_side_data(&b->encoder->coded_side_data,
                    &b->encoder->nb_coded_side_data,
                    AV_PKT_DATA_MASTERING_DISPLAY_METADATA, &mastering,
                    sizeof(mastering))) return -1;
            b->hdr_contract.metadata_written_to |= 4;
        }
        if (hdr_wants_content_light(b)) {
            AVContentLightMetadata light{};
            fill_content_light(b->hdr_metadata, &light);
            if (!add_coded_side_data(&b->encoder->coded_side_data,
                    &b->encoder->nb_coded_side_data,
                    AV_PKT_DATA_CONTENT_LIGHT_LEVEL, &light, sizeof(light)))
                return -1;
            b->hdr_contract.metadata_written_to |= 4;
        }
    }

    const int rc = avcodec_open2(b->encoder, found, nullptr);
    if (rc < 0) { set_av_error("open NVENC P010 encoder", rc); return -1; }

    /* The 10-bit declaration the container should carry. Asking NVENC for it
     * before the open makes the wrapper compare it against the CUDA pixel
     * format's own descriptor depth (0) and log a mismatch: the encoder takes
     * its real depth from the pool's P010LE software format, so the number is
     * recorded after a successful open, still in time for
     * avcodec_parameters_from_context to put it on the stream. */
    b->encoder->bits_per_raw_sample = contract.bits_per_sample;

    // What the opened encoder actually reports, then the refusal if it is not
    // what this entry promised. A driver or build that cannot do Main10 P010
    // fails here with its own error rather than encoding NV12 anyway.
    auto& c = b->hdr_contract;
    c.actual_pix_fmt = b->encoder->pix_fmt;
    c.actual_sw_pix_fmt = b->encoder->sw_pix_fmt;
    c.actual_profile = b->encoder->profile;
    c.actual_bits_per_raw_sample = b->encoder->bits_per_raw_sample;
    c.actual_color_range = b->encoder->color_range;
    c.actual_color_primaries = b->encoder->color_primaries;
    c.actual_color_trc = b->encoder->color_trc;
    c.actual_color_space = b->encoder->colorspace;
    std::snprintf(c.actual_sw_pix_fmt_name, sizeof(c.actual_sw_pix_fmt_name),
                  "%s", av_get_pix_fmt_name(b->encoder->sw_pix_fmt) ?
                      av_get_pix_fmt_name(b->encoder->sw_pix_fmt) : "none");
    if (c.actual_pix_fmt != AV_PIX_FMT_CUDA) {
        std::snprintf(last_error, sizeof(last_error),
                      "HDR10 encode refused: encoder pixel format is %s, "
                      "CUDA hardware input required",
                      av_get_pix_fmt_name(b->encoder->pix_fmt));
        return -1;
    }
    if (c.actual_sw_pix_fmt != contract.sw_format) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s encode refused: encoder software format is %s, "
                      "not P010LE (no NV12 fallback is taken)",
                      contract.label, c.actual_sw_pix_fmt_name);
        return -1;
    }
    if (c.actual_profile != profile_value) {
        std::snprintf(last_error, sizeof(last_error),
                      "%s encode refused: encoder reports profile %d, "
                      "requested %s (%d)", contract.label, c.actual_profile,
                      contract.profile_name, profile_value);
        return -1;
    }
    if (c.actual_color_range != contract.color_range ||
        c.actual_color_primaries != contract.color_primaries ||
        c.actual_color_trc != contract.color_trc ||
        c.actual_color_space != contract.color_space) {
        std::snprintf(last_error, sizeof(last_error),
                      "HDR10 encode refused: encoder colour declaration is "
                      "range %d primaries %d trc %d space %d",
                      c.actual_color_range, c.actual_color_primaries,
                      c.actual_color_trc, c.actual_color_space);
        return -1;
    }
    return 0;
}

// The stream is what a muxer writes its colour and mastering elements from, so
// the metadata has to be reachable there as well as on every frame.
int attach_stream_hdr_metadata(NvencBridge* b) {
    if (!b->format || !b->stream) { fail("HDR stream missing", EINVAL); return -1; }
    AVCodecParameters* par = b->stream->codecpar;
    if (hdr_wants_mastering(b)) {
        AVMasteringDisplayMetadata mastering{};
        fill_mastering(b->hdr_metadata, &mastering);
        if (!add_coded_side_data(&par->coded_side_data, &par->nb_coded_side_data,
                AV_PKT_DATA_MASTERING_DISPLAY_METADATA, &mastering,
                sizeof(mastering))) return -1;
        b->hdr_contract.metadata_written_to |= 2;
    }
    if (hdr_wants_content_light(b)) {
        AVContentLightMetadata light{};
        fill_content_light(b->hdr_metadata, &light);
        if (!add_coded_side_data(&par->coded_side_data, &par->nb_coded_side_data,
                AV_PKT_DATA_CONTENT_LIGHT_LEVEL, &light, sizeof(light)))
            return -1;
        b->hdr_contract.metadata_written_to |= 2;
    }
    return 0;
}

int create_ffmpeg_hdr(NvencBridge* b, const NvencHdrOptions* options,
                      const P010ColorContract& contract, bool with_muxer = true) {
    ContextScope scope(b);
    if (!scope.ok) return -1;
    int rc = av_hwdevice_ctx_create(&b->hw_device, AV_HWDEVICE_TYPE_CUDA,
        nullptr, nullptr, AV_CUDA_USE_CURRENT_CONTEXT);
    if (rc < 0) { set_av_error("create CUDA device context (HDR)", rc); return -1; }
    b->hw_frames = av_hwframe_ctx_alloc(b->hw_device);
    if (!b->hw_frames) { fail("alloc HDR frames", AVERROR(ENOMEM)); return -1; }
    auto* frames = reinterpret_cast<AVHWFramesContext*>(b->hw_frames->data);
    frames->format = AV_PIX_FMT_CUDA;
    frames->sw_format = static_cast<enum AVPixelFormat>(contract.sw_format);
    frames->width = static_cast<int>(b->width);
    frames->height = static_cast<int>(b->height);
    frames->initial_pool_size = kHdrPoolSize;
    rc = av_hwframe_ctx_init(b->hw_frames);
    if (rc < 0) { set_av_error("init CUDA P010 frames", rc); return -1; }
    b->hdr_contract.hw_frames_sw_format = frames->sw_format;
    b->hdr_contract.hw_frames_pool_size = frames->initial_pool_size;
    if (configure_hdr_encoder(b, options, contract) < 0) return -1;
    if (!with_muxer) return 0;  // Real driver capability probe; no output file.
    if (open_muxer(b, b->output_path.c_str()) < 0) return -1;
    std::snprintf(b->hdr_contract.container, sizeof(b->hdr_contract.container),
                  "%s", b->format && b->format->oformat && b->format->oformat->name
                      ? b->format->oformat->name : "none");
    // Container colour tags: what the stream itself declares.
    if (b->stream) {
        b->stream->codecpar->color_range = b->encoder->color_range;
        b->stream->codecpar->color_primaries = b->encoder->color_primaries;
        b->stream->codecpar->color_trc = b->encoder->color_trc;
        b->stream->codecpar->color_space = b->encoder->colorspace;
        b->stream->codecpar->bits_per_raw_sample =
            b->encoder->bits_per_raw_sample;
    }
    b->hdr_contract.metadata_written_to |= 8;
    if (b->hdr_mode && (hdr_wants_mastering(b) || hdr_wants_content_light(b))) {
        b->hdr_contract.metadata_written_to |= 1;  // attached per frame
        if (attach_stream_hdr_metadata(b) < 0) return -1;
    }
    b->frame = av_frame_alloc();
    b->packet = av_packet_alloc();
    if (!b->frame || !b->packet) {
        fail("alloc HDR frame or packet", AVERROR(ENOMEM));
        return -1;
    }
    return 0;
}

int encode_p010_frame(NvencBridge* b, uint64_t y_ptr, uint64_t y_pitch,
                      uint64_t uv_ptr, uint64_t uv_pitch, int64_t pts) {
    ContextScope scope(b);
    if (!scope.ok) return -1;
    av_frame_unref(b->frame);
    b->frame->hw_frames_ctx = av_buffer_ref(b->hw_frames);
    if (!b->frame->hw_frames_ctx) { fail("ref HDR frame", AVERROR(ENOMEM)); return -1; }
    b->frame->format = AV_PIX_FMT_CUDA;
    b->frame->width = static_cast<int>(b->width);
    b->frame->height = static_cast<int>(b->height);
    int rc = av_hwframe_get_buffer(b->hw_frames, b->frame, 0);
    if (rc < 0) { set_av_error("get CUDA P010 frame", rc); return -1; }
    if (!b->frame->data[0] || !b->frame->data[1]) {
        fail("CUDA P010 frame has no planes", 0); return -1;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(b->width) * 2;
    if (b->frame->linesize[0] < static_cast<int>(row_bytes) ||
        b->frame->linesize[1] < static_cast<int>(row_bytes)) {
        std::snprintf(last_error, sizeof(last_error),
            "encoder P010 plane pitches (%d, %d) are under the frame row size %llu",
            b->frame->linesize[0], b->frame->linesize[1],
            static_cast<unsigned long long>(row_bytes));
        return -1;
    }
    if (!b->cuda.memcpy_2d) { fail("cuMemcpy2D_v2 unavailable", 0); return -1; }

    // Device to device, both planes, into the encoder-owned surface. Row
    // padding is never written because WidthInBytes stops at the payload, and
    // the call is synchronous, so the borrowed planes are free to be recycled
    // the moment this function returns.
    CUDA_MEMCPY2D plane{};
    plane.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    plane.dstMemoryType = CU_MEMORYTYPE_DEVICE;
    plane.srcDevice = static_cast<CUdeviceptr>(y_ptr);
    plane.srcPitch = static_cast<size_t>(y_pitch);
    plane.dstDevice = reinterpret_cast<CUdeviceptr>(b->frame->data[0]);
    plane.dstPitch = static_cast<size_t>(b->frame->linesize[0]);
    plane.WidthInBytes = static_cast<size_t>(row_bytes);
    plane.Height = b->height;
    CUresult crc = b->cuda.memcpy_2d(&plane);
    if (crc != CUDA_SUCCESS) { fail("cuMemcpy2D P010 luma", crc); return -1; }
    plane.srcDevice = static_cast<CUdeviceptr>(uv_ptr);
    plane.srcPitch = static_cast<size_t>(uv_pitch);
    plane.dstDevice = reinterpret_cast<CUdeviceptr>(b->frame->data[1]);
    plane.dstPitch = static_cast<size_t>(b->frame->linesize[1]);
    plane.Height = b->height / 2;
    crc = b->cuda.memcpy_2d(&plane);
    if (crc != CUDA_SUCCESS) { fail("cuMemcpy2D P010 chroma", crc); return -1; }
    crc = b->cuda.sync();
    if (crc != CUDA_SUCCESS) { fail("cuCtxSynchronize (P010 copy)", crc); return -1; }
    b->hdr_counters.p010_device_copy_bytes +=
        row_bytes * (static_cast<uint64_t>(b->height) + b->height / 2);

    b->frame->pts = pts;
    b->frame->color_range = static_cast<enum AVColorRange>(b->p010_color_range);
    b->frame->color_primaries =
        static_cast<enum AVColorPrimaries>(b->p010_color_primaries);
    b->frame->color_trc = static_cast<enum AVColorTransferCharacteristic>(
        b->p010_color_trc);
    b->frame->colorspace = static_cast<enum AVColorSpace>(b->p010_color_space);
    if (b->hdr_mode && attach_frame_hdr_metadata(b, b->frame) < 0) return -1;
    rc = avcodec_send_frame(b->encoder, b->frame);
    if (rc < 0) { set_av_error("send P010 frame", rc); return -1; }
    ++b->counters.frames_submitted;
    return write_packet(b);
}

bool capture_cuda_adapter(NvencBridge* b) {
    ContextScope scope(b);
    if (!scope.ok) return false;
    CUdevice cuda_device = 0;
    char luid[8]{};
    unsigned int node_mask = 0;
    CUresult rc = b->cuda.ctx_device(&cuda_device);
    if (rc == CUDA_SUCCESS)
        rc = b->cuda.device_luid(luid, &node_mask, cuda_device);
    if (rc != CUDA_SUCCESS) { fail("cuDeviceGetLuid (HDR)", rc); return false; }
    std::memcpy(b->hdr_contract.cuda_adapter_luid, luid, sizeof(luid));
    b->hdr_contract.cuda_adapter_node_mask = node_mask;
    return true;
}

int hdr_option_count_for(AVCodecContext* probe, const char* const* names,
                         size_t count) {
    int found = 0;
    for (size_t i = 0; i < count; ++i) {
        if (av_opt_find(probe, names[i], nullptr, 0, AV_OPT_SEARCH_CHILDREN))
            ++found;
    }
    return found;
}

}

extern "C" NVENC_BRIDGE_API uint32_t nvenc_abi_version(void) {
    return NVENC_BRIDGE_ABI_VERSION;
}
extern "C" NVENC_BRIDGE_API const char* nvenc_last_error(void) {
    return last_error;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_option_available(const char* codec,
                                                            const char* name) {
    last_error[0] = 0;
    if (!codec || !*codec || !name || !*name) return -1;
    const AVCodec* found = avcodec_find_encoder_by_name(codec);
    if (!found) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder %s not available in this FFmpeg build", codec);
        return -1;
    }
    // NVENC keeps preset/tune/rc settings in the private data of a context, so
    // the lookup has to go through a real one rather than the AVClass alone.
    AVCodecContext* probe = avcodec_alloc_context3(found);
    if (!probe) { fail("alloc probe context", AVERROR(ENOMEM)); return -1; }
    const int32_t rc = av_opt_find(probe, name, nullptr, 0,
                                   AV_OPT_SEARCH_CHILDREN) ? 1 : 0;
    avcodec_free_context(&probe);
    return rc;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_option_value_available(
    const char* codec, const char* name, const char* value) {
    last_error[0] = 0;
    if (!codec || !*codec || !name || !*name || !value || !*value) return -1;
    const AVCodec* found = avcodec_find_encoder_by_name(codec);
    if (!found) {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder %s not available in this FFmpeg build", codec);
        return -1;
    }
    AVCodecContext* probe = avcodec_alloc_context3(found);
    if (!probe) { fail("alloc option value probe", AVERROR(ENOMEM)); return -1; }
    // Unsupported values are an expected capability answer, not an export error.
    // Change only this disposable context's verbosity, never the global logger.
    probe->log_level_offset = AV_LOG_MAX_OFFSET;
    void* target = nullptr;
    const AVOption* option = av_opt_find2(probe, name, nullptr, 0,
                                         AV_OPT_SEARCH_CHILDREN, &target);
    int rc = AVERROR_OPTION_NOT_FOUND;
    if (option && option->type == AV_OPT_TYPE_INT && option->unit &&
        std::strspn(value, "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == std::strlen(value) &&
        (value[0] < '0' || value[0] > '9')) {
        // Named integer constants such as UHQ: the expression parser otherwise
        // logs a global error even with this context's log offset suppressed.
        const AVOption* constant = av_opt_find(target, value, option->unit, 0, 0);
        if (constant && constant->type == AV_OPT_TYPE_CONST)
            rc = av_opt_set_int(target, name, constant->default_val.i64, 0);
    } else if (option) {
        rc = av_opt_set(probe, name, value, AV_OPT_SEARCH_CHILDREN);
    }
    avcodec_free_context(&probe);
    return rc >= 0 ? 1 : 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_open(void* d3d12_device,
    void* cuda_context, uint32_t width, uint32_t height, uint32_t fps_num,
    uint32_t fps_den, int32_t time_base_num, int32_t time_base_den,
    const char* utf8_output_path, const NvencBridgeOptions* options,
    NvencBridge** out_bridge) {
    last_error[0] = 0;
    if (out_bridge) *out_bridge = nullptr;
    if (!d3d12_device || !cuda_context || !out_bridge || !utf8_output_path ||
        !options || options->struct_size != sizeof(NvencBridgeOptions) ||
        !width || !height || (width & 1) || (height & 1) ||
        width > (UINT32_MAX - 255u) / 4u || !fps_num || !fps_den ||
        time_base_num <= 0 || time_base_den <= 0) {
        std::snprintf(last_error, sizeof(last_error), "invalid nvenc_open arguments");
        return -1;
    }
    if (static_cast<uint64_t>(width) * height > UINT64_MAX / 8) {
        std::snprintf(last_error, sizeof(last_error), "frame size overflows");
        return -1;
    }
    auto* b = new (std::nothrow) NvencBridge;
    if (!b) { fail("alloc bridge", ERROR_NOT_ENOUGH_MEMORY); return -1; }
    b->counters.struct_size = sizeof(NvencTransferCounters);
    b->width = width;
    b->height = height;
    b->stage_pitch = align256(width * 4);
    b->fps = AVRational{static_cast<int>(fps_num), static_cast<int>(fps_den)};
    b->time_base = AVRational{time_base_num, time_base_den};
    b->codec_name = options->codec && *options->codec ? options->codec : "av1_nvenc";
    b->output_path = utf8_output_path;
    b->context = static_cast<CUcontext>(cuda_context);
    b->device = static_cast<ID3D12Device*>(d3d12_device);
    if (options->color_matrix == NVENC_COLOR_MATRIX_BT601) {
        b->matrix = k601;
        b->colorspace = AVCOL_SPC_SMPTE170M;
        b->primaries = AVCOL_PRI_SMPTE170M;
        b->transfer = AVCOL_TRC_SMPTE170M;
    } else if (options->color_matrix != NVENC_COLOR_MATRIX_BT709) {
        std::snprintf(last_error, sizeof(last_error), "unknown color matrix %d",
                      options->color_matrix);
        release_all(b); delete b; return -1;
    }

    int rc = 0;
    if (!load_driver(&b->cuda)) rc = -1;
    else if (!same_adapter(b)) rc = -1;
    if (!rc) {
        b->options_copy = *options;
        rc = load_kernel(b);
    }
    if (!rc) rc = create_command_graph(b);
    if (!rc) rc = create_staging(b);
    if (!rc) rc = create_ffmpeg(b);
    if (rc) { release_all(b); delete b; return -1; }
    *out_bridge = b;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_encode_rgba(NvencBridge* b,
    uint64_t rgba_ptr, size_t rgba_pitch, int64_t pts) {
    last_error[0] = 0;
    if (!b || b->failed || !rgba_ptr ||
        rgba_pitch < static_cast<size_t>(b->width) * 4 ||
        rgba_pitch > UINT32_MAX) {
        std::snprintf(last_error, sizeof(last_error), "invalid encode_rgba arguments");
        return -1;
    }
    if (encode_frame(b, rgba_ptr, static_cast<uint32_t>(rgba_pitch), pts) < 0) {
        b->failed = true;
        return -1;
    }
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_encode_texture(NvencBridge* b,
    void* rgba_texture, void* producer_fence, uint64_t producer_fence_value,
    int64_t pts) {
    last_error[0] = 0;
    if (!b || b->failed || !rgba_texture) {
        std::snprintf(last_error, sizeof(last_error), "invalid encode_texture arguments");
        return -1;
    }
    if (stage_texture(b, static_cast<ID3D12Resource*>(rgba_texture),
        static_cast<ID3D12Fence*>(producer_fence), producer_fence_value) < 0) {
        b->failed = true;
        return -1;
    }
    if (encode_frame(b, b->staging_mapped, b->stage_pitch, pts) < 0) {
        b->failed = true;
        return -1;
    }
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_get_last_nv12_planes(NvencBridge* b,
    uint64_t* out_y_ptr, uint32_t* out_y_pitch, uint64_t* out_uv_ptr,
    uint32_t* out_uv_pitch) {
    if (!b || !out_y_ptr || !out_y_pitch || !out_uv_ptr || !out_uv_pitch) return -1;
    if (!b->last_y || !b->last_uv) {
        std::snprintf(last_error, sizeof(last_error), "no conversion yet");
        return -1;
    }
    *out_y_ptr = b->last_y;
    *out_y_pitch = b->last_y_pitch;
    *out_uv_ptr = b->last_uv;
    *out_uv_pitch = b->last_uv_pitch;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_get_transfer_counters(
    NvencBridge* b, NvencTransferCounters* counters) {
    if (!b || !counters || counters->struct_size != sizeof(NvencTransferCounters))
        return -1;
    *counters = b->counters;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_finish(NvencBridge* b) {
    last_error[0] = 0;
    if (!b) return -1;
    if (b->finished) return b->failed ? -1 : 0;
    b->finished = true;
    ContextScope scope(b);
    if (!scope.ok) { b->failed = true; return -1; }
    av_frame_unref(b->frame);
    const int rc = avcodec_send_frame(b->encoder, nullptr);
    if (rc < 0 && rc != AVERROR(EINVAL) && rc != AVERROR_EOF) {
        set_av_error("flush encoder", rc);
        b->failed = true;
        return -1;
    }
    if (write_packet(b) < 0) { b->failed = true; return -1; }
    return 0;
}

extern "C" NVENC_BRIDGE_API void nvenc_close(NvencBridge* b) {
    if (!b) return;
    if (!b->finished) nvenc_finish(b);
    if (b->format && b->header_written && !b->wrote_trailer) {
        // A failed encode still gets a trailer so the packets already written
        // stay probeable; the caller checks the return code, not the file.
        const int rc = av_write_trailer(b->format);
        if (rc < 0) set_av_error("write trailer", rc);
        else b->wrote_trailer = true;
    }
    release_all(b);
    delete b;
}

// ---------------------------------------------------------------------------
// HDR10 entry points.
// ---------------------------------------------------------------------------

extern "C" NVENC_BRIDGE_API uint32_t nvenc_hdr_abi_version(void) {
    return NVENC_HDR_ABI_VERSION;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_get_defaults(
    NvencHdrDefaults* out_defaults) {
    last_error[0] = 0;
    if (!out_defaults ||
        out_defaults->struct_size != sizeof(NvencHdrDefaults)) {
        fail("nvenc_hdr_get_defaults struct_size", EINVAL);
        return -1;
    }
    *out_defaults = NvencHdrDefaults{sizeof(NvencHdrDefaults)};
    out_defaults->sw_pixel_format = kHdrSwFormat;
    out_defaults->color_range = kHdrColorRange;
    out_defaults->color_primaries = kHdrColorPrimaries;
    out_defaults->color_trc = kHdrColorTrc;
    out_defaults->color_space = kHdrColorSpace;
    std::snprintf(out_defaults->profile, sizeof(out_defaults->profile), "%s",
                  kHdrProfileName);
    return 0;
}

static int32_t p010_capability_impl(const char* codec_name, void* cuda_context,
    uint32_t width, uint32_t height, NvencHdrCapability* out_capability) {
    last_error[0] = 0;
    const bool av1 = codec_name && std::strcmp(codec_name, "av1_nvenc") == 0;
    if (!av1 && (!codec_name || std::strcmp(codec_name, kHdrCodec))) {
        fail("unsupported P010 codec", EINVAL); return -1;
    }
    const int profile_value = av1 ? AV_PROFILE_AV1_MAIN : kHdrProfile;
    const char* profile_name = av1 ? "main" : kHdrProfileName;
    if (!out_capability ||
        out_capability->struct_size != sizeof(NvencHdrCapability)) {
        fail("nvenc_hdr_capability struct_size", EINVAL);
        return -1;
    }
    *out_capability = NvencHdrCapability{sizeof(NvencHdrCapability)};
    std::snprintf(out_capability->encoder_name,
                  sizeof(out_capability->encoder_name), "%s", codec_name);

    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name);
    if (codec) out_capability->encoder_present = 1;
    else {
        std::snprintf(last_error, sizeof(last_error),
                      "encoder %s not available in this FFmpeg build", codec_name);
        return -1;
    }

    const AVPixelFormat* formats = nullptr;
    int count = 0;
    if (avcodec_get_supported_config(nullptr, codec, AV_CODEC_CONFIG_PIX_FORMAT,
                                     0, reinterpret_cast<const void**>(&formats),
                                     &count) >= 0 && formats) {
        for (int i = 0; i < count; ++i) {
            if (formats[i] == AV_PIX_FMT_P010LE)
                out_capability->p010_supported = 1;
            if (formats[i] == AV_PIX_FMT_CUDA)
                out_capability->cuda_supported = 1;
        }
    }
    for (const AVProfile* profile = codec->profiles;
         profile && profile->profile != AV_PROFILE_UNKNOWN; ++profile) {
        if (profile->profile == profile_value) {
            out_capability->main10_profile_supported = 1;
            std::snprintf(out_capability->profile_name,
                          sizeof(out_capability->profile_name), "%s",
                          profile->name ? profile->name : "");
        }
    }

    // Does the installed encoder expose any way to ask for HDR10 static
    // metadata? Enumerated from its own option table rather than assumed.
    AVCodecContext* probe = avcodec_alloc_context3(codec);
    if (probe) {
        /* hevc_nvenc leaves AVCodec::profiles empty, so the advertised-profile
         * scan above can come back blank on a build that does encode Main10.
         * The encoder's own option table is the authority here: the option word
         * "main10" is a const of the child class, and setting it through
         * AV_OPT_SEARCH_CHILDREN writes the child's copy while
         * AVCodecContext::profile stays AV_PROFILE_UNKNOWN, so the plain search
         * is what lands on the generic field NVENC actually switches on. */
        if (!out_capability->main10_profile_supported) {
            int rc = av1 ? av_opt_set_int(probe, "profile", profile_value, AV_OPT_SEARCH_CHILDREN)
                         : av_opt_set(probe, "profile", profile_name, AV_OPT_SEARCH_CHILDREN);
            if (probe->profile != profile_value) {
                const int plain = av1 ? av_opt_set_int(probe, "profile", profile_value, 0)
                                      : av_opt_set(probe, "profile", profile_name, 0);
                if (rc < 0) rc = plain;
            }
            const char* resolved = avcodec_profile_name(codec->id,
                                                       probe->profile);
            if (rc >= 0 && probe->profile == profile_value && resolved) {
                out_capability->main10_profile_supported = 1;
                std::snprintf(out_capability->profile_name,
                              sizeof(out_capability->profile_name), "%s",
                              resolved);
            } else {
                std::snprintf(out_capability->profile_name,
                              sizeof(out_capability->profile_name), "%s",
                              resolved ? resolved : "");
                std::snprintf(last_error, sizeof(last_error),
                    "Main 10 profile not resolved: av_opt_set(profile=%s) rc=%d, "
                    "encoder landed on %d (%s)", profile_name, rc,
                    probe->profile, resolved ? resolved : "unnamed");
            }
        }
        static const char* const candidates[] = {
            "mastering_display", "master-display", "mdcv", "st2086",
            "max_cll", "maxcll", "max_fall", "maxfall", "content_light",
            "hdr10", "hdr_metadata"};
        out_capability->hdr_metadata_options = hdr_option_count_for(
            probe, candidates, sizeof(candidates) / sizeof(candidates[0]));
        avcodec_free_context(&probe);
    }

    // A real CUDA device object on the queried context: this is what proves the
    // installed driver can be talked to at all, as opposed to a static table.
    if (cuda_context && width && height) {
        Driver driver;
        if (load_driver(&driver) && driver.push) {
            CUcontext previous = nullptr;
            if (driver.push(static_cast<CUcontext>(cuda_context)) == CUDA_SUCCESS) {
                AVBufferRef* device = nullptr;
                const int rc = av_hwdevice_ctx_create(&device,
                    AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr,
                    AV_CUDA_USE_CURRENT_CONTEXT);
                if (rc >= 0 && device) {
                    out_capability->cuda_device_created = 1;
                    av_buffer_unref(&device);
                } else {
                    set_av_error("HDR capability CUDA device", rc);
                }
                driver.pop(&previous);
            } else {
                fail("cuCtxPushCurrent (HDR capability)", -1);
            }
        }
        if (driver.module) FreeLibrary(driver.module);
    }

    out_capability->supported =
        out_capability->encoder_present && out_capability->p010_supported &&
        out_capability->cuda_supported &&
        out_capability->main10_profile_supported &&
        out_capability->cuda_device_created ? 1 : 0;
    return 0;
}

static int32_t nvenc_p010_open_impl(void* cuda_context,
    uint32_t width, uint32_t height, uint32_t fps_num, uint32_t fps_den,
    int32_t time_base_num, int32_t time_base_den, const char* utf8_output_path,
    const NvencHdrOptions* options, const NvencHdrMetadata* metadata,
    NvencBridge** out_bridge, bool is_hdr, bool legacy = true) {
    P010ColorContract contract = is_hdr ? kHdrP010Contract : kSdr10P010Contract;
    const bool av1 = !legacy && options && options->codec && std::strcmp(options->codec, "av1_nvenc") == 0;
    if (av1) { contract.codec = "av1_nvenc"; contract.profile_name = "main"; }
    last_error[0] = 0;
    if (out_bridge) *out_bridge = nullptr;
    if (!cuda_context || !out_bridge || !utf8_output_path || !options ||
        options->struct_size != sizeof(NvencHdrOptions) || !width || !height ||
        (width & 1) || (height & 1) || !fps_num || !fps_den ||
        time_base_num <= 0 || time_base_den <= 0) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid P010 encoder open arguments");
        return -1;
    }
    if (metadata && metadata->struct_size != sizeof(NvencHdrMetadata)) {
        std::snprintf(last_error, sizeof(last_error),
                      "NvencHdrMetadata struct_size is %u, expected %u",
                      metadata->struct_size,
                      static_cast<uint32_t>(sizeof(NvencHdrMetadata)));
        return -1;
    }
    if (!is_hdr && metadata) {
        const auto* words = reinterpret_cast<const uint32_t*>(metadata);
        for (size_t i = 1; i < sizeof(*metadata) / sizeof(uint32_t); ++i) {
            if (words[i] != 0) {
                std::snprintf(last_error, sizeof(last_error),
                              "SDR10 P010 refuses HDR metadata fields");
                return -1;
            }
        }
    }
    if (options->constant_quality < 0 || options->constant_quality > (av1 ? 63 : 51)) {
        fail("P010 CQ out of codec range", EINVAL); return -1;
    }
    // P010 stores two bytes per sample, so a row is width*2 bytes and an
    // AVFrame linesize has to stay inside a positive int.
    if (width > INT32_MAX / 2) {
        std::snprintf(last_error, sizeof(last_error),
                      "P010 row width %u does not fit a frame linesize", width);
        return -1;
    }
    if (static_cast<uint64_t>(width) * height > UINT64_MAX / 8) {
        std::snprintf(last_error, sizeof(last_error), "frame size overflows");
        return -1;
    }
    if (!hdr_name_ok(options->codec, contract.codec)) {
        std::snprintf(last_error, sizeof(last_error),
                      "codec %s is not the %s entry's %s (this bridge does "
                      "not fall back to another codec or to NV12)",
                      options->codec && *options->codec ? options->codec : "",
                      contract.label, contract.codec);
        return -1;
    }
    if (!hdr_name_ok(options->profile, contract.profile_name)) {
        std::snprintf(last_error, sizeof(last_error),
                      "profile %s is not the %s entry's %s",
                      options->profile && *options->profile ? options->profile
                                                            : "",
                      contract.label, contract.profile_name);
        return -1;
    }
    if (!hdr_number_ok(options->sw_pixel_format, contract.sw_format,
                       "requested software pixel format") ||
        !hdr_number_ok(options->color_range, contract.color_range,
                       "requested color range") ||
        !hdr_number_ok(options->color_primaries, contract.color_primaries,
                       "requested color primaries") ||
        !hdr_number_ok(options->color_trc, contract.color_trc,
                       "requested transfer characteristic") ||
        !hdr_number_ok(options->color_space, contract.color_space,
                       "requested color space")) {
        return -1;
    }
    if (!hdr_locked_option_absent(options->extra_options)) return -1;

    auto* b = new (std::nothrow) NvencBridge;
    if (!b) { fail("alloc HDR bridge", ERROR_NOT_ENOUGH_MEMORY); return -1; }
    b->counters.struct_size = sizeof(NvencTransferCounters);
    b->hdr_counters.struct_size = sizeof(NvencHdrCounters);
    b->hdr_contract.struct_size = sizeof(NvencHdrContract);
    b->p010_mode = true;
    b->hdr_mode = is_hdr;
    b->p010_color_range = contract.color_range;
    b->p010_color_primaries = contract.color_primaries;
    b->p010_color_trc = contract.color_trc;
    b->p010_color_space = contract.color_space;
    b->width = width;
    b->height = height;
    b->fps = AVRational{static_cast<int>(fps_num), static_cast<int>(fps_den)};
    b->time_base = AVRational{time_base_num, time_base_den};
    b->codec_name = contract.codec;
    b->output_path = utf8_output_path;
    b->context = static_cast<CUcontext>(cuda_context);
    if (metadata) b->hdr_metadata = *metadata;
    else b->hdr_metadata.struct_size = sizeof(NvencHdrMetadata);
    // The in-bitstream writer decides here, once, before any encoder, muxer or
    // output file exists: a declared complete mastering display that cannot be
    // placed on the ST 2086 grid refuses the open and writes no file.
    b->metadata_status.struct_size = sizeof(NvencP010MetadataStatus);
    b->metadata_status.codec_kind = av1 ? 2 : 1;
    b->metadata_status.color_mode = is_hdr ? 1 : 0;
    b->metadata_status.writer_available = 1;
    Bytes av1_mastering, av1_light;
    const bool metadata_ok = !is_hdr || (av1 ?
        av1_prepare(b->hdr_metadata, b->metadata_status, av1_mastering, av1_light) : hdr_sei_prepare(b) == 0);
    if (av1 && is_hdr) b->hdr_sei_wanted = b->metadata_status.metadata_requested != 0;
    if (!metadata_ok) {
        std::snprintf(last_error, sizeof(last_error),
                      "nvenc_hdr_open refused: the declared HDR10 static "
                      "metadata cannot be authored into the bitstream (%s)",
                      av1 ? b->metadata_status.detail : b->hdr_sei.detail);
        delete b;
        return -1;
    }
    if (!is_hdr) {
        b->hdr_sei.struct_size = sizeof(NvencHdrSeiStatus);
        b->hdr_sei.writer_available = 1;
        b->hdr_sei.error = NVENC_HDR_SEI_NOTHING_TO_DO;
        std::snprintf(b->hdr_sei.detail, sizeof(b->hdr_sei.detail),
                      "SDR10 output; HDR SEI metadata is not emitted");
    }

    auto& c = b->hdr_contract;
    c.width = width;
    c.height = height;
    c.requested_pix_fmt = AV_PIX_FMT_CUDA;
    c.requested_sw_pix_fmt = contract.sw_format;
    c.requested_profile = av1 ? AV_PROFILE_AV1_MAIN : kHdrProfile;
    c.requested_bits_per_raw_sample = kHdrBitsPerSample;
    c.requested_color_range = contract.color_range;
    c.requested_color_primaries = contract.color_primaries;
    c.requested_color_trc = contract.color_trc;
    c.requested_color_space = contract.color_space;
    c.has_mastering = is_hdr && hdr_wants_mastering(b) ? 1 : 0;
    c.has_content_light = is_hdr && hdr_wants_content_light(b) ? 1 : 0;
    std::snprintf(c.codec, sizeof(c.codec), "%s", contract.codec);
    std::snprintf(c.profile_name, sizeof(c.profile_name), "%s",
                  contract.profile_name);

    int rc = 0;
    if (!load_driver(&b->cuda)) rc = -1;
    else if (!b->cuda.memcpy_2d) {
        fail("cuMemcpy2D_v2 unavailable in nvcuda.dll", 0);
        rc = -1;
    } else if (!capture_cuda_adapter(b)) rc = -1;
    if (!rc) rc = create_ffmpeg_hdr(b, options, contract);
    if (rc) { release_all(b); delete b; return -1; }
    *out_bridge = b;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_open(void* cuda_context,
    uint32_t width, uint32_t height, uint32_t fps_num, uint32_t fps_den,
    int32_t time_base_num, int32_t time_base_den, const char* utf8_output_path,
    const NvencHdrOptions* options, const NvencHdrMetadata* metadata,
    NvencBridge** out_bridge) {
    return nvenc_p010_open_impl(cuda_context, width, height, fps_num, fps_den,
        time_base_num, time_base_den, utf8_output_path, options, metadata,
        out_bridge, true);
}

extern "C" NVENC_BRIDGE_API uint32_t nvenc_sdr10_abi_version(void) {
    return NVENC_SDR10_ABI_VERSION;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_sdr10_open(void* cuda_context,
    uint32_t width, uint32_t height, uint32_t fps_num, uint32_t fps_den,
    int32_t time_base_num, int32_t time_base_den, const char* utf8_output_path,
    const NvencHdrOptions* options, const NvencHdrMetadata* metadata,
    NvencBridge** out_bridge) {
    return nvenc_p010_open_impl(cuda_context, width, height, fps_num, fps_den,
        time_base_num, time_base_den, utf8_output_path, options, metadata,
        out_bridge, false);
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_encode_p010(NvencBridge* b,
    uint64_t y_ptr, size_t y_pitch, uint64_t uv_ptr, size_t uv_pitch,
    int64_t pts) {
    last_error[0] = 0;
    if (!b || b->failed) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid encode_p010 arguments");
        return -1;
    }
    if (!b->p010_mode) {
        std::snprintf(last_error, sizeof(last_error),
                      "nvenc_encode_p010 needs an HDR10 or SDR10 P010 bridge");
        return -1;
    }
    if (!y_ptr || !uv_ptr) {
        std::snprintf(last_error, sizeof(last_error),
                      "encode_p010 was given a null device plane");
        return -1;
    }
    const uint64_t row_bytes = static_cast<uint64_t>(b->width) * 2;
    const uint64_t y_span = row_bytes * (b->height - 1);
    const uint64_t uv_span = row_bytes * (b->height / 2 - 1);
    if ((y_pitch & 1) || (uv_pitch & 1) || y_pitch < row_bytes ||
        uv_pitch < row_bytes || y_pitch > UINT64_MAX - y_span ||
        uv_pitch > UINT64_MAX - uv_span) {
        std::snprintf(last_error, sizeof(last_error),
            "P010 pitches (%llu, %llu) are not both at least the %llu byte "
            "row and below the plane span, or they overflow one",
            static_cast<unsigned long long>(y_pitch),
            static_cast<unsigned long long>(uv_pitch),
            static_cast<unsigned long long>(row_bytes));
        return -1;
    }
    if (encode_p010_frame(b, y_ptr, y_pitch, uv_ptr, uv_pitch, pts) < 0) {
        b->failed = true;
        return -1;
    }
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_get_counters(NvencBridge* b,
    NvencHdrCounters* counters) {
    last_error[0] = 0;
    if (!b || !counters || counters->struct_size != sizeof(NvencHdrCounters)) {
        fail("nvenc_hdr_get_counters struct_size", EINVAL);
        return -1;
    }
    if (!b->p010_mode) {
        std::snprintf(last_error, sizeof(last_error),
                      "nvenc_hdr_get_counters needs a P010 bridge");
        return -1;
    }
    *counters = b->hdr_counters;
    // One source of truth for the two counts the ABI 2 struct already keeps.
    counters->frames_submitted = b->counters.frames_submitted;
    counters->packets_written = b->counters.packets_written;
    counters->host_pixel_bytes = b->counters.host_pixel_bytes;
    counters->rgb_conversion_bytes = b->counters.conversion_output_bytes;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_get_contract(NvencBridge* b,
    NvencHdrContract* contract) {
    last_error[0] = 0;
    if (!b || !contract || contract->struct_size != sizeof(NvencHdrContract)) {
        fail("nvenc_hdr_get_contract struct_size", EINVAL);
        return -1;
    }
    if (!b->p010_mode) {
        std::snprintf(last_error, sizeof(last_error),
                      "nvenc_hdr_get_contract needs a P010 bridge");
        return -1;
    }
    *contract = b->hdr_contract;
    return 0;
}

/* ------------------------------------------------------------------ *
 * Public entries of the in-bitstream metadata writer. They are additive:
 * no struct above moves, and none of them is reachable from the SDR
 * entries. The two packet-level entries run on host-compressed bytes only;
 * they never take a device pointer, never decode and never re-encode.
 * ------------------------------------------------------------------ */

extern "C" NVENC_BRIDGE_API uint32_t nvenc_hdr_sei_abi_version(void) {
    return NVENC_HDR_SEI_ABI_VERSION;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_sei_capability(
    NvencHdrSeiCapability* out_capability) {
    last_error[0] = 0;
    if (!out_capability ||
        out_capability->struct_size != sizeof(NvencHdrSeiCapability)) {
        fail("nvenc_hdr_sei_capability struct_size", EINVAL);
        return -1;
    }
    NvencHdrSeiCapability cap{0};
    cap.struct_size = sizeof(NvencHdrSeiCapability);
    cap.writer_available = 1;
    cap.encoder_offers_metadata_option = 0;
    cap.supports_partial_mastering = 0;
    cap.primary_order = 1;   // 1 = green, blue, red, as Annex D numbers c
    cap.chroma_denominator = static_cast<int32_t>(kSeiChromaDen);
    cap.luma_denominator = static_cast<int32_t>(kSeiLumaDen);
    cap.payload_type_mastering = kSeiTypeMastering;
    cap.payload_type_content_light = kSeiTypeContentLight;
    cap.prefix_sei_nal_type = kSeiNalPrefix;
    cap.mastering_payload_size = kSeiMasteringBytes;
    cap.content_light_payload_size = kSeiContentLightBytes;
    // Both messages, fully escaped, plus the two-byte NAL header and a
    // four-byte start code or length field.
    cap.max_nal_bytes = 2u + 33u + 11u + 1u;
    cap.max_insert_bytes = cap.max_nal_bytes + 4u;
    *out_capability = cap;
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_sei_encode_nal(
    const NvencHdrMetadata* metadata, uint8_t* out_nal, uint32_t out_capacity,
    uint32_t* out_nal_size, NvencHdrSeiInfo* out_info) {
    last_error[0] = 0;
    if (!metadata || metadata->struct_size != sizeof(NvencHdrMetadata) ||
        !out_nal || !out_nal_size || !out_info ||
        out_info->struct_size != sizeof(NvencHdrSeiInfo)) {
        fail("nvenc_hdr_sei_encode_nal arguments", EINVAL);
        return -1;
    }
    *out_nal_size = 0;
    out_info->error = NVENC_HDR_SEI_OK;
    out_info->detail[0] = '\0';
    out_info->nal_bytes = 0;
    const bool complete = sei_mastering_complete(*metadata);
    const bool light = metadata->has_content_light != 0;
    if (!complete && !light) {
        // An empty prefix SEI NAL would be a payload a decoder reads as no
        // metadata at all, so nothing is built and nothing is claimed.
        sei_note(out_info, NVENC_HDR_SEI_NOTHING_TO_DO,
                 "the source declared no complete mastering display and no "
                 "content light level: no SEI NAL is built");
        return 0;
    }
    std::vector<uint8_t> nal, rbsp, escaped;
    if (!sei_build_nal(*metadata, complete, light, nal, rbsp, escaped,
                       out_info)) {
        std::snprintf(last_error, sizeof(last_error), "%s", out_info->detail);
        return -1;
    }
    if (nal.size() > out_capacity) {
        sei_note(out_info, NVENC_HDR_SEI_CAPACITY,
                 "the prefix SEI NAL is %u bytes and the caller offered %u",
                 static_cast<uint32_t>(nal.size()), out_capacity);
        std::snprintf(last_error, sizeof(last_error), "%s", out_info->detail);
        return -1;
    }
    if (!nal.empty()) std::memcpy(out_nal, nal.data(), nal.size());
    *out_nal_size = static_cast<uint32_t>(nal.size());
    out_info->nal_bytes = static_cast<uint32_t>(nal.size());
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_sei_apply_packet(
    const NvencHdrMetadata* metadata, const uint8_t* in, uint32_t in_size,
    uint8_t* out, uint32_t out_capacity, uint32_t* out_size,
    NvencHdrSeiInfo* out_info) {
    last_error[0] = 0;
    if (!metadata || metadata->struct_size != sizeof(NvencHdrMetadata) ||
        !out || !out_size || !out_info ||
        out_info->struct_size != sizeof(NvencHdrSeiInfo)) {
        fail("nvenc_hdr_sei_apply_packet arguments", EINVAL);
        return -1;
    }
    *out_size = 0;
    std::vector<uint8_t> packet, nal, rbsp, escaped;
    const int rc = sei_apply_bytes(*metadata, in, in_size, packet, nal, rbsp,
                                   escaped, out_info, nullptr);
    if (rc < 0) {
        std::snprintf(last_error, sizeof(last_error), "%s", out_info->detail);
        return -1;
    }
    // Nothing to add to this access unit: the packet is handed back as it
    // arrived, byte for byte, so a caller never has to special-case it.
    const uint8_t* src = packet.empty() ? in : packet.data();
    const size_t size = packet.empty() ? in_size : packet.size();
    if (size > out_capacity) {
        sei_note(out_info, NVENC_HDR_SEI_CAPACITY,
                 "the packet is %u bytes and the caller offered %u",
                 static_cast<uint32_t>(size), out_capacity);
        std::snprintf(last_error, sizeof(last_error), "%s", out_info->detail);
        return -1;
    }
    if (size) std::memcpy(out, src, size);
    *out_size = static_cast<uint32_t>(size);
    return 0;
}

extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_sei_get(NvencBridge* b,
    NvencHdrSeiStatus* out_status) {
    last_error[0] = 0;
    if (!b || !out_status ||
        out_status->struct_size != sizeof(NvencHdrSeiStatus)) {
        fail("nvenc_hdr_sei_get struct_size", EINVAL);
        return -1;
    }
    if (!b->p010_mode) {
        std::snprintf(last_error, sizeof(last_error),
                      "nvenc_hdr_sei_get needs a P010 bridge");
        return -1;
    }
    *out_status = b->hdr_sei;
    return 0;
}

extern "C" NVENC_BRIDGE_API uint32_t nvenc_p010_abi_version(void) { return NVENC_P010_ABI_VERSION; }
extern "C" NVENC_BRIDGE_API int32_t nvenc_hdr_capability(void* context,
    uint32_t width, uint32_t height, NvencHdrCapability* out) {
    return p010_capability_impl(kHdrCodec, context, width, height, out);
}
extern "C" NVENC_BRIDGE_API int32_t nvenc_p010_capability(void* context,
    uint32_t width, uint32_t height, const char* codec, NvencP010Capability* out) {
    if (!out || out->struct_size != sizeof(*out)) { fail("P010 capability layout",EINVAL); return -1; }
    static_assert(sizeof(NvencP010Capability) == sizeof(NvencHdrCapability), "capability layout");
    NvencHdrCapability temp{}; temp.struct_size=sizeof(temp);
    const int rc=p010_capability_impl(codec,context,width,height,&temp);
    if (rc == 0 && temp.supported) {
        NvencBridge probe{};
        probe.context=static_cast<CUcontext>(context); probe.width=width; probe.height=height;
        probe.fps={25,1}; probe.time_base={1,25}; probe.p010_mode=true;
        P010ColorContract contract=kSdr10P010Contract;
        const bool av1=std::strcmp(codec,"av1_nvenc") == 0;
        contract.codec=codec; contract.profile_name=av1 ? "main" : "main10";
        NvencP010Options options{}; options.struct_size=sizeof(options);
        options.codec=codec; options.profile=contract.profile_name;
        options.preset="p4"; options.tune="hq"; options.rate_control="vbr";
        options.constant_quality=20;
        temp.supported=load_driver(&probe.cuda) &&
            create_ffmpeg_hdr(&probe,&options,contract,false) == 0;
        release_all(&probe);
    }
    std::memcpy(out,&temp,sizeof(temp)); return rc;
}
extern "C" NVENC_BRIDGE_API int32_t nvenc_p010_open(void* context,
    uint32_t width,uint32_t height,uint32_t fps_num,uint32_t fps_den,
    int32_t tb_num,int32_t tb_den,const char* path,const NvencP010Options* options,
    const NvencHdrMetadata* metadata,int32_t color_mode,NvencBridge** out) {
    if (color_mode != 0 && color_mode != 1) { if(out) *out=nullptr;
        fail("P010 color mode must be SDR10=0 or HDR10=1",EINVAL); return -1; }
    return nvenc_p010_open_impl(context,width,height,fps_num,fps_den,tb_num,tb_den,
        path,options,metadata,out,color_mode == 1,false);
}
extern "C" NVENC_BRIDGE_API int32_t nvenc_p010_get_contract(NvencBridge* b,NvencP010Contract* out) {
    return nvenc_hdr_get_contract(b,out);
}
extern "C" NVENC_BRIDGE_API int32_t nvenc_p010_get_metadata_status(NvencBridge* b,NvencP010MetadataStatus* out) {
    if(!b || !out || out->struct_size != sizeof(*out) || !b->p010_mode) return -1;
    if(b->codec_name != "av1_nvenc") {
        auto& s=b->metadata_status; const auto& old=b->hdr_sei;
        s.metadata_requested=old.metadata_requested; s.mastering_serialized=old.mastering_serialized;
        s.content_light_serialized=old.content_light_serialized; s.error=old.error;
        s.packets_seen=old.packets_seen; s.mastering_inserted=old.mastering_inserted;
        s.content_light_inserted=old.content_light_inserted;
        s.mastering_already_present=old.mastering_already_present;
        s.content_light_already_present=old.content_light_already_present; s.bytes_inserted=old.bytes_inserted;
        std::snprintf(s.detail,sizeof(s.detail),"%s",old.detail);
    }
    *out=b->metadata_status; return 0;
}
extern "C" NVENC_BRIDGE_API int32_t nvenc_av1_metadata_apply_packet(
    const NvencHdrMetadata* md,const uint8_t* in,uint32_t size,uint8_t* out,uint32_t capacity,
    uint32_t* out_size,NvencP010MetadataStatus* status) {
    if(!md || md->struct_size != sizeof(*md) || !in || !size || !out_size ||
       !status || status->struct_size != sizeof(*status)) return -1;
    *status=NvencP010MetadataStatus{}; *out_size=0;
    Bytes result;
    if(!av1_apply(*md,in,size,result,*status)) return -1;
    *out_size=static_cast<uint32_t>(result.size());
    if(!out || capacity < result.size()) return 1;
    std::memcpy(out,result.data(),result.size()); return 0;
}
namespace {
int hdr_av1_apply(NvencBridge* b) {
    Bytes result;
    if(!av1_apply(b->hdr_metadata,b->packet->data,static_cast<uint32_t>(b->packet->size),result,b->metadata_status)) {
        std::snprintf(last_error,sizeof(last_error),"AV1 HDR metadata: %s",b->metadata_status.detail);
        b->failed=true; return -1;
    }
    if(result.size() == static_cast<size_t>(b->packet->size)) return 0;
    AVBufferRef* ref=av_buffer_alloc(result.size()+AV_INPUT_BUFFER_PADDING_SIZE);
    if(!ref) { fail("AV1 metadata packet allocation",ENOMEM); return -1; }
    std::memcpy(ref->data,result.data(),result.size());
    std::memset(ref->data+result.size(),0,AV_INPUT_BUFFER_PADDING_SIZE);
    av_buffer_unref(&b->packet->buf); b->packet->buf=ref; b->packet->data=ref->data;
    b->packet->size=static_cast<int>(result.size()); b->hdr_contract.metadata_written_to |= 32;
    return 0;
}
}
