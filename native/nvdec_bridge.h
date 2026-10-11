#ifndef NVDEC_BRIDGE_H
#define NVDEC_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef _WIN32
#  ifdef NVDEC_BRIDGE_EXPORTS
#    define NVDEC_BRIDGE_API __declspec(dllexport)
#  else
#    define NVDEC_BRIDGE_API __declspec(dllimport)
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 3 adds NvdecTransferCounters, nvdec_bridge_get_transfer_counters and the
 * counted diagnostic readback. Every ABI 2 entry point keeps its signature and
 * behaviour, and NvdecDeviceFrame is unchanged, so an ABI 2 caller's struct
 * still has the layout it was compiled against.
 *
 * P010 borrows extend the same contract with additive open entries; nothing in
 * ABI 3 changes signature, layout or behaviour, so an existing caller loads and
 * runs against this DLL exactly as before. nvdec_bridge_open_hdr and
 * nvdec_bridge_open_p010 opt a bridge into reporting 10-bit CUDA surfaces; an
 * SDR-opened bridge still refuses them, so a 10-bit source cannot quietly be
 * read as NV12 by a caller compiled before P010 existed. */
#define NVDEC_BRIDGE_ABI_VERSION 3u

typedef struct NvdecBridge NvdecBridge;

/* The pointers are borrowed CUDA device addresses. They remain valid until
 * the next nvdec_bridge_next call or close, whichever comes first. Calling
 * nvdec_bridge_next creates and downloads no host frame; the counted readback
 * below is the only call in this API that moves frame pixels to host memory.
 * PTS uses time_base_num/den. */
typedef struct NvdecDeviceFrame {
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    /* 1 = NV12, 2 = P010 (AV_PIX_FMT_P010LE: 16-bit little-endian samples,
     * left-aligned 10-bit, semi-planar UV interleaved at half height). Both are
     * semi-planar, so data[1] holds the UV pair plane either way; what changes
     * is the sample width. A 2 is only ever reported to a bridge opened by
     * nvdec_bridge_open_hdr or nvdec_bridge_open_p010. */
    uint32_t software_format;
    uint64_t y_device_ptr;
    uint64_t uv_device_ptr;
    /* Both pitches are bytes, straight from the AVFrame linesize: an NV12 Y row
     * holds width samples, a P010 Y row holds width 16-bit samples, and a P010
     * UV row holds width samples like an NV12 UV row does because each chroma
     * sample is twice as wide. */
    int32_t y_pitch_bytes;
    int32_t uv_pitch_bytes;
    void* cuda_context;
    int64_t pts;
    int32_t time_base_num;
    int32_t time_base_den;
    int32_t color_range;
    int32_t color_primaries;
    int32_t color_transfer;
    int32_t color_space;
} NvdecDeviceFrame;

/* Every field is incremented by the calls that ran, so a zero is a measurement
 * of a boundary that was not crossed rather than a default. The production
 * decode path this bridge feeds hands out borrowed device surfaces only:
 * frames_decoded and device_frame_count rise together, host_frame_copy_count
 * and host_frame_copy_bytes stay at zero because nothing calls the readback
 * below, and scale_device_output_bytes stays at zero unless a caller opened the
 * scaling decoder. */
typedef struct NvdecTransferCounters {
    uint32_t struct_size;
    uint64_t frames_decoded;            /* frames nvdec_bridge_next returned */
    uint64_t device_frame_count;        /* of those, borrowed CUDA NV12 surfaces */
    uint64_t bitstream_bytes_submitted; /* compressed bytes handed to the decoder
                                         * API, which uploads them itself: packet
                                         * traffic, not frame traffic */
    uint64_t host_frame_copy_count;     /* explicit frame downloads to host */
    uint64_t host_frame_copy_bytes;     /* NV12 payload bytes those wrote */
    uint64_t scale_device_output_bytes; /* scale_cuda output surface payload, one
                                         * frame's worth per scaled frame */
} NvdecTransferCounters;

NVDEC_BRIDGE_API uint32_t nvdec_bridge_abi_version(void);
NVDEC_BRIDGE_API int32_t nvdec_bridge_open(const char* utf8_path,
                                           NvdecBridge** out_bridge);
/* Decode and scale NV12 inside CUDA. Both dimensions must be positive and even.
 * The returned CUDA surface has the requested dimensions; source PTS and color
 * metadata remain attached to each frame. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_open_scaled(const char* utf8_path,
    uint32_t output_width, uint32_t output_height, NvdecBridge** out_bridge);
/* 1 = frame, 0 = end of stream, negative = error. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_next(NvdecBridge* bridge,
                                           NvdecDeviceFrame* out_frame);
/* Preview seek, in microseconds on the source media timeline relative to the
 * video's stream start. Flushes decode history; subsequent next calls may
 * return preroll frames from the preceding keyframe. Never used by exports. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_seek_us(NvdecBridge* bridge,
                                            int64_t relative_microseconds);

/* Copies the most recent frame's NV12 payload into host memory and counts it in
 * host_frame_copy_count/host_frame_copy_bytes. Y occupies height rows, then UV
 * height/2 rows, each row_stride bytes wide with the frame's own width bytes
 * written per row, so the buffer needs row_stride*(height + height/2) bytes; a
 * row_stride below the frame width is refused. The device surface the caller
 * was handed stays valid. Diagnostic only - production export never calls it -
 * and it must run on the thread that called nvdec_bridge_next. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_readback_last_frame(NvdecBridge* bridge,
    uint8_t* host, size_t row_stride);

NVDEC_BRIDGE_API int32_t nvdec_bridge_get_transfer_counters(NvdecBridge* bridge,
    NvdecTransferCounters* counters);
NVDEC_BRIDGE_API void nvdec_bridge_cancel(NvdecBridge* bridge);
NVDEC_BRIDGE_API void nvdec_bridge_close(NvdecBridge* bridge);
NVDEC_BRIDGE_API const char* nvdec_bridge_last_error(void);

/* Opens for an HDR10 borrow: identical to nvdec_bridge_open except that a CUDA
 * frame whose software format is P010 is reported instead of refused. The
 * scaling variants are not part of this contract: scale_cuda only resizes 8-bit
 * NV12 here, and the HDR chain downsamples in its own FP16 stage, so an
 * nvdec_bridge_open_hdr bridge never builds a scaler and
 * scale_device_output_bytes stays at zero. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_open_hdr(const char* utf8_path,
                                               NvdecBridge** out_bridge);
/* P010-capable decoder for SDR 10-bit input. Like open_hdr this only selects
 * whether P010 CUDA surfaces may be returned; frame color tags are passed
 * through from the decoded AVFrame unchanged and no scaler is enabled. */
NVDEC_BRIDGE_API int32_t nvdec_bridge_open_p010(const char* utf8_path,
                                                NvdecBridge** out_bridge);

/* Static HDR metadata as the source declares it, field for field. Luminances
 * and chromaticities are the source's own rationals, kept exact rather than
 * rounded through a float, because the encoder writes them back out again and
 * the point of this struct is that a round trip through the chain does not edit
 * the mastering display. has_* is 0 when the source carries no such side data;
 * nothing here is invented to fill a gap.
 *
 * Valid on the thread that called nvdec_bridge_next, for as long as that frame
 * is the current one, for the same reason the borrowed surfaces are: it reads
 * the side data attached to it. Querying a frame's metadata copies no pixels. */
typedef struct NvdecHdrMetadata {
    uint32_t struct_size;
    int32_t has_mastering;
    /* CIE 1931 xy of the display primaries in r, g, b order, then the white
     * point: the six x,y pairs and two of ST 2086, as numerator/denominator. */
    int32_t primary_x_num[3];
    int32_t primary_x_den[3];
    int32_t primary_y_num[3];
    int32_t primary_y_den[3];
    int32_t white_x_num, white_x_den;
    int32_t white_y_num, white_y_den;
    /* cd/m^2. */
    int32_t min_luminance_num, min_luminance_den;
    int32_t max_luminance_num, max_luminance_den;
    /* The ST 2086 flags: a mastering side datum can carry primaries without
     * luminance or the other way round, and that distinction is the source's,
     * not this bridge's to smooth over. */
    int32_t mastering_has_primaries;
    int32_t mastering_has_luminance;
    int32_t has_content_light;
    /* nits, as CTA-861.3 defines them: 0 is "not specified" in that format and
     * is reported as 0 here rather than as a value the source did not give. */
    uint32_t max_cll;
    uint32_t max_fall;
} NvdecHdrMetadata;

NVDEC_BRIDGE_API int32_t nvdec_bridge_get_hdr_metadata(NvdecBridge* bridge,
    NvdecHdrMetadata* out_metadata);

#ifdef __cplusplus
}
#endif

#endif
