#ifndef NVENC_BRIDGE_H
#define NVENC_BRIDGE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#  ifdef NVENC_BRIDGE_EXPORTS
#    define NVENC_BRIDGE_API __declspec(dllexport)
#  else
#    define NVENC_BRIDGE_API __declspec(dllimport)
#  endif
#endif

#define NVENC_BRIDGE_ABI_VERSION 2u
#define NVENC_COLOR_MATRIX_BT709 0
#define NVENC_COLOR_MATRIX_BT601 1

typedef struct NvencBridge NvencBridge;

/* Encoder and pixel-format request. Strings are copied during open. A zero or
 * NULL numeric/string leaves the corresponding FFmpeg/NVENC option at its own
 * default; rate_control is an NVENC name such as "vbr" or "cbr". */
typedef struct NvencBridgeOptions {
    uint32_t struct_size;
    const char* codec;          /* av1_nvenc (default), hevc_nvenc, h264_nvenc */
    const char* preset;         /* p1..p7 */
    const char* tune;           /* hq, ll, ultralowlat, lossless, bloom */
    const char* multipass;      /* none, quarter, full */
    const char* rate_control;   /* vbr, cbr, cbr_bd, lossless */
    int32_t constant_quality;   /* cq, 0 = unset */
    int64_t bitrate;            /* bits/s, 0 = unset */
    int32_t gop;                /* keyframe interval, 0 = unset */
    int32_t color_matrix;       /* NVENC_COLOR_MATRIX_* ; selects the RGB->NV12
                                 * coefficients and the declared matrix */
    /* ABI 2. Newline-separated "name=value" pairs applied to the encoder after
     * the fields above, so an option this struct does not model still reaches
     * the codec instead of being silently dropped. A name the encoder does not
     * have fails the open; query it first with nvenc_option_available. */
    const char* extra_options;
} NvencBridgeOptions;

/* Every counter in this struct is a device-side or host-side byte total owned
 * by this bridge. host_pixel_bytes has no code path that can raise it: image
 * bytes never cross to the host in this bridge. */
typedef struct NvencTransferCounters {
    uint32_t struct_size;
    uint64_t frames_submitted;
    uint64_t packets_written;
    uint64_t staging_gpu_copy_bytes;   /* D3D12 texture -> D3D12 staging buffer */
    uint64_t conversion_output_bytes;  /* CUDA RGB->NV12 writes: w*h*3/2 each */
    uint64_t host_pixel_bytes;
} NvencTransferCounters;

NVENC_BRIDGE_API uint32_t nvenc_abi_version(void);
NVENC_BRIDGE_API const char* nvenc_last_error(void);

/* 1 if the named encoder exists and exposes that AVOption, 0 if it does not,
 * negative on a lookup failure. Lets a caller decide before opening a GPU
 * session whether a requested encoding setting can be honoured. */
NVENC_BRIDGE_API int32_t nvenc_option_available(const char* codec,
                                                const char* name);

/* d3d12_device is a borrowed ID3D12Device* that produces encode_texture inputs;
 * cuda_context is a borrowed CUcontext validated against the same adapter LUID.
 * This call must run on the thread whose current CUDA context is cuda_context
 * or has it pushable, and both objects outlive the bridge. The output file is
 * opened immediately, so a later close() truncates rather than hides a failure.
 * time_base_num/den are the units of the pts values passed to the encode calls;
 * fps_num/fps_den are written as the stream frame rate. */
NVENC_BRIDGE_API int32_t nvenc_open(void* d3d12_device, void* cuda_context,
    uint32_t width, uint32_t height, uint32_t fps_num, uint32_t fps_den,
    int32_t time_base_num, int32_t time_base_den, const char* utf8_output_path,
    const NvencBridgeOptions* options, NvencBridge** out_bridge);

/* One RGBA8 frame that already lives in cuda_context as a device allocation.
 * The caller owns the allocation and must keep the pixels unchanged until this
 * call returns; the bridge only reads it. No staging copy is performed. */
NVENC_BRIDGE_API int32_t nvenc_encode_rgba(NvencBridge* bridge,
    uint64_t rgba_ptr, size_t rgba_pitch, int64_t pts);

/* One RGBA8 D3D12 TEXTURE2D owned by d3d12_device, left by its producer in
 * D3D12_RESOURCE_STATE_COPY_SOURCE. This bridge never transitions a borrowed
 * resource. producer_fence/producer_fence_value are optional (NULL or 0 means
 * the texture is already complete); when given, the staging queue waits on
 * them before copying. The bridge owns the linear staging buffer, which is a
 * CUDA-imported D3D12 shared resource, and returns only after the staging copy
 * and the RGB->NV12 conversion have both completed on the GPU. */
NVENC_BRIDGE_API int32_t nvenc_encode_texture(NvencBridge* bridge,
    void* rgba_texture, void* producer_fence, uint64_t producer_fence_value,
    int64_t pts);

/* Diagnostic only: the NV12 planes of the most recent conversion, valid in
 * cuda_context until the next encode call. Exists so a test can verify the
 * kernel against a host reference; never feed these addresses anywhere else. */
NVENC_BRIDGE_API int32_t nvenc_get_last_nv12_planes(NvencBridge* bridge,
    uint64_t* out_y_ptr, uint32_t* out_y_pitch, uint64_t* out_uv_ptr,
    uint32_t* out_uv_pitch);

NVENC_BRIDGE_API int32_t nvenc_get_transfer_counters(NvencBridge* bridge,
    NvencTransferCounters* counters);
/* Sends the flush signal and drains the encoder, but does not write the
 * container trailer; call once after the last frame. */
NVENC_BRIDGE_API int32_t nvenc_finish(NvencBridge* bridge);
/* Writes the container trailer (if not finished), releases encoder, staging,
 * and queue resources. Borrowed device/context objects are not released. */
NVENC_BRIDGE_API void nvenc_close(NvencBridge* bridge);

/* ==================================================================== *
 * Experimental HDR10 entry. Additive: NvencBridgeOptions (ABI 2),
 * NvencTransferCounters and every prototype above keep its signature,
 * layout and behaviour, and nothing here is reachable from nvenc_open,
 * so an SDR caller neither sees nor depends on this section.
 *
 * Scope of this entry, stated plainly: HEVC NVENC Main10 fed with P010
 * CUDA surfaces on the caller's own context, plus the HDR10 colour
 * declaration (limited range, BT.2020 primaries, PQ transfer,
 * BT.2020-NCL matrix) and the mastering-display / MaxCLL / MaxFALL
 * handoff. A combination this entry cannot honour is refused with the
 * encoder's own error; it never degrades to NV12, never drops an option
 * it was given, and never tonemaps.
 * ==================================================================== */
#define NVENC_HDR_ABI_VERSION 1u
#define NVENC_SDR10_ABI_VERSION 1u

/* Static HDR10 metadata exactly as the caller's source declared it. The
 * field order, types and meanings are the ones nvdec_bridge.h's
 * NvdecHdrMetadata reports, so a decode -> encode handoff copies the
 * source's own rationals without passing them through a float. has_* and
 * the two mastering flags stay as read; a zero here means the source did
 * not say it, and the bridge then attaches nothing rather than inventing
 * a default. Container rationals are retained in side data. The HEVC SEI
 * serializer converts valid values to the nearest 1/50000 chromaticity or
 * 1/10000 luminance unit (ties upward); exact-grid values are unchanged. */
typedef struct NvencHdrMetadata {
    uint32_t struct_size;
    int32_t has_mastering;
    int32_t primary_x_num[3];
    int32_t primary_x_den[3];
    int32_t primary_y_num[3];
    int32_t primary_y_den[3];
    int32_t white_x_num, white_x_den;
    int32_t white_y_num, white_y_den;
    int32_t min_luminance_num, min_luminance_den;
    int32_t max_luminance_num, max_luminance_den;
    int32_t mastering_has_primaries;
    int32_t mastering_has_luminance;
    int32_t has_content_light;
    uint32_t max_cll;
    uint32_t max_fall;
} NvencHdrMetadata;

/* Encoder request for the HDR10 entry. The strings carry the same NVENC
 * names as in NvencBridgeOptions and reach the codec the same way; a name
 * the encoder does not have fails the open. The numeric fields below that
 * line state the contract the entry insists on: 0 takes the HDR10 value,
 * and any other value must equal the HDR10 value or the open is refused.
 * There is no field with which a caller can ask for NV12, 8-bit, BT.709
 * or H.264 through this entry. */
typedef struct NvencHdrOptions {
    uint32_t struct_size;
    const char* codec;           /* NULL or "hevc_nvenc" only */
    const char* profile;         /* NULL or "main10" only */
    const char* preset;
    const char* tune;
    const char* multipass;
    const char* rate_control;
    int32_t constant_quality;    /* cq, 0 = unset */
    int64_t bitrate;             /* bits/s, 0 = unset */
    int32_t gop;                 /* keyframe interval, 0 = unset */
    const char* extra_options;   /* newline-separated name=value */
    int32_t sw_pixel_format;     /* 0 or AV_PIX_FMT_P010LE */
    int32_t color_range;         /* 0 or the limited-range enumerator */
    int32_t color_primaries;     /* 0 or the BT.2020 enumerator */
    int32_t color_trc;           /* 0 or the SMPTE 2084 enumerator */
    int32_t color_space;         /* 0 or the BT.2020-NCL enumerator */
} NvencHdrOptions;

/* What the entry requires, so a caller can compare its own request against
 * the installed contract without hardcoding FFmpeg enumerator values. */
typedef struct NvencHdrDefaults {
    uint32_t struct_size;
    int32_t sw_pixel_format;
    int32_t color_range;
    int32_t color_primaries;
    int32_t color_trc;
    int32_t color_space;
    char profile[32];            /* the encoder's own name for Main10 */
} NvencHdrDefaults;

/* Capacity of the installed build and driver, read back from FFmpeg and
 * NVENC rather than assumed: the encoder's advertised pixel formats and
 * profiles, whether a CUDA device object can actually be created on the
 * queried context, and how many HDR10-named options the installed encoder
 * exposes. Read hdr_metadata_options for exactly what it is: the encoder
 * offers no option by which it could be asked to author an ST 2086 or
 * CTA-861.3 SEI itself. It is NOT a statement that the bitstream cannot
 * carry one - the writer in the section below authors those two payloads
 * from the compressed bytes of the encoder's own packets, so the required
 * in-bitstream contract is met without any option the encoder has. Ask
 * nvenc_hdr_sei_capability for the writer's own availability, and verify
 * the elementary stream either way: container colour tags are not
 * metadata. */
typedef struct NvencHdrCapability {
    uint32_t struct_size;
    int32_t encoder_present;
    int32_t p010_supported;
    int32_t cuda_supported;
    int32_t main10_profile_supported;
    int32_t cuda_device_created;
    int32_t hdr_metadata_options;
    int32_t supported;           /* every requirement above met */
    char encoder_name[64];
    char profile_name[32];
} NvencHdrCapability;

/* The contract the bridge submitted, and the contract the opened encoder
 * reports back. requested_* is what this call asked for; actual_* is read
 * from the AVCodecContext after avcodec_open2, so a negotiated fallback
 * would be visible here rather than assumed away. */
typedef struct NvencHdrContract {
    uint32_t struct_size;
    char codec[64];
    char container[32];
    char profile_name[32];
    int32_t requested_pix_fmt;
    int32_t actual_pix_fmt;
    int32_t requested_sw_pix_fmt;
    int32_t actual_sw_pix_fmt;
    char actual_sw_pix_fmt_name[32];
    int32_t requested_profile;
    int32_t actual_profile;
    int32_t requested_bits_per_raw_sample;
    int32_t actual_bits_per_raw_sample;
    int32_t requested_color_range;
    int32_t requested_color_primaries;
    int32_t requested_color_trc;
    int32_t requested_color_space;
    int32_t actual_color_range;
    int32_t actual_color_primaries;
    int32_t actual_color_trc;
    int32_t actual_color_space;
    uint32_t width, height;
    int32_t hw_frames_pool_size;
    int32_t hw_frames_sw_format;
    /* Where the bridge wrote the metadata it was given: 1 frame side data,
     * 2 stream coded side data, 4 encoder coded side data, 8 container
     * colour tags, 16 an in-bitstream prefix SEI NAL authored by the writer
     * below and actually inserted into a packet. A bit that is clear was not
     * attempted. Only 16 is proof about the elementary stream. */
    int32_t metadata_written_to;
    int32_t has_mastering;
    int32_t has_content_light;
    /* cuDeviceGetLuid of the borrowed context's device, so a caller can
     * prove the encoder and its own producer are on one adapter. */
    uint8_t cuda_adapter_luid[8];
    uint32_t cuda_adapter_node_mask;
} NvencHdrContract;

/* Additive HDR counters. The struct above (NvencTransferCounters) keeps its
 * layout and its meanings; the four shared fields here (frames_submitted,
 * packets_written, host_pixel_bytes, rgb_conversion_bytes) are reported from
 * that same single source of truth, so an HDR bridge and its own counters can
 * never disagree. p010_device_copy_bytes, mastering_side_data and
 * content_light_side_data are new measurements that only this entry makes. The
 * two zeros are boundaries this entry cannot cross: no host pixel ever appears,
 * and the RGB->NV12 kernel is never launched, so rgb_conversion_bytes stays 0
 * because conversion_output_bytes does. */
typedef struct NvencHdrCounters {
    uint32_t struct_size;
    uint64_t frames_submitted;
    uint64_t packets_written;
    uint64_t p010_device_copy_bytes;  /* device-to-device plane payload */
    uint64_t host_pixel_bytes;        /* always 0 here */
    uint64_t rgb_conversion_bytes;    /* always 0 here */
    uint64_t mastering_side_data;     /* frames carrying ST 2086 side data */
    uint64_t content_light_side_data; /* frames carrying CTA-861.3 side data */
} NvencHdrCounters;

NVENC_BRIDGE_API uint32_t nvenc_hdr_abi_version(void);
/* Fills the struct with the contract this entry insists on. 0 on success. */
NVENC_BRIDGE_API int32_t nvenc_hdr_get_defaults(NvencHdrDefaults* out_defaults);
/* Queries the installed encoder without opening an NVENC session.
 * cuda_context may be NULL, in which case cuda_device_created is reported
 * as 0 and the rest of the static capacity is still answered. */
NVENC_BRIDGE_API int32_t nvenc_hdr_capability(void* cuda_context,
    uint32_t width, uint32_t height, NvencHdrCapability* out_capability);
/* Opens the HDR10 entry. cuda_context is a borrowed CUcontext that must be
 * current on this thread or pushable by it, and must outlive the bridge;
 * the P010 pointers handed to nvenc_encode_p010 have to come from it. No
 * D3D12 device is needed or taken: this path neither stages nor converts.
 * metadata may be NULL, meaning the source declared none, which the
 * encoder then does not invent. nvenc_finish and nvenc_close drain and
 * release it exactly as the SDR path does. */
NVENC_BRIDGE_API int32_t nvenc_hdr_open(void* cuda_context, uint32_t width,
    uint32_t height, uint32_t fps_num, uint32_t fps_den, int32_t time_base_num,
    int32_t time_base_den, const char* utf8_output_path,
    const NvencHdrOptions* options, const NvencHdrMetadata* metadata,
    NvencBridge** out_bridge);
/* SDR 10-bit entry: same borrowed-context and P010-plane lifecycle as the HDR
 * entry, but fixed to limited-range BT.709 primaries, transfer and matrix.
 * Main10 HEVC and P010 are still required. Nonzero HDR metadata and HDR option
 * overrides are refused; no HDR SEI or mastering/content-light metadata is
 * written. The NvencHdrOptions/NvencHdrMetadata layouts are reused unchanged. */
NVENC_BRIDGE_API uint32_t nvenc_sdr10_abi_version(void);
NVENC_BRIDGE_API int32_t nvenc_sdr10_open(void* cuda_context, uint32_t width,
    uint32_t height, uint32_t fps_num, uint32_t fps_den, int32_t time_base_num,
    int32_t time_base_den, const char* utf8_output_path,
    const NvencHdrOptions* options, const NvencHdrMetadata* metadata,
    NvencBridge** out_bridge);
/* One P010 frame that already lives in the opened context as two device
 * planes (left-aligned 10-bit samples, semi-planar UV at half height), with
 * their real byte pitches and PTS. The caller owns both planes and must keep
 * them unchanged until this call returns: the copy is synchronous, and the
 * bridge only reads. The encoder's own pooled surface is what gets handed
 * to the codec, so the borrowed planes are never registered with NVENC. */
NVENC_BRIDGE_API int32_t nvenc_encode_p010(NvencBridge* bridge,
    uint64_t y_ptr, size_t y_pitch, uint64_t uv_ptr, size_t uv_pitch,
    int64_t pts);
NVENC_BRIDGE_API int32_t nvenc_hdr_get_counters(NvencBridge* bridge,
    NvencHdrCounters* counters);
NVENC_BRIDGE_API int32_t nvenc_hdr_get_contract(NvencBridge* bridge,
    NvencHdrContract* contract);

/* ==================================================================== *
 * HDR10 in-bitstream metadata writer. Additive to the HDR section
 * above and to ABI 2: no existing struct moves.
 *
 * Why this exists: the installed hevc_nvenc wrapper takes the mastering
 * and content-light side data and puts it in the container, and the
 * installed FFmpeg (7.1.1) offers no public bitstream filter that authors
 * an ST 2086 mastering_display_colour_volume or a CTA-861.3
 * content_light_level_info SEI - hevc_metadata edits parameter sets and
 * VUI only. So the elementary stream the encoder produced carried no SEI
 * at all. This writer authors exactly those two payloads itself, on the
 * compressed bytes of an already-encoded packet, at the boundary between
 * avcodec_receive_packet and the muxer. It never touches an image
 * surface, never decodes, never re-encodes and never converts a pixel.
 *
 * Syntax is written from ITU-T H.265 Annex D (D.2.27, D.2.28) and Annex B
 * (emulation prevention), plus the NAL header of 7.3.2:
 *   prefix SEI       nal_unit_type 39, nuh_layer_id 0,
 *                    nuh_temporal_id_plus1 1
 *   mastering        payloadType 137, payloadSize 24:
 *                    for c = 0..2: display_primaries_x[c] u(16),
 *                                   display_primaries_y[c] u(16),
 *                    then white_point_x u(16), white_point_y u(16),
 *                    then max_display_mastering_luminance u(32),
 *                         min_display_mastering_luminance u(32)
 *                    HEVC orders the primaries green, blue, red, not the
 *                    red, green, blue of AVMasteringDisplayMetadata, and
 *                    quantises chromaticity in 1/50000 and luminance in
 *                    1/10000 cd/m2.
 *   content light    payloadType 144, payloadSize 4:
 *                    max_content_light_level u(16),
 *                    max_pic_average_light_level u(16)
 * A prefix SEI NAL unit is inserted immediately before the first VCL NAL
 * unit of every access unit; AUD, the parameter sets, every other NAL and
 * all slice bytes are copied through unchanged. Repeating the static
 * declaration per picture is valid: it applies to the whole coded video
 * sequence, and association with the access unit is what Annex D requires.
 * ==================================================================== */
#define NVENC_HDR_SEI_ABI_VERSION 1u

/* Framing actually found in a received packet. Determined per packet, not
 * assumed: an Annex-B packet is given an Annex-B NAL and a length-prefixed
 * packet a length-prefixed one. Anything else is refused. */
#define NVENC_HDR_SEI_FRAMING_UNKNOWN 0
#define NVENC_HDR_SEI_FRAMING_ANNEXB4 1
#define NVENC_HDR_SEI_FRAMING_ANNEXB3 2
#define NVENC_HDR_SEI_FRAMING_LP4 3

/* Why a call did or did not author a payload. */
#define NVENC_HDR_SEI_OK 0
#define NVENC_HDR_SEI_NOTHING_TO_DO 1        /* source declared no metadata */
#define NVENC_HDR_SEI_PARTIAL_MASTERING 2    /* primaries without luminance
                                              * or the reverse: the payload
                                              * has no way to say "half", so
                                              * nothing is invented and the
                                              * required-SEI contract is NOT
                                              * met. The container still gets
                                              * what the source did declare. */
#define NVENC_HDR_SEI_BAD_RATIONAL 3         /* zero/negative denominator,
                                              * negative value, or overflow */
#define NVENC_HDR_SEI_OUT_OF_RANGE 4         /* beyond the syntax width or
                                              * beyond what a conforming
                                              * decoder accepts */
#define NVENC_HDR_SEI_CONFLICT 5             /* the AU already carries the
                                              * payload with different values */
#define NVENC_HDR_SEI_FRAMING_UNSUPPORTED 6
#define NVENC_HDR_SEI_NO_VCL 7               /* packet carries no VCL NAL, so
                                              * no access unit to attach to */
#define NVENC_HDR_SEI_MALFORMED 8
#define NVENC_HDR_SEI_CAPACITY 9

/* What the writer can and cannot do, independent of any encoder or device:
 * these are properties of this code, so a caller can tell "the encoder
 * offers no metadata option" (encoder_offers_metadata_option) apart from
 * "the bitstream cannot be authored" (writer_available == 0). The first is
 * 0 on this build and says nothing about the second. */
typedef struct NvencHdrSeiCapability {
    uint32_t struct_size;
    int32_t writer_available;                /* 1 in any build with this API */
    int32_t encoder_offers_metadata_option;  /* 0 here: NVENC has no such knob */
    int32_t supports_partial_mastering;      /* 0: see PARTIAL_MASTERING */
    int32_t primary_order;                   /* 1 = green, blue, red */
    int32_t chroma_denominator;              /* 50000 */
    int32_t luma_denominator;                /* 10000 */
    int32_t payload_type_mastering;          /* 137 */
    int32_t payload_type_content_light;      /* 144 */
    int32_t prefix_sei_nal_type;             /* 39 */
    uint32_t mastering_payload_size;         /* 24 */
    uint32_t content_light_payload_size;     /* 4 */
    uint32_t max_nal_bytes;                  /* worst-case escaped NAL size */
    uint32_t max_insert_bytes;               /* plus framing */
} NvencHdrSeiCapability;

/* Result of one serialisation or one packet, with the reason when it is not
 * NVENC_HDR_SEI_OK. */
typedef struct NvencHdrSeiInfo {
    uint32_t struct_size;
    int32_t error;                  /* NVENC_HDR_SEI_* */
    int32_t framing;                /* NVENC_HDR_SEI_FRAMING_* */
    int32_t nal_count;              /* NAL units seen in the packet */
    int32_t vcl_index;              /* first VCL NAL, -1 when there is none */
    int32_t mastering_present;      /* 137 already in the AU */
    int32_t content_light_present;  /* 144 already in the AU */
    int32_t mastering_equal;        /* and carrying these exact values */
    int32_t content_light_equal;
    int32_t mastering_inserted;
    int32_t content_light_inserted;
    uint32_t nal_bytes;             /* bytes of the NAL built by this call */
    uint32_t bytes_inserted;        /* bytes added to the packet */
    char detail[192];
} NvencHdrSeiInfo;

/* Everything the bridge measured about its own metadata authoring, per
 * encode session. The two hashes are FNV-1a/64 folds over the bytes as
 * received from the encoder, before any insertion, taken over the whole
 * session in packet order: vcl_hash covers every VCL NAL body and
 * existing_nal_hash covers every NAL body that arrived. A caller can
 * recompute both over the demuxed elementary stream and get the same number,
 * which is what proves the coded slices and every unrelated NAL survived. */
typedef struct NvencHdrSeiStatus {
    uint32_t struct_size;
    int32_t writer_available;
    int32_t metadata_requested;      /* 1 when the source declared anything */
    int32_t mastering_serialized;    /* 1 when a 137 payload was built */
    int32_t content_light_serialized;
    int32_t error;                   /* NVENC_HDR_SEI_* of the open decision */
    char detail[192];
    int32_t framing;                 /* framing of the first packet seen */
    char packet_sample[64];          /* hex of its first bytes, as received */
    uint64_t packets_seen;
    uint64_t nals_inserted;
    uint64_t bytes_inserted;
    uint64_t mastering_inserted;
    uint64_t content_light_inserted;
    uint64_t mastering_already_present;
    uint64_t content_light_already_present;
    uint64_t packets_without_vcl;
    uint64_t vcl_bytes;
    uint64_t vcl_hash;
    uint64_t existing_nal_bytes;
    uint64_t existing_nal_hash;
} NvencHdrSeiStatus;

NVENC_BRIDGE_API uint32_t nvenc_hdr_sei_abi_version(void);
/* Static properties of this writer. 0 on success. */
NVENC_BRIDGE_API int32_t nvenc_hdr_sei_capability(
    NvencHdrSeiCapability* out_capability);
/* Serialise the prefix SEI NAL unit for the metadata - the two-byte NAL
 * header plus the emulation-prevention-encoded payload, with no start code
 * and no length prefix. Host-only: no encoder, no device, no pixel. Returns
 * 0, or -1 with NVENC_HDR_SEI_* and a reason in info and nvenc_last_error. */
NVENC_BRIDGE_API int32_t nvenc_hdr_sei_encode_nal(
    const NvencHdrMetadata* metadata, uint8_t* out_nal,
    uint32_t out_capacity, uint32_t* out_nal_size, NvencHdrSeiInfo* out_info);
/* Parse one encoded access-unit packet, find any mastering/content-light it
 * already carries, add what the source declared and the AU does not have,
 * and write the result into out. in and out must not overlap. Returns 0 when
 * the packet was handled (possibly with nothing to add), -1 when it cannot be
 * handled safely - in which case out is untouched and *out_size is 0. */
NVENC_BRIDGE_API int32_t nvenc_hdr_sei_apply_packet(
    const NvencHdrMetadata* metadata, const uint8_t* in, uint32_t in_size,
    uint8_t* out, uint32_t out_capacity, uint32_t* out_size,
    NvencHdrSeiInfo* out_info);
/* The session's own measurement. Fills a struct of the caller's size. */
NVENC_BRIDGE_API int32_t nvenc_hdr_sei_get(NvencBridge* bridge,
    NvencHdrSeiStatus* out_status);

#ifdef __cplusplus
}
#endif
#endif
