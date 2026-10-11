#define NVDEC_BRIDGE_EXPORTS
#include "nvdec_bridge.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
#include <libavutil/mastering_display_metadata.h>
}

#include <atomic>
#include <cstdio>
#include <cstring>
#include <new>

struct NvdecBridge {
    AVFormatContext* format = nullptr;
    AVCodecContext* codec = nullptr;
    AVBufferRef* cuda_device = nullptr;
    AVPacket* packet = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* scaled_frame = nullptr;
    AVFilterGraph* filters = nullptr;
    AVFilterContext* filter_source = nullptr;
    AVFilterContext* filter_sink = nullptr;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    /* P010-enabled opens report a P010 CUDA surface; ordinary SDR opens refuse
     * it so a 10-bit source cannot be read as NV12. */
    bool allow_p010 = false;
    int stream_index = -1;
    AVRational time_base{0, 1};
    std::atomic<bool> cancelled{false};
    bool packet_pending = false;
    bool input_done = false;
    bool drain_sent = false;
    NvdecTransferCounters counters{sizeof(NvdecTransferCounters)};
    NvdecHdrMetadata hdr{sizeof(NvdecHdrMetadata)};
};

namespace {
thread_local char last_error[512] = "";

void set_error(const char* operation, int code) {
    char detail[AV_ERROR_MAX_STRING_SIZE] = "";
    av_strerror(code, detail, sizeof(detail));
    std::snprintf(last_error, sizeof(last_error), "%s: %s (%d)",
                  operation, detail, code);
}

int interrupt_read(void* opaque) {
    return static_cast<NvdecBridge*>(opaque)->cancelled.load() ? 1 : 0;
}

void capture_hdr_metadata(NvdecBridge* bridge, const AVFrame* frame);

AVPixelFormat choose_cuda_format(AVCodecContext*, const AVPixelFormat* formats) {
    for (auto* format = formats; *format != AV_PIX_FMT_NONE; ++format) {
        if (*format == AV_PIX_FMT_CUDA) return AV_PIX_FMT_CUDA;
    }
    return AV_PIX_FMT_NONE; // no silent CPU decoder fallback
}

bool decoder_has_cuda_output(const AVCodec* decoder) {
    if (!decoder) return false;
    for (int index = 0;; ++index) {
        const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, index);
        if (!config) break;
        if (config->pix_fmt == AV_PIX_FMT_CUDA &&
            (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX))
            return true;
    }
    return false;
}

const AVCodec* find_cuda_decoder(enum AVCodecID id) {
    const AVCodec* preferred = nullptr;
    const AVCodec* first = nullptr;
    void* opaque = nullptr;
    while (const AVCodec* candidate = av_codec_iterate(&opaque)) {
        if (!av_codec_is_decoder(candidate) || candidate->id != id ||
            !decoder_has_cuda_output(candidate)) continue;
        if (!first) first = candidate;
        // CUVID implementations are the explicit NVDEC path. In particular,
        // the generic AV1 decoder is often libdav1d and has no CUDA output.
        if (candidate->name && std::strstr(candidate->name, "_cuvid")) {
            preferred = candidate;
            break;
        }
    }
    return preferred ? preferred : first;
}

int init_scaler(NvdecBridge* bridge) {
    bridge->filters = avfilter_graph_alloc();
    if (!bridge->filters) return AVERROR(ENOMEM);
    const AVFilter* scale_cuda = avfilter_get_by_name("scale_cuda");
    if (!scale_cuda) return AVERROR(ENOSYS);
    char source_args[256];
    const AVRational aspect = bridge->frame->sample_aspect_ratio.num > 0 &&
        bridge->frame->sample_aspect_ratio.den > 0 ?
        bridge->frame->sample_aspect_ratio : AVRational{1, 1};
    std::snprintf(source_args, sizeof(source_args),
        "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
        bridge->frame->width, bridge->frame->height, AV_PIX_FMT_CUDA,
        bridge->time_base.num, bridge->time_base.den, aspect.num, aspect.den);
    int rc = avfilter_graph_create_filter(&bridge->filter_source,
        avfilter_get_by_name("buffer"), "nvdec_input", source_args,
        nullptr, bridge->filters);
    if (rc < 0) return rc;
    AVBufferSrcParameters* params = av_buffersrc_parameters_alloc();
    if (!params) return AVERROR(ENOMEM);
    params->format = AV_PIX_FMT_CUDA;
    params->width = bridge->frame->width;
    params->height = bridge->frame->height;
    params->time_base = bridge->time_base;
    params->sample_aspect_ratio = aspect;
    params->hw_frames_ctx = bridge->frame->hw_frames_ctx;
    params->color_space = bridge->frame->colorspace;
    params->color_range = bridge->frame->color_range;
    rc = av_buffersrc_parameters_set(bridge->filter_source, params);
    av_free(params);
    if (rc < 0) return rc;
    char scale_args[128];
    std::snprintf(scale_args, sizeof(scale_args), "w=%u:h=%u:format=nv12",
        bridge->output_width, bridge->output_height);
    AVFilterContext* scaler = nullptr;
    rc = avfilter_graph_create_filter(&scaler,
        scale_cuda, "gpu_scale", scale_args,
        nullptr, bridge->filters);
    if (rc < 0) return rc;
    rc = avfilter_graph_create_filter(&bridge->filter_sink,
        avfilter_get_by_name("buffersink"), "nvdec_output", nullptr,
        nullptr, bridge->filters);
    if (rc < 0) return rc;
    rc = avfilter_link(bridge->filter_source, 0, scaler, 0);
    if (rc < 0) return rc;
    rc = avfilter_link(scaler, 0, bridge->filter_sink, 0);
    if (rc < 0) return rc;
    return avfilter_graph_config(bridge->filters, nullptr);
}

}

extern "C" NVDEC_BRIDGE_API uint32_t nvdec_bridge_abi_version(void) {
    return NVDEC_BRIDGE_ABI_VERSION;
}

extern "C" NVDEC_BRIDGE_API const char* nvdec_bridge_last_error(void) {
    return last_error;
}

extern "C" NVDEC_BRIDGE_API void nvdec_bridge_cancel(NvdecBridge* bridge) {
    if (bridge) bridge->cancelled.store(true);
}

extern "C" NVDEC_BRIDGE_API void nvdec_bridge_close(NvdecBridge* bridge) {
    if (!bridge) return;
    avfilter_graph_free(&bridge->filters);
    av_frame_free(&bridge->scaled_frame);
    av_frame_free(&bridge->frame);
    av_packet_free(&bridge->packet);
    avcodec_free_context(&bridge->codec);
    av_buffer_unref(&bridge->cuda_device);
    avformat_close_input(&bridge->format);
    delete bridge;
}

static int32_t open_impl(const char* utf8_path, bool allow_p010,
                         NvdecBridge** out_bridge) {
    last_error[0] = 0;
    if (!utf8_path || !*utf8_path || !out_bridge) return -1;
    *out_bridge = nullptr;
    auto* bridge = new (std::nothrow) NvdecBridge;
    if (!bridge) return -1;
    bridge->allow_p010 = allow_p010;
    const auto fail = [&](const char* operation, int code) {
        set_error(operation, code);
        nvdec_bridge_close(bridge);
        return -1;
    };
    bridge->format = avformat_alloc_context();
    if (!bridge->format) return fail("allocate format context", AVERROR(ENOMEM));
    bridge->format->interrupt_callback = {interrupt_read, bridge};
    int rc = avformat_open_input(&bridge->format, utf8_path, nullptr, nullptr);
    if (rc < 0) return fail("open input", rc);
    rc = avformat_find_stream_info(bridge->format, nullptr);
    if (rc < 0) return fail("find stream info", rc);
    bridge->stream_index = av_find_best_stream(bridge->format, AVMEDIA_TYPE_VIDEO,
                                                -1, -1, nullptr, 0);
    if (bridge->stream_index < 0) return fail("find video stream", bridge->stream_index);
    AVStream* stream = bridge->format->streams[bridge->stream_index];
    bridge->time_base = stream->time_base;
    const AVCodec* decoder = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!decoder_has_cuda_output(decoder))
        decoder = find_cuda_decoder(stream->codecpar->codec_id);
    if (!decoder)
        return fail("no decoder with CUDA device output is available",
                    AVERROR_DECODER_NOT_FOUND);
    rc = av_hwdevice_ctx_create(&bridge->cuda_device, AV_HWDEVICE_TYPE_CUDA,
                                nullptr, nullptr, 0);
    if (rc < 0) return fail("create CUDA device", rc);
    bridge->codec = avcodec_alloc_context3(decoder);
    if (!bridge->codec) return fail("allocate decoder", AVERROR(ENOMEM));
    rc = avcodec_parameters_to_context(bridge->codec, stream->codecpar);
    if (rc < 0) return fail("copy codec parameters", rc);
    bridge->codec->get_format = choose_cuda_format;
    bridge->codec->hw_device_ctx = av_buffer_ref(bridge->cuda_device);
    if (!bridge->codec->hw_device_ctx) return fail("retain CUDA device", AVERROR(ENOMEM));
    rc = avcodec_open2(bridge->codec, decoder, nullptr);
    if (rc < 0) return fail("open CUDA decoder", rc);
    bridge->packet = av_packet_alloc();
    bridge->frame = av_frame_alloc();
    bridge->scaled_frame = av_frame_alloc();
    if (!bridge->packet || !bridge->frame || !bridge->scaled_frame)
        return fail("allocate frame", AVERROR(ENOMEM));
    *out_bridge = bridge;
    return 0;
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_open(
    const char* utf8_path, NvdecBridge** out_bridge) {
    return open_impl(utf8_path, false, out_bridge);
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_open_hdr(
    const char* utf8_path, NvdecBridge** out_bridge) {
    return open_impl(utf8_path, true, out_bridge);
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_open_p010(
    const char* utf8_path, NvdecBridge** out_bridge) {
    return open_impl(utf8_path, true, out_bridge);
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_open_scaled(
    const char* utf8_path, uint32_t output_width, uint32_t output_height,
    NvdecBridge** out_bridge) {
    if (!out_bridge || !output_width || !output_height ||
        (output_width & 1) || (output_height & 1) ||
        output_width > INT32_MAX || output_height > INT32_MAX) {
        if (out_bridge) *out_bridge = nullptr;
        std::snprintf(last_error, sizeof(last_error), "invalid NV12 output size");
        return -1;
    }
    const int32_t rc = nvdec_bridge_open(utf8_path, out_bridge);
    if (rc == 0) {
        (*out_bridge)->output_width = output_width;
        (*out_bridge)->output_height = output_height;
    }
    return rc;
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_seek_us(
    NvdecBridge* bridge, int64_t relative_microseconds) {
    last_error[0] = 0;
    if (!bridge || relative_microseconds < 0 || bridge->output_width ||
        bridge->cancelled.load()) {
        std::snprintf(last_error, sizeof(last_error), "invalid preview seek");
        return -1;
    }
    const AVStream* stream = bridge->format->streams[bridge->stream_index];
    const int64_t offset = stream->start_time == AV_NOPTS_VALUE ? 0 : stream->start_time;
    const int64_t target = av_rescale_q(relative_microseconds,
        AVRational{1, 1000000}, bridge->time_base) + offset;
    const int rc = av_seek_frame(bridge->format, bridge->stream_index, target,
                                 AVSEEK_FLAG_BACKWARD);
    if (rc < 0) { set_error("seek video", rc); return -1; }
    avcodec_flush_buffers(bridge->codec);
    av_packet_unref(bridge->packet);
    av_frame_unref(bridge->frame);
    av_frame_unref(bridge->scaled_frame);
    bridge->packet_pending = false;
    bridge->input_done = false;
    bridge->drain_sent = false;
    bridge->hdr = NvdecHdrMetadata{sizeof(NvdecHdrMetadata)};
    return 0;
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_next(
    NvdecBridge* bridge, NvdecDeviceFrame* out_frame) {
    last_error[0] = 0;
    if (!bridge || !out_frame || out_frame->struct_size != sizeof(NvdecDeviceFrame)) {
        std::snprintf(last_error, sizeof(last_error), "invalid frame arguments");
        return -1;
    }
    // Releasing this AVFrame ends the previous caller's borrowed surface lifetime.
    av_frame_unref(bridge->frame);
    av_frame_unref(bridge->scaled_frame);
    for (;;) {
        if (bridge->cancelled.load()) {
            std::snprintf(last_error, sizeof(last_error), "decode cancelled");
            return -1;
        }
        int rc = avcodec_receive_frame(bridge->codec, bridge->frame);
        if (rc == 0) {
            if (bridge->frame->format != AV_PIX_FMT_CUDA ||
                !bridge->frame->hw_frames_ctx ||
                !bridge->frame->data[0] || !bridge->frame->data[1]) {
                std::snprintf(last_error, sizeof(last_error),
                              "decoder did not return a CUDA frame");
                return -1;
            }
            if (bridge->output_width) {
                if (!bridge->filters) {
                    rc = init_scaler(bridge);
                    if (rc < 0) { set_error("initialize CUDA scaler", rc); return -1; }
                }
                rc = av_buffersrc_add_frame_flags(bridge->filter_source,
                    bridge->frame, AV_BUFFERSRC_FLAG_KEEP_REF);
                if (rc < 0) { set_error("submit CUDA scaler frame", rc); return -1; }
                rc = av_buffersink_get_frame(bridge->filter_sink,
                    bridge->scaled_frame);
                if (rc < 0) { set_error("receive CUDA scaler frame", rc); return -1; }
                // The scaler wrote a whole new surface on the device.
                bridge->counters.scale_device_output_bytes +=
                    static_cast<uint64_t>(bridge->output_width) *
                    bridge->output_height * 3 / 2;
            }
            AVFrame* result = bridge->output_width ? bridge->scaled_frame : bridge->frame;
            if (result->format != AV_PIX_FMT_CUDA ||
                !result->hw_frames_ctx ||
                !result->data[0] || !result->data[1]) {
                std::snprintf(last_error, sizeof(last_error),
                              "decoder did not return a CUDA frame");
                return -1;
            }
            const auto* frames = reinterpret_cast<const AVHWFramesContext*>(
                result->hw_frames_ctx->data);
            /* One of the two semi-planar CUDA layouts, and which one it is goes
             * out in software_format rather than being smoothed into NV12: a
             * P010 plane read as 8-bit would be a wrong picture that looked
             * plausible. The scaled decoder is not part of the P010 contract -
             * scale_cuda only resizes 8-bit here - so a request that combined
             * the two is refused instead of silently encoding the wrong width. */
            const bool nv12 = frames->sw_format == AV_PIX_FMT_NV12;
            const bool p010 = frames->sw_format == AV_PIX_FMT_P010LE;
            if (!nv12 && !p010) {
                std::snprintf(last_error, sizeof(last_error),
                    "CUDA frame software format is neither NV12 nor P010");
                return -1;
            }
            if (p010 && !bridge->allow_p010) {
                std::snprintf(last_error, sizeof(last_error),
                    "CUDA frame is P010 and this bridge was not opened by "
                    "nvdec_bridge_open_hdr or nvdec_bridge_open_p010");
                return -1;
            }
            if (bridge->output_width && p010) {
                std::snprintf(last_error, sizeof(last_error),
                    "the CUDA scaler downscales NV12 only; the HDR chain "
                    "downsamples in its own FP16 stage");
                return -1;
            }
            const auto* device = reinterpret_cast<const AVHWDeviceContext*>(
                frames->device_ref->data);
            const auto* cuda = reinterpret_cast<const AVCUDADeviceContext*>(device->hwctx);
            *out_frame = NvdecDeviceFrame{
                sizeof(NvdecDeviceFrame), static_cast<uint32_t>(result->width),
                static_cast<uint32_t>(result->height), nv12 ? 1u : 2u,
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result->data[0])),
                static_cast<uint64_t>(reinterpret_cast<uintptr_t>(result->data[1])),
                result->linesize[0], result->linesize[1],
                cuda->cuda_ctx, result->best_effort_timestamp,
                bridge->time_base.num, bridge->time_base.den,
                result->color_range, result->color_primaries,
                result->color_trc, result->colorspace};
            capture_hdr_metadata(bridge, result);
            bridge->counters.frames_decoded += 1;
            bridge->counters.device_frame_count += 1;
            return 1;
        }
        if (rc == AVERROR_EOF) return 0;
        if (rc != AVERROR(EAGAIN)) { set_error("receive CUDA frame", rc); return -1; }
        if (bridge->packet_pending) {
            rc = avcodec_send_packet(bridge->codec, bridge->packet);
            if (rc == AVERROR(EAGAIN)) continue;
            if (rc < 0) { set_error("send packet", rc); return -1; }
            // Counted only here, where the decoder accepted the packet on the
            // first try; an EAGAIN retry would otherwise count it twice.
            bridge->counters.bitstream_bytes_submitted +=
                static_cast<uint64_t>(bridge->packet->size);
            av_packet_unref(bridge->packet);
            bridge->packet_pending = false;
            continue;
        }
        if (bridge->input_done) {
            if (bridge->drain_sent) return 0;
            rc = avcodec_send_packet(bridge->codec, nullptr);
            if (rc == AVERROR(EAGAIN)) continue;
            if (rc < 0) { set_error("drain decoder", rc); return -1; }
            bridge->drain_sent = true;
            continue;
        }
        rc = av_read_frame(bridge->format, bridge->packet);
        if (rc == AVERROR_EOF) { bridge->input_done = true; continue; }
        if (rc < 0) { set_error("read input", rc); return -1; }
        if (bridge->packet->stream_index == bridge->stream_index) {
            bridge->packet_pending = true;
        } else {
            av_packet_unref(bridge->packet);
        }
    }
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_readback_last_frame(
    NvdecBridge* bridge, uint8_t* host, size_t row_stride) {
    last_error[0] = 0;
    if (!bridge || !host) {
        std::snprintf(last_error, sizeof(last_error), "invalid readback arguments");
        return -1;
    }
    AVFrame* source = bridge->output_width ? bridge->scaled_frame : bridge->frame;
    if (!source || !source->data[0] || source->format != AV_PIX_FMT_CUDA) {
        std::snprintf(last_error, sizeof(last_error),
                      "no decoded CUDA frame to read back");
        return -1;
    }
    if (row_stride < static_cast<size_t>(source->width)) {
        std::snprintf(last_error, sizeof(last_error),
                      "row stride %zu is smaller than the frame width %d",
                      row_stride, source->width);
        return -1;
    }
    AVFrame* host_frame = av_frame_alloc();
    if (!host_frame) {
        set_error("allocate host frame", AVERROR(ENOMEM));
        return -1;
    }
    int rc = av_hwframe_transfer_data(host_frame, source, 0);
    if (rc < 0) {
        set_error("download CUDA frame", rc);
        av_frame_free(&host_frame);
        return -1;
    }
    if (host_frame->format != AV_PIX_FMT_NV12 ||
        host_frame->width != source->width || host_frame->height != source->height) {
        std::snprintf(last_error, sizeof(last_error),
                      "downloaded frame is not NV12 at the decoded size");
        av_frame_free(&host_frame);
        return -1;
    }
    const int width = host_frame->width;
    const int height = host_frame->height;
    const int rows = height + height / 2;
    for (int row = 0; row < rows; ++row) {
        const int plane = row < height ? 0 : 1;
        const int plane_row = row < height ? row : row - height;
        std::memcpy(host + static_cast<size_t>(row) * row_stride,
                    host_frame->data[plane] +
                        static_cast<size_t>(plane_row) * host_frame->linesize[plane],
                    static_cast<size_t>(width));
    }
    av_frame_free(&host_frame);
    // The two counts below describe the copy just made, so they are written
    // after it succeeded and never before.
    bridge->counters.host_frame_copy_count += 1;
    bridge->counters.host_frame_copy_bytes +=
        static_cast<uint64_t>(width) * rows;
    return 0;
}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_get_transfer_counters(
    NvdecBridge* bridge, NvdecTransferCounters* counters) {
    if (!bridge || !counters ||
        counters->struct_size != sizeof(NvdecTransferCounters)) return -1;
    *counters = bridge->counters;
    return 0;
}

namespace {

/* One side datum, exactly as large as the struct it names, or nothing. A demuxer
 * or encoder that wrote a short payload is treated as not having declared the
 * metadata rather than being read past its end. */
const void* side_data_payload(const uint8_t* data, size_t size, size_t need) {
    if (!data || size < need) return nullptr;
    return data;
}

/* Captured while the frame it belongs to is still the current one, and reported
 * by nvdec_bridge_get_hdr_metadata, so a caller never has to hold an AVFrame. */
void capture_hdr_metadata(NvdecBridge* bridge, const AVFrame* frame) {
    NvdecHdrMetadata out{sizeof(NvdecHdrMetadata)};
    const AVFrameSideData* mastering = av_frame_get_side_data(frame,
        AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    const void* payload = mastering ? side_data_payload(mastering->data,
        mastering->size, sizeof(AVMasteringDisplayMetadata)) : nullptr;
    /* A container that tags the track rather than the bitstream declares the
     * same metadata once for the whole stream; the decoder context carries it
     * after avcodec_parameters_to_context, so a source with no in-bitstream SEI
     * is still reported with the primaries it was authored with. */
    if (!payload) {
        const AVPacketSideData* sd = av_packet_side_data_get(
            bridge->codec->coded_side_data, bridge->codec->nb_coded_side_data,
            AV_PKT_DATA_MASTERING_DISPLAY_METADATA);
        if (sd && side_data_payload(sd->data, sd->size,
                sizeof(AVMasteringDisplayMetadata))) payload = sd->data;
    }
    if (const AVMasteringDisplayMetadata* m =
            static_cast<const AVMasteringDisplayMetadata*>(payload)) {
        out.has_mastering = 1;
        for (int i = 0; i < 3; ++i) {
            out.primary_x_num[i] = m->display_primaries[i][0].num;
            out.primary_x_den[i] = m->display_primaries[i][0].den;
            out.primary_y_num[i] = m->display_primaries[i][1].num;
            out.primary_y_den[i] = m->display_primaries[i][1].den;
        }
        out.white_x_num = m->white_point[0].num;
        out.white_x_den = m->white_point[0].den;
        out.white_y_num = m->white_point[1].num;
        out.white_y_den = m->white_point[1].den;
        out.min_luminance_num = m->min_luminance.num;
        out.min_luminance_den = m->min_luminance.den;
        out.max_luminance_num = m->max_luminance.num;
        out.max_luminance_den = m->max_luminance.den;
        out.mastering_has_primaries = m->has_primaries;
        out.mastering_has_luminance = m->has_luminance;
    }
    const AVFrameSideData* light = av_frame_get_side_data(frame,
        AV_FRAME_DATA_CONTENT_LIGHT_LEVEL);
    payload = light ? side_data_payload(light->data, light->size,
        sizeof(AVContentLightMetadata)) : nullptr;
    if (!payload) {
        const AVPacketSideData* sd = av_packet_side_data_get(
            bridge->codec->coded_side_data, bridge->codec->nb_coded_side_data,
            AV_PKT_DATA_CONTENT_LIGHT_LEVEL);
        if (sd && side_data_payload(sd->data, sd->size,
                sizeof(AVContentLightMetadata))) payload = sd->data;
    }
    if (const AVContentLightMetadata* c =
            static_cast<const AVContentLightMetadata*>(payload)) {
        out.has_content_light = 1;
        out.max_cll = c->MaxCLL;
        out.max_fall = c->MaxFALL;
    }
    bridge->hdr = out;
}

}

extern "C" NVDEC_BRIDGE_API int32_t nvdec_bridge_get_hdr_metadata(
    NvdecBridge* bridge, NvdecHdrMetadata* out_metadata) {
    last_error[0] = 0;
    if (!bridge || !out_metadata ||
        out_metadata->struct_size != sizeof(NvdecHdrMetadata)) {
        std::snprintf(last_error, sizeof(last_error),
                      "invalid HDR metadata arguments");
        return -1;
    }
    *out_metadata = bridge->hdr;
    return 0;
}
