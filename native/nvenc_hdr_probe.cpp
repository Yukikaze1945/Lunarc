/* HDR10 entry probe: ABI layout checks, installed-capacity readout and the
 * encoder's own option surface.
 *
 * This is a compile-and-run gate for the additive nvenc_bridge HDR10 API. It
 * links against nvenc_bridge.lib, so the exported names and the header
 * declarations are checked together, and the static assertions below fail the
 * build if the ABI 2 layout a caller was compiled against ever moves.
 *
 * No GPU work happens here: the capacity readout is the encoder's own answer,
 * and the one thing this proves without a device is that the SDR ABI and the
 * HDR10 entry can coexist in one header.
 *
 * Run from native/:
 *     nvenc_hdr_probe.exe <path for the raw encoder option dump>
 */
#include "nvenc_bridge.h"
#include "nvdec_bridge.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
}

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

#define CHECK_OFFSET(type, field, expected)                                 \
    static_assert(offsetof(type, field) == (expected),                      \
                  #type "::" #field " moved: an ABI 2 caller would read "   \
                  "the wrong member")

/* ABI 2 was laid out by these offsets; restating them is what keeps a later
 * edit of NvencBridgeOptions from silently breaking a caller that already
 * compiled against it. */
CHECK_OFFSET(NvencBridgeOptions, struct_size, 0);
CHECK_OFFSET(NvencBridgeOptions, codec, 8);
CHECK_OFFSET(NvencBridgeOptions, preset, 16);
CHECK_OFFSET(NvencBridgeOptions, tune, 24);
CHECK_OFFSET(NvencBridgeOptions, multipass, 32);
CHECK_OFFSET(NvencBridgeOptions, rate_control, 40);
CHECK_OFFSET(NvencBridgeOptions, constant_quality, 48);
CHECK_OFFSET(NvencBridgeOptions, bitrate, 56);
CHECK_OFFSET(NvencBridgeOptions, gop, 64);
CHECK_OFFSET(NvencBridgeOptions, color_matrix, 68);
CHECK_OFFSET(NvencBridgeOptions, extra_options, 72);
static_assert(sizeof(NvencBridgeOptions) == 80,
              "NvencBridgeOptions grew: ABI 2 callers pass a fixed size");

CHECK_OFFSET(NvencTransferCounters, struct_size, 0);
CHECK_OFFSET(NvencTransferCounters, frames_submitted, 8);
CHECK_OFFSET(NvencTransferCounters, packets_written, 16);
CHECK_OFFSET(NvencTransferCounters, staging_gpu_copy_bytes, 24);
CHECK_OFFSET(NvencTransferCounters, conversion_output_bytes, 32);
CHECK_OFFSET(NvencTransferCounters, host_pixel_bytes, 40);
static_assert(sizeof(NvencTransferCounters) == 48,
              "NvencTransferCounters grew: ABI 2 callers pass a fixed size");

/* The HDR10 metadata struct is only a handoff if it is the decoder's struct.
 * Same size, same offsets, so a caller can copy the fields across without a
 * reinterpretation step that could round or drop something. */
static_assert(sizeof(NvencHdrMetadata) == sizeof(NvdecHdrMetadata),
              "NvencHdrMetadata is not layout-compatible with the decoder's");
CHECK_OFFSET(NvencHdrMetadata, struct_size, 0);
CHECK_OFFSET(NvencHdrMetadata, has_mastering, 4);
CHECK_OFFSET(NvencHdrMetadata, primary_x_num, 8);
CHECK_OFFSET(NvencHdrMetadata, white_x_num, 56);
CHECK_OFFSET(NvencHdrMetadata, min_luminance_num, 72);
CHECK_OFFSET(NvencHdrMetadata, mastering_has_primaries, 88);
CHECK_OFFSET(NvencHdrMetadata, has_content_light, 96);
CHECK_OFFSET(NvencHdrMetadata, max_cll, 100);
CHECK_OFFSET(NvencHdrMetadata, max_fall, 104);
static_assert(offsetof(NvdecHdrMetadata, max_fall) ==
              offsetof(NvencHdrMetadata, max_fall),
              "the decoder and encoder metadata layouts diverged");

/* The new counter and contract structs are self-describing through
 * struct_size, so a mismatch is a runtime refusal; these assertions just keep
 * the sizes the smoke binds to honest. The smoke builds its ctypes structs from
 * these same numbers, which is why they are printed as well as asserted. */
CHECK_OFFSET(NvencHdrCounters, struct_size, 0);
CHECK_OFFSET(NvencHdrCounters, frames_submitted, 8);
CHECK_OFFSET(NvencHdrCounters, packets_written, 16);
CHECK_OFFSET(NvencHdrCounters, p010_device_copy_bytes, 24);
CHECK_OFFSET(NvencHdrCounters, host_pixel_bytes, 32);
CHECK_OFFSET(NvencHdrCounters, rgb_conversion_bytes, 40);
CHECK_OFFSET(NvencHdrCounters, mastering_side_data, 48);
CHECK_OFFSET(NvencHdrCounters, content_light_side_data, 56);
static_assert(sizeof(NvencHdrCounters) == 64, "NvencHdrCounters size moved");
CHECK_OFFSET(NvencHdrOptions, struct_size, 0);
CHECK_OFFSET(NvencHdrOptions, codec, 8);
CHECK_OFFSET(NvencHdrOptions, rate_control, 48);
CHECK_OFFSET(NvencHdrOptions, constant_quality, 56);
CHECK_OFFSET(NvencHdrOptions, bitrate, 64);
CHECK_OFFSET(NvencHdrOptions, gop, 72);
CHECK_OFFSET(NvencHdrOptions, extra_options, 80);
CHECK_OFFSET(NvencHdrOptions, color_space, 104);
static_assert(sizeof(NvencHdrOptions) == 112, "NvencHdrOptions size moved");
CHECK_OFFSET(NvencHdrDefaults, profile, 24);
static_assert(sizeof(NvencHdrDefaults) == 56, "NvencHdrDefaults size moved");
CHECK_OFFSET(NvencHdrCapability, encoder_name, 32);
CHECK_OFFSET(NvencHdrCapability, profile_name, 96);
static_assert(sizeof(NvencHdrCapability) == 128,
              "NvencHdrCapability size moved");

/* The in-bitstream writer's own structs are new, but they are appended after a
 * published section, so their sizes are pinned the same way: the smoke binds
 * ctypes mirrors to the numbers printed below. */
CHECK_OFFSET(NvencHdrSeiCapability, struct_size, 0);
CHECK_OFFSET(NvencHdrSeiCapability, writer_available, 4);
CHECK_OFFSET(NvencHdrSeiCapability, max_nal_bytes, 48);
CHECK_OFFSET(NvencHdrSeiCapability, max_insert_bytes, 52);
static_assert(sizeof(NvencHdrSeiCapability) == 56,
              "NvencHdrSeiCapability size moved");
CHECK_OFFSET(NvencHdrSeiInfo, struct_size, 0);
CHECK_OFFSET(NvencHdrSeiInfo, error, 4);
CHECK_OFFSET(NvencHdrSeiInfo, nal_bytes, 44);
CHECK_OFFSET(NvencHdrSeiInfo, bytes_inserted, 48);
CHECK_OFFSET(NvencHdrSeiInfo, detail, 52);
static_assert(sizeof(NvencHdrSeiInfo) == 244, "NvencHdrSeiInfo size moved");
CHECK_OFFSET(NvencHdrSeiStatus, struct_size, 0);
CHECK_OFFSET(NvencHdrSeiStatus, writer_available, 4);
CHECK_OFFSET(NvencHdrSeiStatus, error, 20);
CHECK_OFFSET(NvencHdrSeiStatus, detail, 24);
CHECK_OFFSET(NvencHdrSeiStatus, framing, 216);
CHECK_OFFSET(NvencHdrSeiStatus, packet_sample, 220);
CHECK_OFFSET(NvencHdrSeiStatus, packets_seen, 288);
CHECK_OFFSET(NvencHdrSeiStatus, existing_nal_hash, 376);
static_assert(sizeof(NvencHdrSeiStatus) == 384, "NvencHdrSeiStatus size moved");

void print_sei_capability() {
    NvencHdrSeiCapability c{sizeof(NvencHdrSeiCapability)};
    const int32_t rc = nvenc_hdr_sei_capability(&c);
    std::printf("sei abi=%u rc=%d writer_available=%d "
                "encoder_offers_metadata_option=%d supports_partial_mastering=%d "
                "primary_order=%d chroma_den=%d luma_den=%d type_mastering=%d "
                "type_light=%d nal_type=%d mastering_bytes=%u light_bytes=%u "
                "max_nal=%u max_insert=%u\n",
                nvenc_hdr_sei_abi_version(), rc, c.writer_available,
                c.encoder_offers_metadata_option, c.supports_partial_mastering,
                c.primary_order, c.chroma_denominator, c.luma_denominator,
                c.payload_type_mastering, c.payload_type_content_light,
                c.prefix_sei_nal_type, c.mastering_payload_size,
                c.content_light_payload_size, c.max_nal_bytes,
                c.max_insert_bytes);
    /* The public entries must refuse a bad struct_size and a null handle
     * without touching any buffer the caller owns. */
    NvencHdrSeiInfo info{sizeof(NvencHdrSeiInfo)};
    NvencHdrMetadata md{sizeof(NvencHdrMetadata)};
    uint8_t nal[64] = {0};
    uint32_t nal_size = 7;
    uint8_t packet[64] = {0};
    uint32_t packet_size = 7;
    NvencHdrSeiStatus status{sizeof(NvencHdrSeiStatus)};
    const int32_t null_args = nvenc_hdr_sei_apply_packet(
        nullptr, nullptr, 0, nullptr, 0, nullptr, nullptr);
    const int32_t null_metadata = nvenc_hdr_sei_encode_nal(
        nullptr, nal, sizeof(nal), &nal_size, &info);
    const int32_t null_bridge = nvenc_hdr_sei_get(nullptr, &status);
    /* An empty metadata struct declares nothing, so no NAL is built and the
     * entry still succeeds: absence is not an error, and an empty payload a
     * decoder would read as no metadata is never written. */
    const int32_t nothing = nvenc_hdr_sei_encode_nal(
        &md, nal, sizeof(nal), &nal_size, &info);
    std::printf("sei guards null_args=%d null_metadata=%d null_bridge=%d "
                "nothing rc=%d error=%d nal_size=%u\n",
                null_args, null_metadata, null_bridge, nothing, info.error,
                nal_size);
    /* A zero-length packet is malformed, not empty-of-metadata: the framing
     * is unprovable, so the reason is reported rather than a silent pass. */
    std::vector<uint8_t> dummy(8, 0);
    const int32_t malformed = nvenc_hdr_sei_apply_packet(
        &md, dummy.data(), 0, packet, sizeof(packet), &packet_size, &info);
    std::printf("sei malformed rc=%d error=%d out_size=%u detail=%s\n",
                malformed, info.error, packet_size, info.detail);
}

void print_defaults() {
    NvencHdrDefaults d{sizeof(NvencHdrDefaults)};
    if (nvenc_hdr_get_defaults(&d) != 0) {
        std::printf("defaults FAIL %s\n", nvenc_last_error());
        return;
    }
    std::printf("defaults sw_pix_fmt=%d color_range=%d primaries=%d trc=%d "
                "space=%d profile=%s\n", d.sw_pixel_format, d.color_range,
                d.color_primaries, d.color_trc, d.color_space, d.profile);
}

void print_capability() {
    NvencHdrCapability c{sizeof(NvencHdrCapability)};
    const int32_t rc = nvenc_hdr_capability(nullptr, 1920, 1080, &c);
    std::printf("capability rc=%d encoder_present=%d p010=%d cuda=%d "
                "main10=%d cuda_device_created=%d hdr_metadata_options=%d "
                "supported=%d encoder=%s profile_name=%s error=%s\n",
                rc, c.encoder_present, c.p010_supported, c.cuda_supported,
                c.main10_profile_supported, c.cuda_device_created,
                c.hdr_metadata_options, c.supported, c.encoder_name,
                c.profile_name, nvenc_last_error());
}

/* The encoder's own option table, dumped verbatim: this is the runtime evidence
 * for what the installed NVENC wrapper can be asked to do, as opposed to what
 * an older or newer build might offer. */
void dump_encoder_options(const char* path) {
    const AVCodec* codec = avcodec_find_encoder_by_name("hevc_nvenc");
    if (!codec) { std::printf("options FAIL hevc_nvenc absent\n"); return; }
    AVCodecContext* ctx = avcodec_alloc_context3(codec);
    if (!ctx) { std::printf("options FAIL alloc\n"); return; }
    /* The advertised-profile table this build carries for the encoder: empty is
     * what makes the option-table resolution above the load-bearing check. */
    std::printf("codec->profiles=%p\n", static_cast<const void*>(codec->profiles));
    char* buffer = nullptr;
    const int rc = av_opt_serialize(ctx, AV_OPT_FLAG_ENCODING_PARAM, 0, &buffer,
                                    '=', '\n');
    FILE* out = nullptr;
    const bool opened = fopen_s(&out, path, "wb") == 0 && out;
    if (rc >= 0 && buffer && opened) {
        std::fwrite(buffer, 1, std::strlen(buffer), out);
        std::printf("options rc=%d bytes=%zu path=%s\n", rc,
                    std::strlen(buffer), path);
    } else {
        std::printf("options rc=%d wrote=%d\n", rc, opened ? 1 : 0);
    }
    if (out) std::fclose(out);
    av_free(buffer);

    /* How the installed build resolves the profile name this entry needs.
     * Each search starts from AV_PROFILE_UNKNOWN so the number printed is that
     * call's own effect on AVCodecContext::profile, which is the field NVENC
     * actually switches on. */
    ctx->profile = AV_PROFILE_UNKNOWN;
    const int with_children = av_opt_set(ctx, "profile", "main10",
                                         AV_OPT_SEARCH_CHILDREN);
    std::printf("profile children rc=%d generic_value=%d name=%s\n",
                with_children, ctx->profile,
                avcodec_profile_name(AV_CODEC_ID_HEVC, ctx->profile));
    ctx->profile = AV_PROFILE_UNKNOWN;
    const int plain = av_opt_set(ctx, "profile", "main10", 0);
    std::printf("profile plain rc=%d generic_value=%d name=%s\n", plain,
                ctx->profile, avcodec_profile_name(AV_CODEC_ID_HEVC,
                                                   ctx->profile));
    ctx->profile = AV_PROFILE_HEVC_MAIN_10;
    std::printf("profile forced value=%d name=%s\n", ctx->profile,
                avcodec_profile_name(AV_CODEC_ID_HEVC, ctx->profile));
    avcodec_free_context(&ctx);
}

}

int main(int argc, char** argv) {
    const char* dump_path = argc > 1 ? argv[1] : "nvenc_hdr_options.txt";
    std::printf("abi nvenc=%u nvenc_hdr=%u nvdec=%u nvenc_hdr_sei=%u\n",
                nvenc_abi_version(), nvenc_hdr_abi_version(),
                nvdec_bridge_abi_version(), nvenc_hdr_sei_abi_version());
    /* Printed so the ctypes binding in the smoke can be checked against the
     * compiled struct rather than against a hand count of the header. */
    std::printf("sizes options=%zu counters=%zu defaults=%zu capability=%zu "
                "contract=%zu metadata=%zu\n",
                sizeof(NvencHdrOptions), sizeof(NvencHdrCounters),
                sizeof(NvencHdrDefaults), sizeof(NvencHdrCapability),
                sizeof(NvencHdrContract), sizeof(NvencHdrMetadata));
    std::printf("sei sizes capability=%zu info=%zu status=%zu\n",
                sizeof(NvencHdrSeiCapability), sizeof(NvencHdrSeiInfo),
                sizeof(NvencHdrSeiStatus));
    print_defaults();
    print_capability();
    print_sei_capability();
    dump_encoder_options(dump_path);
    /* A bridge opened for SDR must refuse the P010 entry, and a null handle
     * must be refused everywhere, without touching the device. */
    NvencHdrCounters counters{sizeof(NvencHdrCounters)};
    NvencHdrContract contract{sizeof(NvencHdrContract)};
    std::printf("null guards counters=%d contract=%d encode=%d\n",
                nvenc_hdr_get_counters(nullptr, &counters),
                nvenc_hdr_get_contract(nullptr, &contract),
                nvenc_encode_p010(nullptr, 1, 2, 3, 4, 0));
    NvencHdrOptions bad{sizeof(NvencHdrOptions)};
    bad.codec = "h264_nvenc";
    NvencBridge* bridge = nullptr;
    const int32_t refused = nvenc_hdr_open(nullptr, 1920, 1080, 25, 1, 1, 1000,
        "must-not-exist.h264", &bad, nullptr, &bridge);
    std::printf("refusal codec rc=%d bridge=%p error=%s\n", refused,
                static_cast<void*>(bridge), nvenc_last_error());
    return 0;
}
