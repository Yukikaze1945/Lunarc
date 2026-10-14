/* Original AV1 low-overhead Metadata OBU writer; AOM syntax sections 5/6.
 * No dependency on a third-party bitstream implementation. */
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
using Bytes = std::vector<uint8_t>;
bool av1_error(NvencP010MetadataStatus& s, const char* text) {
    s.error = -1;
    std::snprintf(s.detail, sizeof(s.detail), "%s", text);
    return false;
}
bool av1_wire(int32_t num, int32_t den, uint64_t scale, uint32_t limit,
              uint32_t& value, double& max_error, NvencP010MetadataStatus& s) {
    if (num < 0 || den <= 0) return av1_error(s, "Invalid AV1 HDR rational");
    const uint64_t product = uint64_t(num) * scale;
    if (product > uint64_t(limit) * uint64_t(den))
        return av1_error(s, "AV1 HDR fixed-point value out of range");
    const uint64_t rounded = (product + uint64_t(den) / 2) / uint64_t(den);
    if (rounded > limit) return av1_error(s, "AV1 HDR rounding overflow");
    value = static_cast<uint32_t>(rounded);
    max_error = (std::max)(max_error, std::abs(double(value) / double(scale) - double(num) / den));
    return true;
}
void av1_be(Bytes& bytes, uint32_t value, int count) {
    for (int n = count - 1; n >= 0; --n) bytes.push_back(uint8_t(value >> (8 * n)));
}
void av1_leb(Bytes& bytes, uint64_t value) {
    do { const auto byte = uint8_t(value & 127); value >>= 7;
         bytes.push_back(uint8_t(byte | (value ? 128 : 0))); } while (value);
}
bool av1_read_leb(const uint8_t* data, size_t end, size_t& pos, uint64_t& value) {
    value = 0;
    for (unsigned n = 0; n < 8; ++n) {
        if (pos == end) return false;
        const auto byte = data[pos++]; value |= uint64_t(byte & 127) << (7 * n);
        if (!(byte & 128)) return value <= UINT32_MAX;
    }
    return false;
}
Bytes av1_obu(const Bytes& payload) {
    Bytes result{0x2a};  // OBU_METADATA, has_size_field, no extension.
    av1_leb(result, payload.size());
    result.insert(result.end(), payload.begin(), payload.end());
    return result;
}
bool av1_prepare(const NvencHdrMetadata& md, NvencP010MetadataStatus& s,
                 Bytes& mastering, Bytes& light) {
    s.struct_size = sizeof(s); s.codec_kind = 2; s.color_mode = 1;
    s.writer_available = 1; s.metadata_requested = md.has_mastering || md.has_content_light;
    if (md.has_mastering) {
        if (!md.mastering_has_primaries || !md.mastering_has_luminance)
            return av1_error(s, "AV1 MDCV requires both primaries and luminance");
        const int32_t nums[] = {md.primary_x_num[0],md.primary_y_num[0],md.primary_x_num[1],md.primary_y_num[1],
            md.primary_x_num[2],md.primary_y_num[2],md.white_x_num,md.white_y_num,
            md.max_luminance_num,md.min_luminance_num};
        const int32_t dens[] = {md.primary_x_den[0],md.primary_y_den[0],md.primary_x_den[1],md.primary_y_den[1],
            md.primary_x_den[2],md.primary_y_den[2],md.white_x_den,md.white_y_den,
            md.max_luminance_den,md.min_luminance_den};
        mastering.push_back(2);
        for (int i = 0; i < 10; ++i) {
            const uint64_t scale = i < 8 ? 65536 : i == 8 ? 256 : 16384;
            if (!av1_wire(nums[i], dens[i], scale, i < 8 ? 65535 : UINT32_MAX,
                          s.wire_values[i], i < 8 ? s.max_chromaticity_error : s.max_luminance_error, s)) return false;
            av1_be(mastering, s.wire_values[i], i < 8 ? 2 : 4);
        }
        if (int64_t(md.min_luminance_num) * md.max_luminance_den >
            int64_t(md.max_luminance_num) * md.min_luminance_den)
            return av1_error(s, "AV1 HDR minimum exceeds maximum luminance");
        mastering.push_back(0x80); s.mastering_serialized = 1;
    }
    if (md.has_content_light) {
        if (md.max_cll > 65535 || md.max_fall > 65535)
            return av1_error(s, "AV1 HDR CLL/FALL out of 16-bit range");
        light.push_back(1); av1_be(light,md.max_cll,2); av1_be(light,md.max_fall,2);
        light.push_back(0x80); s.wire_values[10]=md.max_cll; s.wire_values[11]=md.max_fall;
        s.content_light_serialized = 1;
    }
    return true;
}
bool av1_apply(const NvencHdrMetadata& md, const uint8_t* in, uint32_t size,
               Bytes& out, NvencP010MetadataStatus& status) {
    Bytes mastering, light;
    if (!av1_prepare(md,status,mastering,light)) return false;
    bool has_mastering=false, has_light=false, has_frame=false;
    size_t pos=0, insertion=0;
    while (pos < size) {
        const size_t start=pos; const uint8_t header=in[pos++];
        const unsigned type=(header >> 3) & 15;
        if ((header & 0x81) || !(header & 2) || type == 0 || (type > 8 && type < 15))
            return av1_error(status,"Malformed or unsupported AV1 OBU header/framing");
        if (header & 4) {
            if (pos == size || (in[pos] & 7)) return av1_error(status,"Malformed AV1 extension");
            ++pos;
        }
        uint64_t length=0;
        if (!av1_read_leb(in,size,pos,length) || length > size-pos)
            return av1_error(status,"Truncated or invalid AV1 OBU size");
        const size_t end=pos+size_t(length);
        if (type == 5) {
            size_t payload=pos; uint64_t metadata_type=0;
            if (!av1_read_leb(in,end,payload,metadata_type))
                return av1_error(status,"Invalid AV1 metadata type");
            if (metadata_type == 1 || metadata_type == 2) {
                const auto& expected=metadata_type == 2 ? mastering : light;
                const size_t required=metadata_type == 2 ? 25 : 5;
                if (end-payload != required || in[end-1] != 0x80)
                    return av1_error(status,"Malformed AV1 static HDR payload");
                if (expected.empty() || expected.size()-1 != end-payload ||
                    !std::equal(expected.begin()+1,expected.end(),in+payload))
                    return av1_error(status,"Conflicting AV1 static HDR metadata");
                bool& present=metadata_type == 2 ? has_mastering : has_light;
                if (present) return av1_error(status,"Duplicate AV1 static HDR metadata");
                present=true;
            }
        }
        if ((type == 3 || type == 6) && !has_frame) { has_frame=true; insertion=start; }
        pos=end;
    }
    ++status.packets_seen;
    if (has_mastering) ++status.mastering_already_present;
    if (has_light) ++status.content_light_already_present;
    out.clear();
    if (!has_frame) { out.assign(in,in+size); return true; }
    Bytes added;
    if (!mastering.empty() && !has_mastering) {
        auto bytes=av1_obu(mastering); added.insert(added.end(),bytes.begin(),bytes.end());
        ++status.mastering_inserted;
    }
    if (!light.empty() && !has_light) {
        auto bytes=av1_obu(light); added.insert(added.end(),bytes.begin(),bytes.end());
        ++status.content_light_inserted;
    }
    if (uint64_t(size)+added.size() > INT32_MAX) return av1_error(status,"AV1 packet size overflow");
    out.insert(out.end(),in,in+insertion); out.insert(out.end(),added.begin(),added.end());
    out.insert(out.end(),in+insertion,in+size); status.bytes_inserted += added.size();
    return true;
}
} // namespace
