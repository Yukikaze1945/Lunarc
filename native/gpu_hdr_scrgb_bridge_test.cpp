/* Stage-B HDR10 scRGB bridge: bring-up and numerical gates.

 * Opens the bridge on a real D3D12 device whose adapter LUID is matched by a
 * CUDA context, then measures the two things the stage-B contract claims:

 *  gate 1  a P010 -> linear scRGB -> P010 roundtrip on legal neutral and colour
 *          codes, judged twice: against an exact-arithmetic reference of the same
 *          documented path, and against the code error the FP16 store can itself
 *          account for, with the maximum and mean reported;
 *  gate 2  the proxy and the reconstruction against the same inputs, dumped so
 *          the corrected CPU reference module can judge them, plus the identity
 *          and strength-zero gates and the domain refusals.

 * Every readback here is explicit correctness instrumentation on synthetic
 * data, never the production chain, and its bytes are counted separately by the
 * bridge. Scratch artifacts of failed attempts are left in place. */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "include/cuda.h"
#include "gpu_hdr_scrgb_bridge.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what);
    if (!ok) ++g_failures;
}

std::string g_failure_notes;
void note(const std::string& line) {
    std::printf("       %s\n", line.c_str());
    g_failure_notes += line + "\n";
}

/* --- CUDA driver entry points the harness needs -------------------------- */

struct Cuda {
    HMODULE lib = nullptr;
    CUresult(CUDAAPI* init)(unsigned) = nullptr;
    CUresult(CUDAAPI* device_get_count)(int*) = nullptr;
    CUresult(CUDAAPI* device_get)(CUdevice*, int) = nullptr;
    CUresult(CUDAAPI* device_luid)(char*, unsigned*, CUdevice) = nullptr;
    CUresult(CUDAAPI* ctx_create)(CUcontext*, unsigned, CUdevice) = nullptr;
    CUresult(CUDAAPI* ctx_destroy)(CUcontext) = nullptr;
    CUresult(CUDAAPI* mem_alloc)(CUdeviceptr*, size_t) = nullptr;
    CUresult(CUDAAPI* mem_free)(CUdeviceptr) = nullptr;
    CUresult(CUDAAPI* memcpy_htod)(CUdeviceptr, const void*, size_t) = nullptr;
    CUresult(CUDAAPI* memcpy_dtoh)(void*, CUdeviceptr, size_t) = nullptr;

    bool load() {
        lib = LoadLibraryA("nvcuda.dll");
        if (!lib) { note("nvcuda.dll is not loadable"); return false; }
        auto get = [this](const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(lib, name));
        };
        init = reinterpret_cast<decltype(init)>(get("cuInit"));
        device_get_count = reinterpret_cast<decltype(device_get_count)>(
            get("cuDeviceGetCount"));
        device_get = reinterpret_cast<decltype(device_get)>(get("cuDeviceGet"));
        device_luid = reinterpret_cast<decltype(device_luid)>(
            get("cuDeviceGetLuid"));
        ctx_create = reinterpret_cast<decltype(ctx_create)>(
            get("cuCtxCreate_v2"));
        ctx_destroy = reinterpret_cast<decltype(ctx_destroy)>(
            get("cuCtxDestroy_v2"));
        mem_alloc = reinterpret_cast<decltype(mem_alloc)>(get("cuMemAlloc_v2"));
        mem_free = reinterpret_cast<decltype(mem_free)>(get("cuMemFree_v2"));
        memcpy_htod = reinterpret_cast<decltype(memcpy_htod)>(
            get("cuMemcpyHtoD_v2"));
        memcpy_dtoh = reinterpret_cast<decltype(memcpy_dtoh)>(
            get("cuMemcpyDtoH_v2"));
        const bool ready = init && device_get_count && device_get &&
            device_luid && ctx_create && ctx_destroy && mem_alloc && mem_free &&
            memcpy_htod && memcpy_dtoh;
        if (!ready) note("the CUDA driver entry points are missing");
        return ready;
    }
};

struct DeviceBuffer {
    CUdeviceptr ptr = 0;
    size_t bytes = 0;
    bool alloc(Cuda& cuda, size_t size) {
        bytes = size;
        return cuda.mem_alloc(&ptr, size) == CUDA_SUCCESS;
    }
    void free(Cuda& cuda) {
        if (ptr) cuda.mem_free(ptr);
        ptr = 0;
    }
    bool upload(Cuda& cuda, const void* src, size_t size) {
        return cuda.memcpy_htod(ptr, src, size) == CUDA_SUCCESS;
    }
};

/* The D3D12 default device and a CUDA context on the same physical adapter,
 * which is what the bridge insists on. */
struct Machine {
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Cuda cuda;
    CUcontext context = nullptr;

    bool open() {
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device)))) {
            note("no D3D12 device");
            return false;
        }
        if (!cuda.load()) return false;
        const LUID dx = device->GetAdapterLuid();
        if (cuda.init(0) != CUDA_SUCCESS) { note("cuInit failed"); return false; }
        int count = 0;
        cuda.device_get_count(&count);
        for (int i = 0; i < count; ++i) {
            CUdevice d = 0;
            char luid[8]{};
            unsigned node_mask = 0;
            if (cuda.device_get(&d, i) != CUDA_SUCCESS) continue;
            if (cuda.device_luid(luid, &node_mask, d) != CUDA_SUCCESS) continue;
            if (std::memcmp(luid, &dx, sizeof(luid)) != 0) continue;
            if (cuda.ctx_create(&context, 0, d) == CUDA_SUCCESS) return true;
        }
        note("no CUDA device matches the D3D12 adapter LUID");
        return false;
    }

    void close() {
        if (context && cuda.ctx_destroy) cuda.ctx_destroy(context);
        context = nullptr;
        device.Reset();
    }
};

/* --- The contract -------------------------------------------------------- */

/* A two-tap box plan over the extents under test. The gates that measure
 * resampling quality hand the bridge the CPU reference's real Lanczos-2 and
 * Catmull-Rom plans; these gates use a plan the reference can reproduce exactly
 * in numpy, so the proxy comparison is about colour maths and not about a
 * filter. */
struct Plan {
    GpuHdrTapTable table{};
    std::vector<int32_t> vertical_indices;
    std::vector<float> vertical_weights;
    std::vector<int32_t> horizontal_indices;
    std::vector<float> horizontal_weights;

    Plan(uint32_t src_w, uint32_t src_h, uint32_t dst_w, uint32_t dst_h) {
        vertical_indices.resize(size_t(dst_h) * 2u);
        vertical_weights.resize(size_t(dst_h) * 2u);
        horizontal_indices.resize(size_t(dst_w) * 2u);
        horizontal_weights.resize(size_t(dst_w) * 2u);
        for (uint32_t y = 0; y < dst_h; ++y) {
            const int32_t s = std::min<int32_t>(int32_t(2u * y), int32_t(src_h) - 1);
            const int32_t s2 = std::min<int32_t>(s + 1, int32_t(src_h) - 1);
            vertical_indices[size_t(y) * 2u] = s;
            vertical_indices[size_t(y) * 2u + 1] = s2;
            vertical_weights[size_t(y) * 2u] = 0.5f;
            vertical_weights[size_t(y) * 2u + 1] = 0.5f;
        }
        for (uint32_t x = 0; x < dst_w; ++x) {
            const int32_t s = std::min<int32_t>(int32_t(2u * x), int32_t(src_w) - 1);
            const int32_t s2 = std::min<int32_t>(s + 1, int32_t(src_w) - 1);
            horizontal_indices[size_t(x) * 2u] = s;
            horizontal_indices[size_t(x) * 2u + 1] = s2;
            horizontal_weights[size_t(x) * 2u] = 0.5f;
            horizontal_weights[size_t(x) * 2u + 1] = 0.5f;
        }
        table.struct_size = sizeof(GpuHdrTapTable);
        table.taps = 2u;
        table.vertical_src_len = src_h;
        table.vertical_dst_len = dst_h;
        table.vertical_indices = vertical_indices.data();
        table.vertical_weights = vertical_weights.data();
        table.horizontal_src_len = src_w;
        table.horizontal_dst_len = dst_w;
        table.horizontal_indices = horizontal_indices.data();
        table.horizontal_weights = horizontal_weights.data();
    }
};

const char* last_error() { return gpu_hdr_last_error(); }

void show_contract(GpuHdrContract& c) {
    c.struct_size = sizeof(GpuHdrContract);
    gpu_hdr_fill_contract(&c);
    std::printf("contract ABI %u\n", gpu_hdr_abi_version());
    std::printf("  scRGB per cd/m2      %.17g (1.0 scRGB = %.1f cd/m2)\n",
        c.scrgb_per_nit, 1.0 / c.scrgb_per_nit);
    std::printf("  BT.2020 -> scRGB     %.17g %.17g %.17g\n", c.to_scrgb[0],
        c.to_scrgb[1], c.to_scrgb[2]);
    std::printf("                       %.17g %.17g %.17g\n", c.to_scrgb[3],
        c.to_scrgb[4], c.to_scrgb[5]);
    std::printf("                       %.17g %.17g %.17g\n", c.to_scrgb[6],
        c.to_scrgb[7], c.to_scrgb[8]);
    std::printf("  scRGB -> BT.2020     %.17g %.17g %.17g\n", c.to_bt2020[0],
        c.to_bt2020[1], c.to_bt2020[2]);
    std::printf("                       %.17g %.17g %.17g\n", c.to_bt2020[3],
        c.to_bt2020[4], c.to_bt2020[5]);
    std::printf("                       %.17g %.17g %.17g\n", c.to_bt2020[6],
        c.to_bt2020[7], c.to_bt2020[8]);
    std::printf("  PQ M1 %.17g M2 %.17g C1 %.17g C2 %.17g C3 %.17g\n",
        c.pq_m1, c.pq_m2, c.pq_c1, c.pq_c2, c.pq_c3);
    std::printf("  Kr %.6f Kg %.6f Kb %.6f\n", c.kr, c.kg, c.kb);
    std::printf("  proxy knee %.6f span %.6f K %.6f\n", c.proxy_knee,
        c.proxy_shoulder_span, c.proxy_shoulder_k);
    std::printf("  saturation %.17g actionable %.17g\n",
        c.proxy_saturation_relative, c.proxy_actionable_relative);
    std::printf("  max stops %.6f min slope per stop %.17g\n", c.max_stops,
        c.min_slope_per_stop);
    check(c.nits_per_scrgb == 80.0, "reference white is 80 cd/m2");
    check(c.proxy_actionable_relative > 1.0 &&
          c.proxy_actionable_relative < c.proxy_saturation_relative,
          "the measurable band is below the pinning point");
    check(std::fabs(c.proxy_saturation_relative - 1.808731372940256) < 1e-12,
          "saturation matches the stage-A closed form");
    check(c.proxy_actionable_relative == 1.4091796875,
          "the first dead level matches the stage-A enumeration");
    /* The published constants of the two standards, checked against the
     * derivation rather than trusted: the Y row of Rec.2020's own RGB->XYZ. */
    check(std::fabs(c.kr - 0.2627) < 5e-4 && std::fabs(c.kg - 0.6780) < 5e-4 &&
          std::fabs(c.kb - 0.0593) < 5e-4, "the derived luma weights are the "
          "published BT.2020 Kr, Kg, Kb");
    double row_sum = c.to_scrgb[3] + c.to_scrgb[4] + c.to_scrgb[5];
    check(std::fabs(row_sum - 1.0) < 1e-12,
          "a primary conversion row preserves a neutral white");
}

/* --- Synthetic P010 ------------------------------------------------------ */

/* Legal limited-range 10-bit codes, left-aligned in the 16-bit unit as
 * AV_PIX_FMT_P010LE requires: luma 64..940, colour difference 512 +- 448. One
 * sample per 2x2 block, so the encoder's box mean of the chroma is exact.
 *
 * Both tables are generated, not guessed: each row is the forward BT.2020 NCL
 * encode of an encoded primary triple that lies inside [0, 1], which is exactly
 * the condition the bridge's inverse has to be able to undo. A row typed by hand
 * with a peak luma code *and* a maximum chroma code decodes to an encoded primary
 * above 1.0, and PQ has no domain there - the earlier draft of this fixture did
 * that and the roundtrip measured the extrapolation instead of the maths.
 * scratch/hdr_fixture_probe.py reproduces every number below, including the
 * scRGB each row reaches after the gamut conversion. */
struct Sample { uint16_t y, cb, cr; const char* name; };

/* Job 1: a 1000 cd/m2 declared domain. Every component of every row is inside
 * the 12.5 scRGB the job declares, and three rows are gamut-negative so the
 * signed master is exercised. */
const Sample kJob1Samples[] = {
    {64, 512, 512, "coded black"},
    {717, 512, 512, "inside the ceiling"},
    {722, 512, 512, "at the ceiling"},
    {490, 512, 512, "reference white 80 nits"},
    {573, 512, 512, "SGOT 203 nits"},
    {281, 512, 512, "near black 5 nits"},
    {224, 425, 824, "gamut-negative red"},
    {499, 276, 210, "gamut-negative green"},
    {102, 843, 485, "gamut-negative blue"},
    {485, 444, 628, "warm mid"},
    {517, 551, 440, "cool mid"},
    {576, 411, 520, "yellow low"},
};

/* The sweep: the whole legal code range end to end, including the three
 * BT.2020 primaries, whose converted components reach 207 scRGB. It exists to
 * measure the roundtrip where no proxy band can be in the way. */
const Sample kSweepSamples[] = {
    {64, 512, 512, "coded black"},
    {940, 512, 512, "PQ peak white"},
    {431, 512, 512, "40-nit grey"},
    {490, 512, 512, "reference white 80 nits"},
    {573, 512, 512, "SGOT 203 nits"},
    {723, 512, 512, "1000-nit white"},
    {281, 512, 512, "near black 5 nits"},
    {294, 387, 960, "BT.2020 primary red"},
    {658, 189, 100, "BT.2020 primary green"},
    {116, 960, 476, "BT.2020 primary blue"},
    {432, 741, 713, "violet past 709"},
    {627, 396, 682, "bright warm"},
};
constexpr uint32_t kJob1SampleCount =
    uint32_t(sizeof(kJob1Samples) / sizeof(kJob1Samples[0]));
constexpr uint32_t kSweepSampleCount =
    uint32_t(sizeof(kSweepSamples) / sizeof(kSweepSamples[0]));

uint16_t left(uint32_t code) { return uint16_t(code << 6); }

struct P010Frame {
    uint32_t width = 0, height = 0;
    uint32_t y_pitch = 0, uv_pitch = 0;
    std::vector<uint16_t> y;
    std::vector<uint16_t> uv;

    void build(uint32_t w, uint32_t h, const Sample* samples,
               uint32_t count) {
        width = w; height = h;
        y_pitch = w * 2u;                       /* 16-byte multiple for w % 8 */
        uv_pitch = (w / 2u) * 2u * 2u;          /* interleaved Cb, Cr pairs */
        y.assign(size_t(y_pitch) / 2u * h, uint16_t(0));
        uv.assign(size_t(uv_pitch) / 2u * (h / 2u), uint16_t(0));
        for (uint32_t block = 0; block < (w / 2u) * (h / 2u); ++block) {
            const Sample& s = samples[block % count];
            const uint32_t bx = block % (w / 2u), by = block / (w / 2u);
            for (uint32_t dy = 0; dy < 2u; ++dy)
                for (uint32_t dx = 0; dx < 2u; ++dx) {
                    const uint32_t x = bx * 2u + dx, yy = by * 2u + dy;
                    y[size_t(yy) * (size_t(y_pitch) / 2u) + x] = left(s.y);
                }
            /* One row of P010 UV is uv_pitch bytes, so a chroma pair sits at
             * element 2*bx of a row that is uv_pitch/2 uint16 wide. */
            uv[size_t(by) * (size_t(uv_pitch) / 2u) + bx * 2u] = left(s.cb);
            uv[size_t(by) * (size_t(uv_pitch) / 2u) + bx * 2u + 1] =
                left(s.cr);
        }
    }
};

/* --- Half helpers used by the harness's own comparisons ------------------ */

uint16_t to_half(double value) {
    const float wide = float(value);
    uint32_t bits = 0;
    std::memcpy(&bits, &wide, sizeof(bits));
    const uint32_t sign = bits & 0x80000000u;
    const uint32_t rest = bits & 0x7FFFFFFFu;
    uint32_t half = 0;
    if (rest >= 0x38800000u) {
        uint32_t exponent = (rest >> 23) - 127 + 15;
        uint32_t mantissa = rest & 0x7FFFFFu;
        const uint32_t dropped = mantissa & 0x1FFFu;
        mantissa >>= 13;
        const uint32_t guard = 0x1000u;
        if (dropped > guard || (dropped == guard && (mantissa & 1u))) ++mantissa;
        if (mantissa > 0x3FFu) { mantissa = 0; ++exponent; }
        half = (exponent << 10) | mantissa;
    } else {
        half = uint32_t(std::nearbyint(double(wide) * 16777216.0));
    }
    return uint16_t((half << 0) | (sign >> 16));
}

double from_half(uint16_t pattern) {
    const uint32_t e = (pattern >> 10) & 0x1Fu;
    const uint32_t m = pattern & 0x3FFu;
    double value;
    if (e == 0) value = std::ldexp(double(m), -24);
    else if (e == 0x1F) value = m ? std::numeric_limits<double>::quiet_NaN()
                                 : HUGE_VAL;
    else value = std::ldexp(double(m | 0x400u), int(e) - 25);
    return (pattern & 0x8000u) ? -value : value;
}

/* A binary16 pattern as its place on the number line, so two patterns can be
 * compared in grid steps even across the sign bit: adjacent patterns are one
 * apart on either side of zero. */
long half_pattern_index(uint16_t pattern) {
    const long magnitude = long(pattern & 0x7FFFu);
    return (pattern & 0x8000u) ? -magnitude : magnitude;
}

/* One FP16 plane as the bridge hands it back: 8 bytes per sample, RGBA. */
struct PlaneDump {
    uint32_t width = 0, height = 0, pitch = 0;
    std::vector<uint8_t> bytes;
    uint16_t pattern(uint32_t row, uint32_t column, uint32_t channel) const {
        const size_t offset = size_t(row) * pitch + size_t(column) * 8u +
            size_t(channel) * 2u;
        uint16_t value = 0;
        std::memcpy(&value, bytes.data() + offset, sizeof(value));
        return value;
    }
    double at(uint32_t row, uint32_t column, uint32_t channel) const {
        return from_half(pattern(row, column, channel));
    }
    void set(uint32_t row, uint32_t column, uint32_t channel, double value) {
        const size_t offset = size_t(row) * pitch + size_t(column) * 8u +
            size_t(channel) * 2u;
        const uint16_t pattern = to_half(value);
        std::memcpy(bytes.data() + offset, &pattern, sizeof(pattern));
    }
};

/* --- Output for the CPU reference ---------------------------------------- */

std::filesystem::path g_dump_dir;

void write_file(const std::filesystem::path& name, const void* data,
                size_t bytes) {
    const std::filesystem::path path = g_dump_dir / name;
    FILE* handle = nullptr;
    if (fopen_s(&handle, path.string().c_str(), "wb") != 0 || !handle) {
        note("cannot open " + path.string());
        return;
    }
    std::fwrite(data, 1, bytes, handle);
    std::fclose(handle);
}

class Summary {
public:
    void add(const std::string& key, const std::string& value) {
        lines_ += key + " = " + value + "\n";
    }
    void add(const std::string& key, double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.17g", value);
        add(key, buffer);
    }
    void add(const std::string& key, uint64_t value) {
        char buffer[32];
        std::snprintf(buffer, sizeof(buffer), "%llu",
            static_cast<unsigned long long>(value));
        add(key, buffer);
    }
    void write(const char* name) {
        write_file(name, lines_.data(), lines_.size());
    }
    void dump(const PlaneDump& plane, const char* name) {
        char key[96];
        std::snprintf(key, sizeof(key), "%s_pitch", name);
        add(key, uint64_t(plane.pitch));
        std::snprintf(key, sizeof(key), "%s_width", name);
        add(key, uint64_t(plane.width));
        std::snprintf(key, sizeof(key), "%s_height", name);
        add(key, uint64_t(plane.height));
        write_file(name, plane.bytes.data(), plane.bytes.size());
    }
private:
    std::string lines_;
} g_summary;

/* Reports are recorded field by field under their own names. Walking a report as
 * an array of uint32_t would type-pun the doubles inside it into garbage, and a
 * summary the Python side cannot name is not evidence. */

/* --- Refusals ------------------------------------------------------------ */

void expect_refusal(int32_t rc, const char* what) {
    const bool refused = rc != 0;
    check(refused, what);
    if (!refused) return;
    std::string reason = last_error();
    check(!reason.empty(), "the refusal carries a reason");
    note(std::string(what) + " -> " + reason);
}

/* --- The neural input the bridge reads ----------------------------------- */

/* The Feature 18 output as the host leaves it: an RGBA16_FLOAT 2D texture on
 * the same device, in COPY_SOURCE, behind a fence of the same device. Each
 * guarded call names itself, so a refusal says which one and not only what code
 * it came back with. */
#define NEURAL_GUARD(text, call) \
    do { last_step = text; if (FAILED(last_hr = (call))) return false; } \
    while (false)

struct NeuralTexture {
    Microsoft::WRL::ComPtr<ID3D12Resource> texture;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t fence_value = 0;
    HRESULT last_hr = S_OK;
    const char* last_step = "";

    bool build(ID3D12Device* device, uint32_t width, uint32_t height,
               const std::vector<float>& rgba) {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        NEURAL_GUARD("command queue", device->CreateCommandQueue(&queue_desc,
            IID_PPV_ARGS(&queue)));
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        const uint32_t row_pitch = ((width * 8u) + 255u) & ~255u;
        const uint32_t buffer_bytes = row_pitch * height;
        D3D12_HEAP_PROPERTIES default_heap{};
        default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_HEAP_PROPERTIES upload_heap{};
        upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC buffer_desc{};
        buffer_desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer_desc.Width = buffer_bytes;
        buffer_desc.Height = 1;
        buffer_desc.DepthOrArraySize = 1;
        buffer_desc.MipLevels = 1;
        buffer_desc.SampleDesc.Count = 1;
        buffer_desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload;
        NEURAL_GUARD("neural texture", device->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture)));
        NEURAL_GUARD("upload buffer", device->CreateCommittedResource(
            &upload_heap, D3D12_HEAP_FLAG_NONE, &buffer_desc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(&upload)));
        {
            uint8_t* mapped = nullptr;
            NEURAL_GUARD("upload map", upload->Map(0, nullptr,
                reinterpret_cast<void**>(&mapped)));
            for (uint32_t row = 0; row < height; ++row) {
                uint16_t* dst = reinterpret_cast<uint16_t*>(
                    mapped + size_t(row) * row_pitch);
                for (uint32_t column = 0; column < width; ++column) {
                    for (uint32_t channel = 0; channel < 4u; ++channel)
                        dst[column * 4u + channel] = to_half(
                            rgba[size_t(size_t(row) * width + column) * 4u +
                                 channel]);
                }
            }
            upload->Unmap(0, nullptr);
        }
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        NEURAL_GUARD("allocator", device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator)));
        NEURAL_GUARD("command list", device->CreateCommandList(0,
            D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
            IID_PPV_ARGS(&list)));
        /* One copy, linear to tiled: the placed footprint is the upload row and
         * the other side has to be the texture's own subresource. Asking for a
         * footprint on both sides is the invalid call, and it fails the list at
         * Close rather than where the mistake is. */
        D3D12_TEXTURE_COPY_LOCATION destination{}, source{};
        destination.pResource = texture.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        source.pResource = upload.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Offset = 0;
        source.PlacedFootprint.Footprint.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        source.PlacedFootprint.Footprint.Width = width;
        source.PlacedFootprint.Footprint.Height = height;
        source.PlacedFootprint.Footprint.Depth = 1;
        source.PlacedFootprint.Footprint.RowPitch = row_pitch;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture.Get();
        barrier.Transition.Subresource = 0;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &barrier);
        NEURAL_GUARD("close", list->Close());
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        NEURAL_GUARD("fence", device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&fence)));
        fence_value = 1;
        NEURAL_GUARD("signal", queue->Signal(fence.Get(), fence_value));
        HANDLE event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!event) return false;
        while (fence->GetCompletedValue() < fence_value) {
            if (FAILED(fence->SetEventOnCompletion(fence_value, event))) {
                CloseHandle(event);
                return false;
            }
            WaitForSingleObject(event, 5000);
        }
        CloseHandle(event);
        return true;
    }
};

#undef NEURAL_GUARD

/* --- The exact-arithmetic mirror of the decode and encode path -----------
 *
 * ST 2084 has infinite slope at code zero, so no fixed number of codes can
 * bound this roundtrip. A 2020 primary that the source holds at exactly black
 * picks up a few thousandths of a nit from one binary16 step taken by *another*
 * channel, and the toe spends tens of codes on it: measured here, on a BT.2020
 * primary blue, forty-seven. Both halves of the claim are therefore made
 * against arithmetic rather than against a magic number. The mirror recomputes
 * the documented path in doubles from the contract the bridge publishes, so a
 * disagreement localises to the shader; and the box around it is one half step
 * wide per channel, which is exactly the room the store is allowed to take, so
 * the loss is bounded by what the representation can do to that sample and not
 * by an inherited tolerance. This is the gate's reference; no production path
 * calls any of it. */

/* The limited-range 10-bit quantiser of ITU-R BT.709/2020 as ST 2086 uses it. */
constexpr double kLumaBlack = 64.0, kLumaSpan = 876.0;
constexpr double kChromaNeutral = 512.0, kChromaSpan = 896.0;

struct Path {
    const GpuHdrContract* c;
    double relative_to_scrgb = 125.0;
    double cr_gain = 0.0, cb_gain = 0.0, cg_cr = 0.0, cg_cb = 0.0;

    /* scRGB is the plane's own unit, 1.0 == 80 cd/m^2, and one relative PQ step
     * (ST 2084's 0..1 over 10000 cd/m^2) is 10000/80 = 125 of them - the same
     * kScrgbPerRelative colour_header() prints into the shader. This used to be
     * the reciprocal, 1/(scrgb_per_nit * 10000) = 1/125, because decode
     * multiplies by it and encode divides by it and the roundtrip therefore
     * agreed with itself whatever it was. It is not harmless: the mirror then
     * quantized its model plane on a binary16 grid 15625 times finer than the
     * one the device stores, so half_step() was the quantum of a plane that
     * does not exist, and encode() was run on a different number than the one
     * the encoder actually read. */
    explicit Path(const GpuHdrContract& contract) : c(&contract) {
        relative_to_scrgb = contract.scrgb_per_nit * 10000.0;
        cr_gain = 2.0 * (1.0 - contract.kr);
        cb_gain = 2.0 * (1.0 - contract.kb);
        cg_cr = 2.0 * contract.kr * (1.0 - contract.kr) / contract.kg;
        cg_cb = 2.0 * contract.kb * (1.0 - contract.kb) / contract.kg;
    }

    /* The two transfer functions exactly as colour_header writes them, with
     * the same clamp of a negative encoded sample to zero. Doubles here, float
     * series on the GPU: that is the point of comparing them. */
    double eotf(double p) const {
        const double root = std::pow(std::max(p, 0.0), 1.0 / c->pq_m2);
        return std::pow(std::max(root - c->pq_c1, 0.0) /
            (c->pq_c2 - c->pq_c3 * root), 1.0 / c->pq_m1);
    }
    double oetf(double e) const {
        const double em = std::pow(std::max(e, 0.0), c->pq_m1);
        return std::pow((c->pq_c1 + c->pq_c2 * em) / (1.0 + c->pq_c3 * em),
            c->pq_m2);
    }

    /* 10-bit YCbCr codes to linear scRGB, the decode kernel's arithmetic. */
    void decode(const double codes[3], double out[3]) const {
        const double y = (codes[0] - kLumaBlack) / kLumaSpan;
        const double cb = (codes[1] - kChromaNeutral) / kChromaSpan;
        const double cr = (codes[2] - kChromaNeutral) / kChromaSpan;
        const double encoded[3] = { y + cr_gain * cr,
            y - cg_cb * cb - cg_cr * cr, y + cb_gain * cb };
        double linear2020[3];
        for (int i = 0; i < 3; ++i) linear2020[i] = eotf(encoded[i]);
        for (int i = 0; i < 3; ++i)
            out[i] = (c->to_scrgb[i * 3] * linear2020[0] +
                c->to_scrgb[i * 3 + 1] * linear2020[1] +
                c->to_scrgb[i * 3 + 2] * linear2020[2]) * relative_to_scrgb;
    }

    /* Linear scRGB to *unrounded* 10-bit code values, the encode kernel's
     * arithmetic. A chroma box mean is skipped because the gate feeds this a
     * single sample; the four pixels of every fixture box are equal. */
    void encode(const double scrgb[3], double out[3]) const {
        double linear2020[3];
        for (int i = 0; i < 3; ++i)
            linear2020[i] = (c->to_bt2020[i * 3] * scrgb[0] +
                c->to_bt2020[i * 3 + 1] * scrgb[1] +
                c->to_bt2020[i * 3 + 2] * scrgb[2]) / relative_to_scrgb;
        double encoded[3];
        for (int i = 0; i < 3; ++i) encoded[i] = oetf(linear2020[i]);
        const double y = c->kr * encoded[0] + c->kg * encoded[1] +
            c->kb * encoded[2];
        out[0] = y * kLumaSpan + kLumaBlack;
        out[1] = (encoded[2] - y) / cb_gain * kChromaSpan + kChromaNeutral;
        out[2] = (encoded[0] - y) / cr_gain * kChromaSpan + kChromaNeutral;
    }

    /* pack_p010 clamps into 10 bits and then uses HLSL round(), which breaks
     * halves away from zero. Code values here are never negative before the
     * clamp, so floor(x + 0.5) is the same function. */
    double quantise(double value) const {
        return std::floor(std::clamp(value, 0.0, 1023.0) + 0.5);
    }
};

/* The distance to the next binary16 pattern, which is the store's own quantum
 * at this magnitude. Sign is dropped: the step of -0.5 and of +0.5 are the same
 * number, and the box below is symmetric about the value the plane holds. */
double half_step(double value) {
    const uint16_t pattern = to_half(value);
    const double here = from_half(pattern);
    const double next = from_half(uint16_t(pattern + 1u));
    if (!std::isfinite(next)) return here == 0.0 ? 0.0 : std::abs(here);
    return std::abs(next - here);
}

/* The widest the encode can get from the source codes when every channel of the
 * master is free to sit anywhere the rounding left it: the eight corners of the
 * box whose half-widths are the store's quanta. The encode is monotone in each
 * channel of its input - a channel's own PQ code rises with it, and the luma
 * rises with each primary - so the corners bound the interior. */
void bound_box(const Path& path, const double master[3], const double want[3],
               double out[3]) {
    double low[3], high[3];
    for (int channel = 0; channel < 3; ++channel) {
        const double half = half_step(master[channel]) / 2.0;
        low[channel] = master[channel] - half;
        high[channel] = master[channel] + half;
    }
    out[0] = out[1] = out[2] = 0.0;
    for (int corner = 0; corner < 8; ++corner) {
        const double probe[3] = {
            (corner & 1) ? high[0] : low[0],
            (corner & 2) ? high[1] : low[1],
            (corner & 4) ? high[2] : low[2] };
        double codes[3];
        path.encode(probe, codes);
        for (int component = 0; component < 3; ++component) {
            const double quantised = path.quantise(codes[component]);
            out[component] = std::max(out[component],
                std::abs(quantised - want[component]));
        }
    }
}

/* --- Gate 1: the P010 roundtrip ----------------------------------------- */

constexpr uint32_t kFullWidth = 256u, kFullHeight = 256u;
constexpr uint32_t kWorkingWidth = 128u, kWorkingHeight = 128u;

struct RowError {
    const char* name = "";
    uint32_t want_y = 0, got_y = 0, want_cb = 0, got_cb = 0;
    uint32_t want_cr = 0, got_cr = 0;
    /* What the reference path says the codes should be, how far the store's own
     * quantum can move them, and what the returned codes cost in light. */
    double mirror[3] = { 0.0, 0.0, 0.0 };
    double bound[3] = { 0.0, 0.0, 0.0 };
    /* The binary16 the master plane actually holds for this sample, which is
     * the encoder's real input, and the reference's own rounding of the doubles
     * decode, so the two can be shown side by side. */
    double stored[3] = { 0.0, 0.0, 0.0 };
    double model[3] = { 0.0, 0.0, 0.0 };
    uint32_t pattern_gap = 0;
    double light_error_nits = 0.0;
};

struct Roundtrip {
    uint32_t y_max = 0, y_mean_num = 0;
    uint64_t y_total = 0;
    uint32_t uv_max = 0;
    uint64_t uv_total = 0;
    uint32_t uv_count = 0;
    /* Per-sample aggregates over the rows the fixture laid out in order. */
    uint32_t mirror_max_codes = 0;      /* GPU against the doubles reference */
    uint32_t outside_bound = 0;         /* components worse than their own box */
    uint32_t amplified_rows = 0;        /* rows whose box is over 3 codes wide */
    /* How far the device's stored pattern sits from the pattern the doubles
     * decode rounds to. Separate from mirror_max_codes on purpose: this one is
     * the float32 curve standing on the other side of a round-to-nearest-even
     * midpoint, that one is the encode deviating from exact arithmetic. */
    uint32_t decode_pattern_gap = 0;
    uint32_t decode_pattern_samples = 0;
    double worst_light_nits = 0.0;
    /* One row per fixture sample, read back from the first block of each: the
     * aggregate max above says how bad the roundtrip is, this says which codes
     * it is bad on. A sweep or fixture with more rows than this is not
     * diagnosed, only measured. */
    RowError rows[16] = {};
    uint32_t row_count = 0;
};

/* One fixture row of the roundtrip, with the two halves of the claim kept
 * apart: the codes the device returned against the box the store's own quantum
 * can cost at that sample, and the halves the encoder was actually asked about
 * against the reference's own rounding of the doubles decode. */
void show_row(const RowError& e) {
    std::printf("    %-22s y %4u->%-4u cb %4u->%-4u cr %4u->%-4u "
        "box %4.0f/%4.0f/%4.0f light %7.3f cd/m2\n",
        e.name, e.want_y, e.got_y, e.want_cb, e.got_cb, e.want_cr,
        e.got_cr, e.bound[0], e.bound[1], e.bound[2], e.light_error_nits);
    std::printf("    %-22s   stored %9.4f %9.4f %9.4f  model %9.4f %9.4f "
        "%9.4f  %u step(s)  reference codes %5.0f %5.0f %5.0f\n",
        "", e.stored[0], e.stored[1], e.stored[2], e.model[0], e.model[1],
        e.model[2], e.pattern_gap, e.mirror[0], e.mirror[1], e.mirror[2]);
}

bool run_roundtrip(GpuHdrBridge* bridge, Machine& machine, P010Frame& frame,
                   DeviceBuffer& y_buffer, DeviceBuffer& uv_buffer,
                   double ceiling, const Sample* samples, uint32_t count,
                   const GpuHdrContract& contract, Roundtrip* out) {
    const Path path(contract);
    std::printf("   step: allocate\n");
    if (!y_buffer.alloc(machine.cuda, size_t(frame.y_pitch) * frame.height))
        return false;
    if (!uv_buffer.alloc(machine.cuda, size_t(frame.uv_pitch) * frame.height / 2u))
        return false;
    std::printf("   step: upload\n");
    if (!y_buffer.upload(machine.cuda, frame.y.data(),
            frame.y.size() * sizeof(uint16_t))) return false;
    if (!uv_buffer.upload(machine.cuda, frame.uv.data(),
            frame.uv.size() * sizeof(uint16_t))) return false;
    std::printf("   step: decode y_pitch %u uv_pitch %u\n", frame.y_pitch,
        frame.uv_pitch);
    uint64_t master_ptr = 0;
    uint32_t master_pitch = 0;
    if (gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch, uv_buffer.ptr,
        frame.uv_pitch, 1, 9, 16, 10, ceiling, &master_ptr, &master_pitch) != 0) {
        note(std::string("decode refused: ") + last_error());
        return false;
    }
    std::printf("   step: identity chain\n");
    check(master_ptr != 0 && master_pitch == gpu_hdr_plane_pitch(bridge, 2),
          "decode hands back the master plane's own device address");

    /* The encoder reads the composited plane, so the roundtrip drives the middle
     * of the chain with a Feature 18 output that is the proxy itself: identity in
     * gives a zero residual, a zero residual gives the master back, and what the
     * encode then measures is the decoder and the encoder and nothing else. */
    uint64_t scratch_ptr = 0;
    uint32_t scratch_pitch = 0;
    if (gpu_hdr_downscale(bridge, &scratch_ptr, &scratch_pitch) != 0 ||
        gpu_hdr_proxy(bridge, &scratch_ptr, &scratch_pitch, &scratch_ptr,
            &scratch_pitch) != 0) {
        note(std::string("roundtrip chain refused: ") + last_error());
        return false;
    }
    PlaneDump proxy_plane{};
    proxy_plane.width = kWorkingWidth;
    proxy_plane.height = kWorkingHeight;
    proxy_plane.pitch = gpu_hdr_plane_pitch(bridge, 4);
    proxy_plane.bytes.assign(size_t(proxy_plane.pitch) * kWorkingHeight,
        uint8_t(0));
    if (gpu_hdr_readback_scrgb(bridge, 4, proxy_plane.bytes.data(),
            proxy_plane.pitch) != 0) {
        note(std::string("roundtrip proxy readback refused: ") + last_error());
        return false;
    }
    std::vector<float> identity(size_t(kWorkingWidth) * kWorkingHeight * 4u);
    for (uint32_t row = 0; row < kWorkingHeight; ++row)
        for (uint32_t column = 0; column < kWorkingWidth; ++column)
            for (uint32_t channel = 0; channel < 4u; ++channel)
                identity[(size_t(row) * kWorkingWidth + column) * 4u + channel] =
                    float(proxy_plane.at(row, column, channel));
    NeuralTexture neural;
    if (!neural.build(machine.device.Get(), kWorkingWidth, kWorkingHeight,
        identity)) {
        char reason[96];
        std::snprintf(reason, sizeof(reason), "roundtrip neural input refused "
            "at %s (hr 0x%08lx)", neural.last_step,
            unsigned long(neural.last_hr));
        note(reason);
        return false;
    }
    if (gpu_hdr_reconstruct(bridge, neural.texture.Get(), neural.fence.Get(),
            neural.fence_value, 1.0, &scratch_ptr, &scratch_pitch) != 0) {
        note(std::string("roundtrip reconstruct refused: ") + last_error());
        return false;
    }

    std::printf("   step: encode\n");
    /* The claim above is measured, not asserted: if the composited plane is not
     * the master plane bit for bit, then gate 1 is not a decoder/encoder
     * roundtrip and its code errors mean something else. */
    PlaneDump master_dump{};
    master_dump.width = frame.width;
    master_dump.height = frame.height;
    master_dump.pitch = gpu_hdr_plane_pitch(bridge, 2);
    master_dump.bytes.assign(size_t(master_dump.pitch) * frame.height,
        uint8_t(0));
    bool master_readable = false;
    {
        std::vector<uint8_t> composite_bytes(
            size_t(gpu_hdr_plane_pitch(bridge, 7)) * frame.height);
        master_readable = gpu_hdr_readback_scrgb(bridge, 2,
            master_dump.bytes.data(), master_dump.pitch) == 0;
        if (master_readable &&
            gpu_hdr_readback_scrgb(bridge, 7, composite_bytes.data(),
                gpu_hdr_plane_pitch(bridge, 7)) == 0) {
            uint32_t differing = 0;
            double worst = 0.0;
            const size_t halves = master_dump.bytes.size() / sizeof(uint16_t);
            const uint16_t* a = reinterpret_cast<const uint16_t*>(
                master_dump.bytes.data());
            const uint16_t* b = reinterpret_cast<const uint16_t*>(
                composite_bytes.data());
            for (size_t word = 0; word < halves; ++word) {
                if (a[word] == b[word]) continue;
                ++differing;
                const double gap = std::abs(from_half(a[word]) -
                    from_half(b[word]));
                worst = std::max(worst, gap);
            }
            std::printf("    master vs composited: %u of %zu differing halves, "
                "worst gap %g scRGB\n", differing, halves, worst);
        } else {
            note(std::string("roundtrip plane compare refused: ") +
                last_error());
        }
    }
    uint64_t encode_y = 0, encode_uv = 0;
    uint32_t encode_y_pitch = 0, encode_uv_pitch = 0;
    if (gpu_hdr_encode(bridge, 125.0, &encode_y, &encode_y_pitch, &encode_uv,
        &encode_uv_pitch) != 0) {
        note(std::string("encode refused: ") + last_error());
        return false;
    }
    std::vector<uint8_t> back_y(size_t(encode_y_pitch) * frame.height);
    std::vector<uint8_t> back_uv(size_t(encode_uv_pitch) * frame.height / 2u);
    if (gpu_hdr_readback_encoded_p010(bridge, back_y.data(),
            back_y.size() / frame.height, back_uv.data(),
            back_uv.size() / (frame.height / 2u)) != 0) {
        note(std::string("encode readback refused: ") + last_error());
        return false;
    }
    const uint16_t* out_y = reinterpret_cast<const uint16_t*>(back_y.data());
    const uint16_t* out_uv = reinterpret_cast<const uint16_t*>(back_uv.data());
    const uint32_t out_y_stride = uint32_t(back_y.size() / frame.height) / 2u;
    const uint32_t out_uv_stride =
        uint32_t(back_uv.size() / (frame.height / 2u)) / 2u;
    for (uint32_t row = 0; row < frame.height; ++row) {
        for (uint32_t column = 0; column < frame.width; ++column) {
            const uint32_t want = frame.y[size_t(row) *
                (size_t(frame.y_pitch) / 2u) + column] >> 6;
            const uint32_t got = out_y[size_t(row) * out_y_stride + column] >> 6;
            const uint32_t delta = want > got ? want - got : got - want;
            out->y_max = std::max(out->y_max, delta);
            out->y_total += delta;
        }
    }
    for (uint32_t row = 0; row < frame.height / 2u; ++row) {
        for (uint32_t column = 0; column < frame.width / 2u; ++column) {
            for (uint32_t part = 0; part < 2u; ++part) {
                const uint32_t want = frame.uv[size_t(row) *
                    (size_t(frame.uv_pitch) / 2u) + column * 2u + part] >> 6;
                const uint32_t got = out_uv[size_t(row) * out_uv_stride +
                    column * 2u + part] >> 6;
                const uint32_t delta = want > got ? want - got : got - want;
                out->uv_max = std::max(out->uv_max, delta);
                out->uv_total += delta;
                ++out->uv_count;
            }
        }
    }
    out->row_count = std::min(count, uint32_t(16));
    for (uint32_t index = 0; index < out->row_count; ++index) {
        /* Block `index` of the first chroma row is the sample with this name,
         * because P010Frame::build lays the samples out in order. */
        const uint32_t bx = index % (frame.width / 2u);
        const uint32_t by = index / (frame.width / 2u);
        RowError& slot = out->rows[index];
        slot.name = samples[index].name;
        slot.want_y = frame.y[size_t(by * 2u) * (size_t(frame.y_pitch) / 2u) +
            bx * 2u] >> 6;
        slot.got_y = out_y[size_t(by * 2u) * out_y_stride + bx * 2u] >> 6;
        slot.want_cb = frame.uv[size_t(by) * (size_t(frame.uv_pitch) / 2u) +
            bx * 2u] >> 6;
        slot.got_cb = out_uv[size_t(by) * out_uv_stride + bx * 2u] >> 6;
        slot.want_cr = frame.uv[size_t(by) * (size_t(frame.uv_pitch) / 2u) +
            bx * 2u + 1u] >> 6;
        slot.got_cr = out_uv[size_t(by) * out_uv_stride + bx * 2u + 1u] >> 6;

        const double want[3] = { double(slot.want_y), double(slot.want_cb),
            double(slot.want_cr) };
        const double got[3] = { double(slot.got_y), double(slot.got_cb),
            double(slot.got_cr) };
        /* The master the plane holds, and the reference's own rounding of the
         * doubles decode, side by side. The mirror and the box are asked about
         * the pattern the encoder actually read: where the device's float32
         * curve and these doubles land on opposite sides of a
         * round-to-nearest-even midpoint they differ by a grid step, and the
         * encoder then legitimately returns different codes. */
        double exact[3], master[3];
        path.decode(want, exact);
        uint32_t row_gap = 0u;
        for (int channel = 0; channel < 3; ++channel) {
            const uint16_t model_pattern = to_half(exact[channel]);
            slot.model[channel] = from_half(model_pattern);
            const uint16_t held = master_readable
                ? master_dump.pattern(by * 2u, bx * 2u, uint32_t(channel))
                : model_pattern;
            slot.stored[channel] = from_half(held);
            const long here = half_pattern_index(held);
            const long there = half_pattern_index(model_pattern);
            const long difference = here > there ? here - there
                                                 : there - here;
            const uint32_t steps = uint32_t(difference);
            if (steps > row_gap) row_gap = steps;
            master[channel] = slot.stored[channel];
        }
        slot.pattern_gap = row_gap;
        if (row_gap > out->decode_pattern_gap) out->decode_pattern_gap = row_gap;
        if (row_gap != 0u) ++out->decode_pattern_samples;
        double codes[3];
        path.encode(master, codes);
        double returned[3];
        path.decode(got, returned);
        for (int component = 0; component < 3; ++component) {
            slot.mirror[component] = path.quantise(codes[component]);
            const double gap = std::abs(slot.mirror[component] - got[component]);
            out->mirror_max_codes = std::max(out->mirror_max_codes,
                uint32_t(std::ceil(gap)));
        }
        bound_box(path, master, want, slot.bound);
        for (int component = 0; component < 3; ++component) {
            const double error = std::abs(got[component] - want[component]);
            if (error > slot.bound[component] + 1.0) ++out->outside_bound;
        }
        if (slot.bound[0] > 3.0 || slot.bound[1] > 3.0 || slot.bound[2] > 3.0) {
            ++out->amplified_rows;
        }
        for (int channel = 0; channel < 3; ++channel) {
            slot.light_error_nits = std::max(slot.light_error_nits,
                std::abs(returned[channel] - master[channel]) *
                    contract.nits_per_scrgb);
        }
        out->worst_light_nits = std::max(out->worst_light_nits,
            slot.light_error_nits);
    }
    return true;
}

/* --- Harness ------------------------------------------------------------- */

/* Job 1 declares the domain its own fixture occupies: 12.5 scRGB is
 * 1000 cd/m2, the peak a mastering display is commonly tagged with, and every
 * component of every row of kJob1Samples is inside it - which the source report
 * then measures rather than takes on trust. Its paperWhite is the documented
 * selection rule, ceiling / proxy_actionable_relative, so the whole declared
 * domain lands in the band where the proxy carries a measurable slope. */
constexpr double kJob1Ceiling = 12.5;

/* The sweep declares a domain wide enough to contain every legal code, and runs
 * at the paperWhite stage A audited. Band top there is 4.0 * 1.409 = 5.637 scRGB,
 * so the bright half of the sweep arrives pinned: that is the configuration
 * where the reconstruction must provably keep the master instead of inventing a
 * magnitude, and gate 1 has no proxy in the path to interfere with it. */
constexpr double kSweepCeiling = 256.0;
constexpr double kSweepPaperWhite = 4.0;

/* --- Stage B counter map: sortable()/unsortable_to_bits() proof ---------- */

/* CPU mirrors of the two maps the bridge pairs across the device boundary: the
 * HLSL sortable() that orders float keys for InterlockedMin/Max and the host
 * unsortable_to_bits() that decodes them back. They must be exact inverses and
 * a strict order-embedding of the finite floats. The shipped HLSL complemented
 * only the sign and low bit of a negative instead of every bit, which broke
 * both and reported a phantom source_min. */
uint32_t sortable_bits_bits(uint32_t bits) {
    return bits ^ ((bits >> 31u) ? 0xFFFFFFFFu : 0x80000000u);
}
uint32_t sortable_bits_cpu(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return sortable_bits_bits(bits);
}
uint32_t unsortable_bits_cpu(uint32_t key) {
    return key >= 0x80000000u ? (key ^ 0x80000000u) : ~key;
}

/* The three colour channels of an FP16 plane, measured independently of the
 * bridge's own counters. Alpha (channel 3) is skipped; it carries no volume. */
void measure_plane_extrema(const PlaneDump& plane, double& lo, double& hi) {
    lo = std::numeric_limits<double>::infinity();
    hi = -std::numeric_limits<double>::infinity();
    for (uint32_t row = 0; row < plane.height; ++row)
        for (uint32_t col = 0; col < plane.width; ++col)
            for (uint32_t ch = 0; ch < 3u; ++ch) {
                const double v = plane.at(row, col, ch);
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
}

void run_sortable_gates() {
    std::printf("-- counter map: sortable() / unsortable_to_bits() --\n");

    /* Bijection: the forward map is one-to-one on all 2**32 bit patterns, so
     * the host inverse must recover every one exactly. Comparing bit patterns
     * rather than value == lets a NaN and both signed zeros participate. */
    uint32_t roundtrip_fail = 0;
    auto roundtrip_bits = [&](uint32_t bits) {
        if (unsortable_bits_cpu(sortable_bits_bits(bits)) != bits)
            ++roundtrip_fail;
    };
    roundtrip_bits(0x00000000u);            /* +0 */
    roundtrip_bits(0x80000000u);            /* -0 */
    roundtrip_bits(0x7F800000u);            /* +inf */
    roundtrip_bits(0xFF800000u);            /* -inf */
    roundtrip_bits(0x7FC00000u);            /* +NaN */
    roundtrip_bits(0xFFC00000u);            /* -NaN */
    roundtrip_bits(0x7F7FFFFFu);            /* +FLT_MAX */
    roundtrip_bits(0xFF7FFFFFu);            /* -FLT_MAX */
    roundtrip_bits(0x00000001u);            /* smallest +subnormal */
    roundtrip_bits(0x80000001u);            /* smallest -subnormal */
    for (uint32_t hi = 0; hi < 256u; ++hi)
        for (uint32_t mid = 0; mid < 256u; ++mid) {
            roundtrip_bits((hi << 23) | (mid << 15) | 1u);
            roundtrip_bits(0x80000000u | (hi << 23) | (mid << 15) | 1u);
        }
    check(roundtrip_fail == 0u,
          "unsortable(sortable(x)) == x bit-for-bit for finite floats, both "
          "signed zeros, the infinities and NaN");
    std::printf("  roundtrip failures: %u\n", roundtrip_fail);

    /* Strict order-embedding over the finite floats: sort ascending by value,
     * then every strictly increasing neighbour must strictly increase the key.
     * Equal values are only -0 and +0, handled by their own check below. */
    std::vector<float> xs;
    for (uint32_t hi = 1; hi < 255u; ++hi)   /* exponent 1..254 -> finite */
        for (uint32_t mant = 0; mant < 16u; ++mant) {
            const uint32_t mag = (hi << 23) | (mant << 19);
            float p = 0.0f, n = 0.0f;
            std::memcpy(&p, &mag, sizeof(p));
            const uint32_t nb = mag | 0x80000000u;
            std::memcpy(&n, &nb, sizeof(n));
            xs.push_back(p);
            xs.push_back(n);
        }
    xs.push_back(0.0f);
    xs.push_back(-0.0f);
    std::sort(xs.begin(), xs.end(), [](float a, float b) { return a < b; });
    uint32_t mono_fail = 0;
    for (size_t i = 1; i < xs.size(); ++i)
        if (xs[i - 1] < xs[i] &&
            !(sortable_bits_cpu(xs[i - 1]) < sortable_bits_cpu(xs[i])))
            ++mono_fail;
    check(mono_fail == 0u,
          "sortable is strictly increasing over the finite floats "
          "(x < y  =>  sortable(x) < sortable(y))");
    std::printf("  monotonicity failures: %u\n", mono_fail);

    /* Both signed zeros sit at the join, ordered -0 before +0 despite being
     * numerically equal - the documented total order of the map. */
    check(sortable_bits_cpu(-0.0f) == 0x7FFFFFFFu &&
          sortable_bits_cpu(0.0f) == 0x80000000u,
          "the signed zeros map to the adjacent keys below and above the join");

    /* The exact property whose loss caused the defect: a larger-magnitude
     * negative must carry the SMALLER key, so InterlockedMin finds the true
     * minimum. -6.140625 is the fixture's plane minimum; -250.68 was the
     * phantom the reversed map decoded from a near-zero component. */
    check(sortable_bits_cpu(-6.140625f) < sortable_bits_cpu(-0.001f),
          "a larger-magnitude negative sorts below a near-zero negative "
          "(InterlockedMin finds the true minimum)");

    /* The map extends to order the infinities and NaN: -inf below every finite,
     * +inf above every finite, a +NaN just above +inf and a -NaN just below
     * -inf. Decode counts non-finite samples and the caller refuses the frame,
     * so a real job never feeds these to the counters; the order is defined,
     * not incidental. */
    check(sortable_bits_cpu(-std::numeric_limits<float>::infinity()) <
              sortable_bits_cpu(-65504.0f) &&
          sortable_bits_cpu(65504.0f) <
              sortable_bits_cpu(std::numeric_limits<float>::infinity()),
          "the order extends: -inf below every finite, +inf above every finite");
    check(sortable_bits_cpu(std::numeric_limits<float>::quiet_NaN()) >
              sortable_bits_cpu(std::numeric_limits<float>::infinity()),
          "a positive NaN sits at the top of the total order, above +inf");
}

int gates() {
    run_sortable_gates();
    GpuHdrContract contract{};
    show_contract(contract);
    const double paper_white = kJob1Ceiling / contract.proxy_actionable_relative;
    std::printf("  paperWhite rule: ceiling %.6g scRGB / actionable %.10g ="
        " %.10g scRGB (%.1f cd/m2)\n", kJob1Ceiling,
        contract.proxy_actionable_relative, paper_white, paper_white * 80.0);

    Plan down(kFullWidth, kFullHeight, kWorkingWidth, kWorkingHeight);
    Plan up(kWorkingWidth, kWorkingHeight, kFullWidth, kFullHeight);

    Machine machine;
    if (!machine.open()) return 1;

    /* Refusals that must not allocate anything. */
    {
        GpuHdrBridge* probe = nullptr;
        expect_refusal(gpu_hdr_open(machine.device.Get(), machine.context,
            kFullWidth, kFullHeight, kWorkingWidth, kWorkingHeight,
            &down.table, &up.table, 0.5, &probe),
            "open refuses a paperWhite below the reference white");
        expect_refusal(gpu_hdr_open(machine.device.Get(), machine.context,
            kFullWidth - 4u, kFullHeight, kWorkingWidth, kWorkingHeight,
            &down.table, &up.table, paper_white, &probe),
            "open refuses an extent that is not a multiple of 8");
        expect_refusal(gpu_hdr_open(machine.device.Get(), machine.context,
            kFullWidth, kFullHeight, kWorkingWidth, kWorkingHeight,
            nullptr, &up.table, paper_white, &probe),
            "open refuses a missing tap plan");
        expect_refusal(gpu_hdr_open(machine.device.Get(), machine.context,
            kFullWidth, kFullHeight, kWorkingWidth + 4u, kWorkingHeight,
            &down.table, &up.table, paper_white, &probe),
            "open refuses a plan that does not run over the job's extents");
        check(probe == nullptr, "a refused open leaves no bridge behind");
    }

    GpuHdrBridge* bridge = nullptr;
    if (gpu_hdr_open(machine.device.Get(), machine.context, kFullWidth,
        kFullHeight, kWorkingWidth, kWorkingHeight, &down.table, &up.table,
        paper_white, &bridge) != 0 || !bridge) {
        note(std::string("gpu_hdr_open: ") + last_error());
        ++g_failures;
        machine.close();
        return 1;
    }
    std::printf("[ ok ] gpu_hdr_open on the shared adapter\n");
    check(gpu_hdr_paper_white(bridge) == paper_white,
          "the job's paperWhite is the one that was asked for");

    static const char* kPlaneName[10] = {"source y", "source uv", "master",
        "working", "proxy", "guidance", "residual", "composited", "encode y",
        "encode uv"};
    bool aligned = true;
    for (int32_t p = 0; p < 10; ++p)
        if (gpu_hdr_plane_pitch(bridge, p) % 16u != 0u) aligned = false;
    check(aligned, "every plane row is a 16-byte multiple for a pitched copy");
    for (int32_t p = 0; p < 10; ++p)
        std::printf("  plane %2d %-11s pitch %6u bytes %11llu\n", p,
            kPlaneName[p], gpu_hdr_plane_pitch(bridge, p),
            static_cast<unsigned long long>(gpu_hdr_plane_bytes(bridge, p)));

    /* --- gate 1 --- */
    std::printf("-- gate 1: P010 -> linear scRGB -> P010 --\n");
    P010Frame frame;
    frame.build(kFullWidth, kFullHeight, kJob1Samples, kJob1SampleCount);
    DeviceBuffer y_buffer, uv_buffer;
    Roundtrip round{};
    const bool roundtrip_ok = run_roundtrip(bridge, machine, frame, y_buffer,
        uv_buffer, kJob1Ceiling, kJob1Samples, kJob1SampleCount, contract,
        &round);
    if (roundtrip_ok) {
        for (uint32_t index = 0; index < round.row_count; ++index) {
            show_row(round.rows[index]);
        }
        g_summary.add("roundtrip_y_max_codes", uint64_t(round.y_max));
        g_summary.add("roundtrip_uv_max_codes", uint64_t(round.uv_max));
        std::printf("  job 1 fixture: luma max code error %u, mean %.4f over "
            "%u samples\n", round.y_max,
            double(round.y_total) / (kFullWidth * kFullHeight),
            kFullWidth * kFullHeight);
        std::printf("  chroma max code error %u, mean %.4f over %u samples\n",
            round.uv_max, double(round.uv_total) / round.uv_count,
            round.uv_count);
        /* Two claims, both derived rather than asserted.
         *
         * First, correctness: the GPU must produce the codes the doubles
         * reference produces for the same documented path. That is the statement
         * the gate exists for, and it is exact - one code of slack covers the
         * float32 series standing in for the double pow.
         *
         * Second, loss: the code error against the source is bounded by what one
         * binary16 step can do to *that sample*, corner to corner through the
         * toe. For a neutral or in-gamut sample the box is a code wide, so the
         * roundtrip is required to be lossless there; for a sample that puts a
         * 2020 primary at exactly black the box is wide, and the honest report
         * is the box plus the light it costs, not a number of codes. */
        check(round.mirror_max_codes <= 1u,
              "the GPU's roundtrip equals the exact-arithmetic reference");
        check(round.outside_bound == 0u,
              "every roundtrip error is inside the store's own quantum for "
              "that sample");
        /* Third, kept apart from both of the above: the decode's own grid. The
         * device rounds a float32 curve, the reference rounds doubles, and a
         * value that sits on the round-to-nearest-even midpoint can land on
         * either side. The float32 path is wrong by order 1e-7 of its value and
         * one binary16 step is 5e-4 of it, so at most one step is reachable; a
         * gap of more than one is a decode defect, not a boundary. */
        check(round.decode_pattern_gap <= 1u,
              "the master plane holds the binary16 the doubles decode rounds "
              "to, within one grid step");
        bool lossless_where_conditioned = true;
        for (uint32_t index = 0; index < round.row_count; ++index) {
            const RowError& e = round.rows[index];
            const uint32_t err[3] = { uint32_t(std::abs(long(e.got_y) -
                    long(e.want_y))),
                uint32_t(std::abs(long(e.got_cb) - long(e.want_cb))),
                uint32_t(std::abs(long(e.got_cr) - long(e.want_cr))) };
            if (e.bound[0] <= 1.0 && err[0] > 1u) lossless_where_conditioned = false;
            if (e.bound[1] <= 1.0 && err[1] > 1u) lossless_where_conditioned = false;
            if (e.bound[2] <= 1.0 && err[2] > 1u) lossless_where_conditioned = false;
        }
        check(lossless_where_conditioned, "a sample the half grid resolves "
              "roundtrips to the same code it started on");
        std::printf("  job 1: %u of %u samples sit where the PQ toe amplifies "
            "the grid; worst light error there %g cd/m2, reference agreement "
            "%u code(s)\n", round.amplified_rows, round.row_count,
            round.worst_light_nits, round.mirror_max_codes);
        g_summary.add("roundtrip_y_max_codes", uint64_t(round.y_max));
        g_summary.add("roundtrip_uv_max_codes", uint64_t(round.uv_max));
        g_summary.add("roundtrip_reference_max_codes",
            uint64_t(round.mirror_max_codes));
        g_summary.add("roundtrip_outside_box_components",
            uint64_t(round.outside_bound));
        g_summary.add("roundtrip_amplified_samples",
            uint64_t(round.amplified_rows));
        g_summary.add("roundtrip_decode_pattern_gap_max",
            uint64_t(round.decode_pattern_gap));
        g_summary.add("roundtrip_decode_pattern_samples",
            uint64_t(round.decode_pattern_samples));
        g_summary.add("roundtrip_worst_light_error_nits", round.worst_light_nits);
        g_summary.add("roundtrip_y_mean_codes",
            double(round.y_total) / (kFullWidth * kFullHeight));
        g_summary.add("roundtrip_uv_mean_codes",
            double(round.uv_total) / round.uv_count);
    }

    GpuHdrSourceReport source{};
    source.struct_size = sizeof(source);
    check(gpu_hdr_get_source_report(bridge, &source) == 0,
          "the source report reads back");
    std::printf("  source: pixels %u nonfinite %u negative %u above ceiling "
        "%u range [%g, %g]\n", source.pixels, source.nonfinite_samples,
        source.negative_components, source.above_ceiling_components,
        source.min_component, source.max_component);
    g_summary.add("source_nonfinite", uint64_t(source.nonfinite_samples));
    g_summary.add("source_negative", uint64_t(source.negative_components));
    g_summary.add("source_above_ceiling",
        uint64_t(source.above_ceiling_components));
    g_summary.add("source_min", source.min_component);
    g_summary.add("source_max", source.max_component);
    g_summary.add("source_pixels", uint64_t(source.pixels));
    check(source.nonfinite_samples == 0u, "no sample of the fixture is "
          "unrepresentable in the half grid");
    check(source.above_ceiling_components == 0u,
          "the fixture stays inside its declared ceiling");
    check(source.negative_components > 0u,
          "the fixture does drive a gamut conversion negative, and the bridge "
          "reports it instead of clamping it");

    /* The tags this stage has no maths for, and a ceiling that is not a number.
     * Each is refused with a reason before a single instruction runs. */
    {
        uint64_t ptr = 0;
        uint32_t pitch = 0;
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 1, 9, 18, 10, kJob1Ceiling,
            &ptr, &pitch), "decode refuses HLG");
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 2, 9, 16, 10, kJob1Ceiling,
            &ptr, &pitch), "decode refuses full-range 10-bit");
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 1, 9, 16, 9, kJob1Ceiling,
            &ptr, &pitch), "decode refuses a constant-luminance matrix");
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 1, 1, 16, 10, kJob1Ceiling,
            &ptr, &pitch), "decode refuses BT.709 primaries");
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 1, 9, 16, 10, 0.0, &ptr, &pitch),
            "decode refuses a ceiling that is not a positive finite value");
        expect_refusal(gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch,
            uv_buffer.ptr, frame.uv_pitch, 1, 9, 16, 10,
            std::numeric_limits<double>::infinity(), &ptr, &pitch),
            "decode refuses an infinite ceiling");
    }
    /* Re-decode for the stages below: the refusals above must not have left the
     * bridge holding anything but a decoded master. */
    {
        uint64_t ptr = 0;
        uint32_t pitch = 0;
        if (gpu_hdr_decode(bridge, y_buffer.ptr, frame.y_pitch, uv_buffer.ptr,
            frame.uv_pitch, 1, 9, 16, 10, kJob1Ceiling, &ptr, &pitch) != 0) {
            note(std::string("re-decode refused: ") + last_error());
            ++g_failures;
        }
    }

    /* --- gate 2 --- */
    std::printf("-- gate 2: proxy and reconstruction --\n");
    uint64_t working_ptr = 0, proxy_ptr = 0, guidance_ptr = 0;
    uint32_t working_pitch = 0, proxy_pitch = 0, guidance_pitch = 0;
    bool stage_ok = gpu_hdr_downscale(bridge, &working_ptr,
        &working_pitch) == 0;
    check(stage_ok, "the working master is produced on the device");
    stage_ok = stage_ok && gpu_hdr_proxy(bridge, &proxy_ptr, &proxy_pitch,
        &guidance_ptr, &guidance_pitch) == 0;
    check(stage_ok, "the proxy and the guidance surface are produced");

    PlaneDump master{}, working{}, proxy{}, guidance{}, residual{},
        composited{};
    auto read_fp16 = [&](PlaneDump& plane, int32_t id, uint32_t width,
                         uint32_t height) {
        plane.width = width; plane.height = height;
        plane.pitch = gpu_hdr_plane_pitch(bridge, id);
        plane.bytes.assign(size_t(plane.pitch) * height, uint8_t(0));
        return gpu_hdr_readback_scrgb(bridge, id, plane.bytes.data(),
            plane.pitch) == 0;
    };
    stage_ok = read_fp16(master, 2, kFullWidth, kFullHeight) && stage_ok;
    stage_ok = read_fp16(working, 3, kWorkingWidth, kWorkingHeight) && stage_ok;
    stage_ok = read_fp16(proxy, 4, kWorkingWidth, kWorkingHeight) && stage_ok;
    guidance.pitch = gpu_hdr_plane_pitch(bridge, 5);
    guidance.width = kWorkingWidth;
    guidance.height = kWorkingHeight;
    guidance.bytes.assign(size_t(guidance.pitch) * kWorkingHeight, uint8_t(0));
    /* The guidance surface is the only 8-bit one and it is not part of the
     * colour chain, so it is dumped as raw bytes. */
    stage_ok = stage_ok && gpu_hdr_readback_scrgb(bridge, 5,
        guidance.bytes.data(), guidance.pitch) == 0;
    check(stage_ok, "the planes the gates measure are readable back");
    if (!stage_ok) g_failures += 1;

    /* GPU-reported source extrema against the same master read back and measured
     * on the CPU. The decode orders the pre-store float with the fixed sortable
     * map; the plane holds the RNE half store, so the GPU value can sit at most
     * one half step beyond the measured half. The bound below is two quanta at
     * the observed magnitude, far tighter than the ~244 the phantom missed by.
     * This proves the -250.68 source_min was a counter defect, not a pixel, and
     * is deliberately not a colour-gate claim. */
    if (stage_ok) {
        GpuHdrSourceReport fresh{};
        fresh.struct_size = sizeof(fresh);
        const bool report_ok = gpu_hdr_get_source_report(bridge, &fresh) == 0;
        check(report_ok, "the source report reads back for the extrema check");
        double src_lo = 0.0, src_hi = 0.0;
        measure_plane_extrema(master, src_lo, src_hi);
        const double lo_gap = std::fabs(fresh.min_component - src_lo);
        const double hi_gap = std::fabs(fresh.max_component - src_hi);
        std::printf("  counter check: source GPU [%g, %g] vs measured FP16 "
            "[%g, %g] (gaps %.6g / %.6g)\n", fresh.min_component,
            fresh.max_component, src_lo, src_hi, lo_gap, hi_gap);
        g_summary.add("counterfix_source_gpu_min", fresh.min_component);
        g_summary.add("counterfix_source_gpu_max", fresh.max_component);
        g_summary.add("counterfix_source_measured_min", src_lo);
        g_summary.add("counterfix_source_measured_max", src_hi);
        g_summary.add("counterfix_source_min_abs_gap", lo_gap);
        g_summary.add("counterfix_source_max_abs_gap", hi_gap);
        check(lo_gap <= 0.0078125 && hi_gap <= 0.0078125,
              "the GPU-reported source min and max agree with the independently "
              "measured FP16 master within the half grid");
        check(fresh.min_component > -100.0 && fresh.min_component < 0.0,
              "the reported source minimum is a real in-range negative, not the "
              "-250.68 phantom (telemetry defect resolved)");
    }

    GpuHdrProxyReport proxy_report{};
    proxy_report.struct_size = sizeof(proxy_report);
    check(gpu_hdr_get_proxy_report(bridge, &proxy_report) == 0,
          "the proxy report reads back");
    std::printf("  proxy: pixels %u negative %u pinned %u dead %u above "
        "actionable %u above saturation %u codes [%g, %g]\n",
        proxy_report.pixels, proxy_report.negative_components,
        proxy_report.pinned_components, proxy_report.dead_zone_components,
        proxy_report.above_actionable_components,
        proxy_report.above_saturation_components, proxy_report.min_code,
        proxy_report.max_code);
    g_summary.add("proxy_negative", uint64_t(proxy_report.negative_components));
    g_summary.add("proxy_pinned", uint64_t(proxy_report.pinned_components));
    g_summary.add("proxy_dead_zone",
        uint64_t(proxy_report.dead_zone_components));
    g_summary.add("proxy_above_actionable",
        uint64_t(proxy_report.above_actionable_components));
    g_summary.add("proxy_above_saturation",
        uint64_t(proxy_report.above_saturation_components));
    g_summary.add("proxy_min_code", proxy_report.min_code);
    g_summary.add("proxy_max_code", proxy_report.max_code);
    g_summary.add("proxy_pixels", uint64_t(proxy_report.pixels));
    /* Job 1 selected its paperWhite by the coverage rule, so by construction
     * nothing of its declared domain may be outside the band the proxy can
     * measure. If these counts are not zero the rule or the enumeration is
     * wrong, and the correction would be silently passing those channels
     * through. */
    /* What the coverage rule promises is about the domain above paper white:
     * nothing there may land outside the band the proxy can measure. The toe is
     * a separate documented limit - a channel at exactly zero cannot answer a
     * multiplicative probe at all, so its measured slope is zero and the
     * reconstruction refuses it. Both halves are measured, not assumed. */
    uint64_t toe_channels = 0;
    for (uint32_t row = 0; row < kWorkingHeight; ++row)
        for (uint32_t column = 0; column < kWorkingWidth; ++column)
            for (uint32_t channel = 0; channel < 3u; ++channel)
                if (working.at(row, column, channel) == 0.0) ++toe_channels;
    std::printf("  band: %llu of %u working channels sit at exactly zero\n",
        static_cast<unsigned long long>(toe_channels),
        kWorkingWidth * kWorkingHeight * 3u);
    check(proxy_report.pinned_components == 0u &&
          proxy_report.above_actionable_components == 0u &&
          proxy_report.above_saturation_components == 0u,
          "nothing above paper white lands outside the measurable band");
    check(uint64_t(proxy_report.dead_zone_components) == toe_channels,
          "every dead channel is a toe channel and no other channel is dead");
    g_summary.add("working_zero_channels", toe_channels);

    /* The identity case first: a Feature 18 output equal to the proxy must
     * return the pristine master, exactly. */
    NeuralTexture neural;
    std::vector<float> identity(size_t(kWorkingWidth) * kWorkingHeight * 4u);
    for (uint32_t row = 0; row < kWorkingHeight; ++row)
        for (uint32_t column = 0; column < kWorkingWidth; ++column)
            for (uint32_t channel = 0; channel < 4u; ++channel)
                identity[(size_t(row) * kWorkingWidth + column) * 4u + channel] =
                    float(proxy.at(row, column, channel));
    if (!neural.build(machine.device.Get(), kWorkingWidth, kWorkingHeight,
        identity)) {
        char reason[128];
        std::snprintf(reason, sizeof(reason),
            "cannot build the identity Feature 18 output at %s (hr 0x%08lx)",
            neural.last_step, unsigned long(neural.last_hr));
        note(reason);
        ++g_failures;
    } else {
        uint64_t composited_ptr = 0;
        uint32_t composited_pitch = 0;
        const int32_t rc = gpu_hdr_reconstruct(bridge, neural.texture.Get(),
            neural.fence.Get(), neural.fence_value, 1.0, &composited_ptr,
            &composited_pitch);
        check(rc == 0, "the identity reconstruction runs");
        if (rc != 0) note(std::string("identity reconstruct: ") + last_error());
        read_fp16(composited, 7, kFullWidth, kFullHeight);
        read_fp16(residual, 6, kWorkingWidth, kWorkingHeight);
        uint64_t worst = 0;
        uint64_t differing = 0;
        for (uint32_t row = 0; row < kFullHeight; ++row) {
            for (uint32_t column = 0; column < kFullWidth; ++column) {
                for (uint32_t channel = 0; channel < 3u; ++channel) {
                    const double want = master.at(row, column, channel);
                    const double got = composited.at(row, column, channel);
                    if (to_half(want) != to_half(got)) ++differing;
                    const uint64_t a = uint64_t(std::fabs(want) * 1e9);
                    const uint64_t b = uint64_t(std::fabs(got) * 1e9);
                    worst = std::max(worst, a > b ? a - b : b - a);
                }
            }
        }
        std::printf("  identity: %llu differing samples, worst absolute "
            "difference %.9g scRGB\n",
            static_cast<unsigned long long>(differing), double(worst) / 1e9);
        check(differing == 0u, "an identity output returns the master "
            "bit-exactly");
        g_summary.add("identity_differing", differing);

        /* strength 0 must also return the master exactly. */
        if (gpu_hdr_reconstruct(bridge, neural.texture.Get(),
                neural.fence.Get(), neural.fence_value, 0.0, &composited_ptr,
                &composited_pitch) == 0) {
            read_fp16(composited, 7, kFullWidth, kFullHeight);
            uint64_t off = 0;
            for (uint32_t row = 0; row < kFullHeight; ++row)
                for (uint32_t column = 0; column < kFullWidth; ++column)
                    for (uint32_t channel = 0; channel < 3u; ++channel)
                        if (to_half(master.at(row, column, channel)) !=
                            to_half(composited.at(row, column, channel)))
                            ++off;
            std::printf("  strength 0: %llu differing samples\n",
                static_cast<unsigned long long>(off));
            check(off == 0u, "zero residual strength returns the master");
            g_summary.add("strength_zero_differing", off);
        } else {
            note(std::string("strength 0: ") + last_error());
            ++g_failures;
        }

        /* A real correction: +0.02 of code on the red channel of every working
         * sample, which the reference can reproduce exactly. */
        std::vector<float> nudged(identity);
        for (size_t i = 0; i < nudged.size(); i += 4u)
            nudged[i] = std::min(1.0f, nudged[i] + 0.02f);
        NeuralTexture moved;
        if (!moved.build(machine.device.Get(), kWorkingWidth, kWorkingHeight,
            nudged)) {
            note("cannot build the corrected Feature 18 output");
            ++g_failures;
        } else if (gpu_hdr_reconstruct(bridge, moved.texture.Get(),
                moved.fence.Get(), moved.fence_value, 1.0, &composited_ptr,
                &composited_pitch) != 0) {
            note(std::string("corrected reconstruct: ") + last_error());
            ++g_failures;
        } else {
            read_fp16(composited, 7, kFullWidth, kFullHeight);
            read_fp16(residual, 6, kWorkingWidth, kWorkingHeight);
            GpuHdrReconstructReport rec{};
            rec.struct_size = sizeof(rec);
            check(gpu_hdr_get_reconstruct_report(bridge, &rec) == 0,
                  "the reconstruction report reads back");
            std::printf("  corrected: applied %u gain cap %u pinned %u dead %u "
                "negative %u nonfinite %u max stops %.6f max residual %.6g\n",
                rec.applied_components, rec.gain_cap_components,
                rec.rejected_pinned_components, rec.rejected_dead_components,
                rec.rejected_negative_components, rec.nonfinite_components,
                rec.max_abs_stops, rec.max_abs_residual_scrgb);
            std::printf("  composited: outside declared domain %u nonfinite %u "
                "range [%g, %g]\n", rec.composite_out_of_domain_components,
                rec.composite_nonfinite_components, rec.composite_min_component,
                rec.composite_max_component);
            const uint64_t channels =
                uint64_t(kWorkingWidth) * kWorkingHeight * 3u;
            check(uint64_t(rec.applied_components) +
                  uint64_t(rec.gain_cap_components) +
                  uint64_t(rec.rejected_pinned_components) +
                  uint64_t(rec.rejected_dead_components) +
                  uint64_t(rec.rejected_negative_components) +
                  uint64_t(rec.nonfinite_components) == channels,
                  "every channel lands in exactly one disposition");
            g_summary.add("reconstruct_applied",
                uint64_t(rec.applied_components));
            g_summary.add("reconstruct_gain_cap",
                uint64_t(rec.gain_cap_components));
            g_summary.add("reconstruct_rejected_pinned",
                uint64_t(rec.rejected_pinned_components));
            g_summary.add("reconstruct_rejected_dead",
                uint64_t(rec.rejected_dead_components));
            g_summary.add("reconstruct_rejected_negative",
                uint64_t(rec.rejected_negative_components));
            g_summary.add("reconstruct_nonfinite",
                uint64_t(rec.nonfinite_components));
            g_summary.add("reconstruct_max_abs_stops", rec.max_abs_stops);
            g_summary.add("reconstruct_max_abs_residual",
                rec.max_abs_residual_scrgb);
            g_summary.add("composite_out_of_domain",
                uint64_t(rec.composite_out_of_domain_components));
            g_summary.add("composite_nonfinite",
                uint64_t(rec.composite_nonfinite_components));
            g_summary.add("composite_min_component", rec.composite_min_component);
            g_summary.add("composite_max_component", rec.composite_max_component);

            /* GPU composite extrema against the composited plane measured on the
             * CPU: the same reasoning as the source check and the same defect -
             * the old -250.62 composite_min was a phantom of the reversed map.
             * Again deliberately not a colour-gate claim. */
            if (!composited.bytes.empty()) {
                double comp_lo = 0.0, comp_hi = 0.0;
                measure_plane_extrema(composited, comp_lo, comp_hi);
                const double comp_lo_gap =
                    std::fabs(rec.composite_min_component - comp_lo);
                const double comp_hi_gap =
                    std::fabs(rec.composite_max_component - comp_hi);
                std::printf("  counter check: composite GPU [%g, %g] vs "
                    "measured FP16 [%g, %g] (gaps %.6g / %.6g)\n",
                    rec.composite_min_component, rec.composite_max_component,
                    comp_lo, comp_hi, comp_lo_gap, comp_hi_gap);
                g_summary.add("counterfix_composite_gpu_min",
                    rec.composite_min_component);
                g_summary.add("counterfix_composite_gpu_max",
                    rec.composite_max_component);
                g_summary.add("counterfix_composite_measured_min", comp_lo);
                g_summary.add("counterfix_composite_measured_max", comp_hi);
                g_summary.add("counterfix_composite_min_abs_gap", comp_lo_gap);
                g_summary.add("counterfix_composite_max_abs_gap", comp_hi_gap);
                check(comp_lo_gap <= 0.015625 && comp_hi_gap <= 0.015625,
                      "the GPU-reported composite min and max agree with the "
                      "measured FP16 composite within the half grid");
                check(rec.composite_min_component > -100.0,
                      "the composite minimum is a real in-range value, not the "
                      "-250.62 phantom");
            }
        }
    }

    /* Encode the composited result and hand the planes to the Python side. */
    {
        uint64_t encode_y = 0, encode_uv = 0;
        uint32_t encode_y_pitch = 0, encode_uv_pitch = 0;
        check(gpu_hdr_encode(bridge, 125.0, &encode_y, &encode_y_pitch,
            &encode_uv, &encode_uv_pitch) == 0,
            "the corrected result encodes to P010");
        GpuHdrEncodeReport encode{};
        encode.struct_size = sizeof(encode);
        check(gpu_hdr_get_encode_report(bridge, &encode) == 0,
              "the encode report reads back");
        std::printf("  encode: clipped high %u low %u out of gamut %u "
            "nonfinite %u max linear %.6g codes y [%g, %g] uv [%g, %g]\n",
            encode.clamped_high_components, encode.clamped_low_components,
            encode.out_of_gamut_components, encode.nonfinite_components,
            encode.max_linear_component, encode.min_code_y, encode.max_code_y,
            encode.min_code_uv, encode.max_code_uv);
        g_summary.add("encode_clamped_high",
            uint64_t(encode.clamped_high_components));
        g_summary.add("encode_clamped_low",
            uint64_t(encode.clamped_low_components));
        g_summary.add("encode_out_of_gamut",
            uint64_t(encode.out_of_gamut_components));
        g_summary.add("encode_nonfinite",
            uint64_t(encode.nonfinite_components));
        g_summary.add("encode_max_linear", encode.max_linear_component);
        g_summary.add("encode_min_code_y", encode.min_code_y);
        g_summary.add("encode_max_code_y", encode.max_code_y);
        g_summary.add("encode_min_code_uv", encode.min_code_uv);
        g_summary.add("encode_max_code_uv", encode.max_code_uv);
    }

    GpuHdrTransferCounters counters{};
    counters.struct_size = sizeof(counters);
    check(gpu_hdr_get_transfer_counters(bridge, &counters) == 0,
          "the transfer counters read back");
    std::printf("  counters: process %llu source copy %llu source write %llu "
        "working write %llu guidance write %llu composited write %llu encode "
        "write %llu readback %llu\n",
        static_cast<unsigned long long>(counters.process_count),
        static_cast<unsigned long long>(counters.source_device_copy_bytes),
        static_cast<unsigned long long>(counters.source_device_write_bytes),
        static_cast<unsigned long long>(counters.working_device_write_bytes),
        static_cast<unsigned long long>(counters.guidance_device_write_bytes),
        static_cast<unsigned long long>(counters.composited_device_write_bytes),
        static_cast<unsigned long long>(counters.encode_device_write_bytes),
        static_cast<unsigned long long>(counters.report_readback_bytes));
    check(counters.process_count > 0 && counters.source_device_copy_bytes > 0 &&
          counters.source_device_write_bytes > 0 &&
          counters.working_device_write_bytes > 0 &&
          counters.guidance_device_write_bytes > 0 &&
          counters.composited_device_write_bytes > 0 &&
          counters.encode_device_write_bytes > 0,
          "every device-to-device stage of the chain moved pixels");

    g_summary.add("full_width", uint64_t(kFullWidth));
    g_summary.add("full_height", uint64_t(kFullHeight));
    g_summary.add("working_width", uint64_t(kWorkingWidth));
    g_summary.add("working_height", uint64_t(kWorkingHeight));
    g_summary.add("paper_white", paper_white);
    g_summary.add("support_ceiling", kJob1Ceiling);
    g_summary.add("saturation_relative", contract.proxy_saturation_relative);
    g_summary.add("actionable_relative", contract.proxy_actionable_relative);
    g_summary.add("process_count", counters.process_count);
    g_summary.add("device_source_copy_bytes", counters.source_device_copy_bytes);
    g_summary.add("device_source_write_bytes", counters.source_device_write_bytes);
    g_summary.add("device_working_write_bytes", counters.working_device_write_bytes);
    g_summary.add("device_composited_write_bytes",
        counters.composited_device_write_bytes);
    g_summary.add("device_encode_write_bytes", counters.encode_device_write_bytes);
    g_summary.add("report_readback_bytes", counters.report_readback_bytes);
    g_summary.dump(master, "master");
    g_summary.dump(working, "working");
    g_summary.dump(proxy, "proxy");
    g_summary.dump(residual, "residual");
    g_summary.dump(composited, "composited");
    write_file("guidance", guidance.bytes.data(), guidance.bytes.size());
    g_summary.add("guidance_pitch", uint64_t(guidance.pitch));
    g_summary.write("gate_summary.txt");

    /* --- job 2: the legal code range end to end, pinned by design --------- */
    /* Two things are measured here and nowhere else. The roundtrip runs over
     * every legal code, including the BT.2020 primaries whose converted
     * components reach 207 scRGB, with no proxy band in the way. And the proxy
     * runs at the paperWhite stage A audited, whose band top is 5.637 scRGB, so
     * the bright half of the sweep *must* arrive pinned: the identity gate then
     * proves the reconstruction passes those channels through exactly rather than
     * inverting a saturated code into an invented magnitude. */
    std::printf("-- job 2: full-code sweep at paperWhite %.1f --\n",
        kSweepPaperWhite);
    {
        GpuHdrBridge* sweep = nullptr;
        if (gpu_hdr_open(machine.device.Get(), machine.context, kFullWidth,
            kFullHeight, kWorkingWidth, kWorkingHeight, &down.table,
            &up.table, kSweepPaperWhite, &sweep) != 0 || !sweep) {
            note(std::string("sweep gpu_hdr_open: ") + last_error());
            ++g_failures;
        } else {
            P010Frame wide;
            wide.build(kFullWidth, kFullHeight, kSweepSamples,
                kSweepSampleCount);
            DeviceBuffer sweep_y, sweep_uv;
            Roundtrip sweep_round{};
            const bool ok = run_roundtrip(sweep, machine, wide, sweep_y,
                sweep_uv, kSweepCeiling, kSweepSamples, kSweepSampleCount,
                contract, &sweep_round);
            check(ok, "the full-code sweep roundtrips on the device");
            if (ok) {
                for (uint32_t index = 0; index < sweep_round.row_count; ++index) {
                    show_row(sweep_round.rows[index]);
                }
                std::printf("  sweep: luma max %u mean %.4f over %u, chroma "
                    "max %u mean %.4f over %u\n", sweep_round.y_max,
                    double(sweep_round.y_total) /
                        (kFullWidth * kFullHeight),
                    kFullWidth * kFullHeight, sweep_round.uv_max,
                    double(sweep_round.uv_total) / sweep_round.uv_count,
                    sweep_round.uv_count);
                /* The same two claims as job 1, over the whole legal code range
                 * and the three BT.2020 primaries: the GPU equals the reference,
                 * and what it loses is what the half grid can cost that sample.
                 * The primaries are the samples whose 2020 partners sit at
                 * exactly black, so their boxes are wide by construction; every
                 * neutral row still has to come back on its own code. */
                check(sweep_round.mirror_max_codes <= 1u,
                      "the sweep's roundtrip equals the exact-arithmetic "
                      "reference over the whole legal range");
                check(sweep_round.outside_bound == 0u,
                      "every sweep error is inside the store's own quantum for "
                      "that sample");
                check(sweep_round.decode_pattern_gap <= 1u,
                      "the sweep's master plane holds the binary16 the doubles "
                      "decode rounds to, within one grid step");
                check(sweep_round.worst_light_nits < 200.0,
                      "the sweep's roundtrip stays inside a stated light budget "
                      "of 200 cd/m2, which is where the toe amplification lands");
                g_summary.add("sweep_roundtrip_y_max_codes",
                    uint64_t(sweep_round.y_max));
                g_summary.add("sweep_roundtrip_uv_max_codes",
                    uint64_t(sweep_round.uv_max));
                g_summary.add("sweep_roundtrip_reference_max_codes",
                    uint64_t(sweep_round.mirror_max_codes));
                g_summary.add("sweep_roundtrip_outside_box_components",
                    uint64_t(sweep_round.outside_bound));
                g_summary.add("sweep_roundtrip_amplified_samples",
                    uint64_t(sweep_round.amplified_rows));
                g_summary.add("sweep_roundtrip_decode_pattern_gap_max",
                    uint64_t(sweep_round.decode_pattern_gap));
                g_summary.add("sweep_roundtrip_decode_pattern_samples",
                    uint64_t(sweep_round.decode_pattern_samples));
                g_summary.add("sweep_roundtrip_worst_light_error_nits",
                    sweep_round.worst_light_nits);
                g_summary.add("sweep_roundtrip_y_mean_codes",
                    double(sweep_round.y_total) / (kFullWidth * kFullHeight));
                g_summary.add("sweep_roundtrip_uv_mean_codes",
                    double(sweep_round.uv_total) / sweep_round.uv_count);
            }
            uint64_t sweep_ptr = 0;
            uint32_t sweep_pitch = 0;
            bool stage = gpu_hdr_downscale(sweep, &sweep_ptr, &sweep_pitch) == 0;
            stage = stage && gpu_hdr_proxy(sweep, &sweep_ptr, &sweep_pitch,
                &sweep_ptr, &sweep_pitch) == 0;
            check(stage, "the sweep reaches the proxy");
            GpuHdrProxyReport sweep_proxy{};
            sweep_proxy.struct_size = sizeof(sweep_proxy);
            check(gpu_hdr_get_proxy_report(sweep, &sweep_proxy) == 0 &&
                  sweep_proxy.pixels == kWorkingWidth * kWorkingHeight,
                  "the sweep's proxy report reads back");
            std::printf("  sweep proxy: negative %u pinned %u dead %u above "
                "actionable %u above saturation %u codes [%g, %g]\n",
                sweep_proxy.negative_components, sweep_proxy.pinned_components,
                sweep_proxy.dead_zone_components,
                sweep_proxy.above_actionable_components,
                sweep_proxy.above_saturation_components, sweep_proxy.min_code,
                sweep_proxy.max_code);
            check(sweep_proxy.pinned_components > 0u &&
                  sweep_proxy.above_actionable_components > 0u,
                  "the sweep really does run past this job's band, so the "
                  "preservation path is being measured");
            PlaneDump sweep_master{}, sweep_proxy_plane{}, sweep_kept{};
            PlaneDump sweep_working{};
            /* The job-1 reader closes over job 1's bridge; the sweep is a second
             * bridge and must not be read through it. */
            auto sweep_read = [&](PlaneDump& plane, int32_t id, uint32_t w,
                                  uint32_t h) {
                plane.width = w; plane.height = h;
                plane.pitch = gpu_hdr_plane_pitch(sweep, id);
                plane.bytes.assign(size_t(plane.pitch) * h, uint8_t(0));
                return gpu_hdr_readback_scrgb(sweep, id, plane.bytes.data(),
                    plane.pitch) == 0;
            };
            stage = sweep_read(sweep_master, 2, kFullWidth, kFullHeight) &&
                sweep_read(sweep_proxy_plane, 4, kWorkingWidth,
                    kWorkingHeight) && stage;
            sweep_read(sweep_working, 3, kWorkingWidth, kWorkingHeight);
            g_summary.dump(sweep_master, "sweep_master");
            g_summary.dump(sweep_working, "sweep_working");
            g_summary.dump(sweep_proxy_plane, "sweep_proxy");
            std::vector<float> sweep_identity(
                size_t(kWorkingWidth) * kWorkingHeight * 4u);
            for (uint32_t row = 0; row < kWorkingHeight; ++row)
                for (uint32_t column = 0; column < kWorkingWidth; ++column)
                    for (uint32_t channel = 0; channel < 4u; ++channel)
                        sweep_identity[(size_t(row) * kWorkingWidth +
                            column) * 4u + channel] = float(
                            sweep_proxy_plane.at(row, column, channel));
            NeuralTexture sweep_neural;
            const bool built = sweep_neural.build(machine.device.Get(),
                kWorkingWidth, kWorkingHeight, sweep_identity);
            if (!built) {
                char reason[128];
                std::snprintf(reason, sizeof(reason),
                    "sweep neural input refused at %s (hr 0x%08lx)",
                    sweep_neural.last_step,
                    unsigned long(sweep_neural.last_hr));
                note(reason);
            }
            stage = stage && built;
            check(stage, "the sweep's identity Feature 18 output is built");
            if (stage) {
                uint64_t out_ptr = 0;
                uint32_t out_pitch = 0;
                const int32_t rc = gpu_hdr_reconstruct(sweep,
                    sweep_neural.texture.Get(), sweep_neural.fence.Get(),
                    sweep_neural.fence_value, 1.0, &out_ptr, &out_pitch);
                check(rc == 0, "the sweep's identity reconstruction runs");
                if (rc != 0) note(std::string("sweep identity: ") +
                    last_error());
                sweep_read(sweep_kept, 7, kFullWidth, kFullHeight);
                uint64_t differing = 0, pinned_kept = 0;
                for (uint32_t row = 0; row < kFullHeight; ++row)
                    for (uint32_t column = 0; column < kFullWidth; ++column)
                        for (uint32_t channel = 0; channel < 3u; ++channel) {
                            const uint16_t want = to_half(
                                sweep_master.at(row, column, channel));
                            if (want != to_half(
                                    sweep_kept.at(row, column, channel)))
                                ++differing;
                            /* A pinned channel is one the proxy could not have
                             * carried a magnitude through, so it must be exactly
                             * the master no matter what the network said. */
                            if (sweep_proxy_plane.at(row / 2u, column / 2u,
                                    channel) == 1.0)
                                ++pinned_kept;
                        }
                std::printf("  sweep identity: %llu differing of %u, %llu "
                    "channels sit at the ceiling\n",
                    static_cast<unsigned long long>(differing),
                    kFullWidth * kFullHeight * 3u,
                    static_cast<unsigned long long>(pinned_kept));
                check(differing == 0u,
                      "an identity output returns the master bit-exactly even "
                      "where the proxy is pinned");
                g_summary.add("sweep_identity_differing", differing);
                g_summary.add("sweep_proxy_pinned_components",
                    uint64_t(sweep_proxy.pinned_components));
                g_summary.add("sweep_proxy_dead_components",
                    uint64_t(sweep_proxy.dead_zone_components));
            }
            /* The same content against job 1's declared domain: the ceiling is a
             * claim, and the measurement is what says whether the pixels obey
             * it. The count must be non-zero and the frame must still be a
             * faithful master, because nothing here tone-maps behind the caller's
             * back - the caller aborts on this report. */
            {
                uint64_t ptr = 0;
                uint32_t pitch = 0;
                const int32_t rc = gpu_hdr_decode(sweep, sweep_y.ptr,
                    wide.y_pitch, sweep_uv.ptr, wide.uv_pitch, 1, 9, 16, 10,
                    kJob1Ceiling, &ptr, &pitch);
                check(rc == 0, "a narrow declared ceiling decodes rather than "
                      "silently rescaling");
                GpuHdrSourceReport claim{};
                claim.struct_size = sizeof(claim);
                check(gpu_hdr_get_source_report(sweep, &claim) == 0 &&
                      claim.above_ceiling_components > 0u,
                      "content above the declared ceiling is counted, not "
                      "clamped away");
                std::printf("  over-claim: %u components above %g scRGB, "
                    "peak %g\n", claim.above_ceiling_components, kJob1Ceiling,
                    claim.max_component);
                g_summary.add("sweep_above_ceiling_components",
                    uint64_t(claim.above_ceiling_components));
                g_summary.add("sweep_peak_component", claim.max_component);
            }
            sweep_y.free(machine.cuda);
            sweep_uv.free(machine.cuda);
            gpu_hdr_close(sweep);
        }
    }

    y_buffer.free(machine.cuda);
    uv_buffer.free(machine.cuda);
    gpu_hdr_close(bridge);
    machine.close();
    /* Job 2's keys are added after the job-1 write, and the summary is one
     * accumulating buffer: without this the sweep's roundtrip numbers exist only
     * on the console and are missing from the evidence file. */
    g_summary.write("gate_summary.txt");
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    /* Unbuffered: this harness is worthless if a crash swallows the line that
     * says where it stopped. */
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* dir = argc > 1 ? argv[1] : "../bench_out/hdr_gates";
    std::filesystem::create_directories(dir);
    g_dump_dir = std::filesystem::absolute(dir);
    std::printf("== HDR scRGB bridge gates -> %s ==\n",
        g_dump_dir.string().c_str());
    gates();
    std::printf("== %d failure(s) ==\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
