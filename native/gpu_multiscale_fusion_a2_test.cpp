/* Stage A2 typed-HDR fusion gate: one small synthetic frame, the real entries,
 * and the numbers the receipt quotes.
 *
 * Both bridges are loaded by path, so the identical harness can drive the build
 * under test and an archived older build side by side: the legacy SDR control is
 * the same program against both binaries, and a missing typed group is a
 * capability answer rather than a load failure.
 *
 * What this measures, and what the Python driver judges from the dumps:
 *   typed   the scRGB solve end to end - the fused FP16 texture, its format, its
 *           extent and the state it is left in, plus the solver's own report and
 *           counters; then the fused texture handed to a real gpu_hdr_set_composite
 *           behind the solver's real producer fence, and read out of that bridge's
 *           composited plane; then the A1 observation-ordering semantics on live
 *           bridge state; then the counted out-of-range fallback and the three
 *           failure gates.
 *   sdr     the byte-RGBA8 control on the entry that has always been there.
 *   order   the A1 observation-ordering semantics alone. An archived build of the
 *           HDR bridge carries no multiscale group at all, so this mode is only
 *           meaningful against a bridge that has it; what it asserts there is the
 *           pair of refusals that shows the guard is the proxy's generation and
 *           not the ordering of anything else.
 *
 * Every readback here is explicit correctness instrumentation on synthetic data,
 * never the production chain: this harness copies the fused texture to the host on
 * its own queue, after waiting the producer fence the solver returned. The
 * production entries download no picture, and nothing in the bridges changed to
 * let this one out. */
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "include/cuda.h"
#include "gpu_hdr_scrgb_bridge.h"
#include "gpu_multiscale_fusion_bridge.h"

namespace {

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "[ ok ]" : "[FAIL]", what);
    if (!ok) ++g_failures;
}

std::string g_notes;
void note(const std::string& line) {
    std::printf("       %s\n", line.c_str());
    g_notes += line + "\n";
}

/* --- Opt-in validation diagnostics ---------------------------------------
 *
 * The gates measure on the release runtime; nothing about a run's numbers depends
 * on this, and no gate turns it on. It exists because a failing Close on a
 * command list reports an HRESULT and not the reason, and A2 hands textures to
 * two bridges that never met one another before. Asked for on the command line,
 * the layer records the validator's own sentences, which are printed after a
 * call that already failed and then dropped: an old message is never blamed for
 * a later failure. */
Microsoft::WRL::ComPtr<ID3D12InfoQueue> g_info_queue;

void drain_validation_notes() {
    if (!g_info_queue) return;
    const UINT64 count =
        g_info_queue->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        if (FAILED(g_info_queue->GetMessage(i, nullptr, &size)) || !size)
            continue;
        std::vector<uint8_t> storage(size);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if (SUCCEEDED(g_info_queue->GetMessage(i, message, &size)) &&
            message->pDescription)
            note("validator: " + std::string(message->pDescription));
    }
    g_info_queue->ClearStoredMessages();
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
        const bool ready = init && device_get_count && device_get &&
            device_luid && ctx_create && ctx_destroy && mem_alloc && memcpy_htod;
        if (!ready) note("the CUDA driver entry points are missing");
        return ready;
    }
};

struct DeviceBuffer {
    CUdeviceptr ptr = 0;
    size_t bytes = 0;
    uint32_t pitch = 0;
    uint32_t width = 0, height = 0;

    bool upload(Cuda& cuda, const void* src, size_t size, uint32_t row_pitch,
                uint32_t row_width, uint32_t rows) {
        bytes = size;
        pitch = row_pitch;
        width = row_width;
        height = rows;
        if (cuda.mem_alloc(&ptr, size) != CUDA_SUCCESS) return false;
        return cuda.memcpy_htod(ptr, src, size) == CUDA_SUCCESS;
    }
    void free(Cuda& cuda) {
        if (ptr) cuda.mem_free(ptr);
        ptr = 0;
    }
};

struct Machine {
    Microsoft::WRL::ComPtr<ID3D12Device> device;
    Cuda cuda;
    CUcontext context = nullptr;

    bool open(bool want_debug) {
        Microsoft::WRL::ComPtr<ID3D12Debug> debug;
        if (want_debug &&
            SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
        } else if (want_debug) {
            note("the D3D12 debug layer is not installed on this machine");
        }
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device)))) {
            note("no D3D12 device");
            return false;
        }
        if (want_debug) {
            /* The route this SDK offers: the validated device answers a query for
             * the info queue once the layer is enabled. No SetDebugInterface in
             * these headers, so a failed query is reported rather than guessed at. */
            if (FAILED(device.As(&g_info_queue)))
                note("the validator's messages cannot be reached on this SDK");
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

/* --- The bridges, loaded by path ----------------------------------------- */

/* The entries are reached through GetProcAddress rather than an import library
 * so one harness can drive any build of either bridge. The declarations in the
 * two headers are what makes these signatures checked; nothing here restates a
 * struct. */
template <typename T>
T symbol(HMODULE lib, const char* name) {
    return reinterpret_cast<T>(GetProcAddress(lib, name));
}

struct Fusion {
    HMODULE lib = nullptr;
    std::string path;
    uint32_t (*abi)() = nullptr;
    uint32_t (*hdr_abi)() = nullptr;
    int32_t (*open_sdr)(void*, const GpuFusionOptions*, GpuFusionBridge**) =
        nullptr;
    int32_t (*process_sdr)(GpuFusionBridge*, void*, uint64_t, size_t,
        const GpuFusionObservation*, uint32_t, void**,
        GpuFusionSolveReport*) = nullptr;
    int32_t (*counters_sdr)(GpuFusionBridge*, GpuFusionTransferCounters*) =
        nullptr;
    int32_t (*open_hdr)(void*, const GpuFusionOptions*,
        const GpuFusionHdrOptions*, GpuFusionBridge**) = nullptr;
    int32_t (*process_hdr)(GpuFusionBridge*, void*, uint64_t, size_t,
        const GpuFusionHdrObservation*, uint32_t, void**, void**, uint64_t*,
        GpuFusionHdrReport*) = nullptr;
    int32_t (*counters_hdr)(GpuFusionBridge*, GpuFusionHdrCounters*) = nullptr;
    void (*close)(GpuFusionBridge*) = nullptr;
    const char* (*last_error)() = nullptr;

    bool load(const std::string& file) {
        path = file;
        lib = LoadLibraryA(file.c_str());
        if (!lib) {
            note("cannot load " + file);
            return false;
        }
        abi = symbol<decltype(abi)>(lib, "gpu_fusion_abi_version");
        open_sdr = symbol<decltype(open_sdr)>(lib, "gpu_fusion_open");
        process_sdr = symbol<decltype(process_sdr)>(lib, "gpu_fusion_process");
        counters_sdr = symbol<decltype(counters_sdr)>(lib,
            "gpu_fusion_get_transfer_counters");
        close = symbol<decltype(close)>(lib, "gpu_fusion_close");
        last_error = symbol<decltype(last_error)>(lib, "gpu_fusion_last_error");
        hdr_abi = symbol<decltype(hdr_abi)>(lib, "gpu_fusion_hdr_abi_version");
        open_hdr = symbol<decltype(open_hdr)>(lib, "gpu_fusion_open_hdr");
        process_hdr = symbol<decltype(process_hdr)>(lib,
            "gpu_fusion_process_hdr");
        counters_hdr = symbol<decltype(counters_hdr)>(lib,
            "gpu_fusion_get_hdr_counters");
        if (!abi || !open_sdr || !process_sdr || !close || !last_error) {
            note(file + " is not a multiscale fusion bridge");
            return false;
        }
        return true;
    }
    bool typed() const {
        return hdr_abi && open_hdr && process_hdr && counters_hdr;
    }
    std::string error() const {
        const char* text = last_error ? last_error() : nullptr;
        return text ? std::string(text) : std::string();
    }
};

struct HdrBridge {
    HMODULE lib = nullptr;
    std::string path;
    uint32_t (*abi)() = nullptr;
    uint32_t (*multiscale_abi)() = nullptr;
    void (*fill_contract)(GpuHdrContract*) = nullptr;
    int32_t (*open)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t,
        const GpuHdrTapTable*, const GpuHdrTapTable*, double,
        GpuHdrBridge**) = nullptr;
    int32_t (*import_master)(GpuHdrBridge*, uint64_t, uint32_t, uint32_t,
        uint32_t, uint64_t*, uint32_t*) = nullptr;
    int32_t (*downscale)(GpuHdrBridge*, uint64_t*, uint32_t*) = nullptr;
    int32_t (*proxy)(GpuHdrBridge*, uint64_t*, uint32_t*, uint64_t*,
        uint32_t*) = nullptr;
    int32_t (*observe)(GpuHdrBridge*, void*, void*, uint64_t, uint64_t*,
        uint32_t*) = nullptr;
    int32_t (*set_composite)(GpuHdrBridge*, void*, void*, uint64_t, uint64_t*,
        uint32_t*) = nullptr;
    int32_t (*readback_scrgb)(GpuHdrBridge*, int32_t, uint8_t*, size_t) =
        nullptr;
    uint32_t (*plane_pitch)(GpuHdrBridge*, int32_t) = nullptr;
    int32_t (*multiscale_counters)(GpuHdrBridge*, GpuHdrMultiscaleCounters*) =
        nullptr;
    void (*close)(GpuHdrBridge*) = nullptr;
    const char* (*last_error)() = nullptr;

    bool load(const std::string& file) {
        path = file;
        lib = LoadLibraryA(file.c_str());
        if (!lib) {
            note("cannot load " + file);
            return false;
        }
        auto get = [this](const char* name) {
            return reinterpret_cast<void*>(GetProcAddress(lib, name));
        };
        abi = reinterpret_cast<decltype(abi)>(get("gpu_hdr_abi_version"));
        fill_contract = reinterpret_cast<decltype(fill_contract)>(
            get("gpu_hdr_fill_contract"));
        open = reinterpret_cast<decltype(open)>(get("gpu_hdr_open"));
        import_master = reinterpret_cast<decltype(import_master)>(
            get("gpu_hdr_import_master"));
        downscale = reinterpret_cast<decltype(downscale)>(
            get("gpu_hdr_downscale"));
        proxy = reinterpret_cast<decltype(proxy)>(get("gpu_hdr_proxy"));
        observe = reinterpret_cast<decltype(observe)>(get("gpu_hdr_observe"));
        set_composite = reinterpret_cast<decltype(set_composite)>(
            get("gpu_hdr_set_composite"));
        readback_scrgb = reinterpret_cast<decltype(readback_scrgb)>(
            get("gpu_hdr_readback_scrgb"));
        plane_pitch = reinterpret_cast<decltype(plane_pitch)>(
            get("gpu_hdr_plane_pitch"));
        multiscale_abi = reinterpret_cast<decltype(multiscale_abi)>(
            get("gpu_hdr_multiscale_abi_version"));
        multiscale_counters = reinterpret_cast<decltype(multiscale_counters)>(
            get("gpu_hdr_get_multiscale_counters"));
        close = reinterpret_cast<decltype(close)>(get("gpu_hdr_close"));
        last_error = reinterpret_cast<decltype(last_error)>(
            get("gpu_hdr_last_error"));
        const bool ready = abi && fill_contract && open && import_master &&
            downscale && proxy && observe && set_composite && readback_scrgb &&
            plane_pitch && close && last_error;
        if (!ready) note(file + " does not carry the HDR multiscale entries");
        return ready;
    }
    std::string error() const {
        const char* text = last_error ? last_error() : nullptr;
        return text ? std::string(text) : std::string();
    }
};

/* --- The fixture --------------------------------------------------------- */

constexpr uint32_t kAlpha = 3u;
constexpr uint32_t kPlaneComposited = 7u;

struct Axis {
    std::vector<int32_t> indices;
    std::vector<float> weights;
    GpuFusionAxisPlan plan{};
};

struct Layer {
    uint32_t width = 0, height = 0;
    float weight = 0.0f;
    Axis vertical, horizontal;
    DeviceBuffer residual;
    std::vector<uint16_t> samples;      /* the fixture's own halves, in host form */
};

struct Fixture {
    uint32_t full_width = 0, full_height = 0;
    std::vector<Layer> layers;
    Axis baseline_vertical, baseline_horizontal;
    double beta = 0.0, strength = 0.0, relative_tolerance = 0.0;
    uint32_t max_iterations = 0;
    double support_ceiling_scrgb = 12.5;
    std::vector<uint16_t> master;       /* full extent, tight rows */
    std::vector<uint8_t> sdr_master;    /* full extent, RGBA8 tight rows */
    /* The legacy entry's observation is an R32G32B32A32_FLOAT signed residual in
     * byte units, not a RGBA8 picture: the entry refuses anything else, so the
     * control has to arrive in the shape the SDR chain really hands it. Same
     * scene as the halves above, scaled into the 0..255 domain. */
    std::vector<float> sdr_residual[3];
};

std::vector<uint8_t> read_bytes(const std::filesystem::path& path, bool& ok) {
    FILE* handle = nullptr;
    ok = fopen_s(&handle, path.string().c_str(), "rb") == 0 && handle != nullptr;
    if (!ok) {
        note("cannot read " + path.string());
        return {};
    }
    std::fseek(handle, 0, SEEK_END);
    const long size = std::ftell(handle);
    std::fseek(handle, 0, SEEK_SET);
    const size_t length = size_t(size);
    std::vector<uint8_t> bytes(length);
    const size_t got = std::fread(bytes.data(), 1, bytes.size(), handle);
    std::fclose(handle);
    ok = got == bytes.size();
    if (!ok) note("short read from " + path.string());
    return bytes;
}

struct Manifest {
    std::map<std::string, std::string> values;
    double number(const std::string& key, double fallback = 0.0) const {
        const auto found = values.find(key);
        return found == values.end() ? fallback : std::strtod(
            found->second.c_str(), nullptr);
    }
    uint32_t plain(const std::string& key, uint32_t fallback = 0) const {
        return uint32_t(number(key, double(fallback)));
    }
    std::string text(const std::string& key) const {
        const auto found = values.find(key);
        return found == values.end() ? std::string() : found->second;
    }
};

Manifest load_manifest(const std::filesystem::path& dir) {
    Manifest manifest;
    bool ok = false;
    const auto bytes = read_bytes(dir / "manifest.txt", ok);
    if (!ok) return manifest;
    const std::string body(bytes.begin(), bytes.end());
    size_t start = 0;
    while (start < body.size()) {
        const size_t end = body.find('\n', start);
        std::string line = body.substr(start,
            end == std::string::npos ? std::string::npos : end - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t equals = line.find('=');
        if (equals != std::string::npos) {
            manifest.values[line.substr(0, equals)] =
                line.substr(equals + 1);
        }
        start = end == std::string::npos ? body.size() : end + 1;
    }
    return manifest;
}

/* plans.bin: layer_count, then per layer the extent, the weight and the two
 * axes (vertical first, as the kernels run them), then the Catmull baseline.
 * One axis is src_len, dst_len, taps and taps*dst_len int32 indices followed by
 * the same count of float32 weights - the pipeline's own float32 plan, padded to
 * one common tap count the way gpu_transfer_export.tap_tables pads it. */
bool load_plans(const std::filesystem::path& dir, Fixture& fixture) {
    bool ok = false;
    const auto bytes = read_bytes(dir / "plans.bin", ok);
    if (!ok) return false;
    size_t offset = 0;
    auto take = [&](void* destination, size_t size) {
        if (offset + size > bytes.size()) {
            note("plans.bin is shorter than its own layout");
            return false;
        }
        std::memcpy(destination, bytes.data() + offset, size);
        offset += size;
        return true;
    };
    auto read_axis = [&](Axis& axis) {
        uint32_t header[3]{};
        if (!take(header, sizeof(header))) return false;
        const size_t count = size_t(header[2]) * header[1];
        axis.indices.resize(count);
        axis.weights.resize(count);
        if (!take(axis.indices.data(), count * sizeof(int32_t))) return false;
        if (!take(axis.weights.data(), count * sizeof(float))) return false;
        axis.plan = GpuFusionAxisPlan{header[0], header[1], header[2],
            axis.indices.data(), axis.weights.data()};
        return true;
    };
    uint32_t layer_count = 0;
    if (!take(&layer_count, sizeof(layer_count))) return false;
    fixture.layers.resize(layer_count);
    for (auto& layer : fixture.layers) {
        uint32_t extent[2]{};
        if (!take(extent, sizeof(extent))) return false;
        layer.width = extent[0];
        layer.height = extent[1];
        if (!take(&layer.weight, sizeof(layer.weight))) return false;
        if (!read_axis(layer.vertical) || !read_axis(layer.horizontal)) {
            return false;
        }
    }
    return read_axis(fixture.baseline_vertical) &&
        read_axis(fixture.baseline_horizontal);
}

bool load_pixels(const std::filesystem::path& dir, const char* name,
                 uint32_t width, uint32_t height, uint32_t bytes_per_pixel,
                 std::vector<uint8_t>& out) {
    bool ok = false;
    out = read_bytes(dir / name, ok);
    if (!ok) return false;
    if (out.size() != size_t(width) * height * bytes_per_pixel) {
        note(std::string(name) + " is " + std::to_string(out.size()) +
            " bytes, the fixture's extent needs " +
            std::to_string(size_t(width) * height * bytes_per_pixel));
        return false;
    }
    return true;
}

/* The same file read as RGBA16F: the host's own half patterns, one component per
 * uint16_t, rows tight. The size the byte loader just checked is
 * width*height*8, so it is even by construction. */
bool load_halves(const std::filesystem::path& dir, const char* name,
                 uint32_t width, uint32_t height, std::vector<uint16_t>& out) {
    std::vector<uint8_t> raw;
    if (!load_pixels(dir, name, width, height, 8u, raw)) return false;
    out.resize(raw.size() / 2u);
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

/* The same plane as the RGBA32F signed residual the legacy SDR entry borrows:
 * four float32 channels per sample, rows tight. The bytes are copied rather than
 * reinterpreted in place because the loader hands back a byte vector that would
 * otherwise die with the pointer into it. */
bool load_floats(const std::filesystem::path& dir, const char* name,
                 uint32_t width, uint32_t height, std::vector<float>& out) {
    std::vector<uint8_t> raw;
    if (!load_pixels(dir, name, width, height, 16u, raw)) return false;
    out.resize(raw.size() / 4u);
    std::memcpy(out.data(), raw.data(), raw.size());
    return true;
}

bool load_fixture(const std::filesystem::path& dir, const Manifest& manifest,
                  Fixture& fixture) {
    fixture.full_width = manifest.plain("full_width");
    fixture.full_height = manifest.plain("full_height");
    fixture.beta = manifest.number("beta");
    fixture.strength = manifest.number("residual_strength");
    fixture.relative_tolerance = manifest.number("relative_tolerance");
    fixture.max_iterations = manifest.plain("max_iterations");
    fixture.support_ceiling_scrgb = manifest.number("support_ceiling_scrgb",
        12.5);
    if (!load_plans(dir, fixture)) return false;
    if (fixture.layers.size() < 2 || fixture.layers.size() > 3) {
        note("the fixture must carry two or three layers");
        return false;
    }
    bool ok = true;
    ok = load_halves(dir, "master_half.bin", fixture.full_width,
        fixture.full_height, fixture.master) && ok;
    ok = load_pixels(dir, "sdr_master_rgba8.bin", fixture.full_width,
        fixture.full_height, 4u, fixture.sdr_master) && ok;
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        const char* name = i == 0 ? "residual0_half.bin" :
            (i == 1 ? "residual1_half.bin" : "residual2_half.bin");
        ok = load_halves(dir, name, fixture.layers[i].width,
            fixture.layers[i].height, fixture.layers[i].samples) && ok;
        ok = load_floats(dir, i == 0 ? "residual0_f32.bin" :
            (i == 1 ? "residual1_f32.bin" : "residual2_f32.bin"),
            fixture.layers[i].width, fixture.layers[i].height,
            fixture.sdr_residual[i]) && ok;
    }
    return ok;
}

/* --- Half patterns ------------------------------------------------------- */

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

uint16_t nan_pattern() { return 0x7E00u; }

/* --- Output ------------------------------------------------------------- */

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

/* One key=value line per measured number. Every integer shape goes through the
 * template, which prints it exactly; a uint32_t argument would otherwise be
 * ambiguous between the double and the uint64_t parameter, and an ambiguity here
 * is a build failure that says nothing about the bridge. */
class Summary {
public:
    void add(const std::string& key, const std::string& value) {
        lines_ += key + " = " + value + "\n";
    }
    void add(const std::string& key, const char* value) {
        add(key, std::string(value));
    }
    void add(const std::string& key, double value) {
        char buffer[64];
        std::snprintf(buffer, sizeof(buffer), "%.17g", value);
        add(key, buffer);
    }
    template <typename T>
    typename std::enable_if<std::is_integral<T>::value, void>::type
    add(const std::string& key, T value) {
        if constexpr (std::is_signed<T>::value) add(key, std::to_string(
            int64_t(value)));
        else add(key, std::to_string(uint64_t(value)));
    }
    void write(const char* name) { write_file(name, lines_.data(), lines_.size()); }
    void dump(const std::vector<uint8_t>& bytes, const char* name) {
        write_file(name, bytes.data(), bytes.size());
    }
private:
    std::string lines_;
} g_summary;

/* --- Copying a producer's texture to the host ---------------------------- */

uint32_t sample_bytes(DXGI_FORMAT format) {
    return format == DXGI_FORMAT_R16G16B16A16_FLOAT ? 8u :
        (format == DXGI_FORMAT_R8G8B8A8_UNORM ? 4u :
        (format == DXGI_FORMAT_R32G32B32A32_FLOAT ? 16u : 0u));
}

/* The harness's own queue. It waits the producer fence the solver handed back
 * before it copies, which is the claim gate D is about: a consumer on another
 * queue can wait that fence and then take the texture. */
struct Harvester {
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
    Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t value = 0;
    HANDLE event = nullptr;

    bool open(ID3D12Device* device) {
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)))) {
            note("harness command queue");
            return false;
        }
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)))) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
        /* A new list starts in the recording state, and Reset - which copy()
         * calls before it records - is only legal on one that has been closed. */
        if (FAILED(list->Close())) return false;
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&fence)))) return false;
        event = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        return event != nullptr;
    }

    /* The state is not read from here: D3D12_RESOURCE_DESC::Flags is not a
     * resource state, so the only state this harness reports is the one the
     * bridge itself records. What a successful copy *does* prove is that the
     * texture was in a copy-source state at the moment of the copy, because
     * CopyTextureRegion out of a texture that is not COPY_SOURCE is a validated
     * error and the list below would not close. */
    bool copy(ID3D12Device* device, ID3D12Resource* texture,
              ID3D12Fence* producer_fence, uint64_t producer_value,
              std::vector<uint8_t>& out, uint32_t& out_pitch,
              DXGI_FORMAT& out_format, uint32_t& out_width,
              uint32_t& out_height) {
        const D3D12_RESOURCE_DESC desc = texture->GetDesc();
        const uint32_t bpp = sample_bytes(desc.Format);
        if (!bpp) { note("the harness cannot read this texture format"); return false; }
        out_format = desc.Format;
        out_width = uint32_t(desc.Width);
        out_height = desc.Height;
        const uint32_t pitch = (out_width * bpp + 255u) & ~255u;
        out_pitch = pitch;
        const uint64_t bytes = uint64_t(pitch) * out_height;
        /* A readback heap, and COPY_DEST: the validator's own words for the wrong
         * combination are "the destination resource cannot be on a
         * D3D12_HEAP_TYPE_UPLOAD heap". An upload row is CPU-write/GPU-read, so a
         * texture copy may not land in one, and this harness's whole purpose is
         * the GPU writing a picture out to the host. */
        D3D12_HEAP_PROPERTIES heap{};
        heap.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = bytes;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Microsoft::WRL::ComPtr<ID3D12Resource> readback;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE,
            &buffer, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
            IID_PPV_ARGS(&readback)))) {
            note("readback buffer");
            return false;
        }
        allocator->Reset();
        if (FAILED(list->Reset(allocator.Get(), nullptr))) {
            note("the harness readback list could not be reset");
            return false;
        }
        if (producer_fence) {
            if (FAILED(queue->Wait(producer_fence, producer_value))) {
                note("waiting the producer fence failed");
                return false;
            }
        }
        D3D12_TEXTURE_COPY_LOCATION destination{}, source{};
        destination.pResource = readback.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        destination.PlacedFootprint.Offset = 0;
        destination.PlacedFootprint.Footprint.Format = desc.Format;
        destination.PlacedFootprint.Footprint.Width = out_width;
        destination.PlacedFootprint.Footprint.Height = out_height;
        destination.PlacedFootprint.Footprint.Depth = 1;
        destination.PlacedFootprint.Footprint.RowPitch = pitch;
        source.pResource = texture;
        source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        source.SubresourceIndex = 0;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        const HRESULT closed = list->Close();
        if (FAILED(closed)) {
            char reason[96];
            std::snprintf(reason, sizeof(reason),
                "the readback list did not close (0x%08lX); a copy out of a "
                "texture that is not in COPY_SOURCE is a validated error",
                unsigned(closed));
            note(reason);
            drain_validation_notes();
            return false;
        }
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        ++value;
        queue->Signal(fence.Get(), value);
        while (fence->GetCompletedValue() < value) {
            if (FAILED(fence->SetEventOnCompletion(value, event))) return false;
            if (WaitForSingleObject(event, 30000) != WAIT_OBJECT_0) {
                note("the readback never retired");
                return false;
            }
        }
        uint8_t* mapped = nullptr;
        if (FAILED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
            return false;
        out.assign(mapped, mapped + bytes);
        readback->Unmap(0, nullptr);
        return true;
    }
};

/* --- Producer textures the solver reads ---------------------------------- */

/* One 2D texture, filled from a tight host row array and left in COPY_SOURCE
 * behind a signalled fence of its own: exactly the shape both an SDR observation
 * and a Feature 18 output arrive in. */
struct Producer {
    Microsoft::WRL::ComPtr<ID3D12Resource> texture;
    Microsoft::WRL::ComPtr<ID3D12CommandQueue> queue;
    Microsoft::WRL::ComPtr<ID3D12Fence> fence;
    uint64_t fence_value = 0;

    bool build(ID3D12Device* device, uint32_t width, uint32_t height,
               DXGI_FORMAT format, const std::vector<uint8_t>& samples) {
        const uint32_t bpp = sample_bytes(format);
        if (!bpp) return false;
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        if (FAILED(device->CreateCommandQueue(&queue_desc,
            IID_PPV_ARGS(&queue)))) return false;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        desc.Width = width;
        desc.Height = height;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = format;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
        const uint32_t pitch = (width * bpp + 255u) & ~255u;
        const uint64_t bytes = uint64_t(pitch) * height;
        D3D12_HEAP_PROPERTIES default_heap{}, upload_heap{};
        default_heap.Type = D3D12_HEAP_TYPE_DEFAULT;
        upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = bytes;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        Microsoft::WRL::ComPtr<ID3D12Resource> upload;
        if (FAILED(device->CreateCommittedResource(&default_heap,
            D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST,
            nullptr, IID_PPV_ARGS(&texture)))) return false;
        if (FAILED(device->CreateCommittedResource(&upload_heap,
            D3D12_HEAP_FLAG_NONE, &buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&upload)))) return false;
        uint8_t* mapped = nullptr;
        if (FAILED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))))
            return false;
        for (uint32_t row = 0; row < height; ++row) {
            std::memcpy(mapped + size_t(row) * pitch,
                samples.data() + size_t(row) * width * bpp, size_t(width) * bpp);
        }
        upload->Unmap(0, nullptr);
        Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
        Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
        if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(&allocator)))) return false;
        if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
            allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;
        D3D12_TEXTURE_COPY_LOCATION destination{}, source{};
        destination.pResource = texture.Get();
        destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        destination.SubresourceIndex = 0;
        source.pResource = upload.Get();
        source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        source.PlacedFootprint.Footprint.Format = format;
        source.PlacedFootprint.Footprint.Width = width;
        source.PlacedFootprint.Footprint.Height = height;
        source.PlacedFootprint.Footprint.Depth = 1;
        source.PlacedFootprint.Footprint.RowPitch = pitch;
        list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = texture.Get();
        barrier.Transition.Subresource = 0;
        barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        list->ResourceBarrier(1, &barrier);
        if (FAILED(list->Close())) return false;
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&fence)))) return false;
        fence_value = 1;
        queue->Signal(fence.Get(), fence_value);
        HANDLE done = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!done) return false;
        while (fence->GetCompletedValue() < fence_value) {
            if (FAILED(fence->SetEventOnCompletion(fence_value, done))) {
                CloseHandle(done);
                return false;
            }
            WaitForSingleObject(done, 30000);
        }
        CloseHandle(done);
        return true;
    }
};

/* --- Building the option structs from the fixture ------------------------ */

GpuFusionOptions fusion_options(const Fixture& fixture) {
    GpuFusionOptions options{};
    options.struct_size = sizeof(GpuFusionOptions);
    options.full_width = fixture.full_width;
    options.full_height = fixture.full_height;
    options.layer_count = uint32_t(fixture.layers.size());
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        const auto& layer = fixture.layers[i];
        auto& target = options.layers[i];
        target.width = layer.width;
        target.height = layer.height;
        target.weight = layer.weight;
        target.horizontal = layer.horizontal.plan;
        target.vertical = layer.vertical.plan;
    }
    options.baseline_horizontal = fixture.baseline_horizontal.plan;
    options.baseline_vertical = fixture.baseline_vertical.plan;
    options.beta = float(fixture.beta);
    options.residual_strength = float(fixture.strength);
    options.relative_tolerance = float(fixture.relative_tolerance);
    options.max_iterations = fixture.max_iterations;
    return options;
}

/* The HDR bridge reads the same two tables the solver reads: layer 0's Lanczos-2
 * downscale and the Catmull upscale, so the proxy, the residual and the fused
 * frame are all statements about one filter. */
GpuHdrTapTable hdr_tap_table(const Axis& vertical, const Axis& horizontal,
    uint32_t vertical_src, uint32_t vertical_dst, uint32_t horizontal_src,
    uint32_t horizontal_dst) {
    GpuHdrTapTable table{};
    table.struct_size = sizeof(GpuHdrTapTable);
    table.taps = vertical.plan.taps;
    table.vertical_src_len = vertical_src;
    table.vertical_dst_len = vertical_dst;
    table.vertical_indices = const_cast<int32_t*>(vertical.indices.data());
    table.vertical_weights = const_cast<float*>(vertical.weights.data());
    table.horizontal_src_len = horizontal_src;
    table.horizontal_dst_len = horizontal_dst;
    table.horizontal_indices = const_cast<int32_t*>(horizontal.indices.data());
    table.horizontal_weights = const_cast<float*>(horizontal.weights.data());
    return table;
}

/* --- Gates -------------------------------------------------------------- */

struct TypedRun {
    bool ok = false;
    bool harvested = false;
    int32_t code = 0;
    int32_t retry_code = 0;
    bool retry_texture_null = false;
    GpuFusionBridge* bridge = nullptr;
    void* texture = nullptr;
    void* fence = nullptr;
    uint64_t fence_value = 0;
    GpuFusionHdrReport report{};
    GpuFusionHdrCounters counters{};
    std::vector<uint8_t> bytes;
    uint32_t pitch = 0, width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
};

TypedRun run_typed(Fusion& fusion, Machine& machine, Fixture& fixture,
    const GpuFusionOptions& options, bool expect_failure) {
    TypedRun run;
    GpuFusionBridge* bridge = nullptr;
    const GpuFusionHdrOptions hdr{sizeof(GpuFusionHdrOptions),
        GPU_FUSION_HDR_ABI_VERSION, 8u, 0u};
    run.code = fusion.open_hdr(machine.device.Get(), &options, &hdr, &bridge);
    note("gpu_fusion_open_hdr answered " + std::to_string(run.code));
    if (run.code) {
        note("gpu_fusion_open_hdr -> " + std::to_string(run.code) + ": " +
            fusion.error());
        return run;
    }
    /* The master and every residual are CUDA device planes: a real producer's
     * output is a device pointer the observing bridge retired before handing it
     * over, and this is what that looks like to the solver. */
    DeviceBuffer master;
    const uint32_t master_pitch = fixture.full_width * 8u;
    const auto master_bytes = fixture.master.size() * sizeof(uint16_t);
    std::vector<uint16_t> master_samples = fixture.master;
    if (expect_failure) {
        /* The unrecoverable case: the source plane itself carries a component
         * the half grid cannot hold. */
        master_samples[0] = nan_pattern();
    }
    if (!master.upload(machine.cuda, master_samples.data(), master_bytes,
        master_pitch, fixture.full_width, fixture.full_height)) {
        note("cuMemAlloc/cuMemcpyHtoD of the master failed");
        fusion.close(bridge);
        return run;
    }
    std::vector<DeviceBuffer> residuals(fixture.layers.size());
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        const auto& layer = fixture.layers[i];
        const uint32_t pitch = layer.width * 8u;
        if (!residuals[i].upload(machine.cuda, layer.samples.data(),
            layer.samples.size() * sizeof(uint16_t), pitch, layer.width,
            layer.height)) {
            note("uploading a residual plane failed");
            master.free(machine.cuda);
            fusion.close(bridge);
            return run;
        }
    }
    std::vector<GpuFusionHdrObservation> observations(fixture.layers.size());
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        observations[i] = GpuFusionHdrObservation{
            uint64_t(residuals[i].ptr), residuals[i].pitch,
            fixture.layers[i].width, fixture.layers[i].height};
    }
    note("the master and the residual planes are on the device");
    run.report = GpuFusionHdrReport{};
    run.report.struct_size = sizeof(GpuFusionHdrReport);
    run.report.abi_version = GPU_FUSION_HDR_ABI_VERSION;
    void* texture = nullptr;
    void* fence = nullptr;
    uint64_t value = 0;
    run.code = fusion.process_hdr(bridge, machine.context,
        uint64_t(master.ptr), master_pitch, observations.data(),
        uint32_t(observations.size()), &texture, &fence, &value, &run.report);
    note("gpu_fusion_process_hdr answered " + std::to_string(run.code));
    if (run.code) {
        note("gpu_fusion_process_hdr -> " + std::to_string(run.code) + ": " +
            fusion.error());
        run.texture = texture;
        run.fence = fence;
        /* A second frame on a failed bridge must be refused, not solved: that is
         * what "no silent fallback" looks like from the caller's side. */
        GpuFusionHdrReport retry{};
        retry.struct_size = sizeof(retry);
        retry.abi_version = GPU_FUSION_HDR_ABI_VERSION;
        void* retry_texture = nullptr;
        void* retry_fence = nullptr;
        uint64_t retry_value = 0;
        const int32_t again = fusion.process_hdr(bridge, machine.context,
            uint64_t(master.ptr), master_pitch, observations.data(),
            uint32_t(observations.size()), &retry_texture, &retry_fence,
            &retry_value, &retry);
        run.retry_code = again;
        run.retry_texture_null = retry_texture == nullptr;
        check(again != 0, "a failed bridge refuses the next frame");
        check(retry_texture == nullptr, "the refused frame hands back no texture");
        master.free(machine.cuda);
        for (auto& plane : residuals) plane.free(machine.cuda);
        fusion.close(bridge);
        return run;
    }
    run.texture = texture;
    run.fence = fence;
    run.fence_value = value;
    run.counters = GpuFusionHdrCounters{};
    run.counters.struct_size = sizeof(GpuFusionHdrCounters);
    run.counters.abi_version = GPU_FUSION_HDR_ABI_VERSION;
    const int32_t counter_code = fusion.counters_hdr(bridge, &run.counters);
    check(counter_code == 0, "gpu_fusion_get_hdr_counters answers");
    Harvester harvester;
    if (!harvester.open(machine.device.Get())) {
        note("the harness cannot build its own queue");
    } else {
        run.harvested = harvester.copy(machine.device.Get(),
            static_cast<ID3D12Resource*>(texture),
            static_cast<ID3D12Fence*>(fence), value, run.bytes, run.pitch,
            run.format, run.width, run.height);
        if (!run.harvested) {
            note("the fused texture could not be harvested");
        }
    }
    master.free(machine.cuda);
    for (auto& plane : residuals) plane.free(machine.cuda);
    run.ok = true;
    run.bridge = bridge;
    /* The bridge stays open: gate D consumes the texture before it is closed. */
    g_summary.add("typed_bridge_held_open", 1u);
    return run;
}

/* Gate D: a real gpu_hdr bridge takes the fused texture behind the solver's own
 * fence, and its composited plane is what a downstream encode would read. */
bool gate_handoff(HdrBridge& hdr, Machine& machine,
    Fixture& fixture, const TypedRun& run, double paper_white,
    GpuHdrBridge** keep) {
    if (!run.ok || !run.texture) {
        note("gate D skipped: no fused texture");
        return false;
    }
    GpuHdrTapTable down = hdr_tap_table(fixture.layers[0].vertical,
        fixture.layers[0].horizontal, fixture.full_height,
        fixture.layers[0].height, fixture.full_width, fixture.layers[0].width);
    GpuHdrTapTable up = hdr_tap_table(fixture.baseline_vertical,
        fixture.baseline_horizontal, fixture.layers[0].height,
        fixture.full_height, fixture.layers[0].width, fixture.full_width);
    GpuHdrBridge* bridge = nullptr;
    const int32_t opened = hdr.open(machine.device.Get(), machine.context,
        fixture.full_width, fixture.full_height, fixture.layers[0].width,
        fixture.layers[0].height, &down, &up, paper_white, &bridge);
    if (opened) {
        note("gpu_hdr_open -> " + std::to_string(opened) + ": " + hdr.error());
        return false;
    }
    DeviceBuffer master;
    const uint32_t pitch = fixture.full_width * 8u;
    if (!master.upload(machine.cuda, fixture.master.data(),
        fixture.master.size() * sizeof(uint16_t), pitch, fixture.full_width,
        fixture.full_height)) {
        note("the handoff bridge could not be fed a master");
        hdr.close(bridge);
        return false;
    }
    uint64_t master_ptr = 0;
    uint32_t master_pitch = 0;
    const int32_t imported = hdr.import_master(bridge, uint64_t(master.ptr),
        fixture.full_width, fixture.full_height, pitch, &master_ptr,
        &master_pitch);
    if (imported) {
        note("gpu_hdr_import_master -> " + std::to_string(imported) + ": " +
            hdr.error());
        master.free(machine.cuda);
        hdr.close(bridge);
        return false;
    }
    uint64_t composited_ptr = 0;
    uint32_t composited_pitch = 0;
    const int32_t accepted = hdr.set_composite(bridge, run.texture, run.fence,
        run.fence_value, &composited_ptr, &composited_pitch);
    check(accepted == 0, "gpu_hdr_set_composite accepts the fused texture and "
        "the solver's producer fence");
    if (accepted) {
        note("gpu_hdr_set_composite -> " + std::to_string(accepted) + ": " +
            hdr.error());
        master.free(machine.cuda);
        hdr.close(bridge);
        return false;
    }
    const uint32_t row_stride = hdr.plane_pitch(bridge, int32_t(kPlaneComposited));
    std::vector<uint8_t> plane(size_t(row_stride) * fixture.full_height, 0u);
    const int32_t read = hdr.readback_scrgb(bridge, int32_t(kPlaneComposited),
        plane.data(), row_stride);
    check(read == 0, "the composited plane reads back");
    g_summary.add("composited_pitch", uint64_t(row_stride));
    g_summary.add("composited_width", uint64_t(fixture.full_width));
    g_summary.add("composited_height", uint64_t(fixture.full_height));
    g_summary.add("composited_set_code", accepted);
    g_summary.dump(plane, "composited.bin");
    if (keep) *keep = bridge; else hdr.close(bridge);
    master.free(machine.cuda);
    return read == 0;
}

/* Gate C: the A1 observation ordering, on live bridge state. */
void gate_ordering(HdrBridge& hdr, Machine& machine, Fixture& fixture,
    double paper_white) {
    GpuHdrTapTable down = hdr_tap_table(fixture.layers[0].vertical,
        fixture.layers[0].horizontal, fixture.full_height,
        fixture.layers[0].height, fixture.full_width, fixture.layers[0].width);
    GpuHdrTapTable up = hdr_tap_table(fixture.baseline_vertical,
        fixture.baseline_horizontal, fixture.layers[0].height,
        fixture.full_height, fixture.layers[0].width, fixture.full_width);
    GpuHdrBridge* bridge = nullptr;
    if (hdr.open(machine.device.Get(), machine.context, fixture.full_width,
        fixture.full_height, fixture.layers[0].width, fixture.layers[0].height,
        &down, &up, paper_white, &bridge)) {
        note("gate C could not open an HDR bridge: " + hdr.error());
        return;
    }
    DeviceBuffer first, second;
    const uint32_t pitch = fixture.full_width * 8u;
    std::vector<uint16_t> samples = fixture.master;
    if (!first.upload(machine.cuda, samples.data(),
        samples.size() * sizeof(uint16_t), pitch, fixture.full_width,
        fixture.full_height)) {
        note("gate C could not upload a master");
        hdr.close(bridge);
        return;
    }
    /* A different frame for the second import: the same bytes would let a bridge
     * that compared contents rather than provenance look correct. */
    samples[0] = uint16_t(samples[0] ^ 0x0400u);
    bool second_ready = second.upload(machine.cuda, samples.data(),
        samples.size() * sizeof(uint16_t), pitch, fixture.full_width,
        fixture.full_height);
    uint64_t ptr = 0;
    uint32_t out_pitch = 0;
    uint64_t proxy_ptr = 0, guidance_ptr = 0;
    uint32_t proxy_pitch = 0, guidance_pitch = 0;
    int32_t code = hdr.import_master(bridge, uint64_t(first.ptr),
        fixture.full_width, fixture.full_height, pitch, &ptr, &out_pitch);
    g_summary.add("order_import_first_code", code);
    code = hdr.downscale(bridge, &ptr, &out_pitch);
    g_summary.add("order_downscale_first_code", code);
    code = hdr.proxy(bridge, &proxy_ptr, &proxy_pitch, &guidance_ptr,
        &guidance_pitch);
    g_summary.add("order_proxy_first_code", code);
    if (second_ready) {
        code = hdr.import_master(bridge, uint64_t(second.ptr),
            fixture.full_width, fixture.full_height, pitch, &ptr, &out_pitch);
        g_summary.add("order_import_second_code", code);
        /* The stale-proxy order: proxy still reads the working plane the first
         * master left behind, so the downscale after it supersedes it. */
        code = hdr.proxy(bridge, &proxy_ptr, &proxy_pitch, &guidance_ptr,
            &guidance_pitch);
        g_summary.add("order_proxy_second_code", code);
        code = hdr.downscale(bridge, &ptr, &out_pitch);
        g_summary.add("order_downscale_second_code", code);
        code = hdr.observe(bridge, nullptr, nullptr, 0, &ptr, &out_pitch);
        g_summary.add("order_observe_after_stale_proxy_code", code);
        g_summary.add("order_observe_after_stale_proxy_message",
            hdr.error());
        check(code != 0, "an observation whose proxy belongs to a superseded "
            "working plane is refused");
        const std::string message = hdr.error();
        const bool names_the_proxy = message.find("gpu_hdr_proxy") !=
            std::string::npos;
        g_summary.add("order_refusal_names_proxy", names_the_proxy ? 1u : 0u);
    }
    /* And the correct order: a proxy of the working plane this bridge now holds,
     * then the observation gets as far as the missing Feature 18 output. */
    code = hdr.proxy(bridge, &proxy_ptr, &proxy_pitch, &guidance_ptr,
        &guidance_pitch);
    g_summary.add("order_proxy_third_code", code);
    code = hdr.observe(bridge, nullptr, nullptr, 0, &ptr, &out_pitch);
    g_summary.add("order_observe_after_current_proxy_code", code);
    g_summary.add("order_observe_after_current_proxy_message", hdr.error());
    check(code != 0, "an observation with no Feature 18 output is refused");
    const std::string current = hdr.error();
    g_summary.add("order_current_refusal_names_proxy",
        current.find("gpu_hdr_proxy") != std::string::npos ? 1u : 0u);
    if (hdr.multiscale_counters) {
        GpuHdrMultiscaleCounters counters{};
        counters.struct_size = sizeof(counters);
        const int32_t read = hdr.multiscale_counters(bridge, &counters);
        g_summary.add("multiscale_counters_code", read);
        g_summary.add("pass_dispatches", counters.pass_dispatches);
        g_summary.add("scalar_counter_readback_bytes",
            counters.pass_dispatches * (uint64_t)64u * 4u);
        g_summary.add("master_provenance", counters.master_provenance);
        g_summary.add("master_import_count", counters.master_import_count);
        g_summary.add("observe_count", counters.observe_count);
    } else {
        g_summary.add("pass_dispatches", "the bridge exposes no multiscale counters");
    }
    first.free(machine.cuda);
    second.free(machine.cuda);
    hdr.close(bridge);
}

/* Gate B: the legacy SDR entry, on the same fixture extents and the same
 * float32 plans, composites to bytes. Its output is dumped so a second run
 * against another build of the bridge can be compared with it byte for byte. */
bool gate_sdr(Fusion& fusion, Machine& machine, Fixture& fixture) {
    const GpuFusionOptions options = fusion_options(fixture);
    GpuFusionBridge* bridge = nullptr;
    const int32_t opened = fusion.open_sdr(machine.device.Get(), &options,
        &bridge);
    if (opened) {
        note("gpu_fusion_open -> " + std::to_string(opened) + ": " +
            fusion.error());
        return false;
    }
    DeviceBuffer master;
    const uint32_t pitch = fixture.full_width * 4u;
    if (!master.upload(machine.cuda, fixture.sdr_master.data(),
        fixture.sdr_master.size(), pitch, fixture.full_width,
        fixture.full_height)) {
        note("the SDR control could not upload its source plane");
        fusion.close(bridge);
        return false;
    }
    std::vector<Producer> observations(fixture.layers.size());
    std::vector<GpuFusionObservation> descriptors(fixture.layers.size());
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        const std::vector<uint8_t> bytes(
            reinterpret_cast<const uint8_t*>(fixture.sdr_residual[i].data()),
            reinterpret_cast<const uint8_t*>(fixture.sdr_residual[i].data()) +
            fixture.sdr_residual[i].size() * sizeof(float));
        if (!observations[i].build(machine.device.Get(), fixture.layers[i].width,
            fixture.layers[i].height, DXGI_FORMAT_R32G32B32A32_FLOAT, bytes)) {
            note("the SDR control could not build a signed residual texture");
            master.free(machine.cuda);
            fusion.close(bridge);
            return false;
        }
        descriptors[i] = GpuFusionObservation{observations[i].texture.Get(),
            observations[i].fence.Get(), observations[i].fence_value};
    }
    GpuFusionSolveReport report{};
    report.struct_size = sizeof(report);
    void* texture = nullptr;
    const int32_t code = fusion.process_sdr(bridge, machine.context,
        uint64_t(master.ptr), pitch, descriptors.data(),
        uint32_t(descriptors.size()), &texture, &report);
    g_summary.add("sdr_process_code", code);
    g_summary.add("sdr_iterations", report.iterations);
    g_summary.add("sdr_initial_residual_norm", report.initial_residual_norm);
    g_summary.add("sdr_final_residual_norm", report.final_residual_norm);
    g_summary.add("sdr_relative_residual", report.relative_residual);
    g_summary.add("sdr_converged", report.converged);
    if (code || !texture) {
        note("the SDR control did not composite: " + fusion.error());
        master.free(machine.cuda);
        fusion.close(bridge);
        return false;
    }
    GpuFusionTransferCounters counters{};
    counters.struct_size = sizeof(counters);
    if (fusion.counters_sdr) {
        g_summary.add("sdr_counters_code", fusion.counters_sdr(bridge, &counters));
        g_summary.add("sdr_process_count", counters.process_count);
        g_summary.add("sdr_full_device_copy_bytes", counters.full_device_copy_bytes);
        g_summary.add("sdr_scalar_readback_bytes", counters.scalar_readback_bytes);
        g_summary.add("sdr_output_readback_bytes", counters.output_readback_bytes);
    }
    Harvester harvester;
    std::vector<uint8_t> bytes;
    uint32_t row_pitch = 0, width = 0, height = 0;
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN;
    const bool harvested = harvester.open(machine.device.Get()) &&
        harvester.copy(machine.device.Get(), static_cast<ID3D12Resource*>(texture),
            nullptr, 0, bytes, row_pitch, format, width, height);
    g_summary.add("sdr_harvested", harvested ? 1u : 0u);
    g_summary.add("sdr_output_format", uint32_t(format));
    g_summary.add("sdr_output_width", uint64_t(width));
    g_summary.add("sdr_output_height", uint64_t(height));
    g_summary.add("sdr_output_row_pitch", uint64_t(row_pitch));
    g_summary.dump(bytes, "sdr_composite.bin");
    /* The legacy entry's own contract, kept rather than assumed: an 8-bit
     * picture, clamped and rounded by the composite kernel, at the full extent. */
    check(format == DXGI_FORMAT_R8G8B8A8_UNORM,
        "the SDR control's composite is R8G8B8A8_UNORM");
    const bool bytes_moved = harvested && bytes.size() ==
        size_t(row_pitch) * height &&
        std::any_of(bytes.begin(), bytes.end(),
            [](uint8_t value) { return value != 0u; });
    check(bytes_moved, "the SDR control's composite holds a real byte picture");
    master.free(machine.cuda);
    fusion.close(bridge);
    return bytes_moved;
}

/* --- main --------------------------------------------------------------- */

/* The counted out-of-range fallback, on the same fixture and the same solve with
 * one number changed: a residual_strength wide enough that source +
 * strength*residual leaves the half grid. The store must still report success,
 * must still hold nothing the grid cannot hold, and must put the pristine master
 * sample exactly where it fell back. That is the semantics A2 had to carry over
 * from the single-layer composite rather than redefine, and it is only observable
 * from the outside if the count and the pixels are both read. */
void gate_counted_fallback(Fusion& fusion, Machine& machine, Fixture& fixture,
    const GpuFusionOptions& options) {
    GpuFusionOptions wide = options;
    wide.residual_strength = 1.0e9f;
    const TypedRun run = run_typed(fusion, machine, fixture, wide, false);
    g_summary.add("fallback_stage_code", run.code);
    if (!run.ok) {
        note("the counted fallback case never reached its store");
        if (run.bridge) fusion.close(run.bridge);
        return;
    }
    const uint32_t fell_back = run.report.unrepresentable_components;
    const uint32_t width = run.width, height = run.height;
    g_summary.add("fallback_strength", 1.0e9);
    g_summary.add("fallback_unrepresentable_components", fell_back);
    g_summary.add("fallback_iterations", run.report.iterations);
    g_summary.add("fallback_pixels", uint64_t(width) * height);
    const size_t needed = size_t(run.pitch) * height;
    if (run.bytes.size() < needed) {
        note("the fallback store was not harvested whole");
        fusion.close(run.bridge);
        return;
    }
    auto stored = [&](uint32_t row, uint32_t column, uint32_t component) {
        const size_t offset = size_t(row) * run.pitch + size_t(column) * 8u +
            component * 2u;
        return uint16_t(uint16_t(run.bytes[offset]) |
            (uint16_t(run.bytes[offset + 1u]) << 8));
    };
    size_t kept = 0, non_finite = 0, out_of_range = 0;
    for (uint32_t row = 0; row < height; ++row) {
        for (uint32_t column = 0; column < width; ++column) {
            bool same = true;
            for (uint32_t component = 0; component < 3u; ++component) {
                const uint16_t pattern = stored(row, column, component);
                const double value = from_half(pattern);
                if (!std::isfinite(value)) ++non_finite;
                else if (value > 65504.0 || value < -65504.0) ++out_of_range;
                const size_t source = (size_t(row) * fixture.full_width +
                    column) * 4u + component;
                if (pattern != fixture.master[source]) same = false;
            }
            if (same) ++kept;
        }
    }
    g_summary.add("fallback_pixels_holding_master", uint64_t(kept));
    g_summary.add("fallback_non_finite_components", uint64_t(non_finite));
    g_summary.add("fallback_out_of_range_components", uint64_t(out_of_range));
    check(run.code == 0, "a picture that fell back to the master is still a "
        "success, not an error");
    check(fell_back > 0u, "the store counted the components it could not hold");
    check(fell_back % 3u == 0u, "the count is three components a fallback pixel");
    check(uint64_t(fell_back) / 3u <= kept, "every counted fallback pixel holds "
        "the pristine master sample bit for bit");
    check(non_finite == 0u && out_of_range == 0u, "a counted fallback stores no "
        "Inf and no value the half grid cannot hold");
    g_summary.dump(run.bytes, "fallback.bin");
    fusion.close(run.bridge);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::printf("usage: %s <fixture dir> <out dir> [fusion dll] "
            "[hdr dll] [typed|sdr|order]\n", argv[0]);
        return 2;
    }
    /* Unbuffered: this harness drives bridges that own device memory, and a
     * crash inside one of them would otherwise take the progress lines that
     * name the stage down with it. */
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const std::filesystem::path fixture_dir = argv[1];
    g_dump_dir = argv[2];
    std::error_code ec;
    std::filesystem::create_directories(g_dump_dir, ec);
    const std::string fusion_path = argc > 3 ? argv[3] :
        "gpu_multiscale_fusion_bridge.dll";
    const std::string hdr_path = argc > 4 ? argv[4] :
        "gpu_hdr_scrgb_bridge.dll";
    const std::string mode = argc > 5 ? argv[5] : "typed";

    Fusion fusion;
    if (!fusion.load(fusion_path)) return 2;
    g_summary.add("fusion_dll", fusion_path);
    g_summary.add("fusion_abi", uint64_t(fusion.abi()));
    g_summary.add("fusion_hdr_abi", fusion.hdr_abi ?
        uint64_t(fusion.hdr_abi()) : uint64_t(0));
    g_summary.add("fusion_typed_group", fusion.typed() ? "present" : "absent");
    g_summary.add("mode", mode);

    HdrBridge hdr;
    const bool hdr_ready = hdr.load(hdr_path);
    g_summary.add("hdr_dll", hdr_path);
    if (hdr_ready && hdr.abi) {
        g_summary.add("hdr_abi", uint64_t(hdr.abi()));
        g_summary.add("hdr_multiscale_abi", hdr.multiscale_abi ?
            uint64_t(hdr.multiscale_abi()) : uint64_t(0));
    }

    const Manifest manifest = load_manifest(fixture_dir);
    Fixture fixture;
    if (!load_fixture(fixture_dir, manifest, fixture)) {
        std::printf("[FAIL] the fixture did not load\n");
        g_summary.write("summary.txt");
        return 1;
    }
    g_summary.add("full_width", uint64_t(fixture.full_width));
    g_summary.add("full_height", uint64_t(fixture.full_height));
    g_summary.add("layer_count", uint64_t(fixture.layers.size()));
    for (size_t i = 0; i < fixture.layers.size(); ++i) {
        char key[32];
        std::snprintf(key, sizeof(key), "layer%zu_extent", i);
        g_summary.add(key, std::to_string(fixture.layers[i].width) + "x" +
            std::to_string(fixture.layers[i].height));
        std::snprintf(key, sizeof(key), "layer%zu_taps", i);
        g_summary.add(key, uint64_t(fixture.layers[i].horizontal.plan.taps));
        std::snprintf(key, sizeof(key), "layer%zu_weight", i);
        g_summary.add(key, double(fixture.layers[i].weight));
    }
    g_summary.add("beta", fixture.beta);
    g_summary.add("residual_strength", fixture.strength);
    g_summary.add("relative_tolerance", fixture.relative_tolerance);
    g_summary.add("max_iterations", uint64_t(fixture.max_iterations));

    const bool want_debug = argc > 6 && std::string(argv[6]) == "--debug-layer";
    g_summary.add("debug_layer", want_debug ? "requested" : "off");
    Machine machine;
    if (!machine.open(want_debug)) {
        std::printf("[FAIL] no matched D3D12 device and CUDA context\n");
        g_summary.write("summary.txt");
        return 1;
    }

    /* paperWhite is the contract's own actionable band over the ceiling the job
     * declared, the same way the HDR chain resolves it. */
    GpuHdrContract contract{};
    contract.struct_size = sizeof(contract);
    if (hdr_ready && hdr.fill_contract) hdr.fill_contract(&contract);
    const double actionable = contract.proxy_actionable_relative > 0.0
        ? contract.proxy_actionable_relative : 2.5;
    const double paper_white = fixture.support_ceiling_scrgb / actionable;
    g_summary.add("scrgb_per_nit", contract.scrgb_per_nit);
    g_summary.add("proxy_actionable_relative", actionable);
    g_summary.add("paper_white_scrgb", paper_white);

    const GpuFusionOptions options = fusion_options(fixture);

    if (mode == "sdr") {
        check(gate_sdr(fusion, machine, fixture),
            "the legacy SDR entry composited a byte picture");
    } else if (mode == "order") {
        if (!hdr_ready) {
            std::printf("[FAIL] the ordering gate needs an HDR bridge\n");
            ++g_failures;
        } else {
            gate_ordering(hdr, machine, fixture, paper_white);
        }
    } else {
        if (!fusion.typed()) {
            std::printf("[FAIL] this build of the solver has no typed HDR group\n");
            ++g_failures;
        } else {
            const TypedRun run = run_typed(fusion, machine, fixture, options,
                false);
            check(run.ok, "the typed solve produced a fused picture");
            if (run.ok) {
                g_summary.add("iterations", run.report.iterations);
                g_summary.add("initial_residual_norm",
                    run.report.initial_residual_norm);
                g_summary.add("final_residual_norm",
                    run.report.final_residual_norm);
                g_summary.add("relative_residual", run.report.relative_residual);
                g_summary.add("converged", run.report.converged);
                g_summary.add("unrepresentable_components",
                    run.report.unrepresentable_components);
                g_summary.add("output_format", uint64_t(run.format));
                g_summary.add("output_width", uint64_t(run.width));
                g_summary.add("output_height", uint64_t(run.height));
                g_summary.add("output_state", uint64_t(run.report.output_state));
                g_summary.add("fused_row_pitch", uint64_t(run.pitch));
                g_summary.add("source_device_copy_bytes",
                    run.report.source_device_copy_bytes);
                g_summary.add("observation_device_copy_bytes",
                    run.report.observation_device_copy_bytes);
                g_summary.add("scalar_readback_bytes",
                    run.report.scalar_readback_bytes);
                g_summary.add("counter_process_count", run.counters.process_count);
                g_summary.add("counter_source_device_copy_bytes",
                    run.counters.source_device_copy_bytes);
                g_summary.add("counter_observation_device_copy_bytes",
                    run.counters.observation_device_copy_bytes);
                g_summary.add("counter_scalar_readback_bytes",
                    run.counters.scalar_readback_bytes);
                g_summary.add("counter_unrepresentable_components",
                    run.counters.unrepresentable_components);
                check(run.format == DXGI_FORMAT_R16G16B16A16_FLOAT,
                    "the fused store is R16G16B16A16_FLOAT");
                check(run.width == fixture.full_width &&
                    run.height == fixture.full_height,
                    "the fused store is the full extent");
                check(run.report.output_state ==
                    uint32_t(D3D12_RESOURCE_STATE_COPY_SOURCE),
                    "the fused store is left in COPY_SOURCE");
                check(run.fence != nullptr && run.fence_value != 0,
                    "the store names a producer fence and value");
                /* Nothing that reported success may hold a value the half grid
                 * cannot: a stored Inf or NaN would read as a picture. The scan
                 * needs the harvest, so an unhavested store is failed here
                 * rather than read out of an empty buffer. */
                check(run.harvested, "the harness could harvest the fused store");
                size_t non_finite = 0, above_one = 0, below_zero = 0;
                const uint32_t bytes_per_row = run.width * 8u;
                const size_t needed = size_t(run.pitch) * run.height;
                if (run.harvested && run.bytes.size() == needed) {
                for (uint32_t row = 0; row < run.height; ++row) {
                    for (uint32_t column = 0; column < bytes_per_row / 2u;
                         ++column) {
                        const uint16_t pattern = uint16_t(
                            run.bytes[size_t(row) * run.pitch +
                                size_t(column) * 2u] |
                            (uint16_t(run.bytes[size_t(row) * run.pitch +
                                size_t(column) * 2u + 1u]) << 8));
                        if (column % 4u == kAlpha) continue;
                        const double value = from_half(pattern);
                        if (!std::isfinite(value)) ++non_finite;
                        else if (value > 1.0) ++above_one;
                        if (std::isfinite(value) && value < 0.0) ++below_zero;
                    }
                }
                g_summary.add("fused_non_finite_components",
                    uint64_t(non_finite));
                g_summary.add("fused_components_above_one",
                    uint64_t(above_one));
                g_summary.add("fused_components_below_zero",
                    uint64_t(below_zero));
                check(non_finite == 0, "a successful store holds no Inf or NaN");
                check(above_one > 0 && below_zero > 0, "the fused picture kept "
                    "both headroom above 1.0 and out-of-gamut negatives");
                g_summary.dump(run.bytes, "fused.bin");
                }
                if (hdr_ready) {
                    GpuHdrBridge* held = nullptr;
                    check(gate_handoff(hdr, machine, fixture, run,
                        paper_white, &held),
                        "the fused texture reached a real HDR bridge");
                    gate_ordering(hdr, machine, fixture, paper_white);
                    if (held) hdr.close(held);
                    check(gate_sdr(fusion, machine, fixture),
                        "the legacy SDR entry still composites bytes");
                }
                gate_counted_fallback(fusion, machine, fixture, options);
                /* Closing the bridge that handed the fused texture out is the last
                 * thing done with it: the texture belongs to the bridge, and gate D
                 * read it before this point. */
                fusion.close(run.bridge);
                /* The failure gates: an unconverged solve, an indefinitive
                 * operator and a picture the half grid cannot hold all leave no
                 * texture behind. */
                GpuFusionOptions starved = options;
                starved.max_iterations = 1;
                starved.relative_tolerance = 1e-12f;
                const TypedRun budget = run_typed(fusion, machine, fixture,
                    starved, false);
                check(!budget.ok && budget.code == 12,
                    "a one-iteration budget fails at the iteration stage");
                check(budget.texture == nullptr,
                    "the failed solve handed back no texture");
                g_summary.add("budget_stage_code", budget.code);
                g_summary.add("budget_texture_null",
                    budget.texture == nullptr ? 1u : 0u);
                g_summary.add("budget_retry_code", budget.retry_code);
                const TypedRun unrecoverable = run_typed(fusion, machine,
                    fixture, options, true);
                check(!unrecoverable.ok && unrecoverable.code != 0,
                    "a master carrying a non-finite component fails outright");
                check(unrecoverable.texture == nullptr,
                    "that failure handed back no texture");
                g_summary.add("non_finite_master_stage_code",
                    unrecoverable.code);
                g_summary.add("non_finite_master_texture_null",
                    unrecoverable.texture == nullptr ? 1u : 0u);
                g_summary.add("non_finite_master_retry_code",
                    unrecoverable.retry_code);
            }
        }
    }
    machine.close();
    g_summary.add("failures", uint64_t(g_failures));
    g_summary.add("notes", g_notes.empty() ? "none" : "see notes file");
    g_summary.write("summary.txt");
    write_file("notes.txt", g_notes.data(), g_notes.size());
    std::printf("%s  %d failure(s)\n", g_failures ? "[FAIL]" : "[ ok ]",
        g_failures);
    return g_failures ? 1 : 0;
}
