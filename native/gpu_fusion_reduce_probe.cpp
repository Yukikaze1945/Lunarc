#include "gpu_multiscale_fusion_shader.h"

#include <windows.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
namespace {
constexpr uint32_t kSrvCount = 7, kUavCount = 3, kDescriptors = 10;
constexpr uint32_t kMaxCount = 65537;
constexpr uint64_t kMaxBytes = uint64_t(kMaxCount) * sizeof(float);

struct Constants {
    uint32_t out_width, out_height, in_width, in_height;
    uint32_t full_pitch, group_width, mode;
    float scale;
};

struct Harness {
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    ComPtr<ID3D12Fence> fence;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pipeline;
    ComPtr<ID3D12DescriptorHeap> heap;
    ComPtr<ID3D12Resource> input, scratch_a, scratch_b, upload, readback;
    ComPtr<ID3D12InfoQueue> info_queue;
    uint32_t descriptor_step = 0, descriptor_cursor = 0;
    uint64_t fence_value = 0, scalar_readback_bytes = 0;
    HANDLE event = nullptr;
    bool debug_layer = false;

    ~Harness() { if (event) CloseHandle(event); }

    HRESULT make_buffer(uint64_t bytes, D3D12_HEAP_TYPE type,
        D3D12_RESOURCE_STATES state, ComPtr<ID3D12Resource>* result) {
        D3D12_HEAP_PROPERTIES heap_props{};
        heap_props.Type = type;
        D3D12_RESOURCE_DESC desc{};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = bytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (type == D3D12_HEAP_TYPE_DEFAULT)
            desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        return device->CreateCommittedResource(&heap_props,
            D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
            IID_PPV_ARGS(result->GetAddressOf()));
    }

    bool open() {
        ComPtr<ID3D12Debug> debug;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
            debug->EnableDebugLayer();
            debug_layer = true;
        }
        HRESULT hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0,
            IID_PPV_ARGS(&device));
        if (FAILED(hr)) return false;
        device.As(&info_queue);
        if (info_queue) info_queue->ClearStoredMessages();

        D3D12_DESCRIPTOR_RANGE ranges[2]{};
        ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
        ranges[0].NumDescriptors = kSrvCount;
        ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
        ranges[1].NumDescriptors = kUavCount;
        D3D12_ROOT_PARAMETER parameters[3]{};
        for (uint32_t i = 0; i < 2; ++i) {
            parameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
            parameters[i].DescriptorTable.NumDescriptorRanges = 1;
            parameters[i].DescriptorTable.pDescriptorRanges = &ranges[i];
        }
        parameters[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[2].Constants.ShaderRegister = 0;
        parameters[2].Constants.Num32BitValues = 8;
        D3D12_ROOT_SIGNATURE_DESC root_desc{};
        root_desc.NumParameters = 3;
        root_desc.pParameters = parameters;
        ComPtr<ID3DBlob> root_blob, errors;
        hr = D3D12SerializeRootSignature(&root_desc,
            D3D_ROOT_SIGNATURE_VERSION_1, &root_blob, &errors);
        if (FAILED(hr)) return false;
        hr = device->CreateRootSignature(0, root_blob->GetBufferPointer(),
            root_blob->GetBufferSize(), IID_PPV_ARGS(&root));
        if (FAILED(hr)) return false;
        ComPtr<ID3DBlob> code;
        hr = D3DCompile(fusion_shader, sizeof(fusion_shader) - 1,
            "gpu_multiscale_fusion", nullptr, nullptr, "Reduce", "cs_5_0",
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_IEEE_STRICTNESS,
            0, &code, &errors);
        if (FAILED(hr)) {
            if (errors) std::fprintf(stderr, "%s\n",
                static_cast<const char*>(errors->GetBufferPointer()));
            return false;
        }
        D3D12_COMPUTE_PIPELINE_STATE_DESC pso{};
        pso.pRootSignature = root.Get();
        pso.CS = {code->GetBufferPointer(), code->GetBufferSize()};
        hr = device->CreateComputePipelineState(&pso, IID_PPV_ARGS(&pipeline));
        if (FAILED(hr)) return false;

        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{};
        heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        heap_desc.NumDescriptors = 512;
        heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
        hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
        if (FAILED(hr)) return false;
        descriptor_step = device->GetDescriptorHandleIncrementSize(
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);

        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        hr = device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue));
        if (SUCCEEDED(hr)) hr = device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
        if (SUCCEEDED(hr)) hr = device->CreateCommandList(0,
            D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr,
            IID_PPV_ARGS(&list));
        if (SUCCEEDED(hr)) hr = list->Close();
        if (SUCCEEDED(hr)) hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
            IID_PPV_ARGS(&fence));
        if (FAILED(hr)) return false;
        event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        if (!event) return false;

        hr = make_buffer(kMaxBytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON, &input);
        if (SUCCEEDED(hr)) hr = make_buffer(kMaxBytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON, &scratch_a);
        if (SUCCEEDED(hr)) hr = make_buffer(kMaxBytes, D3D12_HEAP_TYPE_DEFAULT,
            D3D12_RESOURCE_STATE_COMMON, &scratch_b);
        if (SUCCEEDED(hr)) hr = make_buffer(kMaxBytes, D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ, &upload);
        if (SUCCEEDED(hr)) hr = make_buffer(sizeof(float),
            D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST,
            &readback);
        if (FAILED(hr)) return false;
        hr = allocator->Reset();
        if (SUCCEEDED(hr)) hr = list->Reset(allocator.Get(), nullptr);
        if (FAILED(hr)) return false;
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.Get());
        transition(input.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(scratch_a.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        transition(scratch_b.Get(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        return submit_and_wait();
    }

    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }

    void descriptors(ID3D12Resource* source, ID3D12Resource* destination,
                     uint32_t output_count, uint32_t* first_out) {
        const uint32_t first = descriptor_cursor;
        descriptor_cursor += kDescriptors;
        *first_out = first;
        auto cpu = heap->GetCPUDescriptorHandleForHeapStart();
        cpu.ptr += size_t(first) * descriptor_step;

        D3D12_SHADER_RESOURCE_VIEW_DESC raw{};
        raw.Format = DXGI_FORMAT_R32_TYPELESS;
        raw.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
        raw.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        raw.Buffer.NumElements = static_cast<uint32_t>(kMaxCount);
        raw.Buffer.Flags = D3D12_BUFFER_SRV_FLAG_RAW;
        auto at = cpu;
        at.ptr += size_t(5) * descriptor_step;
        device->CreateShaderResourceView(source, &raw, at);

        D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
        uav.Format = DXGI_FORMAT_UNKNOWN;
        uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
        uav.Buffer.NumElements = output_count;
        uav.Buffer.StructureByteStride = sizeof(float);
        at = cpu;
        at.ptr += size_t(kSrvCount + 1) * descriptor_step;
        device->CreateUnorderedAccessView(destination, nullptr, &uav, at);
    }

    bool submit_and_wait() {
        HRESULT hr = list->Close();
        if (FAILED(hr)) return false;
        ID3D12CommandList* commands[] = {list.Get()};
        queue->ExecuteCommandLists(1, commands);
        hr = queue->Signal(fence.Get(), ++fence_value);
        if (FAILED(hr)) return false;
        if (fence->GetCompletedValue() < fence_value) {
            hr = fence->SetEventOnCompletion(fence_value, event);
            if (FAILED(hr) || WaitForSingleObject(event, 30000) != WAIT_OBJECT_0)
                return false;
        }
        hr = allocator->Reset();
        if (SUCCEEDED(hr)) hr = list->Reset(allocator.Get(), nullptr);
        if (FAILED(hr)) return false;
        descriptor_cursor = 0;
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        list->SetDescriptorHeaps(1, heaps);
        list->SetComputeRootSignature(root.Get());
        return true;
    }

    bool run(const std::vector<float>& values, float* result) {
        if (values.empty() || values.size() > kMaxCount) return false;
        void* mapped = nullptr;
        D3D12_RANGE no_read{0, 0};
        HRESULT hr = upload->Map(0, &no_read, &mapped);
        if (FAILED(hr)) return false;
        std::memcpy(mapped, values.data(), values.size() * sizeof(float));
        D3D12_RANGE written{0, values.size() * sizeof(float)};
        upload->Unmap(0, &written);

        list->SetPipelineState(pipeline.Get());
        transition(input.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(input.Get(), 0, upload.Get(), 0,
                               values.size() * sizeof(float));
        transition(input.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        ID3D12Resource* source = input.Get();
        ID3D12Resource* destination = scratch_a.Get();
        uint32_t count = static_cast<uint32_t>(values.size());
        while (count > 1) {
            const uint32_t output_count = (count + 255u) / 256u;
            uint32_t first = 0;
            descriptors(source, destination, output_count, &first);
            transition(source, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            auto gpu = heap->GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += size_t(first) * descriptor_step;
            list->SetComputeRootDescriptorTable(0, gpu);
            gpu.ptr += size_t(kSrvCount) * descriptor_step;
            list->SetComputeRootDescriptorTable(1, gpu);
            Constants constants{count, 1, count, 1, 0, output_count, 0, 0.0f};
            list->SetComputeRoot32BitConstants(2, 8, &constants, 0);
            list->Dispatch(output_count, 1, 1);
            /* This transition orders the SRV read before a later pass may
             * reuse this resource as a UAV destination. */
            transition(source, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            source = destination;
            count = output_count;
            destination = destination == scratch_a.Get()
                ? scratch_b.Get() : scratch_a.Get();
        }
        transition(source, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->CopyBufferRegion(readback.Get(), 0, source, 0, sizeof(float));
        transition(source, D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!submit_and_wait()) return false;
        mapped = nullptr;
        D3D12_RANGE read_range{0, sizeof(float)};
        hr = readback->Map(0, &read_range, &mapped);
        if (FAILED(hr)) return false;
        *result = *static_cast<const float*>(mapped);
        D3D12_RANGE no_write{0, 0};
        readback->Unmap(0, &no_write);
        scalar_readback_bytes += sizeof(float);
        return true;
    }

    bool check_debug_messages() {
        if (!info_queue) return true;
        bool clean = true;
        const UINT64 count = info_queue->GetNumStoredMessagesAllowedByRetrievalFilter();
        for (UINT64 i = 0; i < count; ++i) {
            SIZE_T bytes = 0;
            info_queue->GetMessage(i, nullptr, &bytes);
            std::vector<uint8_t> storage(bytes);
            auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
            if (FAILED(info_queue->GetMessage(i, message, &bytes))) continue;
            if (message->Severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                std::fprintf(stderr, "D3D12 %s: %s\n",
                    message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION
                        ? "CORRUPTION" :
                    message->Severity == D3D12_MESSAGE_SEVERITY_ERROR
                        ? "ERROR" : "WARNING", message->pDescription);
                clean = false;
            }
        }
        return clean;
    }
};

struct Case {
    const char* name;
    std::vector<float> values;
    enum class Expect { finite, nan, positive_infinity, negative_infinity } expect;
};

double compensated_sum(const std::vector<float>& values) {
    double sum = 0.0, correction = 0.0;
    for (float item : values) {
        const double adjusted = double(item) - correction;
        const double next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    return sum;
}

double sum_abs(const std::vector<float>& values) {
    double result = 0.0;
    for (float item : values) result += std::abs(double(item));
    return result;
}

std::vector<float> signed_mix(uint32_t count) {
    std::vector<float> values(count);
    for (uint32_t i = 0; i < count; ++i) {
        const float sign = i & 1u ? -1.0f : 1.0f;
        const float magnitude = 8.0f + float(i % 11u) * 0.25f;
        const float offset = (static_cast<int>(i % 3u) - 1) * 0.0003f;
        values[i] = sign * magnitude + offset;
    }
    return values;
}

bool run_case(Harness& harness, const Case& test) {
    float actual = 0.0f;
    if (!harness.run(test.values, &actual)) {
        std::fprintf(stderr, "[FAIL] %s dispatch/readback\n", test.name);
        return false;
    }
    if (test.expect == Case::Expect::nan) {
        const bool ok = std::isnan(actual);
        std::printf("[%s] %-20s count=%zu result=%s readback=4 bytes\n",
            ok ? " ok " : "FAIL", test.name, test.values.size(),
            std::isnan(actual) ? "NaN" : "not-NaN");
        return ok;
    }
    if (test.expect == Case::Expect::positive_infinity ||
        test.expect == Case::Expect::negative_infinity) {
        const bool positive = test.expect == Case::Expect::positive_infinity;
        const bool ok = std::isinf(actual) && (actual > 0) == positive;
        std::printf("[%s] %-20s count=%zu result=%s readback=4 bytes\n",
            ok ? " ok " : "FAIL", test.name, test.values.size(),
            actual > 0 ? "+Inf" : "-Inf");
        return ok;
    }
    const double expected = compensated_sum(test.values);
    const double error = std::abs(double(actual) - expected);
    const double depth = std::ceil(std::log2(double(test.values.size())));
    /* Pairwise FP32 summation bound, doubled to cover the partial stage plus
     * the GPU reduction tree. This checks the real result against a compensated
     * FP64 sum while allowing the documented FP32 accumulation order. */
    const double bound = (std::max)(1.0e-6,
        2.0 * std::numeric_limits<float>::epsilon() * depth *
        sum_abs(test.values));
    const bool ok = std::isfinite(actual) && error <= bound;
    std::printf("[%s] %-20s count=%zu gpu=%.9g fp64=%.12g abs_error=%.6g bound=%.6g readback=4 bytes\n",
        ok ? " ok " : "FAIL", test.name, test.values.size(),
        actual, expected, error, bound);
    return ok;
}

}  // namespace

int main() {
    Harness harness;
    if (!harness.open()) {
        std::fprintf(stderr, "[FAIL] could not initialize D3D12 reduction probe\n");
        return 2;
    }
    std::printf("D3D12 debug layer: %s\n",
        harness.debug_layer ? "enabled" : "unavailable");
    std::vector<Case> cases;
    cases.push_back({"one", {12.5f}, Case::Expect::finite});
    cases.push_back({"boundary_255", signed_mix(255), Case::Expect::finite});
    cases.push_back({"boundary_256", signed_mix(256), Case::Expect::finite});
    cases.push_back({"boundary_257", signed_mix(257), Case::Expect::finite});
    cases.push_back({"multilevel_65537", signed_mix(65537), Case::Expect::finite});
    std::vector<float> sentinel(257, 0.0f);
    sentinel[0] = 1.0e30f;
    cases.push_back({"fallback_sentinel_1e30", sentinel, Case::Expect::finite});
    std::vector<float> nan_case(257, 0.0f);
    nan_case[17] = std::numeric_limits<float>::quiet_NaN();
    cases.push_back({"nan_propagation", nan_case, Case::Expect::nan});
    std::vector<float> inf_case(257, 0.0f);
    inf_case[19] = std::numeric_limits<float>::infinity();
    cases.push_back({"positive_inf", inf_case, Case::Expect::positive_infinity});
    std::vector<float> negative_inf_case(257, 0.0f);
    negative_inf_case[21] = -std::numeric_limits<float>::infinity();
    cases.push_back({"negative_inf", negative_inf_case,
                     Case::Expect::negative_infinity});

    bool passed = true;
    for (const auto& test : cases) passed = run_case(harness, test) && passed;
    const uint64_t expected_bytes = uint64_t(cases.size()) * sizeof(float);
    const bool bytes_ok = harness.scalar_readback_bytes == expected_bytes;
    std::printf("scalar_readback_bytes=%llu expected=%llu\n",
        static_cast<unsigned long long>(harness.scalar_readback_bytes),
        static_cast<unsigned long long>(expected_bytes));
    passed = bytes_ok && passed;
    if (harness.debug_layer) passed = harness.check_debug_messages() && passed;
    std::printf("%s reduction probe\n", passed ? "[ ok ]" : "[FAIL]");
    return passed ? 0 : 1;
}
