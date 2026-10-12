// Reuse the existing fixture/producer/readback helpers, not a second filter.
#define main hdr_a2_original_main
#include "gpu_multiscale_fusion_a2_test.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    g_dump_dir = argv[2];
    std::filesystem::create_directories(g_dump_dir);
    Fixture fixture;
    if (!load_fixture(argv[1], load_manifest(argv[1]), fixture)) return 2;
    Machine machine;
    if (!machine.open(true) || !g_info_queue) return 2;
    Fusion fusion;
    if (!fusion.load(argv[3])) return 2;
    using Apply = decltype(&gpu_fusion_probe_operator);
    const auto apply = reinterpret_cast<Apply>(GetProcAddress(fusion.lib, "gpu_fusion_probe_operator"));
    if (!apply) return 2;
    GpuFusionBridge* bridge = nullptr;
    const auto options = fusion_options(fixture);
    if (fusion.open_sdr(machine.device.Get(), &options, &bridge)) return 2;
    Harvester harvester;
    if (!harvester.open(machine.device.Get())) return 2;
    for (uint32_t layer = 0; layer < options.layer_count; ++layer) {
        for (uint32_t transpose = 0; transpose < 2; ++transpose) {
            const uint32_t width = transpose ? options.layers[layer].width : options.full_width;
            const uint32_t height = transpose ? options.layers[layer].height : options.full_height;
            std::vector<float> samples(size_t(width)*height*4, 0);
            for (size_t i = 0; i < samples.size(); ++i)
                if (i%4 != 3) samples[i] = static_cast<float>(
                    4*std::sin(double(i)*.037)+2*std::cos(double(i)*.081));
            std::vector<uint8_t> raw(samples.size()*sizeof(float));
            std::memcpy(raw.data(), samples.data(), raw.size());
            const std::string prefix = "layer"+std::to_string(layer)+(transpose ? "_adjoint" : "_forward");
            write_file(prefix+"_input.bin", raw.data(), raw.size());
            Producer producer;
            if (!producer.build(machine.device.Get(), width, height,
                DXGI_FORMAT_R32G32B32A32_FLOAT, raw)) return 2;
            Microsoft::WRL::ComPtr<ID3D12CommandAllocator> allocator;
            Microsoft::WRL::ComPtr<ID3D12GraphicsCommandList> list;
            if (FAILED(machine.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                IID_PPV_ARGS(&allocator))) || FAILED(machine.device->CreateCommandList(0,
                D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.Get(), nullptr, IID_PPV_ARGS(&list)))) return 2;
            D3D12_RESOURCE_BARRIER barrier{};
            barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            barrier.Transition.pResource = producer.texture.Get();
            barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
            barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            list->ResourceBarrier(1, &barrier);
            if (FAILED(list->Close())) return 2;
            ID3D12CommandList* lists[] = {list.Get()};
            producer.queue->ExecuteCommandLists(1, lists);
            producer.queue->Signal(producer.fence.Get(), ++producer.fence_value);
            void* texture = nullptr; void* fence = nullptr; uint64_t value = 0;
            const auto code = apply(bridge, layer, transpose, producer.texture.Get(),
                producer.fence.Get(), producer.fence_value, &texture, &fence, &value);
            if (code) { std::printf("operator failed %d: %s\n", code, fusion.error().c_str()); return 1; }
            std::vector<uint8_t> pixels;
            uint32_t pitch=0, out_width=0, out_height=0;
            DXGI_FORMAT format{};
            if (!harvester.copy(machine.device.Get(), static_cast<ID3D12Resource*>(texture),
                static_cast<ID3D12Fence*>(fence), value, pixels, pitch, format, out_width, out_height)) return 1;
            write_file(prefix+"_output.bin", pixels.data(), pixels.size());
            g_summary.add(prefix+"_width", out_width);
            g_summary.add(prefix+"_height", out_height);
            g_summary.add(prefix+"_pitch", pitch);
            if (g_info_queue) {
                const UINT64 count = g_info_queue->GetNumStoredMessagesAllowedByRetrievalFilter();
                for (UINT64 i=0; i<count; ++i) {
                    SIZE_T size=0; g_info_queue->GetMessage(i,nullptr,&size);
                    std::vector<uint8_t> storage(size);
                    auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());
                    if (SUCCEEDED(g_info_queue->GetMessage(i,message,&size)) &&
                        message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) ++g_failures;
                }
                drain_validation_notes();
            }
        }
    }
    fusion.close(bridge);
    machine.close();
    g_summary.add("failures", uint64_t(g_failures));
    g_summary.add("debug_layer_available", uint64_t(1));
    g_summary.write("summary.txt");
    write_file("notes.txt", g_notes.data(), g_notes.size());
    return g_failures ? 1 : 0;
}
