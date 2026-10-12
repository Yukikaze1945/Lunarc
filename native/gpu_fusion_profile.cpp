// Synthetic solver profiling. Fixture uploads and readbacks are outside timing.
#define main hdr_a2_original_main
#include "gpu_multiscale_fusion_a2_test.cpp"
#undef main
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    g_dump_dir = argv[2];
    std::filesystem::create_directories(g_dump_dir);
    Fixture fixture;
    if (!load_fixture(argv[1], load_manifest(argv[1]), fixture)) return 2;
    Machine machine;
    if (!machine.open(false)) return 2;
    Fusion fusion;
    if (!fusion.load(argv[3])) return 2;
    const auto diagnostics = symbol<decltype(&gpu_fusion_get_solver_diagnostics)>(
        fusion.lib, "gpu_fusion_get_solver_diagnostics");
    if (!diagnostics) return 2;
    const auto options = fusion_options(fixture);
    const GpuFusionHdrOptions hdr{sizeof(GpuFusionHdrOptions), GPU_FUSION_HDR_ABI_VERSION, 8u, 0u};
    GpuFusionBridge* bridge = nullptr;
    if (fusion.open_hdr(machine.device.Get(), &options, &hdr, &bridge)) return 2;
    DeviceBuffer master;
    if (!master.upload(machine.cuda, fixture.master.data(), fixture.master.size()*2,
        fixture.full_width*8, fixture.full_width, fixture.full_height)) return 2;
    std::vector<DeviceBuffer> residuals(fixture.layers.size());
    std::vector<GpuFusionHdrObservation> observations(fixture.layers.size());
    for (size_t i=0; i<fixture.layers.size(); ++i) {
        const auto& layer = fixture.layers[i];
        if (!residuals[i].upload(machine.cuda, layer.samples.data(), layer.samples.size()*2,
            layer.width*8, layer.width, layer.height)) return 2;
        observations[i] = {uint64_t(residuals[i].ptr), residuals[i].pitch, layer.width, layer.height};
    }
    HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event) return 2;
    for (uint32_t frame=0; frame<4; ++frame) {
        GpuFusionHdrReport report{};
        report.struct_size=sizeof(report); report.abi_version=GPU_FUSION_HDR_ABI_VERSION;
        void* texture=nullptr; void* fence=nullptr; uint64_t value=0;
        const auto start=std::chrono::steady_clock::now();
        const int code=fusion.process_hdr(bridge,machine.context,uint64_t(master.ptr),
            master.pitch, observations.data(), uint32_t(observations.size()),
            &texture,&fence,&value,&report);
        if (code) { std::printf("solve %d: %s\n",code,fusion.error().c_str()); return 1; }
        auto* completion=static_cast<ID3D12Fence*>(fence);
        if (completion->GetCompletedValue()<value) {
            if (FAILED(completion->SetEventOnCompletion(value,event)) ||
                WaitForSingleObject(event,30000)!=WAIT_OBJECT_0) return 1;
        }
        const double elapsed=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        GpuFusionSolverDiagnostics diag{}; diag.struct_size=sizeof(diag);
        if (diagnostics(bridge,&diag) || !report.converged) return 1;
        const std::string prefix="frame"+std::to_string(frame)+"_";
        g_summary.add(prefix+"ms",elapsed);
        g_summary.add(prefix+"iterations",report.iterations);
        g_summary.add(prefix+"residual",double(report.relative_residual));
        g_summary.add(prefix+"wait_ms",diag.cpu_wait_ms);
        g_summary.add(prefix+"sync",diag.scalar_sync_count);
        g_summary.add("workspace_logical_bytes",diag.workspace_logical_bytes);
        g_summary.add("workspace_committed_bytes",diag.workspace_committed_bytes);
        g_summary.add("solver_path",diag.solver_path);
        if (frame==3) {
            Harvester readback;
            if (!readback.open(machine.device.Get())) return 1;
            std::vector<uint8_t> pixels; uint32_t pitch=0,width=0,height=0; DXGI_FORMAT format{};
            if (!readback.copy(machine.device.Get(),static_cast<ID3D12Resource*>(texture),
                completion,value,pixels,pitch,format,width,height)) return 1;
            write_file("output.bin",pixels.data(),pixels.size());
            g_summary.add("output_pitch",pitch);
        }
    }
    g_summary.write("summary.txt");
    CloseHandle(event);
    fusion.close(bridge);
    master.free(machine.cuda);
    for (auto& residual:residuals) residual.free(machine.cuda);
    machine.close();
    return 0;
}
