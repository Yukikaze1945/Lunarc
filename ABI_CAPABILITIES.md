# DLSS5 Standalone V2 原生 ABI 与 GPU 资源能力报告 (ABI_CAPABILITIES)

**评估对象**: `dlssnr_host.dll` (Feature 18 / DLSS5 Neural Reconstruction Host)  
**依赖运行时**: `nvngx_dlssnr.dll` (NVIDIA NGX Core Runtime)  
**评估基线**: 2026-09-19

---

## 0. 本文件的历史身份（先读这一段）

本文是按 2026-09-19 的 ABI 现场写的**历史评估**，评估对象 `dlssnr_host.dll` 连同它的 ctypes 封装
`dlss_engine.py` 已从仓库移除（`pipeline.RETIRED_DLSS_BACKENDS == ("host_pointer",)`，
`pipeline.DLSS_BACKENDS == ("gpu_flow_d3d12",)`）。正文里的现在时句子描述的是**那座已不存在的宿主**，
请按历史读数保留，不要当作当前能力清单。当前能力以 `GPU_PIPELINE_PHASE2.md` 与代码为准。

其中三条 BLOCKED 已被后来的代码回答，答案是：

| 本文条目 | 现在的状态 | 由什么回答 |
| :--- | :--- | :--- |
| 1. `DLSSNR_GPU_RESOURCE_INPUT` | 已不再适用 | 自研 `native/dlss_gpu_host.dll` 有设备资源入口：`process_device_flow` 收 D3D12 纹理 + CUDA 描述符，`process_zero_motion` 用宿主常驻零纹理。第 1 节关于 `dlssnr_process(CUdeviceptr)` 触发 0xC0000005 的记录仍然是真的，只是那条路不再有人走。 |
| 2. NVOF→DLSS 的运动交接（下文第 2 条） | 已由设备描述符交接回答 | NVOF 的显存输出以借用的设备描述符直接交给宿主：帧记录 `flow_transport="nvof_device_descriptor"`、`dlss_entry="process_device_flow"`，链路上调的是 `process_device_color_flow_resource`。这条路由上没有任何把 motion 回传到主机内存的调用，自检产物另有一项 `zero:dlss_output_readback_bytes_measured` 的实测零值可查（那是 DLSS 输出侧）。描述符在下一次 NVOF execute 前有效，因此 flow→DLSS 必须原子。 |
| 3. `DLSSNR_CONCURRENT_MULTI_SESSION` | 仍然成立，但行为变了 | 自研宿主也是进程内单例会话（`native/dlss_gpu_host.cpp:611` 的 `active.exchange(true)` 守卫），只是第二次 open 被**明确拒绝**（返回码 2）而不是静默销毁前一个会话。生产仍限 1-Pass：`multi_pass != 1` 是 `gpu_transfer_export.negotiate` 的拒绝理由之一。 |
| 4. `GPU_RESIDUAL_FULL_CHAIN` | 已实现，但默认不启用 | opt-in 的 `export_backend="gpu_transfer"` 把解码/缩放/制导/残差/编码留在显存里（`gpu_transfer_export.py` + 五个桥）。默认仍是 FFmpeg 管道链路；`auto`/`legacy` 都只表示「没点名」。本机通过记录（截至 2026-09-28 的自检产物）：**只有制导档** —— 5 次 1280×720、4~6 帧的打包版运行，`gpu_transfer_dlss_flow_calls` 等于帧数、`zero_motion_calls` 为 0、`dlss_output_readback_bytes_measured=0`、`result=GPU_TRANSFER_OK`；零运动档经由这条链路的记录还没有（那座宿主的 `process_zero_motion` 只在 native 自检里跑过，不在导出链上）。**没有 4K 规模的记录**，且这些自检的判据是计数器与形状，不含画面参考正确性——NGX MVec 的符号与 y 轴方向仍未被任何一次比对校验。 |

第 2 节表格里那句「Production export has not selected those entry points」是同一历史现场的产物：它
描述的是管道链路，不是 `gpu_transfer`。

---

## 1. 原生导出函数与 ABI 原型分析

`dlssnr_host.dll` 导出的全部符号列表：
1. `dlssnr_init(int width, int height, int preset, const wchar_t* dll_path, const wchar_t* log_path) -> int`
2. `dlssnr_create_feature(int width, int height, int preset) -> int`
3. `dlssnr_process(void* rgba_in, void* motion_in, void* depth_in, void* rgba_out, int reset) -> int`
4. `dlssnr_resize(int new_w, int new_h) -> int`
5. `dlssnr_set_options(int preset, int style, float intensity, float local_tone, float local_struct, float skin_struct, int use_auto_mask, int ui_correction, int guidance_mode, int depth_convention, float motion_scale_x, float motion_scale_y) -> void`
6. `dlssnr_shutdown() -> void`

The legacy DLL ABI above is immutable in this repository. The current Python
wrapper no longer exposes real depth or a depth-convention setting; it passes a
session-owned zero plane and the fixed legacy convention value `2`. The
project-owned `dlss_gpu_host.dll` instead binds a persistent zero GPU texture
to `DLSSNR.Depth` and performs no per-frame depth upload.

---

## 2. 资源支持矩阵 (Resource Capability Matrix)

| Resource | CPU Pointer (Host) | CUDA Device Pointer | D3D11 Texture | D3D12 Resource | Shared Handle | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Color Input (RGBA8)** | **Supported** | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** | 内部通过 D3D12 Upload Heap 或 Map 将 Host 字节拷贝至 D3D12 纹理。 |
| **Motion Input (Float2)** | **Supported** | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** | 仅接收 Host CPU float32 指针，底层进行 CPU->GPU 拷贝。 |
| **Depth Input (Float1)** | **Supported** | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** | 仅接收 Host CPU float32 指针。 |
| **Color Output (RGBA8)** | **Supported** | **BLOCKED** | **BLOCKED** | **BLOCKED** | **BLOCKED** | D3D12 Readback Heap 回读至 Host CPU 缓冲区。 |

This matrix describes only the immutable legacy `dlssnr_host.dll` ABI. The
project-owned `dlss_gpu_host.dll` has a separate additive CUDA RGBA color entry
point, which imports a shared D3D12 buffer and waits on a CUDA-produced fence.
It also has a device-output entry point that returns a borrowed D3D12 RGBA8
texture and a completed fence value. Its short native smoke matches the host
output byte-for-byte after an explicitly counted readback; processing through
the device-output entry point itself performs no output D2H transfer. Production
export has not selected those entry points, and it still reads DLSS output back
to CPU memory for residual reconstruction.

`dlss_gpu_host.dll` also has an additive host-motion entry point,
`dlss_gpu_host_process_host_motion`, at the same ABI 3 (capability gated by symbol
presence, not a version bump): host RGBA8 in, a host `float32x2` motion plane in
(the same H×W×2 pixel-space plane the legacy `dlssnr_host.dll` `Live.process` took,
at `width*8` row pitch), host RGBA8 out, depth still the resident zero texture. The
`float32→float16` narrowing happens in a compute pass identical to the guided NVOF
conversion, not in numpy, so the card's single-rounded cast — and the bound
`DXGI_FORMAT_R16G16_FLOAT` MVec resource — decide the bytes, and a host frame and a
flow frame hand DLSS the same motion texture for the same vectors. It answers the
`NVOF_TO_DLSS_ZERO_COPY` constraint above for the CPU-pointer case, not by making
NVOF flow zero-copy but by giving a frame whose motion already lives in host memory
a GPU-host entry that does not load CUDA or an NVOF session at all. An explicit
`dlss_gpu_host_readback_motion` copies the bound half2 plane back for a correctness
gate; it is not on any frame path. A hardware gate proves the plane reaches the bound
resource byte for byte and that each reset frame is byte-identical to the legacy host
for the same input; a CPU-only gate proves the wrapper hands the `float32` plane over
uncast, refuses malformed planes, and refuses an old host that exposes no such symbol.
This entry point is added, validated and now routed for one caller: a preview under
`dlss_backend=gpu_flow_d3d12` with `guidance_mode=1` runs on it
(`_preview_live_kind` → `gpu_host_motion`), gated on `supports_host_motion()` so an
older DLL fails visibly instead of falling back. No export caller was moved onto it.
The clause that followed — that `dlssnr_host.dll` still served every guided host-pointer
frame — was true at the baseline and is not true now: that host and its `host_pointer`
setting were removed from this repository (§0), so no frame is drawn by it. Performance is
`NOT_CHARACTERIZED`. Magpie's `DLSSNRFilter` was read only as a
convention reference; its GPLv3 implementation was not copied into this tree.

`gpu_residual_bridge.dll` is a separate opt-in 2x GPU residual implementation.
Its CUDA full/working color inputs and borrowed D3D12 DLSS output produce a
full-size D3D12 RGBA8 texture. A short hardware gate matches the CPU residual
byte for byte for strengths 1.0 and 0.5 on identical inputs.
`gpu_lanczos_bridge.dll` now prepares 50% working color from full-size CUDA
RGBA8, matching the native CPU Lanczos-2 result byte for byte on two fixture
frames. A two-frame isolated chain through NVOF, DLSS and GPU residual also
matches CPU residual output on the second frame. `nv12_rgba_bridge` ABI 3 and
`gpu_lanczos_bridge` ABI 2 add transfer-counter getters
(`Nv12RgbaTransferCounters`, `GpuLanczosTransferCounters`) whose fields are
written only by calls that returned success, so the export report reads the
conversion and working-color device writes as measured payload figures; neither
bridge observes reads, cache-line traffic or PCIe traffic, and no such figure is
claimed. These bridges, and the decoder, encoder and residual bridges, are the
stages of the opt-in `export_backend="gpu_transfer"` path
(`GPU_PIPELINE_PHASE2.md`); the legacy FFmpeg pipes remain the default.

`gpu_lanczos_bridge` ABI 2 and `gpu_residual_bridge` ABI 1 additionally export
`gpu_lanczos_open_scaled` / `gpu_residual_open_scaled`, which take the resolved
working extents and a `GpuLanczosTapTables` / `GpuResidualTapTable` (one
layout, serialized from `pipeline._build_lanczos2_axis_plan` and
`pipeline._build_catmull_rom_axis_plan`) so either resampler can run at a scale
other than 2x; the plan window is `GPU_LANCZOS_MAX_TAPS` 24 and
`GPU_RESIDUAL_MAX_TAPS` 4 respectively, and a plan that needs more is refused
rather than truncated. `gpu_lanczos_bridge` also exports
`gpu_lanczos_max_taps`, which returns that figure from the DLL that is actually
loaded, so a stale 16-wide build is judged by what it walks instead of by what
this checkout's header says; `load_bridges` records it as the
`lanczos_max_taps` capability. A build older than the getter answers nothing, and
that build is a real install: it added `gpu_lanczos_open_scaled` under the ABI 2
it had already published and took the getter with the widening, so negotiation
measures it at the 16 its own shader was compiled with (the `#define TAPS 16u` of
the DLL `dist/` still ships, mirrored as
`gpu_transfer_export.LEGACY_LANCZOS_MAX_TAPS`) rather than at the 24 of this
checkout's header - a 17..21 tap plan is then refused before the first frame
instead of after it. The residual figure needs no such query: 4 is
the width of a Catmull-Rom row. The exports are new symbols rather than changed
signatures: an existing caller is unaffected, and `load_bridges` records the
presence of each scaled open as the `lanczos_scaled` / `residual_scaled`
capability that `negotiate` requires before it will serve a plan-driven
`working_scale` (every whole percentage from 25% to 99%; 50% on a source it
halves exactly and 100% run on the shapes that need no plan), so a stale DLL
still loads and still serves 0.5 and 1.0 instead of failing to open. The
tap-table upload is filter coefficients and is not counted as frame traffic by
either counter struct.

---

## 3. 约束与 BLOCKED 项清单

1. **`DLSSNR_GPU_RESOURCE_INPUT = BLOCKED`**  
   - `dlssnr_host.dll` 不支持直接传入 CUDA `CUdeviceptr` 或外部 D3D12/D3D11 资源。
   - 任何试图直接将 `nvof_bridge_get_output_deviceptr()` 传入 `dlssnr_process()` 的操作均会导致非法内存访问 (Access Violation 0xC0000005)。

2. **`NVOF_TO_DLSS_ZERO_COPY = BLOCKED`**  
   - 由于 DLSSNR 必须从 Host 内存读取 Motion，NVOF 的 GPU 光流结果必须经过 D2H 回传至 CPU NumPy。
   - 当前已实施的最优方案：在 `working_scale < 1.0` 时直接在 Working Resolution 执行 NVOF 并下载对应尺寸的 Flow，将 D2H 回传量降低 75%（4K 66MB -> 1080p 16MB）。

3. **`DLSSNR_CONCURRENT_MULTI_SESSION = BLOCKED`**  
   - `dlssnr_host.dll` 内部维护全局单例 D3D12 状态，未导出 Session Handle。创建新会话会强制销毁前序会话。
   - 生产环境诚实限制为 1-Pass，UI/测试禁用 2/3 Pass。

4. **`GPU_RESIDUAL_FULL_CHAIN = BLOCKED`**  
   - 由于 DLSSNR 输出在 Host 内存，若把残差重构搬到 GPU，就要先把 4K 全尺寸帧上传 (H2D)、重构完再下载 (D2H) 交给 FFmpeg 编码器，多出的两次 PCIe 往返是纯开销。
   - 上一句是量级推理，不是实测：本仓库没有对这条 GPU 路线做过 A/B 测量，因此不给具体毫秒数。
   - 正式导出默认使用 native 后端 `residual_backend_actual = "cpu_reference_native"`：滤镜 plan、FP32 scratch 与行线程池在会话内复用，逐帧不再分配（见 `MAGPIE_RESIDUAL_PROVENANCE.md`）。它仍然是一条 CPU 内存路径，不是 GPU 直通。
