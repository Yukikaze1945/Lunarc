# DLSS5 Standalone V2 现状架构审计报告 (CURRENT_STATE_AUDIT)

**审计基线 Commit**: `5055e0f5e9399d9fd37c55ce0b7fc5febbe29415`  
**测试平台**: Windows 11 x64, NVIDIA GeForce RTX 5060 Ti, CUDA 13.4, Driver 616.92

> **这份文件是那一天的快照，不是现状。** 下面每一条回答的 true/false 只相对于基线 commit 成立，
> 其中多项在此后的轮次里已被修复或替换（例如 item 1-4 的 int16/`cpu_fallback` 残差路径，现在
> 是 native 后端的 FP32 复用路径）。要读当前行为请看来自主干代码的文档：
> `MAGPIE_RESIDUAL_PROVENANCE.md`（残差后端与计数器语义）、`ABI_CAPABILITIES.md`（BLOCKED 项）
> 和 `TESTING.md`。保留原文是因为它是那轮修复的输入证据，不是因为它的结论仍然成立。

---

## 逐项审计与调用链分析 (20 项逐项回答)

### 1. Residual Reconstruction 当前是否仍使用 CPU OpenCV
- **文件**: `pipeline.py`
- **函数**: `export_dlss_stream()` (Line 1947–1956)
- **调用链**: `export_dlss_stream()` -> `cv2.subtract(workspace.dlss_bgr_i16, workspace.down_cur_bgr_i16, dst=workspace.diff_int16)` -> `cv2.resize(...)` -> `cv2.add(...)`
- **数据位置**: CPU Host RAM (NumPy ndarray / OpenCV cv2)
- **真实 Backend**: `residual_backend = "cpu_fallback"` (CPU OpenCV SIMD int16)
- **是否为生产路径**: 是 (正式导出管线主路径)

### 2. Color downsample 当前是否仍使用 `cv2.resize(... INTER_AREA)`
- **文件**: `dlss_service.py` (Line 881) & `pipeline.py` (Line 1895)
- **函数**: `_guidance()` / `export_dlss_stream()`
- **调用链**: `dlss_service._guidance()` -> `cv2.resize(fr, (live_w, live_h), dst=workspace.down_cur_bgr, interpolation=cv2.INTER_AREA)`
- **数据位置**: CPU Host RAM
- **真实 Backend**: CPU OpenCV `cv2.resize(INTER_AREA)`
- **是否为生产路径**: 是

### 3. Residual upscale 当前是否仍使用 `cv2.resize(... INTER_LINEAR)`
- **文件**: `pipeline.py`
- **函数**: `export_dlss_stream()` (Line 1952)
- **调用链**: `export_dlss_stream()` -> `cv2.resize(workspace.diff_int16, (w, h), dst=workspace.diff_up_int16, interpolation=cv2.INTER_LINEAR)`
- **数据位置**: CPU Host RAM
- **真实 Backend**: CPU OpenCV `cv2.resize(INTER_LINEAR)`
- **是否为生产路径**: 是

### 4. Residual 当前中间格式是否仍为 CPU int16
- **文件**: `pipeline.py`
- **函数**: `FrameWorkspace.__init__()` (Line 1706–1707) & `export_dlss_stream()` (Line 1948–1952)
- **调用链**: `workspace.dlss_bgr_i16`, `workspace.down_cur_bgr_i16`, `workspace.diff_int16`, `workspace.diff_up_int16`
- **数据位置**: CPU Host RAM (NumPy `np.int16`)
- **真实格式**: 有符号 CPU `int16` (signed)
- **是否为生产路径**: 是

### 5. Guidance 当前是否先把 Color 缩到 Working Resolution，然后直接在该分辨率运行 NVOF
- **文件**: `dlss_service.py`
- **函数**: `_guidance()` (Line 880–900)
- **调用链**: `_guidance()` -> `cv2.resize(fr, (live_w, live_h), dst=workspace.down_cur_bgr)` -> `guidance_service.get_guidance(cur_bgr=cur_work, target_w=live_w, target_h=live_h)` -> `NvOFHardwareEngine.compute_flow_sequential()`
- **数据位置**: CPU Host RAM -> GPU VRAM (上传 1080p 帧)
- **真实 Backend**: `guidance_backend = "working_resolution_nvof"`
- **是否为生产路径**: 是

### 6. NVOF 输出是否仍通过 D2H 返回 NumPy
- **文件**: `optical_flow.py` (Line 466) & `native/nvof_bridge.cpp` (Line 940–960)
- **函数**: `NvOFHardwareEngine.compute_flow_sequential()` -> `nvof_bridge_execute_sequential()`
- **调用链**: `nvOFExecute()` -> GPU postprocess kernel (S10.5 to float2) -> `pfn_cuMemcpy2D()` (D2H) -> `flow_out` (Host `float32` pointer) -> NumPy ndarray
- **数据位置**: GPU VRAM -> CPU Host RAM
- **真实 Backend**: GPU NVOF + D2H 拷贝至 CPU
- **是否为生产路径**: 是

### 7. `nvof_bridge_get_output_deviceptr()` 是否被任何生产 DLSSNR 路径消费
- **文件**: `optical_flow.py`
- **函数**: `NvOFHardwareEngine.get_output_deviceptr()`
- **调用链**: 仅在 `optical_flow.py` 中暴露了 ctypes 绑定，在 `dlss_service.py`、`dlss_engine.py` 和 `pipeline.py` 中**从未被消费**。
- **数据位置**: N/A
- **原因**: 下游 `dlssnr_host.dll` 的 ABI 仅接收 Host CPU 指针。
- **是否为生产路径**: 否 (未接入生产)

### 8. `dlssnr_host` 是否支持 GPU resource 或 CUDA device pointer
- **文件**: `dlssnr_host.dll` (ABI: `dlssnr_process(void* rgba_in, void* motion_in, void* depth_in, void* rgba_out, int reset)`)
- **分析**: `dlssnr_host.dll` 内部使用 D3D12 Upload Heap 或 Host-mapped memory，将传入的指针作为 CPU 虚拟地址直接解引用并拷贝至 D3D12 Texture 资源。不接受 CUDA `CUdeviceptr` 或外部 D3D12/D3D11 共享句柄。
- **状态**: `DLSSNR_GPU_RESOURCE_INPUT = BLOCKED`
- **是否为生产路径**: 是 (生产当前只支持 Host CPU 内存)

### 9. NVOF 中哪些 `cuCtxSynchronize` / `cuStreamSynchronize` 是实际数据依赖
- **文件**: `native/nvof_bridge.cpp`
- **位置**: `nvof_bridge_execute_sequential()`
- **分析**: 在执行 `cuMemcpy2D` (D2H) 将 flow 读回 CPU Host 之前，必须确保 GPU 上的 NVOF 硬件执行与 postprocess CUDA 核函数完成，此处的 `cuStreamSynchronize(s->stream)` 属于**实际数据依赖 (Data Dependency)**。在 `nvof_bridge_destroy()` 中的 `cuCtxSynchronize()` 属于资源释放依赖。

### 10. 哪些同步只用于 profiler
- **文件**: `native/nvof_bridge.cpp`
- **位置**: `nvof_bridge_execute_timed()`
- **分析**: 在 `nvOFExecute` 之后、postprocess 之前插入的 `cuStreamSynchronize`，如果纯粹是为了在 CPU 端调用 `QueryPerformanceCounter` 分段获取 `timing->execute_ms`，则属于 **Profiler-Induced Synchronization**。

### 11. `last_frame = cur_work.copy()` 是否仍然每帧执行
- **文件**: `dlss_service.py` (Line 899)
- **函数**: `_guidance()`
- **分析**: 是。每一帧结束时仍无条件执行 `last_frame = cur_work.copy()`。但在 NVOF sequential 稳态模式下，GPU 内部已保存历史 ping-pong buffer，CPU 端的每一帧深拷贝是冗余开销。
- **是否为生产路径**: 是

### 12. `FrameWorkspace.full_flow` 是否仍有生产引用
- **文件**: `pipeline.py` (Line 1703)
- **分析**: 在 `FrameWorkspace.__init__()` 中分配了 `(src_h, src_w, 2)` float32 数组 (4K 下约 63 MiB)，但在 `export_dlss_stream()` 生产执行全流程中**没有任何读写引用**。
- **是否为生产路径**: 否 (死内存，应删除)

### 13. `FrameWorkspace.full_depth` 是否仍有生产引用
- **文件**: `pipeline.py` (Line 1704)
- **分析**: 在 `FrameWorkspace.__init__()` 中分配了 `(src_h, src_w)` float32 数组 (4K 下约 32 MiB)，在 `export_dlss_stream()` 生产执行全流程中**没有任何读写引用**。
- **是否为生产路径**: 否 (死内存，应删除)

### 14. `decode_mode=cuda` 是否仍通过 `_supports_av1_nvenc()` 决定 CUDA Decode 能力
- **文件**: `pipeline.py` (Line 655)
- **函数**: `_supports_cuda_decode()`
- **分析**: 部分逻辑曾混用 NVENC 编码器探测推断 NVDEC 解码器能力。需要独立的 CUDA 解码管道探测器（验证 FFmpeg `cuda` hwaccel、`scale_cuda` filter 及实际视频流试解码）。
- **是否为生产路径**: 是

### 15. `test_working_scale_motion_vector_scaling` 是否仍自己复制算法
- **文件**: `tests/unit/test_export_pipeline.py` (Line 67–88)
- **分析**: 是。该测试在函数内直接用 `cv2.resize` 和 `down_flow[..., 0] *= (sw / w)` 重新计算并做 self-assertion，没有调用生产环境的 `FrameGuidanceService` 或 `resolve_working_dims()`。
- **是否为生产路径**: 否 (假测试，需重构)

### 16. `test_residual_pyramid_reconstruction` 是否仍自己复制算法
- **文件**: `tests/unit/test_export_pipeline.py` (Line 90–109)
- **分析**: 是。该测试在函数内直接计算 `diff = dlss - down`，`cv2.resize`，`cv2.add`，没有调用生产代码的 Residual Reconstruction 逻辑。
- **是否为生产路径**: 否 (假测试，需重构)

### 17. `benchmark_4k_scenarios.py` 是否仍自己定义一套 `_guidance()`
- **文件**: `tests/unit/benchmark_4k_scenarios.py` (Line 80–123)
- **分析**: 是。Benchmark 内部自己编写了 `def _guidance(...)`，复制了颜色降采样、guidance service 调用和 RGBA 转换代码。
- **是否为生产路径**: 否 (代码重复，需统一调用生产导出核心)

### 18. Benchmark 文件头是否仍描述已经删除的 2-Pass scenario
- **文件**: `tests/unit/benchmark_4k_scenarios.py` (Line 8–16)
- **分析**: 存在不一致注释，需同步更新为真实的 6 个单 Pass 场景描述。
- **是否为生产路径**: 否

### 19. CPU `cv2.VideoCapture.read()` 卡住时，Cancel 是否能真正中断
- **文件**: `pipeline.py` (`SmartCapture`)
- **分析**: 否。`cv2.VideoCapture.read()` 在底层 C++ 库中同步阻塞，Python 线程设置的 Event 无法强制打断卡死在驱动/C++ 内部的 I/O。正式导出必须由受控的 FFmpeg 子进程提供解码支持，Cancel 时通过强制终止子进程 (`proc.kill()`) 打破管道阻塞。
- **是否为生产路径**: 是

### 20. FFmpeg read/write 卡住时，Cancel 的真正中断机制是什么
- **文件**: `pipeline.py` (`_read_exactly`, `_write_frame_bytes`, `export_dlss_stream`)
- **分析**: 真正的中断机制为：`cancel()` 被触发 -> 调用 `proc.kill()` 强行终止 FFmpeg 子进程 -> OS 管道写/读端失效关闭 -> 阻塞的 `stream.read()` 或 `pipe.write()` 立即抛出 `BrokenPipeError` / 返回空字节 -> 工作线程捕获 `JobCancelled` 并退出 -> 回收子进程，孤儿进程数为 0。
- **是否为生产路径**: 是
