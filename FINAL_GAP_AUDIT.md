# FINAL_GAP_AUDIT.md — DLSS5 Standalone V2 最终收尾差距审计报告

本报告严格基于当前 HEAD（`5ea5e24af8757e3bf50ea18b76b2aa7123b7585e`）的代码真实行为，逐项回答 A 至 Z 的全部核心问题。

---

## 逐项审计结论与证据

### A. `benchmark_4k_scenarios.py` 是否仍自己定义 `_guidance()`？
- **结论**: **是**。
- **证据**: `tests/unit/benchmark_4k_scenarios.py` 第 80-123 行仍内联定义了私有函数 `def _guidance(idx, fr, workspace=None, is_discontinuous=False)`。

### B. Benchmark 是否仍自己执行 color resize / Guidance / previous-frame 管理 / Depth / BGR->RGBA？
- **结论**: **是**。
- **证据**: 在 `_guidance()` 中：
  - Color resize: `cv2.resize(fr, (live_w, live_h), dst=workspace.down_cur_bgr, ...)` (第 85 行)
  - Guidance: `guidance_svc.get_guidance(...)` (第 93 行)
  - Previous-frame: `last_frame = cur_work.copy()` (第 103 行)
  - Depth: `depth_out = np.zeros((live_h, live_w), dtype=np.float32)` (第 113 行)
  - BGR->RGBA: `cv2.cvtColor(cur_work, cv2.COLOR_BGR2RGBA, dst=workspace.live_rgba)` (第 116 行)
  这些逻辑完全脱离生产核心，是重复维护的私有分支。

### C. Benchmark 的 `guidance_backend` 是否根据 scale 人工推断？
- **结论**: **是**。
- **证据**: `tests/unit/benchmark_4k_scenarios.py` 第 152 行：
  ```python
  "guidance_backend": ("working_resolution_nvof" if (live_w, live_h) != (w, h) else "full_resolution_nvof") if gm > 0 else "disabled"
  ```
  该字段是由 Python 字典根据 `(live_w, live_h) != (w, h)` 手动拼装字符串，而非读取真实 Profiler 捕获的运行记录。

### D. Benchmark 的 `residual_backend` 是否根据 settings 人工推断？
- **结论**: **是**。
- **证据**: `tests/unit/benchmark_4k_scenarios.py` 第 153 行：
  ```python
  "residual_backend": "cpu_fallback" if sc.get("pyramid") else "none"
  ```
  直接根据入参 `sc.get("pyramid")` 人工推断。

### E. `FrameGuidanceService` 是否存在 `full_resolution_nvof_gpu_adapter` 但实际内部仍执行 `cv2.resize` 与 NumPy multiply？
- **结论**: **是**。
- **证据**: `optical_flow.py` 第 1055-1070 行：
  ```python
  timing_dict["guidance_backend"] = "full_resolution_nvof_gpu_adapter"
  ...
  cv2.resize(src_flow, (tw, th), dst=out_flow, interpolation=cv2.INTER_LINEAR)
  out_flow[..., 0] *= (tw / float(src_w))
  out_flow[..., 1] *= (th / float(src_h))
  ```
  实际为 CPU OpenCV 与 NumPy 操作，名称却虚标为 `gpu_adapter`。应修正为 `full_resolution_nvof_cpu_adapter`。

### F. NVOF execute timing 是否仍用 CPU timer start -> nvOFExecute() -> CPU timer end？
- **结论**: **是**。
- **证据**: `native/nvof_bridge.cpp` 第 968-980 行：
  ```cpp
  QueryPerformanceCounter(&t_start);
  NV_OF_STATUS of_res = sess->ofFn.nvOFExecute(sess->hOf, &exec_in, &exec_out);
  ...
  QueryPerformanceCounter(&t_end);
  sess->last_timing.execute_ms = (double)(t_end.QuadPart - t_start.QuadPart) * inv_freq_ms;
  ```

### G. 这个值是否仍被叫 `nvof_execute_ms`？
- **结论**: **是**。
- **证据**: `native/nvof_bridge.cpp` 中 `execute_ms`（在 Python 封装和 Profiler 中被映射为 `nvof_execute_ms`）。由于 `nvOFExecute()` 在底层驱动中为异步提交，该耗时实际上是 CPU 提交耗时（`nvof_submit_cpu_ms`），并非 GPU 硬件执行完成时间。

### H. postprocess/upscale timing 是否包含等待前序 NVOF 完成的时间？
- **结论**: **是**。
- **证据**: 在 `native/nvof_bridge.cpp` 第 983-1022 行，调用 `pfn_cuLaunchKernel` 并执行 `pfn_cuStreamSynchronize(sess->stream)`。由于 NVOF 执行和 CUDA postprocess 核函数位于同一 CUDA 上下文/流依赖链中，流同步 `pfn_cuStreamSynchronize` 会阻塞等待前序硬件光流完成，因此 `upscale_ms` 测量中包含了流依赖等待时间。

### I. `test_cancellation_lifecycle_and_process_cleanup` 是否真的启动 Child Process？
- **结论**: **否**。
- **证据**: `tests/unit/test_export_pipeline.py` 第 262-277 行使用 `MockPipe()` 并在内存中将 `cancelled = True` 传入 `_write_frame_bytes`，未生成真实的 `subprocess.Popen`。

### J. 取消测试是否真的制造 blocking read 或 blocking write？
- **结论**: **否**。
- **证据**: 仅在写入前检查了 `cancel_fn()` 并抛出 `JobCancelled`，没有真正的阻塞 I/O 挂起场景。

### K. 取消测试是否检查真实 PID 退出？
- **结论**: **否**。
- **证据**: 当前测试无 PID 追踪与 `proc.poll()` / `proc.kill()` 状态断言。

### L. 正式 Export 的 `decode_mode=cpu` 是否仍可能进入 `cv2.VideoCapture.read()`？
- **结论**: **是（历史代码残留）**。
- **证据**: `pipeline.py` 在部分回退和旧分支中仍使用 `SmartCapture` / `cv2.VideoCapture`，无法通过外部进程 kill 强制中断阻塞。正式 Export sequential decode 必须统一由拥有一级生命周期的 FFmpeg 子进程管道读取。

### M. `test_cuda_decode_probe` 是否只验证返回类型是 bool 和 str？
- **结论**: **是**。
- **证据**: `tests/unit/test_export_pipeline.py` 第 213-218 行：
  ```python
  is_supported, reason = pipeline.probe_cuda_decode_capability()
  assert isinstance(is_supported, bool)
  assert isinstance(reason, str) and len(reason) > 0
  ```
  完全没有使用真实视频流验证 `decode_actual` 和真实帧解码行为。

### N. `FrameGuidanceService` 的 `_last_src_frame` 是否引用一个会在下一帧被覆盖的 Workspace Buffer？
- **结论**: **是（如果外部传入 workspace.down_cur_bgr）**。
- **证据**: `optical_flow.py` 第 1044 行：`self._last_src_frame = cur_bgr`。如果调用方传入的是复用的 `workspace.down_cur_bgr`，下一帧写入时将原地修改同一 ndarray，导致 `_last_src_frame` 与下一帧的 `cur_bgr` 指向相同底层内存（指针 Alias）。

### O. DIS 是否需要真实 CPU previous frame？
- **结论**: **是**。
- **证据**: OpenCV DIS 光流算法必须接收两个独立的 CPU 灰度/BGR 图像 `(prev, cur)` 进行梯度和逆搜索计算。若两者发生内存 Alias，DIS 计算出的光流将错误为零向量。

### P. RAFT 是否需要真实 CPU previous frame？
- **结论**: **是**。
- **证据**: RAFT 深度模型需要将两帧独立的 Tensor `(image1, image2)` 输入神经网络进行相关层计算。

### Q. NVOF sequential 是否使用 Native GPU History？
- **结论**: **是**。
- **证据**: `native/nvof_bridge.cpp` 维护了 `sess->hIn` 和 `sess->hRef` 两个 GPU Ping-Pong 硬件 Buffer。在连续稳态（`reset_temporal_hints = false`）时，只需上传 1 张当前帧到 GPU，内部自动完成 Ping-Pong 切换，CPU 不需要额外缓存或拷贝上一帧。

### R. Residual source downsample 当前使用什么 Filter？
- **结论**: `cv2.INTER_AREA`。
- **证据**: `pipeline.py` 第 1972 行：`cv2.resize(fr, (live_w, live_h), dst=workspace.down_cur_bgr, interpolation=cv2.INTER_AREA)`。

### S. Residual upscale 当前使用什么 Filter？
- **结论**: `cv2.INTER_LINEAR`。
- **证据**: `pipeline.py` 第 1783 行：`cv2.resize(workspace.diff_int16, (src_w, src_h), dst=workspace.diff_up_int16, interpolation=cv2.INTER_LINEAR)`。

### T. Residual 中间格式是什么？
- **结论**: `np.int16`。
- **证据**: `pipeline.py` 第 1780-1788 行使用 `int16` 进行差值与相加。

### U. `ABI_CAPABILITIES.md` 中哪些结论来自真实源码？
- **结论**:
  - `nvof_bridge.cpp` / `nvof_bridge.h` 内部全部内存管理、CUDA PTX 核函数、Ping-Pong 硬件 Buffer 与 Counter 逻辑。
  - `pipeline.py` / `dlss_engine.py` / `optical_flow.py` 的数据调用与内存流转。

### V. 哪些结论来自 DLL Export？
- **结论**:
  - `dlssnr_host.dll` 导出的 6 个 C 符号（`dlssnr_init`, `dlssnr_create_feature`, `dlssnr_process`, `dlssnr_resize`, `dlssnr_set_options`, `dlssnr_shutdown`）。

### W. 哪些结论来自运行时实验？
- **结论**:
  - 向 `dlssnr_process` 传入 GPU 设备指针引发非法内存访问 (0xC0000005)。
  - 并发创建第二个 `dlssnr_init` 会覆盖前序会话状态。
  - NVOF 在 4K 下与 1080p 下的 PCIe 上传/下载实测带宽与耗时。

### X. 哪些结论来自 Binary Analysis？
- **结论**:
  - `dlssnr_host.dll` 导入了 `d3d12.dll`、`dxgi.dll`、`nvngx_dlssnr.dll`，无 CUDA/EGL 互操作导出。

### Y. 哪些结论只是推断？
- **结论**:
  - `dlssnr_host.dll` 内部具体如何创建 D3D12 Upload/Readback Heap、内部 CommandList 的提交时机为 **INFERRED**（因为没有 host dll 的 C++ 源码）。

### Z. 当前生产 Depth 路径究竟是什么？
- **结论**: **全零数组占位 (Zero Float32 Depth)**。
- **证据**:
  - `pipeline.py` 第 1978 行：`workspace.live_depth.fill(0)`
  - `optical_flow.py` 第 1081 行：`depth = self.get_zero_depth(tw, th)`
  - 代码库中不存在 `cv2.cuda`、`bilateralFilter` 或 `GPU Depth`。之前出现的任何此类描述均为无源码支持的虚构描述，在本轮中彻底纠正。

---
