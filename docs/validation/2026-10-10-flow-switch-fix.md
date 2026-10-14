# 光流引擎切换与质量说明修复

后续更正：本报告记录当时未提供 AMD 质量选择的状态。现已补齐性能 / 质量两档，实际分析尺寸与验收见 [AMD 质量补齐](2026-10-10-amd-flow-quality.md)；此前“不存在质量档位”的判断不完整。

## 已复现的问题

1. 原生预览的“光流”视图仍调用旧 `get_optical_flow_engine` 工厂。该工厂没有 FidelityFX，产生 `Unknown optical flow engine: 'fidelityfx'`。
2. 旧 NVOF 创建后将其独立 CUDA 上下文留在调用线程。CPU 图像接口的 RAFT 仍在该上下文中分配 GPU 模型和张量。真实复现：NVOF 创建 → RAFT 推理成功 → NVOF 销毁 → RAFT 再推理报 `CUDA error: an illegal memory access`。
3. 共用的质量问号说明写入了 AMD 固定网格文字，因而显示在 RAFT 中；参数变更后旧错误也会停留到下一帧成功。

## 修复

- `dlss_service.py` 将 NVOF、FidelityFX、RAFT 的光流视图接到现有 `GpuPreview`，与增强视图共用原生会话所有权，不再另外创建旧光流引擎。
- `gpu_preview_engine.py` 在会话线程同步读取实际使用的 float2 运动资源并着色；诊断视图需要运动时，只修改本次设置快照。光流视图与画面视图的显示缓存分开判定。
- 该诊断视图的光流回读明确计入 `flow_diagnostic_readback_bytes`；生产光流计算与导出保持设备路径。没有以零流或其他引擎替代失败。
- `torch_cuda_context.py` 为兼容 RAFT 的模型初始化、预处理和推理显式保留/进入 PyTorch 主上下文，再恢复原生调用者的上下文；PCI 身份用于解析物理 GPU。
- RAFT、DIS 质量说明独立。AMD 保留固定网格说明，不显示不存在的质量档位。参数请求更新时清除已过期的预览错误；当前请求的新错误仍显示。

## 验收

- 原 CUDA 复现流程修复后通过：NVOF 销毁后 RAFT 再次推理成功。
- 同一新进程、4K SDR10、100% 工作分辨率：NVOF → RAFT → FidelityFX → RAFT → NVOF → FidelityFX → RAFT。每次使用 0/1/2 三帧，前两帧光流视图、第三帧增强视图，共 21 帧通过。RAFT 使用长边 1080、6 次迭代；连续帧实际复用上一帧张量。
- 当前源码构建的 Swiss Electron：在光流视图中操作下拉框切换 AMD / RAFT / NVOF，等待真实显示确认；无错误标识。RAFT 问号说明包含 RAFT 预设且不包含 AMD 网格；AMD 没有质量选择器。
- Python 全单元 **1531 passed**；Node **60 passed**；TypeScript/Vite 构建通过。新增上下文异常恢复、CPU 无 CUDA 操作、原生光流视图路由和显示缓存隔离回归检查。

原始证据位于本机忽略提交的 `bench_out/flow_switch_20261010/`：

- `legacy_context_before.txt` / `legacy_context_after.txt`
- `hardware.txt` / `hardware4k.txt` / `hardware.json`
- `ui/report.json` / `ui_final.txt` / `ui/raft_help_fixed.png`
- `unit.txt` / `frontend_tests.txt` / `frontend_build.txt`

已经遭遇非法访问的旧 CUDA 进程不能通过切换引擎可靠修复；需要重启当前源码版程序加载修复。未重置 CUDA 驱动，未终止用户现有后端，未修改用户的参数或队列。未提交或推送。
