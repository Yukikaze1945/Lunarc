# AMD 光流性能 / 质量档位补齐

## 原因与范围

上一轮把“AMD SDK 没有质量预设参数”误解为“不应该提供质量选择”。Magpie Experimental 0.6.9 的两个应用档位实际选择光流分析尺寸：性能为半宽半高，质量为完整尺寸。

参考语义：[Magpie AmdOpticalFlowProvider，固定版本](https://github.com/SAOG0721/Magpie/blob/27c5df91177a29b33be612e98274169f3d2fca49/src/Magpie.Core/AmdOpticalFlowProvider.cpp#L263)。只参考档位和尺寸含义；本项目继续使用已接入的官方 SDK，新增桥接着色器独立实现，不复制 Magpie 算法代码。

## 实现

- Swiss UI 增加 AMD 专用“引擎质量”：性能 / 质量，以及独立问号说明。切换到 AMD 默认质量，已有 `1F` 设置兼容为此前的全尺寸处理。
- 原生桥新增 `ffx_flow_open2`，保留旧入口与 ABI。性能档在 D3D12 中平均缩小运动引导输入，使用半尺寸 SDK history/grid；质量档保持全尺寸分析。输出始终为完整输入尺寸的 CUDA float2，X/Y 位移分别按实际分析尺寸还原到输入像素单位。
- 每层独立 provider 继承所选档位。预览切档重建相应历史，不复用旧档位显示缓存；导出通过同一 provider 工厂传入设置。
- 不改变 HDR/SDR 主色、NR 数学、P010/PQ 或 remux。RGBA8 仅为既有运动引导。不存在用其他引擎或零流代替失败；旧 DLL 不支持新性能入口时明确报错。

## 当前源码验证

- Native `/W4` 编译通过；DLL 已更新。
- 两档各 9 次真实 SDK dispatch，含启动预热、平移、reset。640×480 输入：性能分析 320×240、网格 40×30；质量分析 640×480、网格 80×60。两档完整输出，实测中心中位运动均为 (-8,-4)，对应输入平移 (+8,+4)。证据：`bench_out/amd_quality_20261010/{performance,quality}/fidelityfx_flow_native_gate.json`。
- 性能档 single / dual 50/75 / triple 40/67/85：各 9 帧真实预览和 GPU Transfer 导出通过；各层实际 SDK 网格与半尺寸一致，独立 sessions/history，双/三层 solver 每帧收敛。最终 640×480 视频各 9 packets，PCM 音轨 packet payload SHA256 一致。证据：`bench_out/amd_quality_20261010/pipeline_performance/report.json`。
- 当前 Swiss Electron、独立测试服务器：操作性能 → 质量 → 性能，实际收到并显示确认对应 AMD motion 诊断，分析尺寸 320×240 → 640×480 → 320×240，无错误标识。证据：`bench_out/amd_quality_20261010/ui/report.json`。
- 本轮相关 Python 回归 **23 passed**；参数说明 Node **3 passed**；TypeScript/Vite 构建与 diff whitespace 检查通过。本轮没有重跑全单元套件或 HDR/remux 成功 gate。

## 限制

性能档减少光流分析像素，不保证所有素材速度或画质优势；本轮未进行性能对比 benchmark。每次质量切换需重新预热前六帧。既有设备复制、同步和 12-byte SCD scalar readback 保留；诊断光流视图有明确的 host readback，不能声明总 host traffic 为零。

已有运行实例需要重新启动前后端才能加载新源码/DLL；独立测试实例已关闭。未提交、未推送，保留其他工作树改动。
