# 按实际用途降低显存申请

状态：方案，等待用户确认。没有实施生产代码变更。

## 目标与边界

针对本机当前源码的 4K SDR10、DLAA → NR → TrueHDR、NVOF 引导和静态低频时间稳定路径，优先消除不参与计算的 GPU 分配。
保留分辨率、模型预设、NR/色彩参数、FP16 图片与 FP32 求解/残差计算、光流和时域算法、渲染顺序、P010/色彩/元数据/封装行为。
任何速度或质量取舍在实施前单独向用户确认。此轮只审查代码与现有日志，没有再次启动模型测试。

## 当前证据

- 引用 ChatGPT 对话的审查基于公开 main 31cfd78，不能直接代表当前本机工作树。
- 本机 `dlss_gpu_host.cpp:prepare_textures/process_impl` 已将 CPU color/flow staging 和 readback 改为相应入口首次使用时分配。`motion_stage` 仅在 `host_motion` 非空时创建；GPU 光流路径已没有这一项新增节省。
- `gpu_rgba_exchange.cpp:gpu_rgba_exchange_open2` 仍总是创建输出纹理；DLAA `OutputEffect.input.color()`、当前 TrueHDR 的 D3D 输入 `input.color()` 不使用它。
- `gpu_hdr_scrgb_bridge.cpp:layout_planes/layout_passes/make_heap` 仍创建完整桥的资源与全部 pass 描述符，不区分 GuidanceOnly/DecodeOnly/EncodeOnly。
- 当前 DLAA guidance 使用 `import_master → downscale → proxy`；TrueHDR 编码 Tail 使用 `import_master → set_composite → encode`。两者没有使用全部平面。
- `gpu_temporal_bridge.cpp:make_resources` 总是创建两个 `pixels * 32` 的历史缓冲。仅 scene detection 时不需要它们；当前 static_low_frequency 开启，需要历史，不能直接取消。
- 最新可读故障 a2a0c4dd 成功处理 189 帧后设备 HUNG；DRED `done=7/12 next_op=6` 对应尚未完成的 Dispatch，尚不能确定是哪一个 shader。采集器目前仅输出 ANSI 名称，队列/命令表的 Unicode SetName 可能显示为 unnamed，不能据此断定是其他模块。
- 没有故障瞬间的 DXGI LOCAL budget/current usage 或 allocation failure 记录。因此显存压力是待验证假设，不能把 HUNG 直接判成 OOM。退出工作台后的整卡空闲占用也不能反推故障峰值。

## 第一批：资源用途拆分，计算不变

### 决策 ADR：用途限制只影响分配与可调用入口

采用明确用途，旧入口默认 FullPipeline 兼容。调用不属于该用途的 pass 在提交 GPU 指令前报错，不偷偷切换算法。资源、描述符、CUDA 导入和释放顺序共同裁减，不绑定不存在的平面；所保留 pass 的 shader、tap、数值、采样、fence 同步不变。
不依靠全局资源池跨上下文复用，不移除仍被异步工作借用的资源，不改变每个 Feature 的独立历史。

| 修改 | 4K 等尺寸主要逻辑申请减少量 | 当前适用范围 |
| --- | ---: | --- |
| Exchange 输入用途不创建反向输出纹理 | FP16 每实例 63.28 MiB | 当前 DLAA 输入与 TrueHDR D3D 输入两处约 126.56 MiB；需要 export 的实例保持原资源 |
| DLAA 光流辅助桥 GuidanceOnly | 237.30 MiB/实例 | 当前 wide DLAA＋NVOF；保留 Master/Working/Proxy/Guidance/Scratch 及所需 taps |
| TrueHDR 最终转换桥 EncodeOnly | 435.06 MiB/实例 | 当前导出；保留 Master/Composited/EncodeY/EncodeUV 和计数器，继续执行现有 import/set/encode |
| 独立原图桥 DecodeOnly | 498.34 MiB/实例 | NR 前有效果的高位深预览，或处理尺寸不同的路径；当前等尺寸导出通常没有这份独立桥 |
| 仅场景检测的 temporal observe 用途 | 506.25 MiB 历史＋FP16 输出纹理约 63.28 MiB | scene_detection=true 且 mode=off；当前 static_low_frequency 开启不适用 |

数值是基于 3840×2160 与源码尺寸计算的可取消申请量，未包括纹理/堆实际对齐、驱动额外空间、SDK 工作区；不是实测驻留下降。4K 相关 buffer row pitches 恰好为 256 的倍数。
完整 HDR 桥主要平面为 74 bytes/pixel。GuidanceOnly 删除 30 bytes/pixel；EncodeOnly 保留 19 bytes/pixel、删除 55 bytes/pixel；DecodeOnly 保留 11 bytes/pixel、删除 63 bytes/pixel。CUDA 共享映射不再算第二份同尺寸分配。
当前常见导出链的前三项约 798.93 MiB，即 0.78 GiB。预览不创建最终编码 Tail 桥，不能再把 DecodeOnly 的数字叠加到这份导出合计。其他输入形态、顺序或阶段关闭时重新计算实际实例数。

### 实施顺序与文件

1. `native/gpu_rgba_exchange.cpp`、`native_effects.py`、`truehdr_engine.py`：按方向分配输出纹理。已知双向用途在初始化阶段准备必要资源，避免将首次分配突然推迟到交互操作。
2. `native/gpu_hdr_scrgb_bridge.{h,cpp}`、`hdr_gpu_engine.py`：新增用途配置/入口，保留旧 ABI；建立资源与 pass 依赖清单，按清单创建描述符和同步资源。
3. `native_effects.py`、`gpu_hdr_transfer_export.py`、`truehdr_pipeline.py`：分别选择 GuidanceOnly、DecodeOnly、EncodeOnly。FullPipeline 主处理链不改。
4. `native/gpu_temporal_bridge.cpp`、`temporal_control.py`：observe 初始化只准备 guide/stats/source transport；真正需要 process 时创建历史、candidate transport、output 等资源。沿用上轮修复的初始化事件和 device removal 检查。

质量目标：编码前同参数 RGBA8/FP16 逐比特一致，静态 metadata/PTS/帧数一致。无新增每帧计算或等待，预计稳态速度不下降，但实际首帧和处理耗时必须测量；若出现变慢，在保留该取舍前请用户确认。

## 第二批：存在速度或质量取舍，须单独确认

| 候选 | 内存收益 | 取舍与范围 |
| --- | --- | --- |
| NVENC P010 帧池 16 → 8，依据实际参考/缓冲需求自适应 | 每个 4K P010 surface 约 23.73 MiB，8 个约 189.84 MiB | 有吞吐/排队影响和编码配置兼容风险；`initial_pool_size` 也不一定是全部编码内存。保留 codec、UHQ、B 帧、lookahead 配置，不能靠自动关闭选项让小池通过 |
| 时间历史保存源的原始 RGBA8/FP16 格式，低频残差仍保持 FP32 RGB | FP16 路径理论约 189.84 MiB/实例 | 原始源样本可精确恢复，残差不降精度；数据布局/对齐会影响速度，须验证后由用户确认取舍。普通 SDR8 单独验证，不把原始 8-bit 样本改存 FP16 |
| 降低工作分辨率、光流网格质量、切换 DLAA 模型或关闭时间稳定 | 收益依实际设置/SDK 而定 | 会改变效果，第一批不做；只有用户明确选择才实施 |
| 每帧销毁/重建 DLAA/NR 或把资源迁到 CPU/系统内存 | 未知 | 可能严重变慢及增加传输，当前不推荐 |

## 验收与失败定位

先采同一素材/同一参数的短基线，不以旧 DedicatedUsage 的异常增长记录当作当前显存结论。
资源分配前后记录应用自有 allocation bytes、DXGI LOCAL CurrentUsage/Budget、NON_LOCAL、整卡 NVML，占用统计分开记录；每阶段采样峰值，不能只看作业结束值。
失败时保存 DRED Dispatch 名称（包括 Unicode）、最近阶段、帧号和预算；没有 page fault/allocation failure 证据时不宣布 OOM。

- Exchange：RGBA8/FP16，import-only、export-only、双向及关闭重开，验证实际方向和 fence。
- Bridge：FullPipeline 与专用途同输入/重置序列，编码前图像逐比特对比；signed HDR、SDR10、反向排序、TrueHDR、源图预览覆盖必要条件。
- Scene-only 与 temporal process 切换不丢历史/旧资源状态，不误绑定未初始化 SRV/UAV。
- 当前实际 4K DLAA → NR → TrueHDR 链检查显存峰值和每段时间；保留 NVOF、post/temporal 参数与 AV1 UHQ。遇到性能或像素不一致停止交付该候选，不自动用模型降级绕过。
- 不重新运行已成功的大范围 gate；重建受影响 DLL，只跑对应短 GPU/生命周期检查。可取消申请量与实测驻留节省分别报告。

方案提出时未改生产代码。随后用户批准第一批资源裁减、镜头检测默认开启和独立原图预览精简解码；第二批仍未授权。实施和测量记录见 [显存资源裁减验证](../validation/2026-10-14-vram-resource-purposes.md)。安装版和发布版本未改。
