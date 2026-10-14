# TrueHDR

TrueHDR 将 SDR 转为 HDR10。它与 DLAA、DLSSNR、RTX VSR、NVIDIA VFG 共用渲染顺序列表，拖动左侧手柄或使用上下箭头调整顺序。旧项目中单独启用的 TrueHDR 会迁移到列表末尾。HDR10 素材也可以使用：在 TrueHDR 所处的位置先映射为 SDR，再交给 TrueHDR 生成新的 HDR 画面。

## 控件

| 参数 | 范围 | 默认值 |
| --- | --- | --- |
| 对比度 | 0–200 | 100 |
| 饱和度 | 0–200 | 100 |
| 中灰亮度 | 10–100 | 50 |
| 目标峰值亮度 | 400–2000 nit | 650 nit |
| AI 去色带 | 开 / 关 | 开 |

范围和输入输出格式遵循 [NVIDIA TrueHDR 接口](https://docs.nvidia.com/maxine/vfx/latest/Filters/TrueHDR.html)。

## 运行库

需要 NVIDIA VFX SDK Core 及对应的 `nvvfxtruehdr` 模块。使用现有 VFX 目录查找机制，支持高级选项中的目录覆盖。模块放在 SDK 的 `features/nvvfxtruehdr` 下，保留官方目录结构。缺失组件会明确报错。

NVIDIA 特效 DLL、模型和许可文件由用户安装，不随本项目源码分发。

## 精度与输出

SDR10 的解码、NR 和重建保持原有线性 FP16 链路。TrueHDR 官方接口接收 RGBA8 SDR，所以仅在进入此特效时显式转换；不能把它描述为全程 10-bit 无损转换。特效输出为 BT.2020/PQ RGB10A2，再转回线性 scRGB FP16，供预览映射或最终编码。

TrueHDR 之前的阶段保持输入的色彩域，之后的阶段处理线性 FP16 HDR。HDR10 输入会在进入 TrueHDR 的边界执行固定 203 nit 参考的亮度肩部映射，并转换为 BT.709 SDR；不随帧或原视频 MaxCLL 改变曝光。SDK 输入仍为 RGBA8。NR 位于转换之后时，独立 FP16 会话及既有 HDR 重建／多层融合处理转换后的主画面；VSR 和 VFG 使用 PQ/BT.2020 的 RGB10A2 接口。最终 HDR 编码不会再次执行 TrueHDR 转换。预览不执行导出插帧，但其他阶段保持列表中的顺序。

HDR→SDR→TrueHDR 会重新解释原片的亮度和色彩，不是恢复原 HDR 的无损往返。转换后仍输出 P010 / 10-bit / BT.2020 / PQ；原输入的母版显示和 CLL 描述不再被当作新画面的静态 HDR 元数据。音轨、字幕和章节仍按导出选择封装。诊断记录 `hdr_to_sdr_calls`、映射参数、SDK 转换主机图像计数和标量回读。HLG 保持现有明确拒绝。

HDR10 输入支持明确标记的有限范围或全范围。GPU 解码按每帧实际报告的范围归一化；部分 PS5 WebM 的容器标记为有限范围，而解码帧报告全范围，此时以解码帧为准。输出继续使用有限范围 P010，不通过改标签冒充范围转换。

导出使用现有 P010 / HEVC Main10 / BT.2020 / PQ / limited-range 编码路径，界面限制可选格式。不会为 SDR 源编造 mastering display 或测得的 MaxCLL 元数据。

工作台的 Windows 原生预览可使用 FP16/scRGB 直接呈现 HDR，详情见 [原生预览](native-hdr-preview.md)。需要显示器 HDR 已启用；系统未开启 HDR 时使用 SDR 映射。

图像预览和文件对比页仍传输普通 8-bit JPEG。其显示映射采用固定 203-nit 参考及亮度肩部压缩，GPU 图像预览和 CPU 文件对比使用同一公式 `rgb / (203/80 + max(Y,0))`，随后使用既有显示传递函数。不会按每帧峰值自动调节曝光；不会修改处理或导出的 HDR 主画面。这是未经校准的 SDR 显示方案，不能当作 HDR 显示器的实际亮度呈现。

首次启用需要加载模型。短帧功能检查已通过，尚无新的 4K 性能测量。
