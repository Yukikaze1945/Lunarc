# 原生阶段高精度与 XeSS 帧生成

目标：移除本项目 RGBA8 接口造成的 DLAA 限制，SDR10/HDR10 使用原有线性 FP16 主色、P010 输出和 metadata/remux；对 VSR 测量实际 SDK FP16 格式能力。XeSS 使用 Intel 官方 SDK，核实倍率、HDR、输入及输出 API，不能用重复帧冒充生成帧。

步骤：

1. 分别核对 SDK 与实际本机 API，记录硬限制。已确认 XeSS 3 公共文档：非 Intel GPU 只支持一个生成帧，即 2×；生成通过 XeLL + DXGI proxy swapchain 呈现。不得声称本机 NVIDIA 支持 3×/4×。
2. DLAA 新增兼容的格式感知原生 open 接口与 HDR create flag，交换接口增加 RGBA16F；既有 SDR8 ABI 保留。接现有 HDR/SDR10 前后阶段，浮点数不压到 SDR [0,1]。
3. VSR 实测旧 SDK 拒绝 F16。用户提供新版 F:\DLSSC\VideoFX 后，核对 VFX 1.3 的 RGB10A2 / ImageEncodingMode=1 接口。主色保持 FP16，特效边界在 GPU 编解码 SDR Rec.709 或 HDR PQ/BT.2020 的 packed 10-bit；报告量化与合法编码域裁切，不套用 8-bit 代理。
4. XeSS 运行库官方能力探测与实际适配验证。只有真正可获得的输出才进入预览/导出；需要深度、矩阵、swapchain 的接口不能伪装为普通离线视频插帧 API。真实外部 API 阻塞要报告具体缺口。
5. 相关单测与短 GPU 精度/范围 gate；最新实际 SDR10/HDR10 预览/导出检查格式、帧数、PTS、metadata 和音轨。只重跑受影响短 gate，保留原有工作树；不提交推送，不委派。

## 用户后续选择与实现

用户选择优先实现可导出的 2×–4×，授权更换有离线输出的引擎。采用官方 NVIDIA VFX 1.3 VideoFrameGeneration，安装到用户给出的 SDK 根目录；实际输出 CUDA 图像，渲染链结束后、P010/NV12 编码前执行。名称显示 NVIDIA VFG，不冒充 XeSS。2×/3×/4× 与三档模式可选；预览仍为原始帧率。

均匀源时间戳使用整数倍时间基保留；可变帧率/时间基变化明确拒绝。最后源帧后没有未来帧，片尾 M−1 张明确重复最后帧，保证时长和音轨同步；与真正生成帧分开计数。镜头切换使用 SDK ShotChange/自动检测。CUDA 上保留借用的当前图像再编码中间帧，防止生产者缓冲被回收。所有生产图片边界测量仍为零；另计 RGB10 编码统计的标量回读，既有色彩桥标量字节未测，不能声称总 host traffic 为零。
