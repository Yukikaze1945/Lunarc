# DLAA / RTX VSR 高精度与离线 2×–4× 插帧验收

日期：2026-10-11。工作树验收；未提交或推送，保留原有修改。

## 实现

- DLAA 的加性原生接口、CUDA/D3D12 交换支持 RGBA16F。SDR10 为线性 Rec.709，HDR10 为原有有符号 scRGB；独立 DLAA 直接消费主色，光流代理仅用于运动引导。保留旧 SDR8 ABI。
- 用户给出的 `F:\DLSSC\VideoFX` 为 VFX SDK 1.3。RTX VSR 使用官方 `RGB10A2 / P32 / ImageEncodingMode=1` 输入输出，质量档位 1–4。主色保持 FP16，GPU 上在 VFX 边界转换 Rec.709 或 PQ/BT.2020 packed 10-bit；不经过 RGBA8 主图像。保留既有 P010 编码、静态 HDR metadata 与 FFmpeg remux。
- 渲染顺序保持用户可排序的 DLAA / DLSSNR / VSR。VSR 前置时 NR 按放大后尺寸工作，后置时单独接实际输出尺寸的 P010 色彩桥。预览完整尺寸读取、原图与处理图使用正确的缓冲和幅面。
- 用户选择“优先可导出的 2×–4×”后，接入官方 NVIDIA VideoFrameGeneration。SDK 的独立特效模块安装在用户指定根目录；自动发现与高级目录覆盖可用，外部运行库不打包进软件。VFG 可独立于 VSR 模块加载。
- VFG 位于所选渲染链后、NVENC 前；输出 CUDA 图像，2/3/4 倍率和 Low/Medium/High 三档 Mode 可选。HDR/SDR10 使用 RGB10A2，SDR8 使用 RGBA8。当前借用图像先复制到设备自有缓冲，再生成和编码中间帧，防止编码器回收生产者缓冲。取消、时间戳异常和 SDK 错误直接失败，无静默替代引擎。
- 瑞士风格工作台保留。新增导出插帧、倍率和质量控件，详细限制在圆圈问号中；导出页面显示真实输出尺寸与插帧帧率。

## 短硬件验收

设备：RTX 5060 Ti。证据：`bench_out/wide_native_xess_20261010/`，均为短功能 gate，不是 benchmark。

| Gate | 输入→输出帧 | 输出 | 结论 |
|---|---:|---|---|
| HDR NR→VSR | 4→4 | 3840×2160 Main10，25 fps | PASS |
| HDR VSR→NR | 3→3 | 3840×2160 Main10，25 fps | PASS |
| SDR10 VSR→NR→VFG 2× | 4→8 | 1920×1080 Main10 BT.709，240 fps | PASS |
| SDR8 DLAA→VFG 3× | 4→12 | 1280×720 Main，360 fps | PASS |
| HDR DLAA→VFG 4× | 4→16 | 1920×1080 Main10 PQ/BT.2020，100 fps | PASS |
| HDR NR→VSR→VFG 2× | 3→6 | 3840×2160 Main10 PQ/BT.2020，50 fps | PASS |

此前本轮六个单独 DLAA SDR10/HDR10 原生前置、后置、独立导出及相应预览 gate 保留有效：`sdr10_nr_dlaa_v2`、`sdr10_dlaa_nr`、`sdr10_dlaa`、`hdr_dlaa_nr_flow`、`hdr_nr_dlaa`、`hdr_dlaa`，以及 `wide_preview.json`。

新增 `vsr_preview.json`：SDR10/HDR10 的 VSR 前置与后置，帧 0/1/4，包括跳帧；视口和完整尺寸处理图、原图尺寸检查全部 PASS。实时预览不执行 VFG，保持源帧率。

`vfg_native.json`、`vfg_other_modes.json`：U8 与 packed10 三档真实模型 Load/Run；移动矩形的三个中间相位与两张输入都不同，第一与第三相位也不同。不是重复源帧假装生成。

`rgb10_codec.json`：短浮点渐变测试，SDR 最大绝对往返误差 0.0009765625；HDR 线性范围至 16 scRGB 时最大绝对误差 0.0625，输出最大 16.046875，不钳到 SDR 1。测试素材没有合法编码边界裁切。该测试本身有诊断上传/回读，与生产路径的图片传输计数分开。

## 封装、时序与传输证据

- 所有短输出重新 ffprobe；视频 packet/display 帧数匹配。HDR 4× 的显示 PTS 为 0、10、…、150 ms；4K HDR 2× 为 0、20、…、100 ms。SDR10 120→240 fps 在 Matroska 毫秒时间基下为 0、4、8、13、17、21、25、29 ms。
- 4 帧 4× 包含 9 张真实模型生成帧和 3 张明确的片尾保持帧，总计 16。3 帧 2× 包含 2 张生成帧、1 张片尾保持帧，总计 6。没有未来源帧时片尾保持末帧，保留总时长和音轨同步；诊断分开计数。
- `final_remux_audit.json` 离线审核六个既有输出：流类型/数量、语言/标题/disposition、全局 metadata、章节标题与数量一致；音频和字幕 packet SHA256 为源 packet 的相同前缀。HDR fixture 的 AAC、SubRip、2 个章节及 36-byte 附件保留，附件 SHA256 为 `1a058dd53a5d12a10b13aee437bb990c408a92ed606ed708f533f4972ff13b92`。
- HDR source/final HEVC mastering-display 137、CLL/FALL 144 SEI payload 字节一致，ffprobe mastering 与 content-light side-data 保留。SDR10 输出为 BT.709，未添加 HDR SEI。视频中间 MKV 复用现有 FFmpeg `-c:v copy` 重封装，没有额外视频编码。
- 生产 gate 的已测 NVDEC、DLAA/NR/VSR、VFG 与 NVENC host image 边界为 0。RGB10 adapter 的统计每次 pack 实测 8 bytes 标量回读；既有 HDR 色彩桥统计仍跨 host，字节未测。不能声称总 host traffic 为 0。压缩码流与封装流仍由 host FFmpeg 处理。

## 自动检查

- 全部 Python unit：1590 passed，151.11 s，`final_units_green.txt`。
- Node：65 passed，`final_node.txt`。
- TypeScript/Vite/Electron 构建通过，`final_frontend_build.txt`。
- 原生 VSR/VFG/10-bit adapter `/W4 /WX` 编译通过；生产 DLL 的真实硬件 gate 通过。
- Electron 实际控件验收：DLAA→NR→VSR 排序、4×、High Mode 保存到后端；SDR10 的 120→480 fps 导出说明。使用隔离测试配置，不改用户保存的设置。
- `git diff --check` 通过。成功 gate 没有因说明或检查器修改而重新编码。

## 边界

- 当前帧生成是 NVIDIA VFG，**不是 XeSS**。本机 XeSS 官方 capabilities 探测的 max_interpolations=1；公开接口通过 XeLL/DXGI swapchain 呈现，没有当前离线编码器所需的生成图像输出。用户授权优先离线 2×–4× 后选择 VFG。
- VFG 需要支持的 Ada/Blackwell 显卡及 VFX 1.3 Core/特效模块；本轮为当前机器实测，不代表其他显卡、SDK 或任意素材组合全部适用。非均匀时间戳/时间基变化拒绝，不悄悄转换 VFR。
- 现有 VSR 尺寸选项保持 1080p/4K，目标必须大于源幅面；任意缩放和等尺寸 VSR 不在本轮准入范围内。
- DLAA 使用视频光流和零深度/抖动，预设是请求提示，运行库/驱动可能覆盖。不是具有游戏深度缓冲的完整游戏 DLAA 输入。
- VFX packed10 边界有量化，合法编码域外的色值会裁切并计数；FP16 主色不是无损格式。HDR 预览仍为固定映射的 8-bit 显示，未校准。
- 没有长视频稳定性、模型画质对比或性能 benchmark；**Performance=NOT_CHARACTERIZED**，不承诺速度或画质增益。
