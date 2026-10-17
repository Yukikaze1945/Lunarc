# HDR AVIF 原色与 PQ 标记修复

## 原因

旧命令仅设置输出编码参数，rawvideo 输入帧的原色、传递曲线仍未声明。本机 FFmpeg/libaom 组合最终写出的 AV1 序列头为 `color_primaries=2`、`transfer_characteristics=2`（未指定），矩阵为 9，像素为 yuv444p10le。严格校验因此正确拒绝发布文件。

通过 65×33 固定像素复现并读取 `trace_headers` 证实，非 ffprobe 漏显示。修复在主图编码前增加 `setparams=range=full:color_primaries=bt2020:color_trc=smpte2084:colorspace=bt2020nc`，同时保留原输出色彩选项。该滤镜只设置已转换图像的帧属性，不改变像素，不给独立透明度流套 PQ，不放宽校验。

参考：[FFmpeg setparams 文档](https://ffmpeg.org/ffmpeg-filters.html#setparams-1)、[FFmpeg libaom 编码器源码](https://ffmpeg.org/doxygen/trunk/libaomenc_8c_source.html)。实际复现及验证以本机二进制产物为准。

## 用户授权的 8K 检查

用户明确允许使用该 8K 素材重跑，要求控制测试量。按当前后台设置完整执行一次 **DLAA → NR → TrueHDR → VSR**，默认省显存分步处理。原后台仍加载旧 Python 导出代码，因此未尝试侵入式热补丁；读取当前设置，释放原后台的预览 GPU 资源（保留其 RAM 缓存），在独立验证进程运行更新后的代码。

输入为用户当前 7680×4320 AVIF。保存结果：

- 路径：`build/hdr-avif-fix/MuMu安卓设备-1_MuMuNxDevice_7680x4320_1792085348_HDR_fixed.avif`
- 7680×4320，AV1 High，yuv444p10le，全范围。
- ffprobe 实际读回 BT.2020 原色、SMPTE ST2084/PQ、BT.2020 non-constant 矩阵。
- 最终 AV1 序列头 `color_primaries=9`、`transfer_characteristics=16`、`matrix_coefficients=9`、`color_range=1`；不只依据容器标记。
- 质量 95，文件 1,768,113 字节。该输入不含透明度；本轮未扩大到 8K 透明度组合。
- 证据：`build/hdr-avif-fix/8k-evidence.json`；原参数快照 `live-input.json`；可复用的一次性检查入口 `tests/helpers/hdr_avif_live_image.py`。

10 项 HDR 图片单元检查通过，包括原始精度、色彩声明、独立透明度流及缺字段仍拒绝保存。错误文案仅列出不匹配字段，不再把 ffprobe 整个字典放进界面。

实际结果文件已经可用。运行中的旧后台需要重新启动才能使用修复后的保存代码；本次未发布安装包。
