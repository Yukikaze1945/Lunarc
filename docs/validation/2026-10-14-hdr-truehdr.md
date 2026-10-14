# HDR10 输入的 TrueHDR 转换

开启 TrueHDR 时，在该阶段把线性 scRGB FP16 HDR 映射到 BT.709 SDR，再调用 TrueHDR。参考白固定为 203 nit，亮度肩部为 `rgb / (203/80 + max(Y709, 0))`；映射后在 SDK 的 SDR 边界编码为 RGBA8。其他阶段保留原定处理顺序，TrueHDR 输出继续为线性 FP16 HDR，最终编码为 P010 / PQ / BT.2020 / 10-bit。

HDR10 输入的导出路由与会话精度已一起修正；TrueHDR 阶段执行一次，预览消费和最终编码不重复执行。原输入静态母版和 CLL 描述只保留为输入诊断，不宣称属于重新生成的 HDR 画面。HLG 保持原有拒绝。

## 验收

- 原生桥接 `/W4 /WX` 重建成功，无 pending 标记。`rtx_vsr_bridge.dll` SHA-256：`4ab01f047c3234b4c0c82272544f94a2571ca1f61a897f4117e96fe32e0ff9fd`。
- 317 项相关 Python 路由、格式、会话和 HDR 生命周期检查通过。
- GPU HDR→SDR 输入转换与 CPU 数值参考的 RGBA8 最大误差为 0；含黑、灰、高光、负通道和非中性色样本。测试图像上传和回读分别为 1024/512 字节，仅属于测试；生产转换计数 `host_image_bytes=0`，标量回读单独记录。
- HDR10 素材的三个真实预览顺序通过：TrueHDR；NR→TrueHDR；TrueHDR→NR。每个顺序处理一帧，TrueHDR 与 HDR→SDR 调用数均为 1。
- 两帧真实 AV1 导出通过：1920×1080、`yuv420p10le`、BT.2020 / PQ，TrueHDR 与 HDR→SDR 调用数均为 2，编码器主机图像传输计数为 0。输出未继承原输入的 mastering/CLL side-data。
- 最初复用的 SDR10/UHQ 检查器在编码已经完成后，因为 HDR 报告没有 `sdr10_encoder_tune_adjustment` 而退出；新检查器离线复查其原输出和原报告，没有重新运行已完成的导出。

证据：`build/hdr-truehdr/result.json`、`build/hdr-truehdr/export/result.json`、`build/hdr-truehdr/export/av1_nvenc.mkv`。检查入口：`tests/helpers/hdr_truehdr_gate.py`。

转换会改变原片亮度与色彩，不是 HDR 无损往返；TrueHDR SDK 的 RGBA8 输入边界仍存在。性能未做正式测量，保持 `NOT_CHARACTERIZED`。
