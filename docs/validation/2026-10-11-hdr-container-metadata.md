# MKV 浮点 HDR 元数据兼容性修复

## 原因

用户素材 `D:\浏览器下载\Bulgaria 4K HDR 60P _fantav.com.mkv` 为
3840×2160、HEVC Main10、limited-range PQ / BT.2020。
第一帧和轨道的 mastering metadata 包含 `green_x=2055209/8388608` 等
二进制浮点数对应的分数。原生 SEI 序列化器要求每个分数精确落在
1/50000 色度网格上，因此在开启编码器时误拒绝合法容器元数据。

FFmpeg 的 [Matroska 解封装实现](https://ffmpeg.org/doxygen/trunk/matroskadec_8c_source.html)
用 `av_d2q` 将容器浮点数转换为分数，并不保证分母为 50000。
这段素材短片的原始 HEVC elementary stream 中没有 137/144 SEI，
HDR 静态信息来自 MKV 轨道，不能用“原始 SEI 必须存在”作为验收条件。

## 修改

- `native/nvenc_bridge.cpp`：先验证原始分数范围，再用整数运算转换到
  最接近的 HEVC 元数据单位，中点向上取整。精确网格值字节保持不变。
- 非法分母、负值、范围外数值、缺失字段和冲突的现有 SEI 仍按原逻辑处理；
  不通过钳位伪造合法元数据。
- 容器/side data 保留源分数；仅 bitstream SEI 执行必要的网格转换。
  `hdr_mastering_wire_conversion` 逐项记录源值、编码值与精确误差。
- 重新编译 `native/nvenc_bridge.dll`。不改变 HDR 图像转换、DLSS 模型或显示映射。
- 短导出 checker 支持仅容器声明 HDR 的源文件，独立计算预期 SEI；
  仅解析本次验证的源帧区间，避免将完整长视频当成短 gate 证据。

## 验证

证据目录：`bench_out/hdr_metadata_20261011/`（ignored）。

| 检查 | 结果 |
| --- | --- |
| 原生 DLL 构建（警告视为错误） | PASS |
| CPU SEI gate，包括该素材分数、舍入边界、范围外拒绝和原精确值回读 | 63/63 PASS |
| HDR 生命周期、顺序 VFG、源色彩和新增转换报告单元测试 | 249 PASS |
| 用户原始视频的实际 GPU NR HDR 导出 | 4 帧完成，PASS |
| 输出尺寸、格式与色彩 | 3840×2160、yuv420p10le、Main10、PQ、BT.2020 |
| 输出 display PTS | 0、17、33、50（1/1000） |
| HDR 容器 metadata | 源 mastering 分数完整保留 |
| HEVC bitstream SEI | 137 mastering、144 CLL，每项与独立预期一致 |
| 内容亮度 | MaxCLL 1100、MaxFALL 180 |
| 音频 | 3 个 AAC packet 与源 packet 前缀 payload hash 一致 |
| 已测解码/模型/编码图像 host copy 边界 | 均 0；scalar readback 单独记录 |

源色度与白点共 8 项需要转换，最大绝对误差小于 1e-8，
远小于一个色度编码单位 1/50000。1000 nit / 0 nit 亮度数值不变。
这是元数据表示转换，不是对 HDR 画面做亮度或色彩调整。

`bulgaria_nr/output.mkv`、`diagnostics.json`、`result.json`、`ffprobe.json`
保存实际导出及离线核验；`video.hevc` 可单独读取网格化后的 SEI。
第一次 GPU 导出完成后，旧 checker 在解析整段源文件时被停止，
修改 checker 后只离线复核原成品，未重跑已成功的 GPU 导出。

## 范围

这是该源视频的四帧 single NR 验证，不代表整段视频或所有渲染顺序均已验收。
本次没有性能 benchmark，Performance=NOT_CHARACTERIZED。
