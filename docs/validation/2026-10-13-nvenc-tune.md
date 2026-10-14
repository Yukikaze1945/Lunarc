# TrueHDR 导出 tune 拒绝修复

作业 7cea656a 的原始日志明确给出 `av1_nvenc: Unable to parse option value "uhq"`。
实际加载的 FFmpeg 7.1.1 / libavcodec 61.19.101：HEVC 接受 `uhq`，AV1 不接受。
这是软件选项表的限制，不能据此判断 RTX 5060 Ti 不支持 UHQ。
`b_frames=auto` 保留编码器/预设的选择，没有强制设置 4；B 帧与此次 tune 解析失败无关。

## 修改

- 新增兼容性扩展 `nvenc_option_value_available(codec, option, value)`，在加载的编码库上分配临时 AVCodecContext 并验证选项，不打开 GPU 会话。
- 原生 GPU SDR、SDR10、HDR10 与 TrueHDR 导出共同使用此检查，取消用 CLI 版本替代共享编码库能力的判断。
- 支持 UHQ 的编码器保留原请求；不支持时沿用原有 UHQ → HQ 兼容规则。
- 诊断增加 `*_encoder_tune_adjustment`，明确请求、实际使用值与原因。原设置/预设不改写。
- 旧 DLL 的普通设置仍有效；UHQ 必须使用重建后的 DLL，避免把缺少检查接口当作支持。
- 编码选项错误包含实际 codec、选项名和取值。能力探测的预期拒绝只在临时上下文静音，不更改全局日志。

## 本机证据

- 相关 Python 回归 446 项通过。
- 两个独立 1280×720、2 帧 TrueHDR 完整 GPU 导出成功：AV1 请求 UHQ、实际 HQ；HEVC 请求 UHQ、实际 UHQ。
- ffprobe 确认真实 AV1/HEVC、`yuv420p10le`、BT.2020/PQ/BT.2020nc 和帧数。两次编码入口 `host_pixel_bytes=0`。
- 证据：`build/nvenc-tune-check/result.json`，`build/nvenc-tune-gate.log`，`build/nvenc-tune-hevc-gate.log`。
- AV1 的第一次 GPU 导出已经成功，验收脚本错误访问了报告中已删掉的 `plan` 键。修正检查脚本后离线检查同一输出和原日志，没有重新跑这次 AV1 推理/编码。
- 未运行整段用户视频，也未测量性能提升。当前 AV1 UHQ 仍需要更新配套编码库；没有把 HQ 宣称成 UHQ。

重建 `native/nvenc_bridge.dll`，下次启动后端加载新代码与 DLL；不发布安装包。
