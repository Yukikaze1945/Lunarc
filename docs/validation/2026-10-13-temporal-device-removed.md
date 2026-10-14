# 时间稳定回读处发现 GPU 设备挂起

## 原始证据

作业 41bbef54 在原始 3840×2160 AV1 SDR10 输入上，DLAA → NR → TrueHDR，静态低频时间稳定开启，成功编码 487 帧后报错。
`0x887a0005` 为设备丢失，设备原因 `0x887a0006` 为 GPU 挂起。Windows System 日志在 21:08:26、21:08:31 也记录了 nvlddmkm 153；Python 在 21:08:36 的 scalar Map 才发现重置。
因此回读处是发现设备故障的位置，不足以单独证明该 Map、UHQ、显存不足或时间稳定 shader 就是根因。

## 确定缺陷与修改

- D3D12 围栏在设备移除时返回 `UINT64_MAX`，旧代码用 `>= fence_value` 当作成功；现在先检查该哨兵和设备状态，等待事件返回后也再次检查。
- `gpu_temporal_open` 原先在 GPU 初始化提交后才创建事件；现在先创建事件，再提交和等待 GPU 初始化，消除未初始化事件句柄的竞态。
- 保留 ABI、时间稳定参数和数学；没有关闭用户选中的时间稳定，也没有修改画面/编码格式。
- 原生失败报告可携带 DRED 中未完成的命令表、队列、完成位置与下一条操作。宿主在建 D3D12 device 前按 `LUNARC_GPU_DIAGNOSTICS=1` 开启 breadcrumbs/page-fault；启动脚本当前默认设为 1，可明确设 0 关闭。没有启用昂贵的 GPU-based validation，也没有修改 Windows TDR 设置。
- GPU 设备重置后，预览/导出熔断新 GPU 任务，并明确要求重新启动。普通编码选项错误不熔断。
- 时间稳定关闭先撤销持有句柄，避免异常后的第二次重复释放。
- 重建宿主、时间稳定 DLL；时间稳定 DLL 同样采用暂存/下次启动替换，避免覆盖正在运行的 DLL。

## 检查

恢复本机保存的工作台和编码设置，只把测试输出写到忽略的 build 目录，关闭测试的音字幕/章节复制。视频源、处理链、4K、NVOF 1F、时间稳定、后处理和 AV1 UHQ 编码选项保留。

- 原始视频 470–501 帧短片段，32 帧通过。
- 原始视频从第 0 帧连续 520 帧通过：520 次时间稳定处理、520 次 TrueHDR/P010 编码，覆盖原故障位置。当前测试未发现新的 nvlddmkm 错误。
- 原生 RGBA8/FP16 时间稳定 probe 通过，D3D12 debug layer 无警告/错误。包括静态残差、变化区域拒绝、fenced/completed D3D12 输入、signed HDR 和非法输入拒绝。
- 相关生命周期、导出设备重置与时间稳定 Python 测试 59 项通过。
- 证据：`build/temporal-hang-check/original-chain/result.json`、`long-prefix/result.json`、对应原始日志与 `small-probe.log`。

这是修正确定同步缺陷后的回归证据，不是已经定位原始 GPU hang 唯一原因的证明。没有重跑整段视频，也没有进行性能/画质 benchmark。若再次失败，新增 DRED 能区分具体未完成指令和失效资源。
