# 2026-10-14 显存资源裁减

## 本轮改动

- 纹理交换桥只在第一次输出 D3D 纹理时创建输出纹理。仅输入转换保留共享缓冲，不申请未使用的输出。
- HDR/SDR10 色彩桥增加 guidance、decode、encode 三种资源用途。原 open 接口和 ABI 1 保持完整管线；新版用途入口是新增接口，缺少新版 DLL 时明确要求重建。
- DLAA 引导只创建 master、working、proxy、guidance 和 FP32 scratch；预览独立原图只创建 P010 源平面和 FP16 master；TrueHDR 编码尾部只创建 master、composite 和 P010 输出。
- 镜头检测默认开启，时间稳定默认关闭。界面显示“仅镜头检测”，仍可取消检测。已有显式关闭设置保留；在界面关闭时间稳定时启用镜头检测。
- 镜头检测只申请小 guide/stats 与源图 GPU transport。首次启用时间稳定才申请完整历史、candidate transport 和输出。预览关闭时间稳定后释放这些资源；再次开启重新初始化历史。
- 修正色彩桥对设备的重复 AddRef：ComPtr 赋值已持有引用，额外引用原来没有配对释放。

没有缩小处理尺寸、换模型、改 NR/后处理强度、减少编码帧池或改变时间稳定算法。没有发布安装包。

## 经用户确认的引导修复

实际 GPU 对比发现，HDR 引导 shader 给每像素 4 字节的 RGBA8 缓冲调用 `Store4(uint4(...))`，一次写 16 字节；相邻线程会覆盖其他像素，末尾也越过有效行/缓冲边界。相同输入的结果出现差异。

经用户确认后，将四个 8-bit 通道打包成 uint，以 `Store` 写入 4 字节。亮度公式保持不变；此修复会纠正此前错误的运动引导，可能改变 HDR DLAA/NR 的运动处理结果。本轮前后性能基线都使用修正后的引导 shader，避免把该质量变化混入资源裁减比较。没有据此认定过去的 device hang 已定位或彻底解决。

## 实际分配量

RTX 5060 Ti，3840×2160，FP16；新增诊断调用从实际资源描述与 `GetResourceAllocationInfo` 汇总 DEFAULT heap 分配大小，包含对齐，排除 CPU 计数器 heap。这不是整卡驻留用量。

| 资源 | 完整用途/稳定开启 | 裁减用途 | 减少 |
| --- | ---: | ---: | ---: |
| HDR 光流辅助桥 | 588.06 MiB | 349.38 MiB | 238.69 MiB |
| 独立原图解码桥 | 588.06 MiB | 87.19 MiB | 500.88 MiB |
| TrueHDR 编码尾部 | 588.06 MiB | 150.50 MiB | 437.56 MiB |
| 仅输入 FP16 Exchange | 127.06 MiB | 63.31 MiB | 63.75 MiB / 实例 |
| 时间控制桥 | 696.81 MiB | 63.50 MiB | 633.31 MiB |

不同场景拥有不同实例，不能直接把所有行相加称为整条链的节省量。源图预览与编码尾部不会在本轮预览里同时申请。

## 同素材短测

素材：本机 `12.ユメ語るよりユメ歌おう.mkv`，3840×2160、120 fps、AV1 SDR10。处理链 DLAA L → 单层 NR → TrueHDR。维持 NVOF、原 NR/后处理/TrueHDR 参数；导出为 AV1 Main 10-bit、P7/UHQ、P010、BT.2020/PQ、MKV。两版参数相同。

比较基线使用当前生产数学和已修正的引导，但强制完整色彩桥资源，并加载本轮修改前保存的 Exchange/Temporal DLL。新版使用裁减资源。各版本独立进程、顺序执行，未并发占用 GPU。

| 检查 | 完整资源基线 | 新版 |
| --- | ---: | ---: |
| 6 帧导出，帧末 DXGI LOCAL 进程采样最大值 | 12.932 GiB | 11.427 GiB |
| 预热后 5 帧处理段耗时中位数 | 132.50 ms | 129.57 ms |
| 3 帧预览，镜头检测开启/稳定关闭，帧末 LOCAL 最大值 | 11.285 GiB | 8.977 GiB |
| 预览第 2、3 帧处理/显示墙钟 | 131.74 / 135.79 ms | 119.40 / 126.37 ms |

导出减少约 1.50 GiB，预览减少约 2.31 GiB，均为本次 DXGI LOCAL 进程样本。它们不能等同于 DEFAULT heap 求和，也不使用旧 Windows DedicatedUsage 异常记录推断结果。LOCAL/NON_LOCAL budget、用量、整卡 `nvidia-smi` 起止用量均单独保存；帧末采样不覆盖每个 shader 内部瞬时峰值。

处理段是围绕已完成 GPU 工作的 CPU 墙钟，不是整段导出吞吐或实时播放 FPS。短测未见稳态明显变慢；样本很短、基线先运行且模型/驱动缓存影响首帧，不能将首帧从 14.81 s 降到 6.64 s 全部归因于优化。性能仍为 `NOT_CHARACTERIZED`。

预览三帧都保持 generation=1，关闭后的进程 LOCAL/NON_LOCAL 用量回到 0；新版预览确实使用 DecodeOnly 和 scene-only 资源，而不只改变诊断文字。

## 正确性检查

- GPU 合成输入覆盖 signed HDR、超过 1 的亮度以及 SDR10：完整桥与三种用途的解码、working、proxy、guidance、P010 编码结果逐字节一致。
- 引导 RGB 三通道相等、alpha 全部为 255；RGBA8/FP16 Exchange import/export 逐字节一致，import 不触发输出分配。
- 4K 场景检测不创建完整时间资源；稳定开启时懒分配，关闭后释放，再次观察恢复轻量资源。
- 原生时间处理 RGBA8/FP16 probe：0 failures，包含 cut/duplicate/flash/translation、历史抑制、零权重恒等、D3D candidate、signed HDR 与非法值拒绝；D3D12 debug 检查通过。
- 两版实际导出均为 AV1 10-bit 3840×2160、BT.2020/PQ、6 帧，显示 PTS `[0,8,17,25,33,42]`（1/1000）；解码后逐帧 framemd5 完全一致。
- 编码入口和已测 decoder/NR 图片主机传输保持 0；计数器/镜头检测标量回读单独保留，不能声明总 host traffic 为 0。本次排除音轨字幕元数据 copy，未重复已通过的 remux 功能 gate。
- 关联 Python 检查 391 passed / 1 skipped；Node 14 passed；前端 TypeScript/Vite 构建通过。

基线导出已成功完成 6 帧与全部采样，但采样脚本最后错误查询了被导出管线关闭的 live，导致脚本退出 1；原始错误和结果保存在 `export-full/result.json`。只修正采样脚本，离线复查现存文件、编码计数与 PTS/色彩，不重跑已经成功的 GPU 工作。

## 本机证据（ignored）

- `build/vram-purposes/resource-gate.json`：资源分配、GPU 同值比较。
- `build/vram-purposes/temporal-probe.txt`：原生时间检测/稳定与 debug 检查。
- `build/vram-purposes/export-{full,trimmed}/result.json`：真实导出原始报告及帧末采样。
- `build/vram-purposes/preview-{full,trimmed}/result.json`：原图预览、资源实例、时序、关闭后用量。
- `build/vram-purposes/export-offline-check.json`、`decoded-comparison.json`：最终流检查与逐帧画面比较。
- `build/vram-purposes/unit.txt`、`unit-hdr-final.txt`、`routing-unit-final.txt`、`node.txt`：相关测试。

## 本机 DLL

| DLL | SHA-256 |
| --- | --- |
| gpu_hdr_scrgb_bridge.dll | 88d5bfd8a8e93e9c6dc8298b7093607ddfe80712fd773c6ab01f09587edfd32b |
| gpu_temporal_bridge.dll | 0d6ee321f511ff07aa544c7858bfb17f8bda62a9473e3f7255e969f37c14d50d |
| gpu_rgba_exchange.dll | f49cbd35412ae61fa7c271ee7d88982c1335fb873da83e5f2d03f7921e20751d |

这批裁减不涉及编码池、历史精度或模型降级。device removal 的长时间稳定性尚不能由 6 帧证明；再次出现时仍需保存 DRED、阶段和预算，不能直接称为 OOM。
