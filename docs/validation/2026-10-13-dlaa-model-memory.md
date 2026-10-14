# Windows 11：DLAA 模型切换与显存

## 环境与方法

本机 Windows 11 专业版 Insider Preview，build 26220，RTX 5060 Ti 16 GB。使用已有 3840×2160、120 fps、AV1、BT.709 SDR 10-bit 视频，每次在同一帧切换 DLAA L/M 请求预设。

先前的 NR 工作缩放检查没有覆盖 DLAA 模型切换。Windows 10 的已知计数器问题也不能作为本机 Windows 11 的诊断结论。本轮在同一个测试进程记录三组独立数据：

- Windows `GPU Process Memory` 的 DedicatedUsage、SharedUsage 和 TotalCommitted。
- [IDXGIAdapter3::QueryVideoMemoryInfo](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_4/nf-dxgi1_4-idxgiadapter3-queryvideomemoryinfo)：当前处理设备所在适配器的本进程 LOCAL/NON_LOCAL CurrentUsage 与预算。
- NVIDIA 驱动报告的整卡占用，包含其他程序。

采样先后进行，不是同时取值；预算用量、提交量与整卡驻留统计也不能直接相加。两轮测试的背景负载不同，不用整卡绝对值比较性能。

## 修正前：完整重建

每次更改 `dlaa_preset` 都关闭并重建整个处理管线，包括解码器、D3D12 设备、NR、缓存和显示资源。SDK 的释放计数只统计调用次数，不能证明释放成功；此次诊断宿主另外记录了 ReleaseFeature、DestroyParameters、Shutdown 的真实返回值。

“DLAA → NR”路径，单位 MiB：

| 请求模型 | DedicatedUsage | DXGI LOCAL CurrentUsage |
| --- | ---: | ---: |
| L | 19,030.5 | 6,955.5 |
| M | 22,547.4 | 6,989.7 |
| L | 26,000.6 | 6,960.2 |
| M | 29,517.4 | 6,994.5 |
| L | 32,970.6 | 6,965.0 |
| 关闭管线 | 27,800.3 | 93.4 |

同一进程先检查了单独 DLAA，因此上表 DedicatedUsage 包含前一段的计数；每轮变化约 3,450–3,520 MiB。“每次切换这个数都会涨”在 Windows 11 上确实复现。

关闭后 NON_LOCAL 为约 27.8 MiB，整卡从约 10,889 MiB 回落到 5,720 MiB。记录的 55 次 SDK 清理返回值均成功。这些数据表明该 DedicatedUsage 的数 GB 增量与本进程 DXGI 用量不一致，不代表已经证明 Windows 11 存在和 Windows 10 完全相同的系统缺陷，也不排除其他条件下的泄漏。相同 L 请求的 DXGI 用量仍有数 MiB 的小变化，不能称所有分配都完全归零。

## 修正后：仅重建 DLAA Feature

新宿主在原设备上释放旧 DLAA Feature 和参数，创建请求预设的新 Feature；检查 SDK 返回值并等待创建完成。保留解码器、输入/输出纹理、CUDA 导入、NR 与其他阶段。下一次 Evaluate 强制重置新 Feature 的历史，纹理状态仍沿用实际已有状态。

同路径五轮切换结果，单位 MiB：

| 请求模型 | DedicatedUsage | DXGI LOCAL CurrentUsage | 管线 generation |
| --- | ---: | ---: | ---: |
| L | 10,790.0 | 6,946.6 | 1 |
| M | 10,822.0 | 6,978.6 | 1 |
| L | 10,822.1 | 6,978.6 | 1 |
| M | 10,822.1 | 6,978.7 | 1 |
| L | 10,822.2 | 6,978.7 | 1 |
| 关闭管线 | 5,619.7 | 75.0 | — |

首次换 M 后约增加 32 MiB，之后四轮末端基本稳定，整个管线没有重新创建。单独 DLAA 路径也得到相同趋势。此处的 L/M 是请求提示，运行库/驱动可能重新映射，未获取有效模型名称。

还修正了两个相关问题：

- 单独 FP16 DLAA 已有主会话时，输出阶段误尝试再开一个 DLAA；现在明确复用主会话，不把 DLAA 当 Feature 18 anchor。
- 辅助 DLAA 使用与函数签名匹配的 `Shutdown1(device)`，替换已弃用的无参数全局 Shutdown。

旧宿主不支持局部重建时保留明确的完整重建兼容路径；局部创建失败会报错，不继续显示旧模型冒充新模型。

## 验证与限制

- 两种真实 4K 路径各五轮 L/M 切换；SDK 诊断清理记录没有失败返回。
- RGBA8 和 RGBA16F 各三次 L/M/L：局部重建输出与对应全新 Feature 的首帧输出最大绝对误差均为 0；省略调用者 reset 时仍正确重置新 Feature 历史。
- 56 项相关模型切换、处理顺序、预览缓存和清理测试通过；原生编译 `/W4 /WX` 通过。
- 证据在本机忽略目录 `build/dlaa-memory-20261013`，诊断宿主位于 `native/build/memory`，不随安装包收集。

这是同一台机器的短功能与内存检查，没有测所有预设、显卡或长时间播放。原始 `0x910` 未在本轮复现，不能把其根因确定为显存压力或计数器异常。Performance = NOT_CHARACTERIZED。当前 GitHub 1.0.0 Beta 安装包尚未包含本轮修正。
