<p>
  <img src="docs/assets/lunarc-logo.svg" width="32" height="32" alt="Lunarc 青绿色光弧标志" />
  &nbsp; <strong>LUNARC</strong> &nbsp; / &nbsp; VIDEO ENHANCEMENT WORKBENCH
</p>

![Lunarc · Light in Motion. 让视频增强的每一步，都清晰可控。](docs/assets/lunarc-banner.png)

**Windows x64** &nbsp; · &nbsp; **源码发布** &nbsp; · &nbsp; **GPL-3.0-or-later**

一个把重建、画面对比与导出放在一起的中文视频增强工作台。排好渲染顺序，调整画面，检查结果。

[开始使用](docs/getting-started.md) &nbsp; / &nbsp; [运行库管理](docs/dlssnr-runtime.md) &nbsp; / &nbsp; [验证范围](docs/public-release.md)

---

## 01 / 工作台

![Lunarc 实际工作台：视频预览、可排序渲染链和画质参数](docs/assets/lunarc-workbench.png)

<sub>实际安装版界面，使用原创合成素材。截图展示操作布局，不作为画质或性能对比证据。</sub>

| 渲染链 · Enhance | 画面对比 · Inspect | 导出队列 · Deliver |
| :--- | :--- | :--- |
| 拖动卡片决定处理顺序，按需组合 DLAA、DLSSNR、RTX VSR 与 VFG。 | 原图、处理后与分屏预览；对比页支持两段视频连续播放。 | 配置编码与封装，查看进度、诊断任务、移除完成或失败的条目。 |

## 02 / 你能控制什么

| 功能 | 支持范围 | 状态与条件 |
| :--- | :--- | :--- |
| 神经重建 | DLSSNR 单层 / 双层 / 三层，分层尺寸和权重 | Feature 18 需要匹配的 GPU、驱动及 DLSSNR 运行库；多尺度数据一致性融合为**实验性**。 |
| 光流引导 | NVOF、AMD FidelityFX Optical Flow、RAFT-Large | 引擎能力与质量选项分别检测；RAFT 可选长边 512 / 720 / 1080 分析尺寸。 |
| 原生画质处理 | DLAA 模型预设、RTX VSR 质量档位，按倍率或分辨率输出 | 需要用户另行取得对应 NVIDIA 运行库；支持范围在实际会话中验证。 |
| 导出插帧 | NVIDIA VFG 固定倍率 / 目标帧率，镜头切换检测 | **仅用于导出**，实时预览保持原帧率；固定 5×–8× 为**实验性**。 |
| 色彩链路 | SDR 8-bit、SDR 10-bit、HDR10 | HDR10 为**实验性**，保留 PQ / BT.2020 / P010 / Main10 及静态元数据。 |
| 封装与素材 | 视频 / 图片拖入；音轨、字幕、章节、附件与元数据 | FFmpeg 封装；实际保留范围取决于输入流和所选输出容器。 |
| 运行库管理 | 版本选择、DLL / ZIP 导入、确认来源后下载、独立缓存 | SHA-256、NVIDIA 签名及版本检查；当前已核对 DLSSNR 310.8.0.0。 |
| 环境检测 | 包依赖、CUDA 运算、驱动接口、DLL / ABI、FFmpeg 和模型 | 展示实际检测结果；依赖齐全不等于每种特效已运行成功。 |

### 使用前了解

- 当前公开仓库提供**源码与测试**。预编译桥接 DLL、NVIDIA 专有运行库、模型权重和安装包不在公开发布中。
- HDR 预览通过 8-bit 显示图像进行映射，尚未校准显示器，不能用预览判断最终 HDR 亮度。
- 特效支持受 GPU、驱动、运行库、尺寸与素材影响；缺失组件会明确报告，不静默替换算法。
- 预览、诊断和标量回读仍有主机传输。部分 GPU 图像边界的零回读，不代表整个程序零主机流量。
- 尚未提供跨机器性能保证；**Performance = NOT_CHARACTERIZED**。字幕尾部事件和音频 padding 可能超过视频结尾。

## 03 / 从源码开始

```powershell
git clone https://github.com/Yukikaze1945/Lunarc.git
cd Lunarc
```

准备 Python、Node.js、FFmpeg、C++ 构建工具及所需 SDK，按 [源码运行指南](docs/getting-started.md) 安装依赖、编译桥接库，然后运行根目录 **`run_gui.bat`**。

第一次使用时，打开 **运行与诊断** 检查实际环境，并通过 **DLSSNR 运行库** 导入或选择合法取得的模型。普通 DLSS 超分 DLL 不能代替 DLSSNR。

<details>
<summary>关于本地安装包</summary>

项目提供独立运行环境收集器和离线两文件安装器，可将 CPython、依赖、FFmpeg 与模型放在应用目录中；显卡驱动仍需单独安装。

本地构建方法见 [打包说明](packaging/README.md)。组合安装包当前暂停公开发布；构建脚本、下载来源或文件签名均不授予专有组件的再分发权。本项目保持纯 GPL-3.0-or-later，未添加专有链接例外。

</details>

## 04 / 文档与验证

| 文档 | 内容 |
| :--- | :--- |
| [源码运行](docs/getting-started.md) | 前置环境、依赖安装、原生桥接库构建与测试。 |
| [运行库管理](docs/dlssnr-runtime.md) | 本地导入、版本切换、来源下载、哈希与签名检查。 |
| [原生组件](docs/native-runtime.md) | DLAA、RTX VSR、VFG 等外部运行库配置。 |
| [AMD 光流](docs/AMD_FLOW_DEPENDENCIES.md) | 固定 SDK 版本与官方编译器要求。 |
| [本地打包](packaging/README.md) | 独立环境、离线安装与启动检查。 |
| [验证范围](docs/public-release.md) | 已完成的真实检查、实验性范围和未覆盖的组合。 |

## 05 / 许可与致谢

Lunarc 整体按 **GNU GPL version 3 或任何后续版本** 发布，完整文本见 [LICENSE](LICENSE)。允许按许可证使用、修改与再分发，不提供担保。

第三方组件保留各自许可和版权；NVIDIA 运行库仍适用原协议。见 [第三方说明](THIRD_PARTY_NOTICES.md) 与 [历史 MIT 通知](LICENSES/Lunarc-legacy-MIT.txt)。光流、模型、封装与设计参考的来源保留在对应文档中。

---

<sub>Lunarc · Light in Motion.</sub>
