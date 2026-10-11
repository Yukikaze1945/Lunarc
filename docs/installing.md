# Windows 安装版

[下载 Lunarc 2.1.1](https://github.com/Yukikaze1945/Lunarc/releases/tag/v2.1.1)

## 下载三个文件

- `Lunarc-2.1.1-x64-Installer.zip`：安装入口。
- `lunarc-workbench-2.1.1-x64.nsis.7z.001`：数据第一卷。
- `lunarc-workbench-2.1.1-x64.nsis.7z.002`：数据第二卷。

将三个文件放在同一文件夹，解压安装入口 ZIP，然后双击解压得到的 **`Install-Lunarc.cmd`**。支持 Windows 默认的子文件夹解压方式；脚本会在自身文件夹及其上一级查找完整的数据分卷。

脚本先检查安装程序和分卷的大小、SHA-256，再合并数据包，核对整体 SHA-256，最后打开可选择目录的安装器。合并不需要 Python 或 7-Zip，使用 Windows 自带 PowerShell。也可用支持分卷的工具按原始顺序合并；最终文件名为 `lunarc-workbench-2.1.1-x64.nsis.7z`，与 Setup 放在一起。

数据总计约 2.5 GB。下载目录需要容纳分卷及约 2.5 GB 的合并文件；安装目录另需约 6 GB。首次合并和解压可能需要数分钟。文件缺失、下载不完整或校验失败时，脚本会停止；不会继续使用损坏的数据。

## 第一次启动

需要 Windows x64 和匹配的 NVIDIA GPU / 驱动。无需另装 Python、Node.js、pip 或 CUDA Toolkit。完整包携带用于 RAFT 的 PyTorch / CUDA 环境，这也是体积较大的主要原因。

打开 **运行与诊断 → 运行环境** 检查机器。DLSSNR 不随包携带：在 **DLSSNR 运行库** 导入合法取得的 DLL / ZIP，或确认来源后下载，再选择“使用”。DLAA、RTX VSR、VFG 所需运行库也由用户另行提供。

2.1.1 已在 RTX 5060 Ti 开发机实际安装、隔离环境启动、队列连接、真实 NR GPU 预览及连续播放验收通过。尚未在其他机器全面验证；HDR10、多尺度融合与 5×–8× 插帧仍为实验性，Performance = NOT_CHARACTERIZED。

项目源码为 GPL-3.0-or-later，第三方组件保留各自许可；见 [第三方说明](../THIRD_PARTY_NOTICES.md)。文件 SHA-256 在 Release 的 `SHA256SUMS.txt`，组件清单和来源在安装入口内的 `SOURCE-AND-NOTICES.md`。
