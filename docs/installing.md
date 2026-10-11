# Windows 安装版

[下载 Lunarc 2.1.2](https://github.com/Yukikaze1945/Lunarc/releases/tag/v2.1.2)

## 安装与首次启动

1. 下载 **Lunarc-2.1.2-x64-Setup.exe**，运行安装程序。无需下载分卷或合并文件。
2. 首次启动显示使用教程。先安装 **基础运行环境**，再安装 **视频处理组件**。
3. 组件安装完成后进入工作台。以后可以从侧栏或软件设置重新打开教程。

安装包约 140 MiB，包含桌面界面、项目桥接库与其原生依赖、组件安装工具。Python、基础 Python 依赖和 FFmpeg 命令行程序首次启动时联网下载；PyTorch/CUDA 和 RAFT 权重仅在安装可选的 **RAFT 光流** 时下载。不再提供完整离线包。

组件位于 `%APPDATA%/DaVinci_DLSS5_Studio/components`，独立于软件安装目录。重新安装或更新软件会复用版本仍匹配的组件与已校验的下载文件。首次安装 RAFT 仍有数 GB 的下载与磁盘开销，安装后需要重启软件。

安装需要连接 GitHub、PyPI、Astral 的 Python 分发源和 Gyan FFmpeg；RAFT 还需要连接 PyTorch 官方源。失败时显示具体原因，可重试。暂停保留已经完成的下载；FFmpeg、PyTorch/CUDA 主包和 RAFT 模型支持断点续传。其他 Python 依赖由 uv 管理下载缓存，不保证每个文件都能续传。

## 模型运行库与显卡

Windows x64 和匹配的 NVIDIA GPU / 驱动仍是 GPU 功能的前提。不需要自行安装 Python、Node.js、pip 或 CUDA Toolkit。

DLSSNR 不随安装包携带：在 **运行与诊断 → DLSSNR 运行库** 中导入合法取得的 DLL / ZIP，或确认来源后下载，再选择“使用”。DLAA、RTX VSR 与 NVIDIA VFG 也需要对应 NVIDIA 运行库。

**不装 RAFT 不影响原生 GPU 功能。** 显卡检测、原生 NVIDIA / AMD 光流、DLSSNR 和 GPU 解码编码不依赖 PyTorch；各特效的设备和模型兼容性仍在实际会话中验证。

HDR10、多尺度数据一致性融合和 5×–8× 插帧仍为实验性。预览经过 8-bit 显示映射，不能用来判断显示器上的 HDR 亮度。Performance = NOT_CHARACTERIZED。

## 更新

在 **关于 Lunarc → 软件更新** 中检查、下载和校验新版安装包。退出软件后运行新的 Setup；组件目录会保留。此方式没有安装器增量更新，运行组件也不会自动追随最新版升级。

项目为 GPL-3.0-or-later，第三方组件保留各自许可。Release 提供 `SHA256SUMS.txt` 和 `INSTALLER.json`；安装目录保留组件许可证、来源与固定版本记录。
