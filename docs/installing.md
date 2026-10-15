# 安装 Lunarc

[下载 Lunarc 1.2.2](https://github.com/Yukikaze1945/Lunarc/releases/tag/v1.2.2)

## 首次安装

1. 下载并运行 **Lunarc-1.2.2-x64-Setup.exe**。
2. 启动 Lunarc，先安装 **基础运行环境**，再安装 **视频处理组件**。
3. 按教程进入工作台。教程也可以从侧栏或软件设置重新打开。
4. 在 **运行与诊断 → DLSSNR 运行库** 中下载或导入 DLL / ZIP，再选择“使用”。

无需自行安装 Python、Node.js 或 CUDA Toolkit。DLAA、RTX VSR 和 NVIDIA VFG 需要对应的 NVIDIA 运行库，配置方法见 [运行库配置](native-runtime.md)。

## RAFT 光流

只有使用 RAFT 时才需要安装 PyTorch/CUDA 和模型。在 **软件设置 → 组件管理** 中安装 **RAFT 光流**，完成后重启软件。

首次下载需要数 GB 的流量和磁盘空间。NVIDIA / AMD 原生光流、DLSSNR 及 GPU 解码编码不依赖 PyTorch。

## 下载与存储

组件下载需要连接 GitHub、PyPI、Python 分发源和 Gyan FFmpeg；RAFT 还需要连接 PyTorch 官方源。下载失败后可以重试。

FFmpeg、PyTorch/CUDA 主包和 RAFT 模型支持断点续传。其他 Python 依赖使用下载缓存，部分文件可能需要重新下载。

组件保存在 `%APPDATA%/DaVinci_DLSS5_Studio/components`。重新安装或更新 Lunarc 会保留版本匹配的组件。

## 更新软件

在 **关于 Lunarc → 软件更新** 中检查和下载更新。退出 Lunarc 后运行新版安装程序。1.2.2 是正式版本，无需勾选“包含预发布版本”。

本次从 1.0.0 Beta 重新编号，旧发布版已下架。使用 2.x 的用户需要手动下载安装 1.0.0-beta 一次；此后的更新可通过软件内更新器获取。使用原安装目录即可，已有组件和用户设置会保留。

## 注意事项

- 需要 Windows x64 和支持所选功能的 NVIDIA 显卡、驱动。
- HDR10、多尺度数据一致性融合和 5×–8× 插帧为实验性功能。
- Windows 工作台支持原生 FP16/scRGB 视频预览。显示器需启用 Windows HDR 才能显示 HDR；关闭时映射为 SDR。文件对比页仍使用 SDR 映射。
- 安装程序尚未进行代码签名。`SHA256SUMS.txt` 提供文件完整性校验。

项目使用 GPL-3.0-or-later，第三方组件适用各自的许可证。
