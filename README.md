# Lunarc

Windows 视频增强工作台。以简洁的中文界面组织可排序的渲染链，支持画面对比、实时预览与导出队列。

**公开源码 · GPL-3.0-or-later · Windows x64**

## 功能

- DLSSNR 单层 / 双层 / 三层重建；多尺度数据一致性融合仍为实验性。
- 可排序的 DLAA、DLSSNR、RTX VSR、NVIDIA VFG 卡片；功能是否可用取决于本机 GPU、驱动和运行库。
- NVIDIA VFG 固定倍率与目标帧率模式；5×–8× 标为实验性。VFG 为导出功能，实时预览保持原帧率。
- NVOF、AMD FidelityFX Optical Flow 与 RAFT-Large 引导；RAFT 支持长边 512 / 720 / 1080 分析尺寸。
- SDR 8-bit、SDR 10-bit 与实验性 HDR10 处理；HDR10 保留 PQ / BT.2020 / P010 / Main10 链路及静态元数据。
- FFmpeg 封装原音轨、字幕、章节和附件；队列诊断、素材拖入、连续视频对比与环境检测。

## 获取源码

```powershell
 git clone https://github.com/Yukikaze1945/Lunarc.git
 cd Lunarc
```

本仓库公开源码和测试，不提供预编译桥接 DLL、NVIDIA 专有运行库、模型权重或安装包。新仓库使用独立的源码快照历史。

## 源码运行

准备 Windows x64、支持相关功能的 NVIDIA GPU 和驱动、Python 3.10+、Node.js 22.18+、FFmpeg / FFprobe。RAFT 需要适配 GPU 的 CUDA 版 PyTorch；并非所有显卡都支持 Feature 18 或各项 NVIDIA 特效。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 按 PyTorch 官方说明安装适合 GPU 的 CUDA 版本。
$env:PYTHON_PATH = "$PWD\.venv\Scripts\python.exe"
cd frontend
npm ci
cd ..
```

FFmpeg 与 FFprobe 需能从 PATH 找到；FFmpeg 必须支持 zscale、scale、select、showinfo。RAFT-Large 首次使用可能下载权重，也可提前准备离线模型。已测独立环境使用 Python 3.10.18、torch 2.10.0+cu128、torchvision 0.25.0+cu128；这不是全部受支持版本的保证。

### 编译原生桥接库

安装 Visual Studio C++ Build Tools、Windows SDK；另行取得 NVIDIA DLSS SDK 和含头文件、导入库及共享 DLL 的 FFmpeg 开发包。

```powershell
$env:DLSS_SDK_DIR = "<DLSS SDK 根目录>"
$env:FFMPEG_SDK_DIR = "<FFmpeg 开发包根目录>"
cmd /c native\build_all.bat
```

构建脚本自动查找 Visual Studio。也可在 x64 Native Tools 命令提示符中运行，或设置 `VCVARS64`。运行时还需所链接 FFmpeg 共享 DLL，以及合法取得的匹配 DLSSNR 运行库 `nvngx_dlssnr.dll`；仅有普通 DLSS 超分 DLL 不能代替它。项目不会提供该运行库的下载或再分发授权。

AMD 光流另需 CMake 和与固定 SDK 版本匹配的官方 FidelityFX_SC 编译器：设置 `FFX_SC_EXECUTABLE` 后重新运行 `build_all.bat`。未提供编译器时构建会明确报告该组件未构建。详细版本与接口说明见 [AMD 光流依赖](docs/AMD_FLOW_DEPENDENCIES.md)。

DLAA、RTX VSR、VFG 使用本机另行安装的 NVIDIA 运行库；缺少组件时在诊断页显示实际缺项。路径配置见 [原生组件配置](docs/native-runtime.md)。

运行根目录的 **`run_gui.bat`**。该脚本构建并打开当前 Lunarc 工作台。页面“运行与诊断 → 运行环境”会实际检测导入、CUDA 运算、DLL / ABI、FFmpeg 和模型文件；检测齐全不代表所有输入及特效会话都可运行。

## 独立运行环境与安装包构建

项目提供私有运行环境收集器和离线安装脚本，详见 [打包说明](packaging/README.md)。独立包可携带 CPython、Python 依赖、CUDA 运行库、FFmpeg 与模型，不要求接收机器预装 Python / Node.js / CUDA Toolkit；显卡驱动仍需单独安装。

**打包脚本不是第三方运行库的再分发授权。** 本次公开发布仅包含源码；分发包含专有组件的组合二进制前，需要单独确认全部许可兼容性和所需授权。本项目未授予专有库链接例外。

## 测试

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests/unit -q
node --experimental-strip-types --test tests/node/*.test.ts
cd frontend
npm run build
```

GPU / ABI 测试需要先编译桥接库；真实媒体验收工具位于 `tests/helpers` 和 `tests/unit`，请查看各脚本的 `--help` 并指定自己的素材。开发机验收摘要见 [公开版验证范围](docs/public-release.md)。

## 已知限制

- HDR10、Feature 18 多层融合及高倍率插帧为实验性，不保证所有驱动、尺寸或素材支持。
- 当前预览输出为 8-bit 显示图像；HDR 预览使用显示映射，未经显示器校准，不能用它评判最终 HDR 亮度。
- 光流引导、渲染链特效与导出插帧的实际支持不同；缺少对应组件会明确失败，不以其他算法冒充。
- 部分 GPU 链路已测图像回读边界为零，但预览绘制、诊断和标量回读仍存在主机流量，不能称整条程序没有主机传输。
- 性能取决于输入尺寸、链路顺序和特效设置；公开版没有跨机器性能保证。Performance = NOT_CHARACTERIZED。
- 字幕尾部事件和音频 padding 可能超过视频结尾。

## 许可与致谢

项目整体按 **GNU GPL version 3 或任何后续版本**发布，完整文本见 [LICENSE](LICENSE)。允许按许可证使用、修改与再分发，不提供担保。

第三方源码保留原有许可及版权，不因项目采用 GPL 而被改写。旧项目 MIT 通知保留于 [LICENSES/Lunarc-legacy-MIT.txt](LICENSES/Lunarc-legacy-MIT.txt)。第三方范围、外部 SDK 与权重说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
