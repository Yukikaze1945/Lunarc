# 原生组件配置

源码仓库不包含运行库和预编译桥接 DLL。源码运行前请按构建指南编译原生库。

| 配置 | 用途 |
| --- | --- |
| `DLSS_SDK_DIR` | 编译 DLSS 原生宿主所需的外部 SDK 根目录。 |
| `FFMPEG_SDK_DIR` | 编译桥接库所需的 FFmpeg `include` / `lib` 根目录；运行时也用于查找对应共享 DLL。 |
| `VCVARS64` | 可选，指定 Visual Studio `vcvars64.bat`；默认自动查找安装。 |
| `FFX_SC_EXECUTABLE` | 官方 FidelityFX_SC 编译器路径，用于 AMD 光流构建。 |
| `LUNARC_DLSS_DLL` | 可选，指定本机合法取得的 `nvngx_dlss.dll`，用于 DLAA；不同于 DLSSNR 运行库。 |
| `LUNARC_VSR_RUNTIME` | 可选，指定 NVIDIA VFX SDK 根目录，供 VSR / VFG 发现。也可在高级设置选择目录。 |
| `DLSS5_FFMPEG_DIR` | 可选，指定包含 FFmpeg / FFprobe 程序的目录；不要指向命令代理目录。 |
| `TORCH_HOME` | 开发环境的 PyTorch 模型缓存根目录；安装版使用组件目录中的模型缓存。 |
| `PYTHON_PATH` | 源码启动时选择 Python 解释器；安装版不采用此覆盖。 |

DLSSNR 通过“运行与诊断 → DLSSNR 运行库”管理：本地 DLL / ZIP 导入、可信目录、版本目录下载、缓存版本选择与移除。处理时使用校验后的缓存文件，详见 [运行库管理](dlssnr-runtime.md)。开发环境首次使用可将旧根目录 DLL 校验后迁移到缓存；独立运行模式必须明确导入并选择，取消选择后不自动恢复旧库。

DLAA 默认会检查 NVIDIA NGX 本机缓存，VSR / VFG 会检查本机 VFX 目录。10-bit / HDR VFX 路径要求新版 SDK Core 和对应特效模块。

`nvdec_bridge.dll` / `nvenc_bridge.dll` 的 FFmpeg 导入库版本必须和运行时共享 DLL 匹配。CUDA 驱动 DLL 来自 Windows 已安装的 NVIDIA 驱动，不应从开发机复制显卡驱动到软件目录。

本轮解码/编码桥使用 FFmpeg 8.1.2 full shared。先运行 `python packaging/prepare_ffmpeg_sdk.py`，构建脚本会采用下载到 `build/ffmpeg-sdk` 的匹配 SDK；也可通过 `FFMPEG_SDK_DIR` 明确指定。头文件、导入库和运行时 DLL 必须来自同一版。安装版的视频处理组件同时下载命令行程序与共享库；开发运行时优先从程序的 `runtime/tools` 查找，不依赖其他软件的旧 DLL。

源码运行可从同一下载包安装本地视频运行库：

```powershell
python packaging/prepare_ffmpeg_sdk.py
python component_install.py ffmpeg build/ffmpeg-sdk/ffmpeg-8.1.2-full_build-shared.7z runtime
```

缺少组件时，“运行与诊断 → 运行环境”会列出缺少的组件。特效是否可用还取决于显卡、驱动、运行库版本和输入尺寸。
