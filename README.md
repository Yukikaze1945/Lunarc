<p>
  <img src="docs/assets/lunarc-logo.svg" width="32" height="32" alt="Lunarc" />
  &nbsp; <strong>LUNARC</strong>
</p>

![Lunarc — 视频增强与插帧](docs/assets/lunarc-banner.png)

**Windows x64** &nbsp; · &nbsp; **GPL-3.0-or-later**

Lunarc 是一款视频增强软件，支持 DLSSNR、DLAA、RTX VSR 和 NVIDIA VFG，可调整处理顺序、预览画面并批量导出。

[下载安装版](https://github.com/Yukikaze1945/Lunarc/releases/tag/v2.1.2) &nbsp; / &nbsp; [安装说明](docs/installing.md) &nbsp; / &nbsp; [源码运行](docs/getting-started.md)

## 01 / 界面

![视频预览、渲染顺序与画质设置](docs/assets/lunarc-workbench.png)

<sub>截图中的视频为演示素材。</sub>

| 画面处理 | 视频对比 | 批量导出 |
| :--- | :--- | :--- |
| 拖动卡片调整 DLAA、DLSSNR、RTX VSR 和 VFG 的处理顺序。 | 查看原图、处理结果和分屏预览，对比导出前后的视频。 | 设置编码、画质和保存位置，管理导出任务。 |

## 02 / 功能

| 功能 | 说明 |
| :--- | :--- |
| DLSSNR | 单层、双层和三层处理，可调整各层分辨率与权重。多尺度数据一致性融合为实验性功能。 |
| DLAA / RTX VSR | DLAA 模型预设；RTX VSR 质量档位，按倍率或指定分辨率输出。 |
| 光流 | NVIDIA NVOF、AMD FidelityFX Optical Flow 和 RAFT-Large。RAFT 分析尺寸可选长边 512、720、1080。 |
| NVIDIA VFG | 按固定倍率或目标帧率插帧，支持镜头切换检测。仅用于导出；5×–8× 为实验性选项。 |
| 色彩 | SDR 8-bit、SDR 10-bit 和 HDR10。HDR10 为实验性功能，支持保留静态 HDR 元数据。 |
| 素材与封装 | 拖入视频或图片；导出时保留音轨、字幕、章节、附件与元数据，具体取决于输出容器。 |
| 运行库管理 | 下载、导入和切换 DLSSNR 版本，检查文件摘要、数字签名和版本。 |

## 03 / 安装

1. 在 [Releases](https://github.com/Yukikaze1945/Lunarc/releases/tag/v2.1.2) 下载并运行 `Lunarc-2.1.2-x64-Setup.exe`，约 140 MB。
2. 首次启动按提示安装基础运行环境和视频处理组件。
3. 在 **运行与诊断 → DLSSNR 运行库** 中下载或导入运行库，再选择“使用”。

RAFT 光流需要另外安装 PyTorch/CUDA 和模型，可以在 **软件设置 → 组件管理** 中安装。不使用 RAFT 时无需下载这些文件。更新软件会保留已安装且版本匹配的组件。

### 系统要求与限制

- Windows x64、支持所选功能的 NVIDIA 显卡及驱动。
- DLSSNR、DLAA、RTX VSR 和 VFG 需要对应的 NVIDIA 运行库，安装包不包含这些文件。
- 首次安装组件需要联网；RAFT 下载量为数 GB，安装后需重启软件。
- HDR 预览经过显示映射，不能作为最终 HDR 亮度的判断依据。
- 处理速度取决于显卡、素材分辨率和所选设置。已测试的环境与功能见 [测试记录](docs/public-release.md)。

## 04 / 源码运行

```powershell
git clone https://github.com/Yukikaze1945/Lunarc.git
cd Lunarc
```

按 [源码运行指南](docs/getting-started.md) 安装依赖、编译原生库，然后运行根目录的 **`run_gui.bat`**。

## 05 / 文档

| 文档 | 内容 |
| :--- | :--- |
| [安装说明](docs/installing.md) | 安装、组件下载和更新。 |
| [DLSSNR 运行库](docs/dlssnr-runtime.md) | 下载、本地导入和版本管理。 |
| [DLAA / VSR / VFG 配置](docs/native-runtime.md) | NVIDIA 运行库与 SDK 路径。 |
| [AMD 光流](docs/AMD_FLOW_DEPENDENCIES.md) | 编译依赖与 SDK 版本。 |
| [软件更新](docs/app-updates.md) | 检查和下载新版安装程序。 |
| [多尺度融合](docs/multiscale-solver.md) | 求解方法、实测耗时与限制。 |
| [构建安装包](packaging/README.md) | 打包方法与依赖。 |
| [测试记录](docs/public-release.md) | 测试环境、结果和未覆盖的项目。 |

## 06 / 许可证

Lunarc 使用 **GPL-3.0-or-later**，详见 [LICENSE](LICENSE)。第三方组件适用各自的许可证，见 [第三方说明](THIRD_PARTY_NOTICES.md)。
