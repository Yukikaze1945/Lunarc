# 源码运行

```powershell
git clone https://github.com/Yukikaze1945/Lunarc.git
cd Lunarc
```

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

构建脚本自动查找 Visual Studio。也可在 x64 Native Tools 命令提示符中运行，或设置 `VCVARS64`。运行时还需所链接 FFmpeg 共享 DLL，以及合法取得的匹配 DLSSNR 运行库 `nvngx_dlssnr.dll`；仅有普通 DLSS 超分 DLL 不能代替它。“运行与诊断”提供 [DLSSNR 运行库管理器](dlssnr-runtime.md)，支持本地导入及用户确认的来源下载，文件仍受原许可约束，软件不授予取得或再分发权。

AMD 光流另需 CMake 和与固定 SDK 版本匹配的官方 FidelityFX_SC 编译器：设置 `FFX_SC_EXECUTABLE` 后重新运行 `build_all.bat`。未提供编译器时构建会明确报告该组件未构建。详细版本与接口说明见 [AMD 光流依赖](AMD_FLOW_DEPENDENCIES.md)。

DLAA、RTX VSR、VFG 使用本机另行安装的 NVIDIA 运行库；缺少组件时在诊断页显示实际缺项。路径配置见 [原生组件配置](native-runtime.md)。

运行根目录的 **`run_gui.bat`**。该脚本构建并打开当前 Lunarc 工作台。页面“运行与诊断 → 运行环境”会实际检测导入、CUDA 运算、DLL / ABI、FFmpeg 和模型文件；检测齐全不代表所有输入及特效会话都可运行。


## 独立环境与测试

安装版下载及分卷安装见 [安装说明](installing.md)；自行构建见 [打包说明](../packaging/README.md)。运行库、模型和编译二进制不放入 Git 源码历史。

```powershell
.\.venv\Scripts\python.exe -m pip install pytest
.\.venv\Scripts\python.exe -m pytest tests/unit -q
node --experimental-strip-types --test tests/node/*.test.ts
cd frontend
npm run build
```

GPU / ABI 测试需要先编译桥接库。媒体验收脚本在 `tests/helpers`；请查看各脚本的 `--help` 并指定自己的素材。已完成的检查及边界见 [验证范围](public-release.md)。
