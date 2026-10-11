# Lunarc 独立运行环境

当前 Electron 版本携带独立的 Windows x64 CPython、运行依赖、CUDA 运行库、FFmpeg、RAFT-Large 权重和项目原生桥接库。接收机器不需要安装 Python、Node.js、pip 或 CUDA Toolkit；NVIDIA 显卡驱动仍由机器安装。

## 构建

构建工具使用 Python 3.10+，需要 `pefile`。`--source-python` 指向已验证的 Windows x64 GPU 环境，其中需要 `packaging` 和以下包：numpy、opencv-python、torch、torchvision、fastapi、uvicorn、websockets、pydantic、Pillow、av。构建不会修改源环境。

```powershell
python packaging/build_runtime.py --source-python "<GPU 环境的 python.exe>" --ffmpeg-dir "<完整 FFmpeg 的真实 bin 目录>"
cd frontend
npm run electron:build
# 或生成可选择安装目录的安装程序：
npm run electron:installer
```

完整 GPU 环境超过 NSIS 单个 EXE 的内嵌数据限制，安装版采用离线两文件形式：`Lunarc-<版本>-x64-Setup.exe` 与同目录的 `lunarc-workbench-<版本>-x64.nsis.7z`。保持两者在同一文件夹，双击 Setup。安装器先校验本地数据包，缺失或损坏会停止，不下载环境。

不要传入 Scoop 的 shims 目录；应使用实际 FFmpeg 的 bin 目录，且必须支持 zscale、scale、select、showinfo。桥接库所需的共享 FFmpeg DLL 可以与命令行程序不同；构建器按 PE 导入表从源 Python 的 Library/bin 等目录解析它们。额外查找目录使用 `--dll-dir`。

默认输出 `build/standalone-runtime`。如有独立的官方 RAFT-Large 权重，使用 `--raft-weights`。运行环境的 SHA-256 清单记录实际收集的依赖版本和全部文件；打包前逐文件验证。构建失败的目录带有 `build-in-progress.json`，不能打包。

更新 Python/依赖版本或调整可选组件时，请选择全新的 `--output` 目录，验证后替换生成目录，不要把不同版本混在一起。生成环境和安装包不提交到 Git。

## 专有可选组件

默认不收集本机的 VFX SDK 和 DLAA DLL。接收机器可自行安装合法取得的对应运行库；缺失时诊断页显示对应功能缺少组件，不影响基础工作台启动。

本机私有构建如需携带这些组件，可使用 `--vfx-runtime "<SDK 根目录>"` 和 `--dlaa-dll "<nvngx_dlss.dll>"`。这些开关不是公开再分发授权，公开发布须另行核对实际运行库许可。DLSSNR 不再由 electron-builder 或 PyInstaller 自动收集；用户通过运行库管理器导入或确认来源后下载到独立缓存。

## 启动和检测

- 安装版只使用资源目录里的 Python；不会借用 PYTHON_PATH、系统 Python 或其他软件的环境。
- `_pth` 隔离用户 site、注册表 Python 和 PYTHONHOME/PYTHONPATH；PATH 只带私有运行库和 Windows 系统目录。
- 启动前实际导入基础依赖；失败后显示具体启动状态，日志写入用户数据目录的 `logs/backend.log`。
- WebSocket 传输依赖也在启动和打包前检查中；HTTP 接口正常不能代替预览和队列连接验收。
- 若同一端口被不同运行环境的旧后端占用，提示先退出旧版，不自动结束其进程。
- “运行与诊断 → 运行环境”实际检查 CUDA 运算、驱动接口、DLL 加载/ABI、FFmpeg 功能和模型文件。检测在独立进程中执行，有超时和缓存，点击“重新检测”才重新执行。
- “依赖齐全”不是模型/编码器运行成功。具体 SDK 特效、GPU 架构、输入编码和尺寸支持在创建实际会话时验证；不会把文件存在误报为运行成功。
- GPU 检测失败返回未知值和原因，显存显示本进程的缓存量，不冒充整卡占用，也不重复累计 allocated 和 reserved。

## 本轮验证范围

公开验收摘要：`docs/public-release.md`。本轮隔离了开发路径并实际启动打包程序；未声称已在无驱动的干净 Windows 虚拟机通过 GPU 验收。

## 公开源码许可

项目源码为 GPL-3.0-or-later；第三方组件保留自身许可。Git 仓库保留源码，Windows 安装文件通过 Releases 发布。打包配置不能授予第三方组件的再分发权；完整通知与组件信息见根目录 `THIRD_PARTY_NOTICES.md`，实际包内通知位于 `resources/licenses` 和各 Python 分发包的许可目录。

## GitHub Release 分卷

GitHub 的单个附件须小于 2 GiB。对已验收的离线安装器执行以下命令，按字节分割数据包，不重新压缩或改变安装内容：

```powershell
python packaging/build_release_assets.py --input "<nsis-web 输出目录>" --output "<新分卷目录>" --version 2.1.1
powershell -NoProfile -ExecutionPolicy Bypass -File "<新分卷目录>/Install-Lunarc.ps1" -VerifyOnly
```

验证合并后的整体 SHA-256 后发布两个分卷，并将安装 EXE、两份启动脚本、`installer-payload.json` 与安装说明打成小型安装入口 ZIP。用户将 ZIP 解压到分卷所在目录，运行 `Install-Lunarc.cmd`；源码构建仍可直接使用原离线两文件安装集。
