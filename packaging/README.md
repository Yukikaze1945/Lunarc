# 小安装包与在线组件

默认构建 Windows x64 NSIS 安装器。安装器只携带 Electron 界面、后端源码、项目桥接与最小原生依赖、固定版本 uv；不收集 Python、PyTorch/CUDA 或 RAFT 权重。旧离线收集器仅保留用于历史构建复现，不再用于默认发布。

## 构建

构建机需要 Python、`pefile`、`zstandard`、Node.js 与已编译桥接库。`--native-runtime-prefix` 指向含匹配共享 FFmpeg DLL 和 VC runtime 的 Conda 环境；它仅用于解析原生 PE 依赖，不收集该 Python 环境。

```powershell
python packaging/build_small_resources.py --output build/small-resources --native-runtime-prefix "<原生依赖环境目录>"
cd frontend
npm ci
npm run electron:installer
```

构建器从 PE 导入表收集最小 DLL 闭包，拒绝未解析的依赖，不复制 Windows 驱动或 DLSS/VFX 模型库。Conda 依赖的精确包摘要、许可与 recipe 从对应已校验的官方包取得；构建下载缓存不进入安装器。打包前核对资源清单内每个文件的大小与 SHA-256。

## 组件

- 基础：uv 管理的 CPython 3.10.18；`components/base.lock` 固定基础包版本与 wheel SHA-256。
- 视频：Gyan FFmpeg 8.1.2 essentials，固定归档大小与 SHA-256。解压后执行实际工具与必要过滤器检查。
- 可选 RAFT：`components/raft.lock` 固定 PyTorch 2.10.0+cu128、torchvision 0.25.0+cu128 及依赖；独立 site overlay。主包按 `raft-wheel.json` 校验并续传，安装时仅允许该 Windows wheel 的精确摘要；模型来自 PyTorch 官方源，完整 SHA-256 在 `raft-model.json`。
- uv 0.13.0：固定官方发行归档摘要，执行前再次核对工具摘要；MIT/Apache 许可证保留。

Electron 主进程管理安装，所以后端尚未启动时也可准备环境。完成标记在实际导入/工具检测成功后才写入；必要组件齐全才能启动后端。`-I` 隔离 Python 用户 site 与环境路径，安装版不借用开发机 Python。可选 RAFT 的 overlay 只在重启后按完成标记及锁文件身份启用。

组件和下载缓存在用户数据目录；新版本复用身份未变化的组件。锁版本变化会要求重新安装相应组件。运行组件不会自动升级，不提供未验证镜像源或任意下载地址。

## 验收

`tests/node/components.test.ts`：缺失、错误组件、安装工具损坏、单任务安装、取消与锁文件失效。

`tests/helpers/components_installer_gate.cjs <Lunarc.exe> <新证据目录> <已校验组件缓存> <短测试视频> <外部DLSSNR DLL>`：真实首次启动、组件安装 IPC、教程、隔离后端、原生 GPU 检测与无 Torch NR 预览。缓存复用和全新网络下载分别记录。

`tests/helpers/components_raft_gate.py <resources> <组件目录> <输出JSON>`：使用管理的 Python 与 `-I`，验证可选环境加载和短 CUDA RAFT 运行。`components_raft_ui_gate.cjs` 检查实际安装版重启后启用组件；不代表光流画质或性能验收。

默认发行是单个 Setup EXE。`INSTALLER.json` 标记在线组件格式，并记录安装器大小和 SHA-256；软件更新器验证该声明与 GitHub 资产、校验清单一致后才提供下载。
