# 独立运行环境与真实环境检测验收

## 生成环境

- Windows x64 CPython 3.10.18，PyTorch 2.10.0+cu128，torchvision 0.25.0+cu128。
- 收集 27 个 Python 分发包的实际依赖闭包、CPython/stdlib、PE 导入依赖、FFmpeg/ffprobe、官方 RAFT-Large 权重。
- 运行环境清单 17,067 文件，总计约 5.15 GiB；每个文件记录大小与 SHA-256。完整版本清单在生成目录的 manifest.json。
- `_pth` 隔离外部 Python；运行时无需 Python/Node/pip/CUDA Toolkit 安装。
- 操作系统和驱动 DLL 不复制。VC 运行库和本机桥接库需要的共享 FFmpeg 依赖按导入表收集。

## 自动检查

`tests/unit/test_runtime_environment.py` 与现有打包二进制、HDR 导入和 RAFT 分析测试合计 **38 passed**。

覆盖基础依赖损坏、CUDA 不可用时不伪造 GPU、显存不重复累计、检测超时/失败、结果缓存/手动刷新、原生 stdout 干扰、隔离 FFmpeg 查找、私有 DLL 不能替代启动程序所需同名 DLL。

`tests/node/backend_runtime.test.ts` **3 passed**，验证安装版拒绝外部 Python/PYTHONHOME/PYTHONPATH/FFmpeg SDK 路径，开发环境的显式失效路径不会被当成可用环境，并且源码变化会使旧后端指纹失效。

前端 TypeScript/Vite 构建通过；electron-builder 解压目录构建通过，打包前完整清单 SHA-256 验证通过。

## 实际打包程序

原始证据：`bench_out/standalone_20261011/electron.json` 与 `environment.png`。

使用 `frontend/release/win-unpacked/Lunarc.exe`，独立用户数据目录、测试端口 18927。外部 PYTHON_PATH/PYTHONHOME/PYTHONPATH/FFMPEG_SDK_DIR 指向不存在目录，PATH 不含开发环境。未结束用户端口 18923 上的后端。

通过：

1. 打包 Electron 启动自己的 Python 后端，进入 ready。
2. 返回实际 RTX 5060 Ti、15.9 GiB 显存、进程缓存量、运行环境指纹。
3. 诊断页返回 34 项组件检查、9 项功能依赖；isolated=true、standalone=true。
4. 故意把 DLAA/VFX 覆盖路径设为不存在目录，DLAA/VSR/VFG 明确显示缺少组件；其他检查通过。
5. 加载用户 bounce_24fps_5s.mp4，解码原图与实际 DLSSNR GPU 预览成功，分别返回 121,956 / 129,760 字节 JPEG。
6. 测试 Electron 退出后收回自己的后端进程。

补充模型验收：使用打包 Python、私有 TORCH_HOME、无开发 PATH，禁止权重下载，实际加载内置 RAFT-Large 权重并执行 CUDA 推理。128×128、2 次更新返回 `[1, 2, 128, 128]`，device=cuda:0，所有位移值有限。原始证据：`bench_out/standalone_20261011/raft-runtime.json`。这是运行依赖与模型可用性检查，不是画质或速度评测。

## 离线安装

生成 `C:\Users\18392\Downloads\Lunarc-2.1.0-offline\nsis-web\Lunarc-2.1.0-x64-Setup.exe`，配套同目录 `lunarc-workbench-2.1.0-x64.nsis.7z`（2,613,776,074 字节）。完整环境超过 NSIS 内嵌 EXE 的 2 GiB 数据限制，因此采用本地配套数据包；缺失返回 exit 2，损坏返回 exit 3，先停止再退出，不切换为在线下载。

实际 `/S /currentuser /D=<临时目录>` 安装成功（exit 0），配套数据包保留。随后从**安装目录**运行 Electron，使用独立测试配置和外部失效 Python/FFmpeg 路径，重复通过启动、真实环境检查、视频加载与实际 DLSSNR GPU 预览。环境结果中的 Python 路径为安装目录内私有 Python。首次安装后启动测试把窗口等待从 30 秒调整为 90 秒；原先 30 秒等待不足，不能据此认定程序未启动。

原始证据：`bench_out/standalone_20261011/installation.json`、`installed-electron.json`、`installed-electron-gate.log`。本机首次安装包括数 GB 的校验和解压，花费数分钟，不代表运行时处理速度。

实际静默卸载成功（exit 0），移除本轮临时安装与其注册项，未要求删除用户工作台数据。

## 限制

- 此验收是已有 NVIDIA 驱动机器上的应用环境隔离，不能代替另一台干净 Windows 机器的安装验收。
- 可选 NVIDIA VFX/DLAA 专有运行库默认不收集；安装或合法私有构建另行提供。驱动不随软件打包。
- 环境页显示的是实测 CUDA 运算和依赖/ABI 检查。具体 SDK 模型、HDR 编码参数和 GPU 架构支持仍由实际会话验证。
- 本轮没有重跑完整 HDR/VFG 导出或全量测试套件，没有性能提升结论。
- CPython 与依赖是本轮验证的固定快照，不宣称为最新安全版本；正式公开发布前应检查版本维护与第三方分发条款。
