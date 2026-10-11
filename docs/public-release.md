# 2026-10-11 公开源码版验证范围

## 已完成的本机运行环境验收

这些结果来自公开源码整理前的同日开发机验收，并非对其他机器或公开安装包的保证。

- Windows x64，RTX 5060 Ti；实际报告 15.9 GiB 显存。
- 独立 CPython 3.10.18、torch 2.10.0+cu128、torchvision 0.25.0+cu128；依赖闭包 27 个 Python 分发包。
- 运行环境清单 17,067 文件，约 5.15 GiB，逐文件大小与 SHA-256 校验。
- 运行环境、打包路径、HDR 导入、RAFT 分析相关 Python 检查：38 passed；后端启动环境 Node 检查：3 passed。
- 实际解压目录及安装目录 Electron 启动成功；外部 Python / FFmpeg 路径故意设为无效，PATH 不含开发环境；后端使用包内 Python。
- 环境页返回 34 项组件检查、9 项功能依赖；缺失 DLAA / VFX 覆盖路径正确报告缺项。
- 短视频加载与真实 DLSSNR GPU 预览成功；原图 / 处理后 JPEG 返回 121,956 / 129,760 字节。
- 禁止权重下载后，私有模型缓存中的 RAFT-Large 完成 128×128、2 次更新的 CUDA 推理，返回 `[1, 2, 128, 128]` 且位移有限。
- 离线两文件安装成功、卸载成功（均 exit 0）；数据包缺失 exit 2，损坏 exit 3。

原始本机报告和日志未上传，避免公开机器路径与媒体信息。上述包为本地验收产物，本仓库没有公开分发该二进制。

## 源码发布检查

公开版使用最新工作树的源码快照，保留独立运行环境构建器、环境检测、测试及第三方原始通知。项目许可为 GPL-3.0-or-later；构建脚本改为定位当前目录并发现 Visual Studio，SDK 路径由环境变量提供。

发布目录排除旧 Git 历史、本机配置、账号配置、原始诊断证据、媒体、模型和编译二进制。
从独立的公开源码目录实际验证：

- `native/build_all.bat`：18 个基础原生桥接 DLL 从源码构建成功（exit 0）。Visual Studio 自动发现成功，DLSS / FFmpeg SDK 从环境变量取得，未复制旧目录的桥接 DLL。AMD 构建未在本次重新执行，需要额外提供官方编译器；脚本明确报告跳过。基础 NVOF 编译有既有的编码/宏重复警告，其余构建继续通过。
- `frontend/npm ci` 与 `npm run build`：通过，TypeScript、Vite renderer / Electron main / preload 均生成。
- `test_runtime_environment`、`test_raft_analysis`、`test_comparison_audio`、`test_hdr_gpu_engine`、`test_no_misleading_claims`：合计 **137 passed**。这是相关范围检查，不是全量单元测试结果。
- `tests/node/backend_runtime.test.ts`：**3 passed**。
- 发布文件检查：没有检测到私钥或常见 API token 格式，没有可执行 PE 二进制、模型或媒体文件；个人目录与本机账号配置已排除。该检查不是完整安全审计。

## 限制

已有显卡驱动机器上的隔离环境验收，不能替代另一台干净 Windows 的 GPU 安装验收。文件存在及基础计算成功也不能代替每种素材、SDK 模型及 GPU 架构的实际会话检查。

本次公开整理不重新测量 GPU 渲染速度，不声称完成全量 HDR / VFG 导出回归；Performance = NOT_CHARACTERIZED。专有运行库兼容性及组合二进制分发许可需单独处理。
