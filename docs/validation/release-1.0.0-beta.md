# 1.0.0 Beta 发布检查

日期：2026-10-13。Windows x64，Electron 42.2.0。

## 安装包

- 文件：`Lunarc-1.0.0-beta-x64-Setup.exe`。
- 大小：146,729,942 字节，约 140 MiB。
- SHA-256：`955dc093362e1100849c8cd293178dfb014a3b1885fffe59398a16cd4a2f3f3f`。
- 产品版本与安装器文件版本均为 `1.0.0-beta`。
- 前端与 Electron 构建通过；NSIS 安装程序构建通过。

## 本轮范围

`tests/helpers/release_package_gate.cjs` 检查实际打包目录：

- 当前 Python 模块、前端构建产物及所有随包原生 DLL 的摘要与构建输入一致。
- 组件资源清单中每个文件的大小和 SHA-256 一致。
- GPL、react-range 与原生依赖的许可说明随包保留。
- 不携带 Python 环境、PyTorch/CUDA、RAFT 模型或 NVIDIA 专有模型运行库。
- 使用独立空白配置目录启动打包的 Electron：首次使用教程显示正常，缺失组件被正确识别，更新器报告 `1.0.0-beta`，未发现前端运行错误。

这些检查通过。证据保存在本机忽略目录 `build/release-beta-package-check`。本轮没有重新下载运行组件、改动用户安装目录、运行全量 GPU 回归或测量性能；不能将打包与首次启动检查当作完整功能验收。

原生预览及区间导出的开发检查见 [GPU 预览](../native-hdr-preview.md) 与 [预设和区间预览](../workbench-presets.md)。Performance = NOT_CHARACTERIZED。

## 版本迁移

原 `2.x` 更新器只接受更大的版本号，因此需要手动安装本版一次。应用标识、用户配置目录和组件目录保持兼容；没有清除用户设置或已安装组件。
