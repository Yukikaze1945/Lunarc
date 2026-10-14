# 1.1.0 Beta 发布检查

日期：2026-10-14。Windows x64，Electron 42.2.0。

## 安装包

- 文件：`Lunarc-1.1.0-beta-x64-Setup.exe`。
- 大小：146,679,078 字节（139.9 MiB）。
- SHA-256：`ce0b5aab0c0604a8f871008bb342941d3b4a18f05658e711bd1e02f029fdad4d`。
- 产品版本与安装器文件版本均为 `1.1.0-beta`。
- TypeScript/Vite/Electron 前端构建及 Windows x64 NSIS 安装程序构建通过。
- 打包前资源检查验证组件清单中的文件大小与 SHA-256。

## 本轮范围

本轮确认前端构建、组件资源清单检查和安装器生成通过。没有执行本次安装器的完整安装与首次启动验收、全量 GPU 回归、多机兼容性检查或性能基准。RTX VSR 模式与显存用途的局部设备检查分别记录于 [VSR 模式检查](2026-10-14-vsr-modes.md) 和 [显存用途检查](2026-10-14-vram-resource-purposes.md)。

局部设备检查不代表完整功能验收；Performance = NOT_CHARACTERIZED。
