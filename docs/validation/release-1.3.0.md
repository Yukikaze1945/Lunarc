# 1.3.0 发布检查

日期：2026-10-17。Windows x64，Electron 42.2.0。

## 安装包

- 文件：`Lunarc-1.3.0-x64-Setup.exe`。
- 大小：149,628,673 字节（约 142.7 MiB）。
- SHA-256：`9c73fd678f535cbcf5911571f361238876159cf17f07a7cca0f2d16b3845b7bc`。
- 安装器文件版本与应用 package 版本均为 `1.3.0`。
- Authenticode 状态：未签名。
- 发布目录包含安装器、`INSTALLER.json`、`INSTALL.md` 和 `SHA256SUMS.txt`。

## 构建检查

- `npm run electron:installer` 完成；TypeScript、Vite、Electron 生产构建和 Windows x64 NSIS 构建通过。
- 打包前资源清单、资源文件大小及 SHA-256 检查通过；HDR/JPEG XL 编解码文件与固定清单匹配。
- 为保持已验证的运行时文件集合，本次沿用上一版的原生依赖与许可证归档，并叠加 1.3.0 的组件锁文件和图片编解码资源。
- Vite 对 500 kB 以上的 JavaScript chunk 发出体积提示；构建仍成功。

## 检查范围

本轮确认生产构建、安装包生成、资源校验和本地发布附件摘要。没有重新运行完整自动测试套件、交互式安装验收或 GPU 导出回归；也没有进行多机兼容性或性能测试。功能的局部验证范围见对应的 [2026-10-16](2026-10-16-source-and-hdr-preview.md) 与 [2026-10-17](2026-10-17-mcp-http.md)、[VFG 16×](2026-10-17-vfg-16x.md) 记录。
