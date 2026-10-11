# DLSSNR 运行库管理

打开Lunarc的“运行与诊断 → DLSSNR 运行库”。源码版使用根目录 `run_gui.bat` 启动。

## 使用

- **本地导入**：选择或拖入 DLL / ZIP，也可粘贴路径。ZIP 仅读取其中唯一的 `nvngx_dlssnr.dll`，不执行或安装包内其他文件。原文件保留。
- **版本目录**：列出已核对的 DLSSNR 版本。当前只有 310.8.0.0；下载来源为 DLSS5-Feeder 安装器引用的社区仓库，不是 NVIDIA 官方发布。勾选来源及使用权确认后下载，再选择“使用”。不自动更新或切换模型。
- **可信来源**：添加含 DLL 的本机目录，或 HTTPS DLL / ZIP 地址。下载必须填写 DLL SHA-256；ZIP 下载还需填写 ZIP SHA-256。
- **版本管理**：同版本不同字节按不同 SHA-256 分开缓存；可切换、取消选择、校验或移除。未验收但签名有效的版本默认不可选择，明确勾选实验性使用后才可启用。此选项不能跳过签名、产品、架构及哈希校验。

运行库默认位于应用用户数据目录的 `runtimes/dlssnr`；开发环境通常是 `%LOCALAPPDATA%\Lunarc\runtimes\dlssnr`。Electron 传入其自身用户数据根目录。页面显示实际位置。`LUNARC_DATA_DIR` 可指定根目录；`LUNARC_DLSSNR_DLL` 仅在尚无缓存配置时用于首次校验导入，不覆盖已选版本。

## 校验与会话

导入时先复制到独立临时文件，检查 PE32+ / x64 DLL、SHA-256、Windows Authenticode、NVIDIA Corporation 发布者、NVIDIA DLSSNR 产品信息与四段版本号，再放入以哈希命名的版本目录。签名检查只使用 Windows 自带 PowerShell 模块，不采用开发环境或用户模块路径。

每次会话打开前重新核对缓存哈希。同一进程只对已校验且字节未变的文件复用签名检查结果；“重新校验”会再次检查签名。缓存丢失或被修改时明确失败，不回退其他 DLL。根会话获得缓存路径，独立多层会话继承同一运行库版本，各自保留独立宿主会话。

导出进行中不允许切换或移除运行库。切换前关闭预览会话，GPU 预览在所属线程关闭，下次请求重新创建。取消选择后不会自动启用其他版本。运行库检查通过不代表所有 GPU、分辨率、HDR 或其他渲染组合兼容。

## 下载

HTTPS 下载使用证书及主机名校验、流式写入和大小/超时边界。禁止本机、内网或保留地址；允许 GitHub 发布附件重定向到固定的 GitHub 资产域名。仅已核对的目录 URL 支持 TUN 代理的 198.18.0.0/15 假 IP 映射，依然校验证书和两层固定哈希；任意自定义地址不享有此例外。

先写 `.part`，包哈希通过才保存 ZIP 下载缓存；之后解包并核对 DLL 哈希和签名。已有缓存须重新匹配包哈希才复用。不使用半成品或损坏缓存，不自动转向其他镜像。支持取消，读取或校验阶段结束后响应；单次网络读取最多等待 20 秒，Windows 签名检查最多 30 秒。尚未实现断点续传。

ZIP 不允许越界路径、符号链接、加密、重复目标 DLL 或超限文件；不使用通用解压到目录操作。下载文件上限与解包后的 DLL 上限均为 512 MiB，ZIP 最多 128 个条目。

## 参考机制

这是本项目独立实现的管理器，没有复制上述项目的代码或下载脚本。

- [DLSS Swapper](https://github.com/beeradmoore/dlss-swapper/blob/375e8a43c4a8354cf7d9c8a85e2ba07699fa4760/src/Pages/LibraryPageModel.cs)：版本记录、本地导入、下载缓存分离。其当前 manifest 列出了普通 DLSS / DLSSD / DLSSG 等类型，未提供 DLSSNR 类型，普通 DLSS 清单不能用于 DLSSNR。
- [DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder/blob/34edfe16626d52a8f66c5996c58d5b7c982d3023/tools/Install-DLSS5Feeder.ps1)：本机来源覆盖、临时下载文件、ZIP 内目标文件解析；本项目额外要求固定哈希和 NVIDIA 签名。
- [Overglaze](https://github.com/HANsoA-KevinO/overglaze/blob/4d49c0a64ca96583910c3bcc8694165d9ebc4ccc/docs/MODEL.md)：用户提供模型、复制后按已知 SHA-256 管理、拒绝悄悄替换；本机 DLL 与其记录的 310.8.0.0 哈希一致。

Lunarc 使用 GPL-3.0-or-later。安装包不包含 DLSSNR DLL，运行库适用其原许可证。
