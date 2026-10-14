# FFmpeg 共享编码库升级至 8.1.2

## 原因

系统 Scoop FFmpeg 为 8.1.2，支持 AV1 UHQ。Lunarc 原生桥原先按开发环境 FFmpeg 7.1.1 的头文件和导入库构建，运行时仍加载 libavcodec 61.19.101；该版只支持 HEVC UHQ。
此前视频处理组件只安装 FFmpeg/FFprobe 命令行程序，原生共享库来自另一套环境/安装器依赖，两个版本并不同步。

## 修改

- 使用 Gyan 固定版本 `ffmpeg-8.1.2-full_build-shared.7z`，大小 59,459,100 字节。
- 官方 SHA-256：`cba748035c21ce1431d0823c7a3a711f38616f89f87a265dceddf9b7f6749d2d`。
- 对同一包的头文件、MSVC import libraries 重建 `nvdec_bridge.dll` 与 `nvenc_bridge.dll`；不是将新 DLL 政名为旧 ABI。
- 新导入：avcodec/avformat 62、avfilter 11、avutil 60；在线组件同时安装所需 swresample 6、swscale 9 和 CLI。
- `prepare_ffmpeg_sdk.py` 校验官方包并提供可复现的匹配 SDK；构建环境默认采用项目下载的 SDK，保留显式覆盖。
- 视频组件 readiness 必须包含共享库，CLI-only 旧安装不能报告完整。Windows 自带 tar 可读取官方 7z，无需用户另装 7-Zip。
- 小安装器不再单独收集旧 FFmpeg DLL，避免 CLI 与原生编码库再次分开更新。许可文件随同组件保留。
- 本机源码 `runtime/tools`、当前独立开发环境 `build/standalone-runtime/tools` 与已有组件目录的视频运行库均已更新。不修改 WhisperJAV 或 Scoop 的环境。
- GPU 输出使用新库的真实 UHQ 查询；HEVC 与 AV1 均返回支持。没有降低为 HQ，没有改动 B 帧设置。

## 验收

- 两次 1280×720、2 帧完整 TrueHDR→P010 导出：AV1 UHQ 与 HEVC UHQ 均成功，`tune_adjustment=null`。确认真实 codec、10-bit、BT.2020/PQ/BT.2020nc、帧数；编码器图像主机传输计数为 0。
- 四种编码边界各 4 帧：HEVC/AV1 × SDR10/HDR10 均成功，旧 HEVC open ABI 仍有效；AV1 原始 metadata OBU 检查通过。
- 51 项 Python 相关检查、9 项组件/启动 Node 检查通过；前端构建通过。
- GPU 原始证据：`build/ffmpeg-upgrade/uhq-export/result.json`、`uhq-export.log`、`codec-boundary/boundary.json`、`codec-boundary.log`。
- 未重跑整段用户视频或进行性能 benchmark。本轮不发布安装包。

来源：[Gyan 官方构建与校验](https://www.gyan.dev/ffmpeg/builds/)，[固定包摘要](https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-8.1.2-full_build-shared.7z.sha256)。
