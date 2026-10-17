# UHDR JPEG 与 JPEG XL

图片导出页增加 **UHDR JPEG**、**JXL**。默认继续保存完整尺寸缓存，仅在 CPU 上编码；不加载 DLAA、NR 或 VSR，也不改变视频编码器。

| 格式 | 输出 | 透明度 | 质量 100 |
| --- | --- | --- | --- |
| UHDR JPEG (`.jpg`) | Display-P3 SDR 基图＋多通道 HDR 增益图 | 不支持 | JPEG 仍为有损编码 |
| JXL (`.jxl`)，SDR | 8-bit sRGB RGB(A) | 支持 | 对送入编码器的像素无损 |
| JXL (`.jxl`)，HDR | 16-bit BT.2020/PQ RGB(A) | 支持 | 对量化后的 PQ 像素无损；不逆转色域边界裁剪和量化 |

UHDR 只接受 HDR 渲染结果，不能把普通 SDR 文件直接标成 HDR。普通 JPEG 查看器显示其 SDR 基图，支持增益图的查看器可恢复 HDR。JXL 需要兼容查看器。图片导出页的缩略图仍是 SDR 显示映射。

## 与 Starshot 的关系

用户提供的 `starshot-uhdr-work` 构建日志指向本机 `D:/coding/Starshot` 源码。参考其 `ImageSaver.cs`、`UhdrColor.cs`、`UhdrGainmapEffect.cs`：

- UHDR 在转换工作色域之前不裁掉 scRGB 负通道，采用 Display-P3 工作域，并对域外颜色沿中性点压缩。
- scRGB 的 1.0 固定为 80 nit；libultrahdr 线性 HDR 输入的 1.0 为 203 nit，显式转换，不能直接传递而放大亮度。
- 本实现使用 libultrahdr 2.0.2 的 HDR-only 原始 FP16 API，自动构造 SDR 基图和增益图，开启多通道、原尺寸增益图。没有逐行移植 Starshot 的 D2D 手工增益图流程，两个软件的 SDR 基图不保证逐像素一致。
- UHDR 编码目标亮度取工作域像素的最大分量亮度，下限 203 nit、上限 10000 nit；这是编码器容量参数，不添加伪造的母版显示器信息。
- JXL 使用 Starshot 的 BT.2020/PQ 16-bit 路线，经 libjxl 0.11.1 编码；PQ 转换复用现有 HDR 保存函数。透明度是线性覆盖率，不套 PQ。

## 本机组件与后续打包

本轮组件放在 `runtime/tools/image-codecs/`，与 Starshot 的文件独立。没有改动 Starshot，也不依赖它的安装路径。约 9 MB 的原生图片编码工具不需要 PyTorch/CUDA。

搜索位置：显式环境变量 `LUNARC_IMAGE_CODECS_DIR`（指定后不回退），应用的 `image-codecs/`、当前运行环境的 `tools/image-codecs/`，源码运行时的 `runtime/tools/image-codecs/` 或 `build/standalone-runtime/tools/image-codecs/`。组件缺失时对应格式禁用并给出原因。

确切二进制及许可证文件的 SHA-256 记录在 `packaging/image-codecs.json`，来源为用户现有 Starshot 运行库。`packaging/prepare_image_codecs.py --source <组件目录> --output <目标目录>` 验证全套文件后复制。小安装包资源准备步骤在本机组件存在时收集该小型组件，保留 `licenses/`；本轮没有构建或上传新安装包。

组件包含 libultrahdr、libjxl、Brotli 及相应原始许可/专利通知。本次没有复制截图程序或其模型、GPU 捕获组件和 .NET 环境。

## 验证与限制

`tests/helpers/image_extra_codecs_smoke.py` 使用缓存像素实际保存 UHDR、HDR JXL、SDR JXL，不运行 GPU。检查 UHDR 基图和增益图、原生 HDR 解码后的亮度比例（约 0.994，JPEG/增益图为有损）、JXL 的 16-bit/PQ 标记与 16-bit 解码、透明度以及 SDR JXL 再导入。证据为 `build/image-extra-codecs/evidence.json`。

15 项定向单元检查和前端构建通过。浏览器模拟后端检查覆盖 UHDR 仅 HDR 可选、`.jpg` 扩展名及无透明度、JXL 的 `.jxl`、HDR16 和透明度控件。

当前支持 8-bit sRGB JXL 导入。高位深或 HDR JXL 不会自动降成 SDR 后进入图片处理链；UHDR JPEG 重新导入时与普通 JPEG 一样只读取 SDR 基图。HDR 图片输入是另一个接口范围。

上一个任务已用用户 8K 图片验证原有处理链和 HDR AVIF；本轮新增格式使用小尺寸真实编解码检查，未再重复运行该 GPU 链，也不提供跨机器性能承诺。

参考：[libultrahdr API](https://github.com/google/libultrahdr/blob/v2.0.2/ultrahdr_api.h)、[JPEG XL 色彩与亮度说明](https://libjxl.github.io/libjxl/api_color.html)。
