# 图片模型、多层 VSR 与视野修复

## 改动

- 图片放大磁贴：IllustrationJaNai V3 的 4× DAT2 去噪／细节版；AnimeSharp V4 的 2× RCAN 标准版。
- 原生渲染链用独立 VSR 实例 ID，旧 `vsr` 保持兼容。顺序、开关、每层参数、NR 入口尺寸、VFG 边界尺寸、最终尺寸和诊断都按实际链路解析。
- 图片模型通过明确的 CUDA 上下文设备复制接入 GPU Transfer，不将外部 CUDA 指针直接包装成 Torch 张量，不增加模型阶段的 CPU 图像回读。
- 默认整图 FP32；分块为手动选项，32 像素输入上下文。模型分块结果可能不同；整图处理大图片仍可能显存不足。没有自动切换精度、模型版本或分块。
- 安装版模型组件独立安装，共用已有 PyTorch／CUDA。固定版本与 wheel 哈希；权重从作者发布页下载到用户缓存，校验 SHA-256 和长度。本地导入校验哈希，pth 用 weights_only=True。
- 本机源码运行环境已安装推理依赖，并缓存 AnimeSharp 标准版、IllustrationJaNai 去噪版。细节版可在磁贴中下载／导入；本次未执行其 GPU 检查。
- Alt＋滚轮缩放、左键移动、普通滚轮上下查看、对比线拖动均在前端重绘完整图像；不重新请求模型渲染。视野边界按实际缩放尺寸处理。

## 检查

- 55 个相关 Python 检查通过；7 个组件管理 Node 检查通过；生产构建通过。
- 浏览器隔离接口检查通过：缩放／移动／滚动／对比线不新增 seek；边缘立即回拖；重复 VSR、独立档位和开关、两个图片模型磁贴。
- 实际 RTX 5060 Ti CUDA 导出：65 × 73 SDR RGBA 图片，保持透明度、准确裁掉输入补齐；五种链路通过。每个模型／VSR 阶段 `host_image_bytes=0`。这不包含初始 CPU 图片上传和保存所需的最终显示／导出回读。

| 链路 | 最终尺寸 | 各阶段首次调用墙钟耗时 |
| --- | --- | --- |
| animesharp | 130 × 146 | animesharp 9107.93 ms |
| illustration | 260 × 292 | illustration 14537.76 ms |
| vsr-two | 260 × 292 | vsr 1367.80 ms / vsr:second 230.09 ms |
| model-vsr | 260 × 292 | animesharp 887.09 ms / vsr 1112.75 ms |
| model-nr-vsr | 260 × 292 | animesharp 1092.34 ms / vsr 922.91 ms |

上述耗时含模型加载、资源建立与同步，首次加载时上下文初始化成本很高；它们不是稳态性能基准，不可换算为持续预览 FPS。此处 NR 推理耗时未列入模型／VSR 阶段列。

证据：`build/image-model-gate/result.json`，输出 PNG 位于同目录；浏览器记录 `build/image-export-gate/ui/result.json`。

本轮未改原生 DLL、未发布安装包或推送 GitHub。安装版新组件需要后续重新打包；本机源码版退出后运行 `F:\dlss5standaloneV2\run_gui.bat`。
