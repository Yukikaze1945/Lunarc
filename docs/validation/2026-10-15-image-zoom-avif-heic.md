# Alt 缩放与 AVIF／HEIC

- 工作台与图片导出页支持 Alt＋左键横向缩放，25%–800%。工作台移动期间使用 CSS 缩放已有画面，松开后再提交一次预览更新。浏览器检查验证了移动期间没有新增 seek 请求、松开后 zoom 约为 1.414。
- 导出页缩放、恢复适配、AVIF／HEIC 控件与保存请求均通过隔离浏览器检查。该检查使用模拟 API，不替代真实编码验证。
- AVIF 使用当前 Pillow 12.0.0 的 libavif；HEIC 使用 pillow-heif 1.8.1／libheif 1.23.6／x265。HEIC 所需 6.6 MB wheel 已安装到本机应用独立 Python 环境，保留组件许可证；源码环境未被修改。
- 97×65 RGB 彩色与渐变透明度样本分别保存 AVIF／HEIC，质量 70、100，文件真实解码、尺寸和 ICC 检查通过。质量 70 的透明度解码与原值最大差不超过 1；100 的透明度完全一致，HEIC RGB 无损样本也完全一致。AVIF 100 不据此宣称一般 RGB 无损。
- 复用上一轮通过 NR→DLAA 的 1536×2752 GPU PNG，实际编码并解码 AVIF、HEIC，尺寸正确；没有为相同画面重新执行模型推理。
- ffprobe 确认 AVIF 为 AV1、HEIC 为 HEVC。HEIC 内部编码尺寸可能补齐，实际 HEIF 图片尺寸以显示区域为准，Pillow/libheif 正确裁切。
- 11 个图片源／已有编码相关单元检查通过，TypeScript 与 Vite 构建通过。

证据：`build/image-format-gate/result.json`；`tests/helpers/image_formats_gate.py`；`build/image-export-gate/ui/result.json`。本轮仍提供 8-bit SDR 图片输出，没有新增 HDR 图片或高位深处理承诺，也没有发布安装包。
