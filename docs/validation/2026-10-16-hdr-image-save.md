# HDR 图片保存：实现与待验收范围

本轮只做代码检查和前端编译。用户正在运行 GPU 任务，明确要求**不进行导出测试**。未启动 GPU、模型、图片或视频导出，未重启后端、替换 DLL、安装组件或更新安装包。

已完成：6 个 Python 文件的 AST 语法检查、三种翻译目录的 JSON 与键一致性检查、`npm run build`（TypeScript 与 Vite）均通过。前端仍有原有的大 chunk 提示，不影响编译。上述检查不验证实际 HDR 编码结果。

## 实现

- 图片导出 API 增加 `color_mode`：`sdr`（兼容旧请求）或 `hdr10`。
- HDR 要求渲染链启用 TrueHDR，并使用完整尺寸线性 FP16/scRGB 结果。不会经过旧的 SDR 显示映射或 8-bit RGB 图。
- Rec.709 scRGB（1 = 80 nit）转换为 BT.2020 线性 RGB，再使用 ST2084 PQ 编码。按行转换限制主机内存临时量；不增加全尺寸 GPU 处理纹理。
- AVIF：CPU libaom-av1，10-bit YUV444，AV1 High profile；主图 CICP 9/16/9、全范围，透明度使用独立灰度辅助图。
- HEIC：pillow-heif 16-bit RGB 输入，明确指定 10-bit 输出、4:4:4 和 CICP 9/16/9、全范围；支持源透明度。
- 不附加 sRGB ICC，不添加未测量的 mastering-display 元数据。BT.2020 负值与超过 10000 nit 的分量裁剪计数进入返回报告；非有限值报错。
- 保存时先写临时文件、核对文件实际位深与色彩声明，再原子发布；不覆盖源图，失败清理临时文件。
- 前端区分 SDR / HDR；HDR 仅允许 AVIF、HEIC。格式能力查询只读取编码器帮助与组件信息，不初始化 GPU。
- HDR 保存结果不会混入 SDR 预览缓存。图片导出页暂时显示原图缩略图并明确注明；HDR 图片重新导入仍未支持。

## 后续验收（本轮未执行）

- TrueHDR → AVIF / HEIC：检查亮度、色彩、透明度、尺寸和真实 10-bit 文件声明。
- 外部 HDR 查看器呈现、Windows HDR 开关下的显示行为。
- 带透明度 / 奇数尺寸 / 放大链 / 覆盖保护 / 编码失败恢复。
- 对比 SDR 旧保存行为，确认 HDR 输出不会污染后续预览缓存。
- 新增单元测试在 `tests/unit/test_image_hdr_export.py`，本轮未运行。

## 接口依据

- [pillow-heif 高位深保存](https://pillow-heif.readthedocs.io/en/stable/saving-images.html)
- [pillow-heif 编码位深与 NCLX 参数](https://pillow-heif.readthedocs.io/en/stable/_modules/pillow_heif/misc.html)
- [FFmpeg AVIF 封装器（主图与 alpha 辅助流）](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/movenc.c)
- [FFmpeg libaom-av1 still-picture 接口](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/libaomenc.c)
