# HDR 全范围解码修复

## 原因

用户的 PS5《跑车浪漫旅 7》录像为 VP9 Profile 2、3840×2160、10-bit 4:2:0、BT.2020/PQ/NCL。ffprobe 的流信息为 `color_range=tv`，但第一帧为 `pc`。NVDEC 同样返回 `2/9/16/9`。

旧桥接只接受 range=1。导出准入基于流信息先通过，真正处理解码帧时 range=2 被拒绝。此错误是缺少输入范围支持，不是设备移除或显存分配失败。

## 修改

- HDR 解码允许明确的 range=1 或 2，其余色彩检查保留。
- 全范围以 `Y/1023`、`(Cb−512)/1023`、`(Cr−512)/1023` 归一化，在原有解码 shader 内完成；有限范围保留原有算式。
- 仍使用 PQ→线性 scRGB FP16，保持有符号色彩。编码输出继续采用有限范围 P010。
- 导出准入接受明确全范围标记，未指定范围仍拒绝。SDR10 的范围支持本轮未扩展。
- ABI 与工作纹理布局不变，使用原有根常量中的保留字段。
- DLL 被运行中的后端占用时留待下次启动替换，启动脚本新增颜色桥接的待更新处理。

## 验证

- 326 个相关单元检查通过，1 个跳过。
- MSVC `/W4 /WX` 编译通过；新 DLL 位于 `native/build/hdr_colour/gpu_hdr_scrgb_bridge.dll`，SHA-256：`a5426be7df3f0833bc8659dd93b93c142ac5fe55fe7a72eb52552523c4a3afc4`。
- `tests/helpers/hdr_full_range_gate.py`：全范围→有限范围→全范围，黑位、白位及色度偏移样本，GPU FP16 与双精度参考量化结果完全一致，最大 FP16 步差为 0。范围 0、3 明确拒绝。CPU 图像传输仅用于此测试。
- `tests/helpers/hdr_source_range_gate.py`：实际用户录像第一帧 NVDEC→新 HDR 桥接通过，8294400 像素，非有限值和超域计数均为 0；未回读图像。只测解码，不是完整渲染或导出验收。
- 证据：`build/hdr-full-range/result.json`、`build/hdr-full-range/source-result.json`。

本轮没有测量性能变化，也未宣称全片所有帧、TrueHDR 或最终封装已经重新验收。当前运行中的工作台仍持有旧 DLL；退出后通过 `run_gui.bat` 启动会自动应用新 DLL。
