# 关闭 NR 后单独 DLAA 的光流冲突与退出修正

## 根因

用户只移除 NR，保留 DLAA，预览报 `NVOF CUDA context changed`。

显式渲染链为 `['dlaa']` 时，宽色路径的 `make_live` 特例返回直接 DLAA 主会话，但 `install` 仍将 DLAA 安装为输出阶段。于是主阶段先用主光流执行 DLAA，输出阶段借用同一个 DLAA 会话，改用独立光流再次执行。两个 NVOF 会话的 CUDA 上下文不同，触发宿主的保护检查。错误中的 `0x00000000` 是 detail 字段，不是发现的空指针地址。

移除宽色特例后，显式有序链统一由 Passthrough 持有 DLAA，会话只在有序阶段执行一次。旧的无显式渲染链单独 DLAA 仍作为主阶段执行一次。保留原生上下文保护检查，没有零光流回退。

## 资源寿命

真实 GPU 检查的第一次修正已能完成五次出帧，但关闭时复现 `access violation reading 0x910`，随后解码上下文 pop 也失败。借用主 DLAA 的输出阶段提前关闭了独立光流，而主 DLAA 的 CUDA/D3D12 导入仍借用其上下文。

借用会话的光流现在转交 Passthrough 管理，在主 DLAA 关闭、释放导入之后才关闭光流。Passthrough 的清理同时改为先分离字段，再逐项释放，重复关闭不会重复调用原生句柄。独立拥有 DLAA 的阶段继续先关闭自己的会话，再关闭光流。

这定位并修正了本条单独 DLAA 路径的关闭异常，不能据此断言所有历史 `0x910` 都有相同根因。

## 短检查

本机真实 3840×2160 AV1 SDR 10-bit 素材，RGBA16F 图片链、NVOF 1F 引导、生产 DLL：

| 操作 | 预览 generation | 累计成功 DLAA Evaluate |
| --- | ---: | ---: |
| NR → DLAA，帧 0 | 1 | 1 |
| 关闭 NR，单独 DLAA，帧 0 | 2 | 2 |
| 单独 DLAA，帧 1 | 2 | 3 |
| 同一帧从模型 L 切换到 M | 2 | 4 |
| 跳至帧 5 | 2 | 5 |

每次出帧仅增加一次 DLAA Evaluate。后三次没有重建整条预览链；最终完整关闭成功，进程退出码为 0。79 项相关路由、模型复用、预览交互、TrueHDR 和资源寿命测试通过。

原始本机证据保留在忽略目录 `build/dlaa-memory-20261013/ordered-guidance-lifetime-fixed.json` 和同名 `.log`；首轮关闭失败证据是 `ordered-guidance-fixed.log`。辅助脚本为 `tests/helpers/ordered_dlaa_guidance_gate.py`，后续运行也明确记录关闭结果。

本轮只测上述短序列，没有长时间播放或 HDR 素材完整导出性能测量。Performance = NOT_CHARACTERIZED。共享路由修正适用于预览和导出；尚未重新发布 GitHub 安装包。
