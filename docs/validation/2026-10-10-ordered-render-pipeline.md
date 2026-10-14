# 用户排序渲染链路与 DLAA 模型预设

## 实现

- Swiss 工作台提供 DLAA / DLSSNR / RTX VSR 阶段列表，可添加、删除、拖动或使用上下按钮排序。列表就是执行顺序，预览和导出共用；旧设置保留原有顺序。移除 NR 时隐藏 NR 主面板参数。
- NR 前后分别执行有序 GPU 阶段；VSR 在 NR 前时，下游缩放、各层输入、残差、融合和时序资源均使用放大后的尺寸。源视频场景检测与处理后时序资源独立，先检测 cut，再重置前置模型及 NR 历史。
- DLAA 独立保存默认 / J / K / L / M / E / F，直接传入原生 SuperSampling 的 `DLSS_Hint_Render_Preset_DLAA` 创建参数，不修改 Feature 18 的 NR preset。E/F 标为旧版；A–D 已移除，保留/无效枚举不在菜单内。官方定义参考：[NVIDIA DLSS SDK](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h)。
- VSR 1–4 档传入原生 QualityLevel，1080p / 4K 输出直接参与尺寸规划。
- 自动查找本机 NGX 缓存、应用旁与已安装的 VFX 运行库；手动覆盖只在高级面板。此机器实际使用 NGX 缓存中的 `nvngx_dlss.dll` 与 `F:\Magpie-Experimental-x64` 的 VFX 文件。指定覆盖错误明确失败，不回退。

## 原始验收证据

所有短 GPU 素材为 1280×720 SDR8，6 帧，含日语 PCM 音轨；没有进行性能 benchmark。

- `bench_out/ordered_pipeline_20261010/first/report.json`：DLAA→NR→VSR、VSR→NR→DLAA，真实预览及导出通过。
- `remaining/report.json`：其余四种三阶段排列及无 NR 的 DLAA→VSR、VSR→DLAA，通过。
- `models/dlaa_{5,6,10,11,12,13}/report.json`：新版 native DLL，各预设 DLAA→NR，独立 NVOF history，预览与导出通过。默认预设在排列用例中覆盖。
- `models/vsr_{1,3,4}/report.json`：各档 NR→VSR 通过；档位 2 在排列用例中覆盖。
- `temporal/report.json`：VSR→DLAA(K)→NR，启用场景检测和静态低频时序处理，通过，处理尺寸 1920×1080。另有单测验证源 cut 在前置模型消费历史前传播 reset，处理后观察器使用独立尺寸。
- 含 NR 的模型短 gate 使用 dual 50/75 和 `data_consistency_experimental`，两个不同 NR session，6 帧 solver 收敛。VSR 提前时工作尺寸为 960×544、1440×808。
- `offline_audit.json`：18 个最终文件逐一离线核对音轨 packet payload SHA256 与源相同、视频 6 packets、display PTS 与源相同；解码图片完整性及源结构关联在原 gate 中检查。
- `ui/report.json` 和 `ui/ordered-render.png`：真实 Swiss Electron 操作添加、删除、按钮排序、拖动、模型 M、VSR 档位 4，等待对应实际 GPU 帧的显示确认后通过。
- TypeScript/Vite 构建通过；Node 测试 63 passed。当前默认 Python 环境完整 `python -m pytest tests/unit -q`：**1570 passed，158.10 秒**，结果见 `unit_verified.txt`。此前 DLAA lifecycle 假对象测试已补齐新增预设字段与未知实际模型断言；硬件运行所用 WhisperJAV Python 3.10 缺少旧预览测试使用的 `asyncio.timeout`，不作为完整单测环境。

## 限制与运行

- 当前原生 DLAA/VSR 阶段仍仅验证 SDR8；HDR 和 SDR10 保持既有 FP16 NR 链路，新增原生阶段对这些输入明确拒绝，没有降级。
- DLAA 为视频光流、零 depth/jitter 合约。模型选项是请求提示，驱动/运行库可能替换，API 无法查询实际内部模型；诊断分别记录 `preset_requested` 和未知的 `preset_effective=null`，不将成功调用等同于指定模型生效。
- VSR 需完整 Video Effects 运行库，驱动安装不保证提供。当前只放大、保持宽高比、每轴不超过 4 倍。
- 导出各已测图像 host 边界为 0；solver/时序标量读回保留，预览存在用于显示的图像读回，不能宣称总 host traffic=0。Performance=NOT_CHARACTERIZED。
- 运行中的旧实例需要重启前后端以加载新源码/DLL。本轮独立测试实例已关闭；没有提交、推送或清理其他未提交文件。
