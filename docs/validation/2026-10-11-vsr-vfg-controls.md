# VSR 输出方式与 VFG 固定／动态插帧

## 设置与数据流

- 保留简洁瑞士风格与问号说明。VSR 增加按倍率（>1 至 4×）和按分辨率输出，预设 720p／1080p／1440p／4K／8K 与自定义宽高。输出保持源宽高比、放大双方尺寸、单轴不超过 4×；倍率输出四舍五入到最近偶数。导出尺寸、预览尺寸与后端使用相同规则，不另加二次缩放。
- `NVIDIA VFG（固定）` 使用官方 `FrameMultiplier=2..8` 与 `FrameIndex=1..M−1`；5×–8× 标注实验性。旧 `frame_generation=vfg`、1080p／4K 设置与旧原生入口仍兼容。
- `NVIDIA VFG（动态）` 使用 `FrameMultiplier=0` 与每次 `NvVFX_SetF32(Timestep)`。`framegen_target_fps` 接受目标帧率的十进制或有理数（例如 60、120、60000/1001），须高于源帧率。按目标帧率的均匀时间网格寻找相邻源帧，生成真实中间时刻；编码器使用准确目标有理帧率和对应时间基，未通过重复帧冒充动态插帧。
- 镜头切换检测默认开启，`framegen_scene_detection` 真正传给 SDK `AutomaticShotChangeDetectionEnabled`；关闭时也不向 VFG 转发上游镜头 reset。其他渲染阶段的重置行为保留。
- 插帧仍在用户排列的空间渲染链之后、编码之前；实时预览忽略全部导出插帧参数，编辑这些参数不会重建预览会话或加载 VFG。取消与 SDK 错误直接传出。借用主色先保存在自有 GPU 缓冲，逐张消费生成缓冲后才继续运行模型。
- 固定模式保留原有时间戳逻辑。动态模式检查源 CFR 时间戳后使用其标称有理时间网格，避免容器毫秒取整积累偏移。输出帧数为 `ceil(N × targetFPS / sourceFPS)`；片尾无未来帧时明确保持末帧，诊断单独计数，末端时长误差小于一张输出帧。最终 uncached 校验使用独立计算的期望帧数。

新增设置：`vsr_output_mode`、`vsr_scale`、`vsr_width`、`vsr_height`、`framegen_target_fps`、`framegen_scene_detection`；`frame_generation` 增加 `vfg_dynamic`，固定倍率扩至 8。新增原生入口 `vfx_framegen_open2`、`vfx_framegen_process_time`，保留旧入口。P010／线性 FP16／PQ／BT.2020／静态 HDR metadata、音轨字幕章节附件与现有 FFmpeg 视频 copy remux 保留。

官方接口：[Video Frame Generation](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoFrameGeneration.html)。本机 SDK：`F:\DLSSC\VideoFX`，GPU：RTX 5060 Ti。

## 短硬件证据

证据目录：`bench_out/vfx_modes_20261011/`，仅短功能检查，Performance=NOT_CHARACTERIZED。

| 检查 | 实际输出 | 结果 |
|---|---|---|
| SDR10 VSR 1.5× → VFG 固定 8× | 1280×720 → 1920×1080，120→960 fps，3→24 帧 | PASS |
| HDR DLAA → VFG 动态，镜头检测关闭 | 1920×1080，25→60 fps，4→10 帧 | PASS |
| SDR8 DLAA → VFG 动态 | 1280×720，120→300 fps，3→8 帧 | PASS |
| SDR10 VSR 自定义分辨率，插帧关闭 | 1280×720 → 2560×1440，3→3 帧 | PASS |

- `native.json`：5／6／7／8 倍、动态 0.17／0.43／0.81 时间点的实际模型 Load/Run，RGBA8 与 RGB10A2 均通过；动态镜头检测开／关均通过。移动矩形至少有真实中间图像，首末相位不同；低于约 0.15 的 7×／8× 首个相位在该素材中与前帧逐字节相同，所以不宣称每个相位都能产生不同画面或改善画质。
- 每个导出都有原始 diagnostics／ffprobe／result。packet 与 display 帧数正确，PTS 严格递增。HDR 动态显示 PTS 为 0、17、33、50、67、83、100、117、133、150 ms；Main10、10-bit、PQ／BT.2020 保留，source/final HEVC mastering 137 与 CLL/FALL 144 payload 字节相同，ffprobe side data 保留。音频 packet SHA256 与源前缀一致，字幕、章节、附件进入最终 MKV。
- 已测生产 host image 边界仍为 0；RGB10 adapter 的标量回读与既有 HDR 色彩桥标量统计另计，不能声明总 host traffic 为 0。native 合成素材诊断上传／回读不属于生产边界。
- 实际 Electron 控件验证：倍率／分辨率／自定义宽高、固定 8× 实验性标签、动态有理帧率与镜头检测开关保存到后端；低于源的目标帧率在入队前被禁用，合法动态设置进入队列。截图 `controls.png`、`fixed.png`、`dynamic.png`。

## 限制

最终检查：Python 3.11 完整 unit suite **1608 passed / 151.09 s**（`units_final.txt`），Node **70 passed**（`node.txt`），原生 `/W4 /WX` 编译与最终 TypeScript／Vite／Electron 构建通过。实际 Electron 验收通过（`ui.json`）。固定 8× 的原始输出另经 `video_info_uncached` 校验，确为 24 帧。保留既有未提交修改，未 commit／push。

首次全量检查误选了 Python 3.10：其中五个已有异步测试需要 Python 3.11 `asyncio.timeout`；同时检查捕获了服务参数中的帧率变量名错误，已修复。改用项目此前全量检查使用的 Python 3.11 后，所有测试通过。

- 只改变现有 VSR 与 NVIDIA VFG 的配置和调度，不替换成 XeSS FG 或游戏 DLSS 帧生成。VFG 仍只用于导出。
- 非均匀源时间戳和时间基变化拒绝。任意不保持宽高比或不放大源图的 VSR 尺寸拒绝；8K 预设也受输入尺寸及硬件约束。
- 5×–8× 保持实验性；未做长视频稳定性、各质量档画质对比或性能 benchmark。
- 固定 8× 首次导出完成后发现最终校验中的变量名错误；修复校验后离线重新验证同一输出，未重复编码。原生诊断第一版断言错误地要求所有相位必须区别于源帧；改为检查真实中间图像及不同相位，记录相同源帧现象。
