# VFG 可排序卡片与 VSR 预览核验

日期：2026-10-11。没有委派 Qoder 或子代理；保留已有工作树修改。

## 实现

- VFG 是 `render_pipeline` 的独立阶段，与 DLAA／DLSSNR／VSR 共用拖动手柄、动画位移、上下按钮、删除和添加。支持只保留 `['vfg']`；不会为此开启 NR 或 DLAA 会话。删除卡片会关闭插帧，添加会启用固定模式，排序保留固定／动态及相关参数。旧工程独立开启 VFG 时迁移到列表末尾一次。
- 导出执行真实顺序。VFG 位于 NR 前时，原始解码画面经过前置模型后插帧，生成的 CUDA RGBA8／FP16 主色直接进入既有 NR／多层融合，后续光流使用独立、连续的输出帧索引。没有中间视频编码、再次解码或主机图像回读。位于 NR 后时，先完成 NR 与前置模型，再插帧，后面的模型处理每张原帧、生成帧和明确的片尾保持帧。
- 源 CFR 检查与编码输出时钟分开：下游帧使用生成后的 PTS／时间基；不能再次乘倍率。固定模式与动态目标帧率保留既有 SDK 接口，取消和失败直接传出，借用 GPU 指针同步消费并在失败时清除。
- 实时预览继续跳过插帧、保持素材原帧率；其余模型照常执行。仅 VFG 时显示“原帧预览”。这一点在圆圈问号说明中写明。
- VSR 名称采用本地 Magpie Experimental 简体中文资源：`Resources.language-zh-Hans.resw` 的 `EffectParam_RTXVideo_RTXVideo_VSR_strength_Option_0..3` 为“低／中／高／极高”，UI 标题为“强度”。后端 SDK 数值仍为 1..4，没有改变功能含义。
- VSR 预览显示从实际运行计数器读取的强度和输出尺寸；无 NR 时不再显示虚假的“1 层”提示。新增 `1:1` 原像素查看，关闭视口预缩小并按屏幕像素比例显示裁剪图，支持拖动查看。普通缩放名称改为“放大 2×／3×”，不再把固定裁剪倍率称为 100%／200%。修正拖动画布提示遮挡实际执行信息的问题。
- Passthrough GPU 资源复制不计作 NGX 推理调用。VFG 独立导出的 NR／DLAA 核心推理计数为 0。

## 真实证据

路径：忽略目录 `bench_out/ordered_vfg_20261011/`。以下均为短功能验收，不是性能或画质测评。每个目录有未改写的原始 diagnostics、ffprobe、result，最后离线复核帧／packet 数、PTS、阶段调用数和图像主机边界。

| 场景 | 输出 | 证据目录 |
|---|---|---|
| SDR10 VFG → NR，2× | 3→6 帧，240 fps；NR 调用 6 次 | `vfg_nr_cfr` |
| SDR10 NR → VFG → VSR，2× | 3→6 帧，1920×1080；NR 3 次、VSR 6 次 | `nr_vfg_vsr` |
| SDR10 VSR → VFG → NR，2× | 3→6 帧，1920×1080；VSR 3 次、NR 6 次 | `vsr_vfg_nr` |
| SDR10 VFG → dual NR | 3→6 帧；两个不同 session，各 6 次调用，每帧真实 typed solver 收敛 | `dual_vfg_nr` |
| SDR8 NR → VFG → VSR | 3→6 帧，1920×1080，240 fps | `sdr8_nr_vfg_vsr` |
| SDR8 VFG → NR | 3→6 帧，240 fps | `sdr8_vfg_nr_retry` |
| SDR10 VFG 单独运行 | 3→6 帧；没有 NR／DLAA 核心推理 | `vfg_only_final` |
| SDR10 VFG → NR + NVOF | 3→6 帧，6 次设备光流／引导调用 | `flow_vfg_nr` |
| HDR VFG 动态 → DLAA，25→60 fps | 2→5 帧；DLAA 下游执行 5 次 | `hdr_vfg_dlaa_final` |

- HDR 保留 Main10、P010、PQ／BT.2020；源与最终 mastering 137／CLL 144 HEVC SEI payload 字节相同，ffprobe side data 存在。音频 packet payload 与源前缀一致，字幕和附件保留。
- 导出已测 NVDEC、NGX 与 NVENC 主机图像边界为 0。VSR／VFG 使用设备图像；RGB10 编码边界有量化／裁剪及标量回读，原有 HDR 色彩桥也有标量统计回读。不能声称总 host traffic 为 0。
- `vsr_quality/result.json` 用相同 SDR10 素材帧和 2× 输出测试低与极高。两档均真实加载模型、运行 2 次，SDK quality_level 为 1／4，输出 3840×2160。JPEG 编码前显示图有 3,200,810 个像素不同，三通道平均码值差约 1.41483／255。该数据证明参数和输出有区别，不是画质提高的证明。没有发现最高档未传入 SDK 的问题；缩小显示可能掩盖差异，不能据此承诺任意素材都明显改善。
- 预览确实有显示图像回读；其非零字节在 diagnostics 中单独记录。

## 回归与界面

- `unit_suite_final.txt`：全量 1691 passed。随后新增 passthrough 计数用例并修正该计数字段，相关分组 297 passed 与 HDR 分组 230 passed。
- Node 前端测试 71 passed；前端 TypeScript／Vite 构建通过。
- 新增调度测试覆盖四种阶段的 24 个完整排列、旧设置迁移、输出时间基、连续帧索引、失败后的借用指针清理及 VFG 独立资源所有者。
- Electron 真界面检查拖动上下动画、取消、键盘排序、VFG 移到顶部、只剩 VFG、设置保存、模式／倍率／镜头选项、VSR 中文标签及实际极高预览。
- 界面证据在 `bench_out/pipeline_sort_20261011/`：`vfg_only.png`、`vsr_ultra_preview.png`、`ui.json`、`pixel_view.json`。当前 DPR=1 时，图像 1164×654 像素与显示 1164×654 完全对应。

## 限制与启动

- VFG 仍是导出插帧，不是 XeSS FG。非均匀源时间戳仍明确拒绝；没有偷偷做 CFR 转换。本轮初试的四帧 stream-copy 切片存在非均匀 PTS，已保留失败证据并改用既有 CFR 素材验收，未改用户视频。
- VFG 放在 NR／VSR 前会使后面的模型处理更多帧，计算开销会增加。没有正式性能测评，`Performance=NOT_CHARACTERIZED`。
- 验证了当前 RTX 5060 Ti 与本机 VFX 1.3 运行库；未声称所有显卡／排列／视频均已实机测试。
- 关闭旧窗口后，从 `F:\dlss5standaloneV2\run_davinci_studio.bat` 启动当前工作台。
