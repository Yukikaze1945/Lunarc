# 实时预览、RAFT-Large 与中文术语验收

日期：2026-10-10。基线提交：`a2dfc906ac3e8082ac593b2ffc82e753633a4a6a`。
本轮直接实现，未使用 Qoder 或子代理。以下是短样本实测，不是持续吞吐量或普遍画质保证。

## 实现与耗时归因

- 预览由一个持久工作任务处理，每个客户端只保留最新请求。原生处理真正完成后才释放全局处理槽；取消异步包装不会启动重叠 CUDA/NGX 工作。
- 拖动、换源、参数变化带版本号。过期结果丢弃；小幅向前跳帧继续解码，向后拖动才重新 seek。缓存仅保留一组显示图像，报告历史保留最近 16 帧。
- NR 强度、局部色调/结构、皮肤结构、遮罩以及画质/抗闪烁控制在处理线程更新，复用会话并重置历史。模型预设、尺寸、原生模式、层数等结构变化仍安全重建。
- 原 FP16 回读逐行调用 CUDA，2160 行产生大量调用开销。现在整平面一次二维复制；正常界面更进一步在 GPU 按视口面积重采样、执行显示变换，只回读视口 BGRA8。增强视图不再回读未使用的原图。
- 主色处理仍按完整输入分辨率执行。SDR10/HDR10 主色仍为 FP16；视口缩放、8-bit JPEG 和 HDR 显示映射只服务显示，不进入导出。放大检查时保留完整像素裁剪路径。
- WebSocket 使用二进制 JPEG、单写入者和绘制确认背压。短视频循环用不取模的时间位置推进；绘制确认只释放背压，不跳过帧率等待。Electron 先解码再提交最新画面，不堆积旧图。诊断分别记录解码、已完成处理、显示回读、合成、JPEG、序列化、发送、客户端解码和绘制机会。
- 中文圆圈问号说明支持鼠标/键盘、Escape；浮层不被参数面板裁剪。保留瑞士风格及四个基本滑块，停止播放时不把两次人工操作之间的间隔显示成播放 FPS。冷态保留最后一张图并显示“更新预览”。

## 测量环境与比较口径

Windows，RTX 5060 Ti 16 GB。原生 D3D12/CUDA/Feature 18 管线。
预览基线与优化后的同组测试使用同一 6 帧 3840×2160 / 120 FPS / BT.709 10-bit 素材，单层 50%、关闭光流引导、增强视图。
该短片接近黑场，适合定位传输开销，不用于证明一般画质或压缩效率。

基线编码完整 3840×2160 JPEG，优化后按实际 960×540 显示视口编码；两者的主画面处理尺寸相同。
这包含显示传输策略变化，不应归因于 DLSS 模型本身加速。4 个热态点的 p95 只是样本分位数，不能代表长时间尾延迟。

| 后端请求 | 基线 ms | 优化后 ms |
|---|---:|---:|
| 冷启动帧 0 | 4554.2 | 3657.9 |
| 连续帧 1 | 1175.1 | 20.1 |
| 向前跳至帧 3 | 1216.4 | 18.6 |
| 连续帧 4 | 1250.6 | 18.8 |
| 重复帧 4 | 1272.5 | 1.6（显示缓存） |
| 向后拖至帧 1 | 1251.0 | 76.4 |
| 再向前至帧 4 | 1337.3 | 20.0 |
| 四个连续/向前点 p50 / p95 | 1233.5 / 1324.3 | 19.4 / 20.1 |

同一个连续帧 1 的分段值（有包含关系，不能重复相加）：

| 阶段 | 基线 ms | 优化后 ms |
|---|---:|---:|
| 解码 CPU 墙钟 | 0.44 | 0.34 |
| 已完成原生处理，排除显示消费者 | 15.97 | 15.98 |
| 处理图显示消费者（含其回读） | 572.97 | 1.61 |
| 原图显示消费者 | 560.21 | 0（增强视图不请求原图） |
| 其中旧 FP16 平面回读 | 304.54 | 不适用：已改为视口显示回读 |
| JPEG 编码 | 13.07 | 0.86 |

完整 FP16 回读的中间版本：约 8–9 ms；尚未采用 GPU 视口时，总请求约 99–101 ms。
正常视口的显示消费者约 1.5–1.6 ms，范围包含 GPU 显示变换、设备复制、同步和回读，不冒称纯 PCIe 或核函数耗时。

真实 Electron 960×540 图像解码与两次 requestAnimationFrame 显示机会：

| 请求 | 基线 ms | 优化后同组 ms |
|---|---:|---:|
| 冷启动 | 5642 | 4099 |
| 热态帧 1 | 1459 | 41 |
| 向前帧 3 | 1549 | 33 |
| 快速请求 0/2/4，最终帧 4 | 1534 | 33 |

最后补测传输段：三个热态请求均约 33.1–33.2 ms；客户端 JPEG 解码 1.2–1.5 ms，等待显示机会 9.4–11.1 ms；二进制头序列化约 0.008–0.036 ms；服务端 WebSocket 发送 API 完成约 0.39–0.66 ms。
发送 API 完成不等于纯网络飞行时间，双 RAF 不等于物理显示器扫描输出。基线发送 API 段未单独测量，因此不编造该项前后比值。

当前源码构建的瑞士界面独立验收：NR 滑块变化到显示新图最近一次 72 ms（此前 54 ms）；播放、暂停、拖动、画质参数刷新、圆圈帮助和 RAFT 控件/真实 CUDA 引导均验收。
当前界面还验证了循环播放帧位置连续推进，最终 RAFT 帧 1 已收到显示确认、reset=false、previous_tensor_reused=true。
首次启用画质处理着色器仍有约 0.8–1.0 s 初始化；同一会话后续调整实测单层 10.5 ms、双层 19.9 ms（640×480 短素材，不能与 4K 表格混为同组）。

## RAFT-Large

参考 [DLSS5Tool 的真实实现](https://github.com/banbanzhige/DLSS5Tool/tree/851fb86e966b1cac8c15b6e728c6864328db1cef/dlss5tool)，核对长边分析、拥有上一帧输入/张量和 GPU 共享资源的实际代码。该项目 MIT；本项目实现按自身所有权接口编写。

- 长边上限为 512 / 720 / 1080，保持比例、对齐 8；小图不上采样到目标长边。极窄图用模型所需边界填充，不拉伸至最小高度。
- 默认速度/均衡/质量对应 512/4、720/6、1080/12；尺寸与 1–24 次迭代可独立配置。采用 FP32，当前帧到上一处理帧，恢复位移时分别乘横纵比例。
- CPU 兼容适配器保留上一帧归一化张量，并检查内容指纹，避免可复用 OpenCV 缓冲区被误当旧帧。GPU 生产路径直接取独立运动 RGBA8；主色从未转成 RGBA8。
- 解码上下文与 Torch 主上下文不同，显式设备到设备复制并按 PCI 身份核对 GPU；所有 Torch 分配/推理在所保留主上下文内运行，再恢复解码上下文。修复了会话切换后缓存分配失效的真实错误。
- 输出 float32×2 直接交给现有 CUDA/D3D12 光流入口；CUDA 事件完成后才让原生消费者读取。每层有独立上一帧张量，预览只缓存一套权重，尺寸/迭代改变复用权重。Torchvision 相关金字塔属于临时计算工作区，每次推理结束释放。
- [CUDA 跨上下文复制接口](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__MEM.html)与 [Torchvision RAFT](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_large.html)是实现依据；不把跨上下文指针直接解释为 Torch 张量。

CUDA 实测环境：Python `F:\WhisperJAV\python.exe`，torch 2.10+cu128、torchvision 0.25+cu128。CPU-only Python 明确拒绝该设备链，不静默改为 CPU 或零光流。

旧短边 720 / 1280×720 / 6 次 AMP：两个后期热态点约 143.7、142.9 ms。
新长边 720 / 720×408 / 6 次 FP32 CPU 适配器：后续三个点约 75.9、78.4、89.7 ms。分析尺寸与精度策略已改变，这不是同尺寸模型算子优化比较。

生产设备路径的已知平移测试：1280×720 固定随机纹理，每帧向右 3、向下 2 像素；理论 current→previous 为 (-3,-2)。每档 8 次，首帧/显式 reset 检查零流，第一个推理点单列预热，统计后 5 点：

| 长边 / 实际分析尺寸 | 6 次迭代 p50 ms | p95 ms | 中央流中位数最大单轴误差，原图像素 |
|---|---:|---:|---:|
| 512 / 512×288 | 29.6 | 33.6 | 0.216 |
| 720 / 720×408 | 41.6 | 42.3 | 0.134 |
| 1080 / 1080×608 | 88.0 | 91.2 | 0.092 |

这验证方向、独立轴尺度、有限值和 reset；不是全图 EPE、真实遮挡评价或普遍画质结论。较小分析尺寸/较少迭代有精细运动损失的取舍。
设备路径不回读图像或光流；仍有 RGBA8 guidance 设备复制、原生流导入复制和 CUDA 同步，不能称完全零复制。

## 中文术语与功能边界

核对 [Magpie Experimental 0.6.9 简体中文资源](https://github.com/SAOG0721/Magpie/blob/27c5df91177a29b33be612e98274169f3d2fca49/src/Magpie/Resources.language-zh-Hans.resw)。
对应项使用 NR 强度、局部色调强度、局部结构强度、皮肤结构强度、自动遮罩、NR 界面修正、总强度、明度总强度、色度强度、低频/高频强度及色相/暗部/亮部保护。
速度/均衡/质量只是标签，保留底层 provider ID 和档位。模型自动遮罩与手动 ROI 保护分别说明。总强度是残差增益，不等于模型 NR 强度。
本项目抗闪烁仅支持关闭或静态低频累积，不包含光流重投影；说明与实际功能一致，没有照搬其他项目尚未实现的选项。只参考术语，说明文字原创，未复制 GPL 算法或完整语言资源。

## 验收与原始证据

- Python 全单元：1524 passed；前端 Node：60 passed；TypeScript/Vite 构建通过。
- 原生显示着色器 `/W4 /WX` 构建通过；FP16 SDR、FP16 HDR、RGBA8 各 3 种尺寸共 9 组，GPU 视口显示与 CPU 面积重采样/固定显示映射参考逐字节相同（该随机短样本最大误差 0）。
- 最新 RAFT 设备预览：SDR8 单层，SDR10 单层/双层 50/75/三层 40/67/85，HDR10 双层；每组 3 帧。独立 1/2/3 会话与历史张量、实际 data-consistency solver 收敛、原图/增强视口、同会话参数调整均通过。
- SDR8、SDR10、HDR10 最新处理/预览/导出一致性与 remux：每组 6 帧通过，HEVC packet payload、显示 PTS、完整输出尺寸、音轨/字幕、章节、全局标签、附件均检查。HDR10 保留 Main10/PQ/BT.2020、mastering、MaxCLL=900、MaxFALL=250；SEI 137/144 的 payload 完整一致（源重复 2 次，中间/最终重复 6 次，不要求重复次数相同）。
- 实测原生主图像处理边界的 host pixel 为 0；预览显示回读明确非零，标量报告回读仍存在且字节未测，不声明总 host traffic 为 0。

原始证据目录：`bench_out/preview_raft_20261010/`（忽略提交，保留本机）。

| 证据 | 文件 |
|---|---|
| 同组后端前后记录 | `before/backend.json`、`after_first/backend.json`、`after/backend.json` |
| 同组 Electron 前后记录 | `before/electron.json`、`after/electron.json` |
| WebSocket 分段补测 | `transport_final/electron.json` |
| 当前 Swiss Electron 操作和界面 | `electron_current/report.json`、`electron_current/swiss_help.png`、`electron_current/swiss_raft_controls.png` |
| RAFT CPU 适配器前后 | `before/raft.json`、`after_first/raft.json` |
| RAFT 原图平移/三档尺度 | `raft_device.json`、`raft_device_weights_final.txt` |
| RAFT 最新多层/HDR | `raft_latest_smoke.json`、`raft_latest_smoke_final.txt` |
| 重复调整画质的热态响应 | `hot_controls/report.json` |
| SDR/HDR 导出、metadata、remux | `parity/report.json` 及对应各 case 子目录 |
| 全单元、Node、构建 | `unit_final.txt`、`node_final.txt`、`frontend_final_build.txt` |

可复现集成入口（需要短 fixture、CUDA Torch 和已构建 native DLL）：

```powershell
& 'F:\WhisperJAV\python.exe' tests/helpers/raft_gpu_preview_gate.py --source fixture.mkv --modes single dual triple --directory bench_out/raft-preview-new
& 'C:\Users\18392\AppData\Local\hermes\hermes-agent\venv\Scripts\python.exe' -m pytest tests/unit -q
cd frontend
npm run build
node --test ../tests/node/*.test.ts
```

## 保留的限制

1. 冷态主链/Feature 18/着色器初始化仍需约 3–7 秒（取决于层数）；首次启用新画质通道另有约 0.8–1 秒。界面与请求接收保持异步，显示明确等待；没有把冷态称为已解决的热态速度。最新界面测试中，RAFT 第一对非零光流推理仍用约 1.59 秒；后续热态以独立设备测试表为准。
2. 背压与跟随时钟会跳过过期帧；拖动、跳帧、换源/配置保守重置时间历史。实时预览不是逐帧离线导出的时域质量承诺。没有测量长时间 FPS/VRAM 稳定性。
3. RAFT 大分析尺寸、多层推理仍受 GPU 算量限制，尤其 1080 和三层；不会用较少层数或别的算法伪装所选模式。
4. 光流诊断视图与显式 CPU DIS/RAFT 兼容路径仍有 host 图像/光流与完整显示处理；本轮设备直连针对生产 GPU Transfer/原生预览。
5. HDR 预览仍是固定、未校准 SDR 显示映射；8-bit JPEG 显示不等于主色或导出降为 8-bit。放大像素检查退回完整显示裁剪，传输成本更高。
6. RAFT 设备链依赖 CUDA-enabled PyTorch、torchvision 和权重；当前轻量打包配置不内置 Torch。源项目的当前 Swiss 启动入口是 `run_davinci_studio.bat`，已有运行进程或旧包不会自动加载新的 DLL/代码，需重启当前前后端。

保留原有工作树和测试证据。没有清理、提交或推送本轮变更。
