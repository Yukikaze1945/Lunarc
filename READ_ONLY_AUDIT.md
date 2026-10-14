# DLSS5 Standalone V2 只读代码审计报告

审计日期：2026-09-17  
审计范围：Python 后端、视频处理流水线、Electron/React 前端、启动与依赖配置、自动化测试  
审计方式：静态代码审查、Python AST 解析、TypeScript `--noEmit` 编译、Oxlint、现有测试产物交叉核对  
限制：未执行会生成媒体、测试报告或占用 DLSS/GPU 的 E2E 测试；未修改任何项目源码。本报告是唯一新增文件。

## 结论摘要

当前快照可通过 Python 语法解析与 TypeScript 类型检查，但仍存在多项会影响正确性、可靠性和本机安全边界的缺陷。优先级统计：

| 级别 | 数量 | 含义 |
|---|---:|---|
| P0 | 1 | 本机控制面可被任意网页跨域调用，能够读取本地媒体、启动 GPU 作业并指定输出路径 |
| P1 | 10 | 会导致启动失败、画质/运动向量错误、取消失效、意外关机或功能名实不符 |
| P2 | 4 | 会造成错误完成状态、预览冻结、测试失真或安装不可复现 |

最先应处理的三项是：限制本地 API/WS 来源并增加会话鉴权；恢复真正的高位深/HDR 数据通路；修正阻塞 I/O 与原生超时的进程隔离/取消模型。

## 发现详情

### P0-01 本地高权限 API 对任意网页开放

- 证据：`server.py:52-60` 对所有来源、方法和请求头开放 CORS；所有 REST 与 WebSocket 端点均无鉴权或 Origin 校验（例如 `server.py:170-176`、`server.py:231-257`、`server.py:281-289`、`server.py:520-526`）。
- 影响：用户浏览恶意网页时，网页可尝试调用 `127.0.0.1:18923`，加载可猜测的本地路径、读取帧画面、创建/取消渲染任务、消耗 GPU，并向调用者指定的本地路径写输出。WebSocket 同样没有来源限制。
- 建议：只允许 Electron 的明确来源；为每次启动生成随机会话令牌并校验 REST/WS；校验 `Host`/`Origin`；限制输入与输出到用户通过原生对话框授权的路径；不要同时使用 `allow_origins=["*"]` 与凭据模式。
- **修复说明（2026-09-18，已修复，回归测试 01-1.3 通过）**：`server.py` 新增每启动随机 `DLSS_LOCAL_TOKEN`（`DLSS_LOCAL_TOKEN` 环境变量可覆盖；未设置则生成并打印），除 `GET /api/system` 外全部接口要求 `X-DLSS-Token` 头或 `?token=`；新增 HTTP 中间件同时校验 `Origin`（仅本机/`null`/缺省放行）与 `Host`（防 DNS rebinding，非回环 403）；CORS `allow_origins=["*"]` 改为 `allow_origin_regex` 仅本机；两处 WS 在 `accept` 前同样校验（拒绝码 4401）。`main.ts` 生成 token 经 env 传后端并经 `preload.localToken` 给渲染器；`api.ts` 全量附 token、`buildFrameUrl`/两处 WS 自动拼接；测试侧 `backend_client.js` 固定 `e2e-test-token` 并透传，`01`/`11.4` 裸 WS 改走 `wsUrl()`，新增 `1.3` 断言 401/403。9 项门禁冒烟脚本全过。残留：输入/输出路径沙箱未做（本地用户本就有文件读写权；Electron 路径走原生对话框）。

### P1-01 HDR/10-bit 输入在进入 DLSS 前已被不可逆地降为 8-bit

- 证据：`pipeline.py:603-604` 只定义 8-bit `bgr24`/`nv12` 帧；`pipeline.py:614-622` 的 CUDA 路径固定输出 NV12；`pipeline.py:650-656` 的 CPU 路径固定输出 BGR24；`dlss_service.py:775-776` 再封装为 `uint8` RGBA；编码输入也固定为 `bgr24`（`pipeline.py:1203-1207`）。
- 影响：源视频的 10-bit 精度、PQ/HLG 传递函数及广色域信息在神经处理前已经丢失。之后选择 `p010le`、Main10 或 BT.2020/PQ 只是在 8-bit 数据上重新封装/打标签，不能恢复 HDR，且可能产生色带或错误色彩。
- 建议：建立明确的 10/16-bit 线性或受控传递函数工作格式；让解码、DLSS 输入、缩放、色调映射和编码全链路携带位深/色彩元数据；在不支持时禁用 HDR/Main10 选项而不是静默降级。
- **修复说明（2026-09-18，已诚实化，未实现真 HDR 管线）**：新增 `pipeline.probe_pixel_info()`（ffprobe 读 `pix_fmt`/`bits_per_raw_sample`/色彩三元组，判 `bit_depth` 与 `is_hdr`，fixture 实测 yuv420p/8-bit/非 HDR）；`load_video` 把四字段写入 `video_info` 并对 HDR/10-bit+ 输入记 WARN；`_execute_job` 对 HDR-ish 输出请求（10/12/16-bit 像素格式、BT.2020 系）记 WARN；前端 `DeliverPage` 对 HDR 输入弹警告、把 P010/P016/P210/P216/YUV444P10/16 与 BT.2020 三档置 `disabled`。真 10-bit 管线未做（`dlssnr` 接口即 `uint8` RGBA，属 SDK 侧限制）；`Main10` profile 选项未动（编码器侧接受 8-bit 内容，影响面小）。

### P1-02 “NVOF/AMD 硬件光流”实际均为 OpenCV DIS CPU 光流

- 证据：`optical_flow.py:44-67` 的 `nvof` 与 `amd` 分支都创建 `cv2.DISOpticalFlow_create(...)`，没有调用 NVIDIA Optical Flow SDK、OpenCV CUDA NVOF 或 AMD FidelityFX 接口。
- 影响：UI、README 与日志会把 CPU 算法描述成硬件光流，导致性能预期、显存/CPU 预算和结果质量判断全部失真；高分辨率导出可能远慢于宣称。
- 建议：在真正接入对应硬件后端前，将名称改为 DIS CPU；初始化时返回真实 provider/capability；对硬件后端做能力探测与可验证的回退提示。
- **修复说明（2026-09-18，已修复）**：取值 `nvof`/`amd` 保持不变（API/设置兼容），显示名改为“DIS CPU 光流（nvof/amd 兼容名）”（`DLSSPage.tsx:803-804`），`optical_flow.py` 文档与注释同步更正；新增 `provider_info()`（nvof/amd 返回 `opencv-dis-cpu`/`hardware=False`），`get_optical_flow_engine` 首次创建即打印 provider；`README` 与两处后端 docstring 同步。残留：真硬件后端未接入（与建议一致，属未来工作）。

### P1-03 RAFT 后向光流通过简单取负构造，方向与像素对应关系错误

- 证据：接口要求“当前帧到前一帧”的后向流（`optical_flow.py:92-95`）；RAFT 却以 `prev, cur` 顺序推理（`optical_flow.py:160-167`）得到前向流，然后仅执行 `-fl`（`optical_flow.py:174`）。
- 影响：非刚性运动、遮挡和位移梯度场中，反向流不是前向流逐点取负。错误运动向量会直接降低时序稳定性，产生重影、拖影或细节闪烁。
- 建议：直接以 `cur, prev` 顺序运行 RAFT，或实现带坐标重采样和遮挡处理的严格逆流；增加已知平移、旋转和遮挡场的数值测试。
- **修复说明（2026-09-18，已修复并数值验证）**：RAFT 分支改为喂 `(cur, prev)` 直接求后向流，删除 `-fl` 取负（`optical_flow.py:166-186`）。合成平移验证（白块平移 +12/+4，CUDA RAFT-Fast，本地权重零下载）：方形内中值流 `(-12.30, -3.91)` ≈ 期望 `(-12, -4)`，误差 <0.5px；旧代码会给出 `(+12, +4)`。仅覆盖纯平移场，旋转/遮挡数值测试未补（缺对应 fixture 生成器）。

### P1-04 Electron 后端启动绑定开发者本机路径，且进程启动错误未被捕获

- 证据：`frontend/electron/main.ts:46-48` 默认 Python 固定为 `F:\\WhisperJAV\\python.exe`；`spawn()` 只被同步 `try/catch` 包围（`main.ts:50-59`），但可执行文件不存在时错误通过 ChildProcess 的异步 `error` 事件发出，当前没有监听器。
- 影响：换机器或盘符后，README 所述的 `npm start` 很可能直接无法启动；异步 ENOENT 可能成为未处理的 `error` 事件并终止 Electron 主进程。
- 建议：依次解析显式配置、项目虚拟环境、打包运行时、系统 `py/python`；在启动前验证路径；监听 `error`/`exit` 并向窗口展示可操作的诊断。
- **修复说明（2026-09-18，已修复）**：`main.ts:56-118` 按 `PYTHON_PATH` → 旧本机路径 → `.venv/venv` → `python`/`py` 依次探测并校验文件存在，找不到则记日志+E2E 外弹 `showErrorBox`；`spawn` 补 `error`/`exit` 异步监听（退出码非零 E2E 外同样弹框；退出事件同步 `backend-exit` 给渲染器）。注：`before-quit`/`will-quit` 的 taskkill 兜底与端口等待系同期另一改动，兼容。

### P1-05 Electron 不等待后端就绪，前端关键初始化请求只执行一次

- 证据：`startPythonBackend()` 在 `spawn` 后立即返回（`frontend/electron/main.ts:39-60`），随后马上创建窗口（`main.ts:161-164`）；系统信息、当前视频和设置只在挂载时请求一次，失败后仅打印日志（`frontend/src/App.tsx:42-57`）。
- 影响：Python 冷启动、Torch 导入或 DLL 初始化稍慢时，页面先于 API 就绪。WebSocket 会重连，但系统信息和设置不会重试，UI 可能长期使用默认值或显示未就绪。
- 建议：主进程轮询 `/api/system` 至成功/超时后再加载页面；前端初始化增加有上限的重试与明确错误态；后端子进程退出时同步更新 UI。
- **修复说明（2026-09-18，已修复）**：`main.ts` 在 `createWindow` 前轮询 `/api/system` 至多 60s；`App.tsx` 初始化三请求改为至多 5 次退避重试，失败置状态栏“后端连接失败…”；后端 `exit` 经 `backend-exit` 事件同步 UI（`preload.onBackendExit` + `App` 订阅）。前端构建通过。

### P1-06 所谓超时/取消不能中断正在阻塞的管道读写

- 证据：`pipeline.py:497-530` 在调用 `stream.read(...)` 之前检查超时和取消，但 `read` 本身是阻塞调用；一旦进入就无法继续检查。解码 FFmpeg 进程没有通过 `dlss_service._register_proc()` 注册，取消时只会杀编码进程（`dlss_service.py:165-189`、`pipeline.py:1462-1467`）。编码侧 `proc.stdin.write(...)` 也无超时（`pipeline.py:1593-1598`）。
- 影响：硬解初始化或管道停滞时，取消按钮可能留下后台 FFmpeg/daemon 线程；30 秒 watchdog 也无法打断已阻塞的 `write`。多次重试会积累进程和句柄。
- 建议：把解码与编码都纳入统一的进程所有权；Windows 下使用可取消的异步 I/O/独立工作进程；取消时终止完整进程树并等待回收；不要用“循环前检查”冒充阻塞调用超时。
- **修复说明（2026-09-18，已修复，残留两处原生盲区）**：`iter_frames` 新增 `on_proc_started/exited` 钩子，CUDA 探测与 CPU 回退解码进程与编码进程共用同一注册表，`cancel_active_job` 一并杀死；编码 `stdin.write` 改 1MB 分块（块间查取消/存活）；`_read_exactly` 改后台读取线程+0.2s 监督节拍（取消/滑动超时/存活全响应；干净 EOF 返回短字节，异常退出码才抛）。残留：`cv2.VideoCapture.read()` 与 `live.process()` 原生调用仍不可打断，由 P1-07 熔断兜底；完整进程树（子子进程）回收未做，仅杀直接子进程。

### P1-07 原生调用超时后线程仍继续运行，可能与后续会话并发操作同一 DLL

- 证据：`dlss_service.py:97-122` 在超时后明确留下 daemon 线程；调用者收到异常后会继续清理、重建或处理下一任务（例如 `dlss_service.py:707-742`）。
- 影响：超时并没有停止 `dlssnr_init/shutdown/update`。遗留线程稍后恢复时可能在后台改变全局 DLL/D3D12 状态，与新会话并发，造成访问冲突、设备丢失或不可预测崩溃。
- 建议：把不可中断的原生会话放到独立子进程；超时时终止整个 worker 进程并重新创建，不能在同一进程里继续复用受污染的全局原生状态。
- **修复说明（2026-09-18，部分修复：熔断而非子进程隔离）**：未做独立 worker 进程（需跨进程传帧/重建管线，改动过大，列为后续）。已做：`_native_degraded` 熔断位——`Live` 构建/关闭/更新任一超时即标记，后续 DLSS 任务直接拒绝并提示重启后端，不再复用疑似污染进程；预览路径保持降级源片、不熔断。选择依据：超时后遗留线程不可杀，隔离是唯一彻底解，但熔断已能防止“带病复用导致访问冲突/设备丢失”。

### P1-08 “循环播放”开关是死控件

- 证据：`frontend/src/pages/DLSSPage.tsx:49` 和 `:552-555` 只维护/显示 `loop` 状态；该值从未发送给后端。后端无条件使用 `% total` 回绕（`server.py:431-442`）。
- 影响：用户关闭循环后视频仍会无限循环，UI 状态与实际播放行为相反。
- 建议：在 play 命令中传递 loop；非循环模式到末帧时停止并发送 ended 事件；为开/关两种模式加端到端断言。
- **修复说明（2026-09-18，已修复；E2E 断言未补）**：`play` 接受 `loop`（缺省 `true` 保持旧行为），非循环用线性推进到末帧后发 `{"type":"ended"}` 并停止；前端 `togglePlay` 发送 `loop`，收到 `ended` 落 `isPlaying`、置“播放结束”。开/关双模式 E2E 断言未补（现有 03 覆盖默认循环模式；非循环路径仅代码走读+编译通过）。

### P1-09 队列失败或取消后仍可能执行睡眠/关机动作

- 证据：`dlss_service.py:575-618` 无论任务完成、失败还是因取消跳出，循环结束后都会调用 `_handle_finish_action()`；该函数扫描整个历史队列并取最后一个非 none 动作（`dlss_service.py:625-650`），不检查该任务是否成功，也不限制为本次实际执行的任务。
- 影响：任务失败、用户取消，甚至某个尚未运行的排队任务设置了关机时，仍可能触发睡眠/关机（当 `ALLOW_POWER_ACTIONS=1`）。这是数据与操作安全问题。
- 建议：完成动作只绑定本次队列运行且成功完成的任务集合；任何失败/取消都禁止破坏性动作；执行关机前在 UI 显示倒计时并允许撤销。
- **修复说明（2026-09-18，已修复；倒计时未做）**：`_run_queue` 记录本轮 `completed_this_run`，任一失败/取消即否决（`run_had_failure`，含 failed 分支），`_handle_finish_action` 只认本轮成功任务；`DeliverPage` 启动含 sleep/shutdown 的队列前弹 `confirm`（后端 E2E_TEST 抑制+`ALLOW_POWER_ACTIONS` 双保险保留）。关机前倒计时+撤销未做（被否决机制+确认框替代，风险已收敛）。

### P1-10 预览“播放中”状态可在 WebSocket 不可用时永久冻结

- 证据：`frontend/src/pages/DLSSPage.tsx:203-229` 即使 WebSocket 未连接也会把 `isPlaying` 设为 true 并显示“连续播放中”，但没有 HTTP 播放回退；`ws.onopen` 只记录日志（`:73-75`），重连后不会补发 play 命令。
- 影响：后端启动竞态或短暂断线时，UI 显示正在播放但画面不动；连接恢复后也不会自动恢复，必须用户手动暂停再播放。
- 建议：未连接时禁止进入 playing 或进入 buffering；onopen 根据期望状态补发 play/seek；断线时明确暂停并提示。
- **修复说明（2026-09-18，已修复）**：引入 `wantPlayingRef` 期望态与 `lastPlayRef` 意图快照：未连接点播放只记意图（“等待连接，后台就绪后自动开始…”）而不亮 playing；`onopen` 按意图补发 play；`onclose` 落 playing 并提示重连；`seek`/切页/卸载清意图。HTTP 播放回退未做（seek 仍保留原有图片回退；连续播放无回退 mild 缺口，与建议有差）。

### P2-01 输出完整性校验在两种探测器都无法确认帧数时仍会判定成功

- 证据：`pipeline.py:1665-1689` 在 `video_info()` 失败后回退到 OpenCV；如果 OpenCV 返回 `real_count == 0`，既不报错也不记录不可验证，随后仍返回输出路径（`:1690`）。
- 影响：大于 1 KB 但索引损坏、解码器不支持或无法探测的文件可能被标记为 completed，与“严格物理校验”的 UI/日志承诺不符。
- 建议：两个探测器都不能得到正帧数时应进入 `verification_failed`/failed；至少实际解码首帧、尾帧并校验时长与流信息。
- **修复说明（2026-09-18，已修复；无独立 `verification_failed` 状态）**：`video_info` 返回 `<=0` 即抛“无法确认有效帧数”；cv2 回退 `<=0` 同样抛（此前静默返回路径）。未新增 `verification_failed` 状态（复用 `failed`+错误文案，队列语义不变）；首/尾帧试解码未做（强校验仍以包数+文件存在/大小为准）。

### P2-02 Python 依赖清单无法安装出 README 所述的可运行环境

- 证据：`requirements.txt` 只有 numpy、OpenCV、Torch、Torchvision；`server.py:31-34` 还必需 uvicorn、fastapi、pydantic；`gui.py:31` 必需 Pillow；`pipeline.py:269` 会使用 PyAV 作为解码后端。
- 影响：用户按 README 安装 requirements 后仍可能在启动阶段 `ModuleNotFoundError`，或无提示失去更可靠的 PyAV 解码路径。
- 建议：区分 `requirements-runtime.txt`、可选 GPU/RAFT 依赖和开发依赖；锁定已验证版本；在启动时做依赖自检。
- **修复说明（2026-09-18，已修复；未拆分文件）**：`requirements.txt` 补 `fastapi/uvicorn/pydantic/Pillow` 与可选 `av`（附回退说明）；未拆 `requirements-runtime.txt`（单文件加注释分区，体量尚不值得拆）；版本沿用 `>=` 风格未锁定（与仓库既有风格一致）；`server.main()` 新增 `preflight_check`（缺必需依赖直接中文退出，可选缺失告警，ffmpeg/ffprobe 缺失告警）。

### P2-03 测试套件写死盘符、项目目录与 Python 路径，无法在其他环境复现

- 证据：`tests/e2e/07_command_preview_consistency.spec.ts:43,68,78`、`08_ui_control_audit.spec.ts:46-47,102` 及多个其他测试写死 `F:\\dlss5standaloneV2` 和 `F:\\WhisperJAV\\python.exe`；测试 helper 也使用相同默认路径。
- 影响：CI、其他开发者机器或项目移动目录后测试直接失败；这会掩盖真正的产品回归。
- 建议：用 `import.meta.url`/`process.cwd()` 推导仓库根目录；Python 路径只从环境或可移植探测获得；临时输出使用测试专属临时目录。
- **修复说明（2026-09-18，已修复；临时目录部分采纳）**：新增 `tests/helpers/paths.js`（`ROOT_DIR`/`FIXTURES_DIR`/`fixturePath()`/`pythonExe()`，`PYTHON_PATH` 环境优先，win32 回退旧路径）；`06/07/08/10/11` 硬编码盘符全部替换，helper 统一经该模块；输出文件仍放 `fixtures/`（与输入同目录保证断言确定性与用例自清理），未迁独立临时目录。

### P2-04 现有测试和文档会产生过度置信，测试产物之间也相互矛盾

- 证据：`tests/e2e/08_ui_control_audit.spec.ts:66-87` 仅通过源码中是否出现字段字符串判断“控件已接线”，没有验证行为；字段出现在注释或无效分支也会通过。`TESTING.md:110` 声称 17/17 通过，但 `tests/test-results.json` 仍保存 1 个失败、1 个通过的详细结果，而 `test-results/.last-run.json` 又标记 passed，缺少同一轮完整报告。
- 影响：团队可能把“字符串存在”和不一致的历史产物误认为真实端到端保障，遗漏本报告中的死控件和语义错误。
- 建议：UI 控件测试应观察生成命令或实际输出变化；每轮测试生成带 commit/hash、时间、环境的不可混淆报告；文档结果由测试报告自动生成，不手工维护。
- **修复说明（2026-09-18，已修复；文档自动生成未做）**：`08` 改为行为型——分辨率族走 `resolve_output_dims` 差值、`scale/color/skip/overwrite` 走生成命令差值、`decode_mode` 走 `cpu` 真实解码 2 帧、`finish_action` 走“否决 vs 放行”两条日志断言，另加 Deliver 六组标签 DOM 存在检查；字符串出现降为必要条件。`playwright.config.ts` 每轮写 `test-results/e2e-<时间戳>.json`（另有 `PLAYWRIGHT_RUN_TAG` 覆盖），`test-results/` 与旧 `tests/test-results.json` 进 `.gitignore`。`TESTING.md` 数字与报告自动生成未动（归属原维护者）。

## 其他低优先级观察

- `dlss_engine.py:117-127` 的批量 `run_dlss()` 不检查 `dlssnr_process` 返回值，失败帧仍以零值图像加入结果；若该旧接口仍被使用，应与 `Live.process(..., raise_on_error=True)` 保持一致。**修复（2026-09-18）：新增 `raise_on_error=True` 参数，失败帧抛 `DLSSInferenceError`（附 C++ 日志尾），默认即 fail-loud；仓内无调用方，行为安全。**
- `FrameCache.put()` 在 key 已存在时只移动 LRU 顺序、不替换 value（`dlss_service.py:48-55`）。当前按帧号缓存通常不会重复写入不同内容，但未来若用于重渲染或可变源会返回陈旧帧。**修复（2026-09-18）：存在 key 时同步替换 value。**
- Electron 生产模式发现 `--remote-debugging-port` 时只打印警告，没有移除开关或退出（`frontend/electron/main.ts:87-92`），与注释中的“prohibited”不一致。**修复（2026-09-18）：非 E2E 下直接 `removeSwitch`，名副其实。**
- Oxlint 报告 20 个警告，其中包含多处 React Hook 依赖缺失和渲染期间访问 ref；这些不是本报告主缺陷，但建议在 CI 中把规则基线固定下来。**修复（2026-09-18）：未使用导入/变量全部删除（前后端+测试约 20 处）；`[video]` effect 补齐依赖（真修）；剩余 4 处 intentional pattern（最新值 ref、日志轮询、表单默认值、切页暂停）保留并附 NOTE 注释；`lint` 脚本钉 `--max-warnings 4`，新增警告即失败。实测 oxlint 1.83 不识别 `eslint-disable-next-line`，故未采用注释抑制。当前 `0 error / 4 warning`，构建通过。**

## 验证记录

| 检查 | 结果 |
|---|---|
| Python AST 解析（21 个 `.py` 文件） | 通过 |
| TypeScript `tsc --noEmit --incremental false -p tsconfig.app.json` | 通过 |
| Oxlint | 0 error，20 warning |
| `git diff --check` | 无空白错误；Git 提示部分文件下次写入会 LF→CRLF |
| E2E | 本次未运行，避免写入测试/媒体产物和占用 GPU；仅审阅源码与已有结果 |

## 推荐修复顺序

1. 封闭 API/WS 控制面（P0-01），同时为路径和任务请求引入结构化校验。
2. 明确产品是否真正支持 HDR/10-bit 与硬件光流；在实现前禁用误导选项（P1-01、P1-02、P1-03）。
3. 将 DLSS/FFmpeg 工作负载迁入可终止的独立 worker 进程，统一取消和资源回收（P1-06、P1-07）。
4. 修复 Electron 启动/就绪协议与播放状态机（P1-04、P1-05、P1-08、P1-10）。
5. 修正完成动作和输出验证，再补行为型回归测试（P1-09、P2-01、P2-04）。
6. 清理依赖清单和测试路径，使新机器与 CI 可以一键复现（P2-02、P2-03）。
