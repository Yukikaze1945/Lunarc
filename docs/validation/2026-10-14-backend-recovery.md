# 2026-10-14 后端恢复与显存统计验收

## 修改范围

新增 Electron 健康检查和有次数限制的后端恢复、队列原子存档、Windows 后端子进程管理及所属线程上的显存采样。重建 `native/dlss_gpu_host.dll`，ABI 仍为 3，只增加诊断入口。

DLL SHA-256：`625c525650393a06f3b0d97ade8cf2038cd215e1b701199d47149f9ca63025e8`。

没有修改模型参数、色彩数学、FP16/FP32 精度、编码器队列深度或融合算法。本轮不发布安装包。

## 检查结果

| 检查 | 结果 |
| --- | --- |
| 队列恢复、写入失败、故障分类、子进程释放、预览相关 Python 检查 | 53 通过 |
| 恢复次数限制、进程归属、启动失败、队列消息及运行环境 Node 检查 | 12 通过 |
| 前端 TypeScript / Vite 构建 | 通过；已有 500 kB 分包提示仍存在 |
| 原生宿主 /W4 /WX 编译 | 通过，已复制源码运行 DLL，无 pending 标记 |
| 两帧 SDR10 NR→DLAA GPU 预览 | 两个独立会话，显存查询有效，两会话 CPU 图像暂存均为 0 |
| 隔离 Electron 后端退出 | 自动启动新 PID；运行中记录转为中断，待处理任务保留，之后可删任务及加新任务 |
| 已载入 SDR10 视频的退出恢复 | 视频、guidance_mode=0、帧索引 1 保留；恢复后重新完成该帧原生预览 |
| 短时间第二次后端退出 | 停止自动重启，窗口保留错误提示 |
| 后端死亡时的子进程管理 | Windows Job Object 检查通过，所属测试子进程结束 |

生产帧采样在 GPU 所属线程上执行，最多约每秒一次；诊断 HTTP 只读取缓存。两帧 GPU 测试记录的进程 LOCAL 用量约 1.04 GiB，已归属的资源分配约 125.63 MiB；这是不同口径，不是优化前后显存差，也不是整段峰值。原生呈现恢复检查另外确认新后端实际处理了帧索引 1，并产生非零 DXGI 用量。

## 证据位置

- `build/backend-recovery/memory-gate.json`
- `build/backend-recovery/ui/result.json`、`recovered.png`
- `build/backend-recovery/video-ui/result.json`、`recovered.png`
- `tests/helpers/backend_memory_gate.py`
- `tests/helpers/backend_recovery_ui_gate.cjs`
- `tests/unit/test_backend_recovery.py`、`test_backend_process.py`
- `tests/node/backend_recovery.test.ts`

检查使用隔离用户目录，并通过结束测试后端验证进程恢复；没有主动制造真实 TDR、驱动重置或耗尽显存。GPU 挂起的根因尚不能由这些恢复检查认定为已解决。

视频恢复时暂停旧帧派发，视频和参数恢复完后重新请求当前帧；中断导出不从原帧续编码，待处理队列不自动启动。队列存档不包含逐帧大体积 GPU 报告。缺失 SDK、普通编码选项错误不会触发恢复。

一个旧的消费者测试使用伪造 D3D12 指针，却因之前默认开启的镜头检测进入原生 DLL；已让该测试显式关闭与其检查无关的时间控制。生产默认镜头检测保持开启。

性能仍为 `NOT_CHARACTERIZED`；本轮只建立恢复与诊断能力，没有作吞吐或显存节省承诺。使用说明见 `docs/backend-recovery.md`。
