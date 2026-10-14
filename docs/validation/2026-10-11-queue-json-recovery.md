# 导出完成后的队列 JSON 与素材载入修复

## 原因

HDR/SDR10 导出诊断中的 negotiation.plan.struct 保留了传给原生编码器的
bytes 参数（codec、preset、tune 等）。server 的队列初始消息与广播直接
json.dumps 整个任务，导致 `Object of type bytes is not JSON serializable`。
HTTP 队列接口同样暴露这些内部值；长视频还会反复携带数万条逐帧诊断。

## 修改

- 队列 HTTP、WebSocket 初始消息、后台进度及结束通知统一使用独立快照。
  快照不携带完整 residual_backend_diagnostics，提供 diagnostics_available。
- 完整诊断仍在服务内部保留。诊断接口把 UTF-8 参数转为文本，非文本
  二进制使用含长度的 Base64 对象，原始诊断不修改。
- 自动检查最近任务时请求 summary=true，长数组返回实际条目数；求解
  帧摘要记录真实 all_converged。主动打开诊断仍能读取完整记录。
- 导出删除规则保持：已完成、失败等终态可删，运行中任务必须先停止。

## 验证

- 69 项相关 Python 回归通过（队列、删除/重试、素材载入、诊断接口）。
- 新回归包含 40,000 条逐帧记录及 bytes 的已完成任务：WebSocket 两次
  重连、HTTP 删除并广播、保留另一终态任务、添加新任务、加载及解码
  新视频和中文路径图片；确认内部完整诊断未丢失。
- 4 项前端导出/素材拖入路径检查通过；TypeScript/Vite/Electron 构建通过。
- 用独立 ASGI TestClient 验证接口；没有终止用户后端或现有导出。
  当前运行的旧后端 HTTP 请求超时，需要退出程序后重新启动加载修改。
- 未执行新的 GPU 导出或插帧性能测试；本次不改变画面处理、编码参数。
