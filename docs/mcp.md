# MCP 自动化接口

MCP 客户端可以读取与软件共用的预设、修改本次任务的参数、导出视频或图片，并等待完成或错误。接口使用现有后端及渲染队列，不会另外启动 GPU 推理进程。

## 连接

先启动 Lunarc，在 **软件设置 → 连接智能体和应用** 开启「允许 AI 智能体控制 Lunarc」。默认关闭，选择保存在本机。点击「复制提示词」可让支持的智能体配置连接；「MCP 工具」列出全部工具的用途。关闭开关后新的 MCP 调用会被拒绝，已提交的导出继续执行。

首次升级后打开一次工作台，界面会将原有本地预设迁移到后端；原数据备份保存在本地存储的 `lunarc.presets.migration-backup.v1`。同名后端预设保留，迁移不会覆盖它。

Lunarc 内置 **Streamable HTTP** MCP 服务，不需要额外安装运行时或启动 MCP 子进程。将仓库中的 [配置示例](../mcp-config.example.json) 加入支持 Streamable HTTP 的 MCP 客户端：

```json
{
  "mcpServers": {
    "lunarc": {"url": "http://127.0.0.1:18923/mcp"}
  }
}
```

客户端通过本机地址 `http://127.0.0.1:18923/mcp` 连接；不要将服务暴露到公网。服务仅在软件运行时可用，并受设置中的开关控制。接口适用于包含内置 MCP 服务的版本；旧安装包需要更新后才能使用。

旧运行环境缺少 MCP 依赖时，工作台仍可启动，智能体连接会报告组件不可用。组件版更新基础运行环境即可；源码使用的独立 Python 应安装 `requirements-mcp.txt` 后重启软件。

高级用法仍可通过源码运行 `lunarc_mcp.py` 使用 stdio 适配器，适合只支持 stdio 的 MCP 客户端。它需要单独的 Python 依赖（见 `requirements-mcp.txt`），但默认接入不再需要该适配器。

## 工具

| 工具 | 用途 |
| --- | --- |
| `get_status` | 后端健康状态和精简队列。 |
| `probe_media` | 读取本机素材的尺寸、位深和色彩标记，不启动预览或模型。 |
| `list_presets` / `get_preset` | 列出预设名称或读取一份完整参数。 |
| `get_parameters` | 读取默认值和可选参数名。 |
| `resolve_settings` | 将预设和自定义参数合并校验，不修改界面。嵌套对象按字段合并，数组整体替换。 |
| `save_preset` | 保存预设；覆盖同名预设须显式传入 `overwrite=true`。 |
| `render_video` | 提交 MKV 导出，默认等待最终结果。 |
| `render_image` | 渲染并保存图片；GPU 正忙时明确返回错误。 |
| `wait_for_jobs` | 一次等待指定任务完成、失败、取消或跳过。 |
| `get_job` | 按任务 ID 或 `request_id` 找回结果。 |
| `cancel_job` | 取消指定的待处理或当前活动任务，不取消另一个活动任务。 |
| `start_pending_renders` | 启动仍待处理的 MCP 任务，不启动其他手动队列项。 |

## 示例

调用 `render_video`：

```json
{
  "input_path": "D:/Videos/source.mkv",
  "output_path": "D:/Videos/source_lunarc.mkv",
  "preset": "电影",
  "settings": {"intensity": 1.2, "post_controls": {"chroma_gain": 0.8}},
  "encoding": {"codec": "av1_nvenc", "rate_mode": "cq", "cq": 20, "preset": "p7"},
  "request_id": "source-movie-v1",
  "wait": true,
  "timeout_seconds": 600
}
```

预设名称须实际存在，也可以省略预设、直接提供 `settings`。默认使用 GPU Transfer。视频编码可选 `hevc_nvenc` 或 `av1_nvenc`，色彩、位深及运行库能力由现有导出链检查。输出目录须存在。MCP 不接受编码器 `extra_args`，也不执行睡眠或关机完成动作。

`frame_range` 可设为 `{"start_frame": 100, "end_frame": 200}`，表示从第 100 帧开始、到第 200 帧之前结束。绝对帧索引从 0 开始。

图片使用 `render_image`，提供 `input_path`、`output_path`、`format`、`color_mode` 及可选预设和参数。格式可选 PNG/JPEG/WebP/AVIF/HEIC/UHDR/JXL，实际可用性以已安装组件为准；HDR 保存使用 `color_mode="hdr10"`，处理结果须为 HDR。

## 等待与恢复

- 视频任务保存独立参数快照；之后修改界面或预设不影响已提交任务。图片任务也使用独立参数。
- `render_video(wait=true)` 的等待发生在适配器和后端内部，不会每次进度变化都让模型发起新调用。最终返回状态、输出路径和必要错误，不自动返回逐帧诊断。
- 可设置 `wait=false` 立即取得任务 ID，再调用一次 `wait_for_jobs`。单次等待最长 3600 秒；MCP 客户端自身的请求超时也应相应设置。
- 等待超时或 MCP 断开不会取消渲染。后续用任务 ID 继续等待，或用 `request_id` 查回任务。
- `render_video` 必须提供 `request_id`，由 MCP 客户端为每个新任务取一个稳定的标识。相同 ID 和相同有效参数返回已有任务；不同参数复用同一 ID 会拒绝。网络中断导致不确定是否提交时，先查询该 ID，避免换 ID 重复提交。等待过程中断时，结果保留任务 ID 与 `wait_error`，可继续查询。
- 自动化任务等候当前 GPU 队列结束，并只自动启动标记为 MCP 的任务。手动点击队列开始仍沿用软件原有的全部待处理任务行为。
- 失败任务不会自动重试。明确取消当前活动任务会停止该轮队列；其他待处理 MCP 任务可以用 `start_pending_renders` 继续。
- 队列恢复遵循软件现有持久化规则：重启时正在执行的任务标记为中断，不将它们当作成功。

## 验证范围

已检查 MCP 握手、结构化工具结果、错误返回、预设迁移、参数快照、请求去重、任务筛选、取消和完成等待。渲染执行在自动化测试中使用替身；本轮没有运行实际 GPU 渲染，也没有停止正在进行的导出。

实现使用 [官方 MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)。
