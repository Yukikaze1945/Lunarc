================================================================================
 DLSS5 离线工具 —— 独立移动版 (dlss5standaloneV2)
================================================================================
全模型内置、免安装、用 Python 直跑。拷贝整个文件夹即可运行。

【做什么】
    导入视频 → 生成光流(nvof 硬件 / RAFT-Large / DIS CPU) → 运行 DLSS(Feature 18)
              → 导出视频(mp4)
    制导只有两档：0=关闭、1=仅光流。本版本不做单目深度估计(见【说明】)。

【运行】
    双击  run_gui.bat               (构建当前前端并启动 Lunarc；需 Node.js 和 Python 依赖)
    备用: python3 gui.py            # 经典 Tkinter 界面
    CLI:    python3 pipeline.py <video> --flow --export dlss
            (--flow/--export 可选 dlss|flow；无 --depth，深度链路已从本版本移除)

【目录结构】(本文件夹即"移动版"，模型全在这)
    gui.py             前端(tkinter)
    pipeline.py        深度+光流+缓存+导出
    dlss_gpu_engine.py ctypes 封装 native/dlss_gpu_host.dll，调 Feature 18
    native/dlss_gpu_host.dll 自编 GPU 宿主(D3D12 + 门禁绕法 + Init_Ext)，本版本唯一原生宿主
    nvngx_dlssnr.dll   NVIDIA 签名 DLSS5 神经渲染运行时 (165MB, 模型内置)
    models/
      depth_anything_v2/        DAV2-Large 网络代码包 —— 历史遗留，无运行路径加载
      checkpoints/depth_anything_v2_vitl.pth   深度模型权重 (1.3GB) —— 同上，已不被读取
    torch_home/hub/checkpoints/raft_large_C_T_SKHT_V2-ff5fadd5.pth  光流模型权重 (21MB)
    requirements.txt           运行依赖
    run_gui.bat                GUI 启动器

【本机运行环境】(打包时已验证)
    Python 3.12.10 + torch 2.11.0+cu128 + torchvision + opencv 4.13 + numpy 2.5
    NVIDIA GPU(本机 RTX 4090) + 较新驱动。DLSS5 神经渲染必须 GPU。

【输出目录】(与视频同名，放在视频所在目录)
    <stem>_flow /000000.flo    Middlebury float32 HxWx2 光流
    <stem>_dlss /000000.png    DLSS5 Feature 18 输出
    <stem>_flow.mp4 / _dlss.mp4   预览/结果视频
    (不再有 <stem>_depth/ 与 <stem>_depth.mp4：CLI 的 out_dirs 只返回 flow 与 dlss 两个目录)

【DLSS 设置面板】(gui 内)
    预设(1/2/3) 风格 强度 本地色调/结构/皮肤结构 自动遮罩 UI校正
    时空引导模式(0=关闭 / 1=仅光流) 运动缩放X/Y
    输出视图(处理/差异×10/左右对比) 输出混合
    (没有「深度约定」也没有深度类引导档：深度链路已移除，见【做什么】)

【说明】
    · 本版已把路径改成相对本文件夹，拷贝到任何位置都能跑(不再依赖 E:\games\dlss5\test3)。
    · 光流默认把短边夹到 720(CLI 的 --edge，默认 720)再计算，之后放大回原尺寸。
    · 导出优先 ffmpeg + NVENC(p7)，不可用时回退 libx264，再无 ffmpeg 时回退 opencv mp4v(无音轨)。
    · RAFT 权重已离线内置，本版不会再联网下载。
    · 深度：DLSS 宿主用的是它自己常驻的零深度纹理(帧记录 depth_source="gpu_zero_texture")，
      没有单目深度估计，也没有深度字节跨进程边界；DAV2 权重留在盘上只是没删干净。
================================================================================
