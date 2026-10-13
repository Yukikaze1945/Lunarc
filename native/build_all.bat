@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined DLSS_SDK_DIR (echo Set DLSS_SDK_DIR to your NVIDIA DLSS SDK directory. & exit /b 2)
if not defined FFMPEG_SDK_DIR (echo Set FFMPEG_SDK_DIR to your FFmpeg headers and import libraries directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
for %%B in (build.bat build_nvof_d3d12_interop.bat build_dlss_gpu_host.bat build_residual.bat build_nvdec_bridge.bat build_nv12_rgba_bridge.bat build_gpu_lanczos_bridge.bat build_gpu_residual_bridge.bat build_gpu_low_residual_bridge.bat build_gpu_multiscale_fusion_bridge.bat build_gpu_post_control_bridge.bat build_gpu_temporal_bridge.bat build_nvenc_bridge.bat build_gpu_hdr_scrgb_bridge.bat build_gpu_preview_readback.bat build_gpu_preview_display.bat build_gpu_export_resize.bat build_gpu_rgba_exchange.bat build_rtx_vsr_bridge.bat) do (
  call "%%B"
  if errorlevel 1 exit /b 1
)
if defined FFX_SC_EXECUTABLE (
  call build_fidelityfx_optical_flow.bat || exit /b 1
  call build_fidelityfx_flow_bridge.bat || exit /b 1
) else (
  echo AMD Optical Flow is not built. Set FFX_SC_EXECUTABLE and run its two build scripts to enable it.
)
echo Native source build complete.
exit /b 0
