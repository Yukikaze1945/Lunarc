@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined DLSS_SDK_DIR (echo Set DLSS_SDK_DIR to your NVIDIA DLSS SDK directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /I"%DLSS_SDK_DIR%\include" dlss_gpu_init_probe.cpp nvof_bridge.lib nvof_d3d12_interop.lib d3d12.lib advapi32.lib user32.lib "%DLSS_SDK_DIR%\lib\Windows_x86_64\x64\nvsdk_ngx_s.lib" /Fe:dlss_gpu_init_probe.exe || exit /b 1
