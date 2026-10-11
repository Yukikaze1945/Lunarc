@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "..\build\hdr_a2" mkdir "..\build\hdr_a2" || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /I"include" gpu_temporal_probe.cpp gpu_temporal_bridge.lib nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib user32.lib /Fo:..\build\hdr_a2\ /Fe:..\build\hdr_a2\gpu_temporal_probe.exe || exit /b 1
