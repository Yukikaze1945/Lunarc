@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if errorlevel 1 exit /b 1
cd /d "%~dp0"
cl /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_post_control_smoke.cpp gpu_post_control_bridge.lib d3d12.lib dxgi.lib /Fe:gpu_post_control_smoke.exe
exit /b %errorlevel%
