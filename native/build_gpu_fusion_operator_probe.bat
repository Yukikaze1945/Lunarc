@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "..\build\fusion_optimization" mkdir "..\build\fusion_optimization" || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /I"include" gpu_fusion_operator_probe.cpp d3d12.lib /Fo:..\build\fusion_optimization\ /Fe:..\build\fusion_optimization\gpu_fusion_operator_probe.exe || exit /b 1
