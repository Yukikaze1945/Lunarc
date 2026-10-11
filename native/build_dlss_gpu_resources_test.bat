@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
call build_dlss_gpu_resources.bat || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 dlss_gpu_resources_test.cpp dlss_gpu_resources.lib d3d12.lib dxgi.lib /Fe:dlss_gpu_resources_test.exe || exit /b 1
dlss_gpu_resources_test.exe || exit /b 1
