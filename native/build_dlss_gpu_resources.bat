@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 dlss_gpu_resources.cpp /Fe:dlss_gpu_resources.dll /link /DLL || exit /b 1
