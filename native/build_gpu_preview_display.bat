@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_preview_display.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib /Fe:gpu_preview_display.dll /link /DLL || exit /b 1
