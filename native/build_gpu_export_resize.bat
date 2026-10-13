@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_export_resize.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib /Fe:gpu_export_resize.dll /link /DLL || exit /b 1
