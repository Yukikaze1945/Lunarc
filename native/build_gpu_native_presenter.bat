@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_native_presenter.cpp nvof_d3d12_interop.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib /Fe:gpu_native_presenter.dll /link /DLL || exit /b 1
