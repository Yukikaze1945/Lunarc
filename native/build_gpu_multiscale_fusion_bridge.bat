@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\fusion" mkdir "build\fusion"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_multiscale_fusion_bridge.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib user32.lib /Fo:build\fusion\gpu_multiscale_fusion_bridge.obj /Fe:build\fusion\gpu_multiscale_fusion_bridge.dll /link /DLL /IMPLIB:build\fusion\gpu_multiscale_fusion_bridge.lib || exit /b 1
copy /y "build\fusion\gpu_multiscale_fusion_bridge.dll" "gpu_multiscale_fusion_bridge.dll" >nul
if errorlevel 1 (
    echo pending>"build\fusion\pending.txt"
    echo Compiled successfully. Close Lunarc and restart run_gui.bat to apply the DLL.
    exit /b 1
)
if exist "build\fusion\pending.txt" del "build\fusion\pending.txt"
