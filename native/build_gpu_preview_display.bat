@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\display" mkdir "build\display"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /Fo:build\display\ gpu_preview_display.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib /link /DLL /OUT:build\display\gpu_preview_display.dll /IMPLIB:build\display\gpu_preview_display.lib || exit /b 1
copy /y "build\display\gpu_preview_display.dll" "gpu_preview_display.dll" >nul
if errorlevel 1 (
    echo pending>"build\display\pending.txt"
    echo Compiled successfully. Close Lunarc and restart run_gui.bat to apply the display DLL.
    exit /b 1
)
if exist "build\display\pending.txt" del "build\display\pending.txt"
exit /b 0
