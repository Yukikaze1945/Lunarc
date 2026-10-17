@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\presenter" mkdir "build\presenter"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /Fo:build\presenter\ gpu_native_presenter.cpp nvof_d3d12_interop.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib /link /DLL /OUT:build\presenter\gpu_native_presenter.dll /IMPLIB:build\presenter\gpu_native_presenter.lib || exit /b 1
copy /y "build\presenter\gpu_native_presenter.dll" "gpu_native_presenter.dll" >nul
if errorlevel 1 (
    echo pending>"build\presenter\pending.txt"
    echo Compiled successfully. Close Lunarc and restart run_gui.bat to apply the presenter DLL.
    exit /b 1
)
if exist "build\presenter\pending.txt" del "build\presenter\pending.txt"
exit /b 0
