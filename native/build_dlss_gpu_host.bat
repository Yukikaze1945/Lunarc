@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined DLSS_SDK_DIR (echo Set DLSS_SDK_DIR to your NVIDIA DLSS SDK directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
if not exist "build\host" mkdir "build\host"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /I"%DLSS_SDK_DIR%\include" dlss_gpu_host.cpp nvof_d3d12_interop.lib d3d12.lib dxgi.lib d3dcompiler.lib advapi32.lib user32.lib "%DLSS_SDK_DIR%\lib\Windows_x86_64\x64\nvsdk_ngx_s.lib" /Fo:build\host\dlss_gpu_host.obj /Fe:build\host\dlss_gpu_host.dll /link /DLL /IMPLIB:build\host\dlss_gpu_host.lib || exit /b 1
copy /y "build\host\dlss_gpu_host.dll" "dlss_gpu_host.dll" >nul
if errorlevel 1 (
    echo pending>"build\host\pending.txt"
    echo Compiled successfully. Close Lunarc and restart run_gui.bat to apply the DLL.
    exit /b 1
)
if exist "build\host\pending.txt" del "build\host\pending.txt"
