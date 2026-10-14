@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\vfx" mkdir "build\vfx"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4100 /wd4201 /wd4324 /utf-8 /Fo:build\vfx\ ^
 /I"include" rtx_vsr_bridge.cpp vfx_rgb10_codec.cpp ^
 /link /OUT:build\vfx\rtx_vsr_bridge.dll /IMPLIB:build\vfx\rtx_vsr_bridge.lib || exit /b 1
copy /y "build\vfx\rtx_vsr_bridge.dll" "rtx_vsr_bridge.dll" >nul
if errorlevel 1 (
    echo pending>"build\vfx\pending.txt"
    echo Compiled successfully. Close Lunarc and restart run_gui.bat to apply the DLL.
    exit /b 1
)
if exist "build\vfx\pending.txt" del "build\vfx\pending.txt"
exit /b 0
