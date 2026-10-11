@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 ^
 /I"include" gpu_hdr_scrgb_bridge.cpp ^
 d3d12.lib d3dcompiler.lib user32.lib ^
 /link /OUT:gpu_hdr_scrgb_bridge.dll || exit /b 1
