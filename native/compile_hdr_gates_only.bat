@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /I"include" ^
 gpu_hdr_scrgb_bridge_test.cpp gpu_hdr_scrgb_bridge.lib d3d12.lib ^
 /Fe:gpu_hdr_scrgb_bridge_test.exe || exit /b 1
