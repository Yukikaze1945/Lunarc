@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4100 /wd4201 /wd4324 /utf-8 ^
 /I"include" rtx_vsr_bridge.cpp vfx_rgb10_codec.cpp ^
 /link /OUT:rtx_vsr_bridge.dll || exit /b 1
exit /b 0
