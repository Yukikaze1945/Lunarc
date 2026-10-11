@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 ^
 /I"include" nv12_rgba_bridge.cpp /link /OUT:nv12_rgba_bridge.dll || exit /b 1
