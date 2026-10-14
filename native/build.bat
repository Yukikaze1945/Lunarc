@echo off
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
cl.exe /LD /O2 /EHsc /DNVOF_BRIDGE_EXPORTS /I. /Iinclude nvof_bridge.cpp /Fe:nvof_bridge.dll /link /DLL
