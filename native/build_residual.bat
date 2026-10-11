@echo off
rem Builds the native residual hot path as residual_bridge.dll next to this script.
rem Needs the x64 C++ toolchain; the path below matches the one used for nvof_bridge.
call "%~dp0build_env.bat" || exit /b 1
cd /d %~dp0
cl.exe /nologo /LD /O2 /Oy- /EHsc /std:c++17 /DRESIDUAL_BRIDGE_EXPORTS /I. residual_bridge.cpp /Fe:residual_bridge.dll /link /DLL || exit /b 1
echo built %~dp0residual_bridge.dll
