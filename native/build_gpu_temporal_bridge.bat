@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\temporal" mkdir "build\temporal"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /Fo"build\temporal\gpu_temporal_bridge.obj" gpu_temporal_bridge.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib user32.lib /link /DLL /OUT:build\temporal\gpu_temporal_bridge.dll /IMPLIB:gpu_temporal_bridge.lib || exit /b 1
copy /y "build\temporal\gpu_temporal_bridge.dll" "gpu_temporal_bridge.dll" >nul 2>nul
if errorlevel 1 (
  echo Restart Lunarc to apply the rebuilt temporal bridge.
  echo pending>"build\temporal\pending.txt"
) else (
  if exist "build\temporal\pending.txt" del "build\temporal\pending.txt"
)
exit /b 0
