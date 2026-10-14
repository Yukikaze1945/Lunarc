@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\exchange" mkdir "build\exchange" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 /Fo"build\exchange\gpu_rgba_exchange.obj" gpu_rgba_exchange.cpp nvof_d3d12_interop.lib d3d12.lib /link /DLL /OUT:build\exchange\gpu_rgba_exchange.dll /IMPLIB:gpu_rgba_exchange.lib || exit /b 1
copy /y "build\exchange\gpu_rgba_exchange.dll" "gpu_rgba_exchange.dll" >nul 2>nul
if errorlevel 1 (
  echo Restart Lunarc to apply the rebuilt texture exchange.
  echo pending>"build\exchange\pending.txt"
) else (
  if exist "build\exchange\pending.txt" del "build\exchange\pending.txt"
)
exit /b 0
