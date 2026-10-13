@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\hdr_colour" mkdir "build\hdr_colour" || exit /b 1
rem Link away from the live DLL: a failed/locked rebuild must not remove it.
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 ^
 /I"include" gpu_hdr_scrgb_bridge.cpp ^
 d3d12.lib d3dcompiler.lib user32.lib ^
 /Fe:"build\hdr_colour\gpu_hdr_scrgb_bridge.dll" ^
 /link /OUT:"build\hdr_colour\gpu_hdr_scrgb_bridge.dll" || exit /b 1
copy /y "build\hdr_colour\gpu_hdr_scrgb_bridge.dll" "gpu_hdr_scrgb_bridge.dll" >nul || (
  echo Close applications using the colour bridge before replacing the DLL.
  exit /b 1
)
