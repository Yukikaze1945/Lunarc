@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
if not exist "build\post_control" mkdir "build\post_control" || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 compile_gpu_post_control_shader.cpp d3dcompiler.lib /Fe:"build\post_control\compile_shader.exe" || exit /b 1
"build\post_control\compile_shader.exe" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /utf-8 gpu_post_control_bridge.cpp nvof_d3d12_interop.lib d3d12.lib d3dcompiler.lib user32.lib /Fe:"build\post_control\gpu_post_control_bridge.dll" /link /DLL || exit /b 1
copy /y "build\post_control\gpu_post_control_bridge.dll" "gpu_post_control_bridge.dll" >nul || (
  echo Close applications using the post-control bridge before replacing the DLL.
  exit /b 1
)
