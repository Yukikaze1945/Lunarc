@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
call build_nvof_d3d12_interop.bat || exit /b 1
cl.exe /nologo /O2 /EHsc /std:c++17 /W4 /WX /utf-8 nvof_d3d12_interop_test.cpp nvof_d3d12_interop.lib nvof_bridge.lib d3d12.lib /Fe:nvof_d3d12_interop_test.exe || exit /b 1
nvof_d3d12_interop_test.exe || exit /b 1
