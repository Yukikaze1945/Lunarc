@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
cd /d "%~dp0" || exit /b 1
set "ROOT=%~dp0.."
set "SDK=%ROOT%\native\vendor\fidelityfx\sdk"
set "LIBS=%ROOT%\bench_out\amd_flow_20261010\ffx_sdk"
set "OUT=%ROOT%\bench_out\amd_flow_20261010\bridge"
if not exist "%OUT%" mkdir "%OUT%"
if not exist "%LIBS%\ffx_opticalflow_x64.lib" (echo Missing built FidelityFX optical-flow library & exit /b 2)
cl.exe /nologo /LD /MD /O2 /EHsc /std:c++17 /W4 /utf-8 /Fo"%OUT%\\" /I"%SDK%\include" /I"include" ^
 fidelityfx_flow_bridge.cpp /link /OUT:"%OUT%\fidelityfx_flow_bridge.dll" ^
 /IMPLIB:"%OUT%\fidelityfx_flow_bridge.lib" ^
 /LIBPATH:"%LIBS%" /LIBPATH:"%~dp0" ffx_opticalflow_x64.lib ffx_backend_dx12_x64.lib ^
 nvof_d3d12_interop.lib d3d12.lib dxgi.lib d3dcompiler.lib dxguid.lib || exit /b 1
copy /y "%OUT%\fidelityfx_flow_bridge.dll" "%~dp0fidelityfx_flow_bridge.dll" >nul
if errorlevel 1 exit /b 1
echo Built FidelityFX Optical Flow bridge: %OUT%\fidelityfx_flow_bridge.dll
exit /b 0
