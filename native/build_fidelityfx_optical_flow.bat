@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
for %%I in ("%~dp0..") do set "PROJECT_ROOT=%%~fI"
set "SDK_ROOT=%PROJECT_ROOT%\native\vendor\fidelityfx\sdk"
set "EVIDENCE=%PROJECT_ROOT%\bench_out\amd_flow_20261010"
if not defined FFX_SC_EXECUTABLE set "FFX_SC_EXECUTABLE=%EVIDENCE%\compiler\FidelityFX_SC.exe"
if not exist "%FFX_SC_EXECUTABLE%" (echo Missing official FidelityFX_SC.exe; set FFX_SC_EXECUTABLE to the tool from the pinned SDK commit. & exit /b 2)
cmake -S "%SDK_ROOT%" -B "%EVIDENCE%\cmake_build" -G "Visual Studio 17 2022" -A x64 ^
 -DFFX_API_BACKEND=DX12_X64 -DFFX_OF=ON -DFFX_ALL=OFF -DFFX_AUTO_COMPILE_SHADERS=ON ^
 -DFFX_FSR1=OFF -DFFX_FSR2=OFF -DFFX_FSR3=OFF -DFFX_FSR3UPSCALER=OFF -DFFX_FI=OFF ^
 -DFFX_CACAO=OFF -DFFX_CAS=OFF -DFFX_LPM=OFF -DFFX_SPD=OFF -DFFX_VRS=OFF ^
 -DFFX_BLUR=OFF -DFFX_DOF=OFF -DFFX_LENS=OFF -DFFX_PARALLELSORT=OFF -DFFX_DENOISER=OFF ^
 -DFFX_SSSR=OFF -DFFX_BRIXELIZER=OFF -DFFX_BRIXELIZER_GI=OFF -DFFX_CLASSIFIER=OFF -DFFX_BREADCRUMBS=OFF ^
 -DFFX_SC_EXECUTABLE="%FFX_SC_EXECUTABLE%" -DFFX_DISABLE_PIX_CAPTURES=ON ^
 -DFFX_ARTIFACT_DIR="%EVIDENCE%\ffx_sdk"
if errorlevel 1 exit /b %errorlevel%
cmake --build "%EVIDENCE%\cmake_build" --config Release --target ffx_opticalflow_x64 ffx_backend_dx12_x64 -- /m
if errorlevel 1 exit /b %errorlevel%
echo Built Optical Flow static effect and DX12 backend under %EVIDENCE%\ffx_sdk
