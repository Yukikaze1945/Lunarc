@echo off
rem Builds a residual_bridge DLL with RR_TEST_FAULT_INJECTION defined, so the create path
rem can be interrupted at a named step. Same translation unit as the production build and
rem the same optimisation flags - only the macro differs - which is the point: a test build
rem that drifted from the shipped source would prove nothing about the shipped source.
rem
rem Goes to scratch, never to native\: the artifact is a test fixture, not something to
rem ship or commit. Returns non-zero if the compile or link fails.
rem
rem Usage:  build_residual_test.bat [output-directory]

setlocal
set "OUTDIR=%~dp0..\scratch\native_test"
if not "%~1"=="" set "OUTDIR=%~1"
call "%~dp0build_env.bat" || exit /b 1
if not exist "%OUTDIR%" mkdir "%OUTDIR%" || exit /b 1
cd /d "%OUTDIR%" || exit /b 1
cl.exe /nologo /LD /O2 /Oy- /EHsc /std:c++17 /DRESIDUAL_BRIDGE_EXPORTS /DRR_TEST_FAULT_INJECTION /I"%~dp0." "%~dp0residual_bridge.cpp" /Fe:residual_bridge_test.dll /link /DLL || exit /b 1
echo built %OUTDIR%\residual_bridge_test.dll
endlocal
