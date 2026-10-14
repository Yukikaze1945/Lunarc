@echo off
rem Builds nvof_bridge.dll with NVOF_TEST_DEBUG_EXPORTS defined, which adds the one entry
rem point that copies an NVOF input surface back to the host. The byte-map kernel's output is
rem only observable that way, and the claim it has to answer - "the bytes that entered NVOF
rem are the bytes cv2.cvtColor(BGR2RGBA) would have produced" - cannot be answered from the
rem flow field, because two different inputs can produce the same flow.
rem
rem Same translation unit as the production build and the same optimisation flags; only the
rem macro differs, which is the point: a test build that drifted from the shipped source
rem would prove nothing about the shipped source. The production DLL does not export this
rem symbol, and the test that loads this one checks that.
rem
rem Goes to scratch, never to native\: the artifact is a test fixture, not something to ship
rem or commit. Returns non-zero if the compile or link fails.
rem
rem Usage:  build_nvof_test.bat [output-directory]

setlocal
set "OUTDIR=%~dp0..\scratch\native_test"
if not "%~1"=="" set "OUTDIR=%~1"
call "%~dp0build_env.bat" || exit /b 1
if not exist "%OUTDIR%" mkdir "%OUTDIR%" || exit /b 1
cd /d "%OUTDIR%" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /DNVOF_BRIDGE_EXPORTS /DNVOF_TEST_DEBUG_EXPORTS /I"%~dp0." /I"%~dp0include" "%~dp0nvof_bridge.cpp" /Fe:nvof_bridge_test.dll /link /DLL || exit /b 1
echo built %OUTDIR%\nvof_bridge_test.dll
endlocal
