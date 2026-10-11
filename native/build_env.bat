@echo off
rem Reuse an x64 Native Tools prompt, or discover MSVC without a machine path.
if defined VSCMD_VER exit /b 0
if defined VCVARS64 (
  call "%VCVARS64%" || exit /b 1
  exit /b 0
)
set "LUNARC_VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%LUNARC_VSWHERE%" (
  echo Install Visual Studio C++ Build Tools and Windows SDK, or set VCVARS64.
  exit /b 2
)
set "LUNARC_VS_ROOT="
for /f "usebackq tokens=*" %%I in (`"%LUNARC_VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "LUNARC_VS_ROOT=%%I"
if not defined LUNARC_VS_ROOT exit /b 2
call "%LUNARC_VS_ROOT%\VC\Auxiliary\Build\vcvars64.bat" || exit /b 1
exit /b 0
