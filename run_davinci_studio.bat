@echo off
chcp 65001 >nul
title Lunarc
cd /d "%~dp0"

echo ================================================================
echo   Lunarc
echo ================================================================
echo.

rem Respect the existing local Python override without changing its contents.
if not defined PYTHON_PATH if exist "python_path.txt" (
    for /f "usebackq delims=" %%A in ("python_path.txt") do (
        if exist "%%~A" set "PYTHON_PATH=%%~A"
    )
)
echo [INFO] Building the current workbench...
cd /d "%~dp0frontend"
call npm run build
if errorlevel 1 goto :FAILED
echo [INFO] Starting Lunarc...
call npm start
if errorlevel 1 goto :FAILED
exit /b 0

:FAILED
set "LUNARC_EXIT_CODE=%errorlevel%"
echo.
echo [ERROR] Launch failed with code: %LUNARC_EXIT_CODE%
pause
exit /b %LUNARC_EXIT_CODE%
