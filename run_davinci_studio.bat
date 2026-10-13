@echo off
chcp 65001 >nul
title Lunarc
cd /d "%~dp0"

rem Apply successfully compiled DLLs deferred while a running backend held them.
if exist "%~dp0native\build\display\pending.txt" (
    copy /y "%~dp0native\build\display\gpu_preview_display.dll" "%~dp0native\gpu_preview_display.dll" >nul
    if errorlevel 1 (
        echo [ERROR] Close the existing Lunarc window and backend before restarting.
        goto :FAILED
    )
    del "%~dp0native\build\display\pending.txt"
)
if exist "%~dp0native\build\vfx\pending.txt" (
    copy /y "%~dp0native\build\vfx\rtx_vsr_bridge.dll" "%~dp0native\rtx_vsr_bridge.dll" >nul
    if errorlevel 1 (
        echo [ERROR] Close the existing Lunarc window and backend before restarting.
        goto :FAILED
    )
    del "%~dp0native\build\vfx\pending.txt"
)
if exist "%~dp0native\build\fusion\pending.txt" (
    copy /y "%~dp0native\build\fusion\gpu_multiscale_fusion_bridge.dll" "%~dp0native\gpu_multiscale_fusion_bridge.dll" >nul
    if errorlevel 1 (
        echo [ERROR] Close the existing Lunarc window and backend before restarting.
        goto :FAILED
    )
    del "%~dp0native\build\fusion\pending.txt"
)

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
