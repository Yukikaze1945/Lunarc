@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined FFMPEG_SDK_DIR (echo Set FFMPEG_SDK_DIR to the FFmpeg SDK directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
if not exist "build\encoder" mkdir "build\encoder"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4244 /utf-8 ^
 /I"%FFMPEG_SDK_DIR%\include" /I"include" /Fo"build\encoder\nvenc_bridge.obj" nvenc_bridge.cpp ^
 /link /OUT:build\encoder\nvenc_bridge.dll /LIBPATH:"%FFMPEG_SDK_DIR%\lib" /LIBPATH:"%~dp0" ^
 nvof_d3d12_interop.lib d3d12.lib ^
 avcodec.lib avformat.lib avutil.lib || exit /b 1
copy /y "build\encoder\nvenc_bridge.dll" "nvenc_bridge.dll" >nul 2>nul
if errorlevel 1 (
  echo Restart Lunarc to apply the rebuilt encoder.
  echo pending>"build\encoder\pending.txt"
) else (
  if exist "build\encoder\pending.txt" del "build\encoder\pending.txt"
)
exit /b 0
