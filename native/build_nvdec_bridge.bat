@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined FFMPEG_SDK_DIR (echo Set FFMPEG_SDK_DIR to the FFmpeg SDK directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
if not exist "build\decoder" mkdir "build\decoder"
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4244 /utf-8 ^
 /I"%FFMPEG_SDK_DIR%\include" /I"include" /Fo"build\decoder\nvdec_bridge.obj" nvdec_bridge.cpp ^
 /link /OUT:build\decoder\nvdec_bridge.dll /LIBPATH:"%FFMPEG_SDK_DIR%\lib" ^
 avcodec.lib avformat.lib avfilter.lib avutil.lib || exit /b 1
copy /y "build\decoder\nvdec_bridge.dll" "nvdec_bridge.dll" >nul 2>nul
if errorlevel 1 (
  echo Restart Lunarc to apply the rebuilt decoder.
  echo pending>"build\decoder\pending.txt"
) else (
  if exist "build\decoder\pending.txt" del "build\decoder\pending.txt"
)
exit /b 0
