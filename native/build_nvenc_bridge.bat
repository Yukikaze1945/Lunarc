@echo off
setlocal
call "%~dp0build_env.bat" || exit /b 1
if not defined FFMPEG_SDK_DIR (echo Set FFMPEG_SDK_DIR to the FFmpeg SDK directory. & exit /b 2)
cd /d "%~dp0" || exit /b 1
cl.exe /nologo /LD /O2 /EHsc /std:c++17 /W4 /WX /wd4244 /utf-8 ^
 /I"%FFMPEG_SDK_DIR%\include" /I"include" nvenc_bridge.cpp ^
 /link /OUT:nvenc_bridge.dll /LIBPATH:"%FFMPEG_SDK_DIR%\lib" /LIBPATH:"%~dp0" ^
 nvof_d3d12_interop.lib d3d12.lib ^
 avcodec.lib avformat.lib avutil.lib || exit /b 1
