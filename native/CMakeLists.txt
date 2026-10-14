cmake_minimum_required(VERSION 3.18)
project(nvof_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(nvof_bridge SHARED
    nvof_bridge.cpp
    nvof_bridge.h
)

target_include_directories(nvof_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(nvof_bridge PROPERTIES
    OUTPUT_NAME "nvof_bridge"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_library(residual_bridge SHARED
    residual_bridge.cpp
    residual_bridge.h
)

target_include_directories(residual_bridge PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(residual_bridge PROPERTIES
    OUTPUT_NAME "residual_bridge"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
)

if(WIN32)
    add_library(nv12_rgba_bridge SHARED nv12_rgba_bridge.cpp nv12_rgba_bridge.h)
    target_include_directories(nv12_rgba_bridge PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
    set_target_properties(nv12_rgba_bridge PROPERTIES
        OUTPUT_NAME "nv12_rgba_bridge"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")

    if(DEFINED FFMPEG_SDK_DIR)
        add_library(nvdec_bridge SHARED nvdec_bridge.cpp nvdec_bridge.h)
        target_include_directories(nvdec_bridge PRIVATE
            "${FFMPEG_SDK_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(nvdec_bridge PRIVATE
            "${FFMPEG_SDK_DIR}/lib/avcodec.lib"
            "${FFMPEG_SDK_DIR}/lib/avformat.lib"
            "${FFMPEG_SDK_DIR}/lib/avfilter.lib"
            "${FFMPEG_SDK_DIR}/lib/avutil.lib")
        if(MSVC)
            target_compile_options(nvdec_bridge PRIVATE /wd4244)
        endif()
        set_target_properties(nvdec_bridge PROPERTIES
            OUTPUT_NAME "nvdec_bridge"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")

        add_library(nvenc_bridge SHARED nvenc_bridge.cpp nvenc_bridge.h)
        target_include_directories(nvenc_bridge PRIVATE
            "${FFMPEG_SDK_DIR}/include" "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(nvenc_bridge PRIVATE
            nvof_d3d12_interop d3d12
            "${FFMPEG_SDK_DIR}/lib/avcodec.lib"
            "${FFMPEG_SDK_DIR}/lib/avformat.lib"
            "${FFMPEG_SDK_DIR}/lib/avutil.lib")
        if(MSVC)
            target_compile_options(nvenc_bridge PRIVATE /wd4244)
        endif()
        set_target_properties(nvenc_bridge PROPERTIES
            OUTPUT_NAME "nvenc_bridge"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    if(DEFINED DLSS_SDK_DIR)
        add_library(dlss_gpu_host SHARED dlss_gpu_host.cpp dlss_gpu_host.h)
        target_include_directories(dlss_gpu_host PRIVATE "${DLSS_SDK_DIR}/include")
        target_link_libraries(dlss_gpu_host PRIVATE
            nvof_d3d12_interop d3d12 dxgi d3dcompiler advapi32 user32
            "${DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_s.lib")
        if(MSVC)
            set_property(TARGET dlss_gpu_host PROPERTY MSVC_RUNTIME_LIBRARY
                "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        endif()
        set_target_properties(dlss_gpu_host PROPERTIES
            OUTPUT_NAME "dlss_gpu_host"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
            RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")
    endif()

    add_library(nvof_d3d12_interop SHARED
        nvof_d3d12_interop.cpp
        nvof_d3d12_interop.h
    )
    target_link_libraries(nvof_d3d12_interop PRIVATE d3d12 d3dcompiler)
    set_target_properties(nvof_d3d12_interop PROPERTIES
        OUTPUT_NAME "nvof_d3d12_interop"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    add_library(gpu_residual_bridge SHARED
        gpu_residual_bridge.cpp gpu_residual_bridge.h)
    target_link_libraries(gpu_residual_bridge PRIVATE
        nvof_d3d12_interop d3d12 d3dcompiler user32)
    set_target_properties(gpu_residual_bridge PROPERTIES
        OUTPUT_NAME "gpu_residual_bridge"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")

    add_library(gpu_lanczos_bridge SHARED
        gpu_lanczos_bridge.cpp gpu_lanczos_bridge.h)
    target_link_libraries(gpu_lanczos_bridge PRIVATE
        nvof_d3d12_interop d3d12 d3dcompiler user32)
    set_target_properties(gpu_lanczos_bridge PROPERTIES
        OUTPUT_NAME "gpu_lanczos_bridge"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}")

    add_library(gpu_native_presenter SHARED gpu_native_presenter.cpp)
    target_link_libraries(gpu_native_presenter PRIVATE nvof_d3d12_interop d3d12 dxgi d3dcompiler user32)
    set_target_properties(gpu_native_presenter PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}")

    add_library(dlss_gpu_resources SHARED
        dlss_gpu_resources.cpp
        dlss_gpu_resources.h
    )
    target_include_directories(dlss_gpu_resources PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(dlss_gpu_resources PROPERTIES
        OUTPUT_NAME "dlss_gpu_resources"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_CURRENT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endif()
