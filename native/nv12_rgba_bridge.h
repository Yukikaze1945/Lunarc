#ifndef NV12_RGBA_BRIDGE_H
#define NV12_RGBA_BRIDGE_H

#include <stdint.h>

#ifdef _WIN32
#  ifdef NV12_RGBA_BRIDGE_EXPORTS
#    define NV12_RGBA_API __declspec(dllexport)
#  else
#    define NV12_RGBA_API __declspec(dllimport)
#  endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 3 adds Nv12RgbaTransferCounters and nv12_rgba_get_transfer_counters. Every
 * ABI 2 entry point keeps its signature and behaviour, so an ABI 2 caller that
 * never reads the counters is unaffected. */
#define NV12_RGBA_BRIDGE_ABI_VERSION 3u
#define NV12_RGBA_COLOR_MATRIX_BT601 1
#define NV12_RGBA_COLOR_MATRIX_BT709 2
typedef struct Nv12RgbaBridge Nv12RgbaBridge;

NV12_RGBA_API uint32_t nv12_rgba_abi_version(void);
/* The CUDA context and source surfaces remain owned by the caller. The caller
 * must close this converter before destroying that context. Width and height
 * must be positive and even. */
NV12_RGBA_API int32_t nv12_rgba_open(void* cuda_context, uint32_t width,
    uint32_t height, Nv12RgbaBridge** out_bridge);
/* Select the limited-range YUV matrix. The original open entry point retains
 * its BT.601 behaviour for existing callers. */
NV12_RGBA_API int32_t nv12_rgba_open_matrix(void* cuda_context, uint32_t width,
    uint32_t height, int32_t color_matrix, Nv12RgbaBridge** out_bridge);
/* Synchronously convert a borrowed NV12 CUDA frame to a persistent RGBA8 CUDA
 * allocation. The returned pointer and pitch belong to this converter and
 * remain valid until the next convert or close. No host copy occurs. */
NV12_RGBA_API int32_t nv12_rgba_convert(Nv12RgbaBridge* bridge,
    uint64_t y_device_ptr, uint32_t y_pitch_bytes,
    uint64_t uv_device_ptr, uint32_t uv_pitch_bytes,
    uint64_t* out_device_ptr, uint32_t* out_pitch_bytes);

/* Every field is incremented by the conversions that ran, so a zero is a
 * measurement of a boundary that was not crossed rather than a default. The
 * kernel writes the whole RGBA8 output once per conversion; only those writes
 * are counted, because the source reads are per-pixel loads served through
 * caches and the physical bytes they move are not observable from here. The
 * convert call itself performs no host copy and no device-to-device copy. */
typedef struct Nv12RgbaTransferCounters {
    uint32_t struct_size;
    uint64_t convert_count;             /* conversions that returned 0 */
    uint64_t output_device_write_bytes; /* RGBA8 bytes the kernel wrote: the
                                         * w*h*4 output payload per conversion */
} Nv12RgbaTransferCounters;

NV12_RGBA_API int32_t nv12_rgba_get_transfer_counters(Nv12RgbaBridge* bridge,
    Nv12RgbaTransferCounters* counters);
NV12_RGBA_API void nv12_rgba_close(Nv12RgbaBridge* bridge);
NV12_RGBA_API const char* nv12_rgba_last_error(void);

#ifdef __cplusplus
}
#endif

#endif
