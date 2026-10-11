/*
 * Minimal CUDA Driver API definitions for NVOF Bridge
 * Provides the required CUDA types for nvOpticalFlowCuda.h
 */
#ifndef _CUDA_H_
#define _CUDA_H_

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32)
#define CUDAAPI __stdcall
#else
#define CUDAAPI
#endif

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUstream_st* CUstream;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct _CUarray* CUarray;
typedef unsigned __int64 CUdeviceptr;

typedef enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_UNKNOWN = 999
} CUresult;

typedef enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
} CUmemorytype;

typedef struct CUDA_MEMCPY2D_st {
    size_t srcXInBytes;
    size_t srcY;
    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;

    size_t dstXInBytes;
    size_t dstY;
    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;

    size_t WidthInBytes;
    size_t Height;
} CUDA_MEMCPY2D;

typedef struct CUDA_MEMCPY3D_PEER_st {
    size_t srcXInBytes, srcY, srcZ, srcLOD;
    CUmemorytype srcMemoryType;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    CUcontext srcContext;
    size_t srcPitch, srcHeight;
    size_t dstXInBytes, dstY, dstZ, dstLOD;
    CUmemorytype dstMemoryType;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    CUcontext dstContext;
    size_t dstPitch, dstHeight;
    size_t WidthInBytes, Height, Depth;
} CUDA_MEMCPY3D_PEER;

#endif /* _CUDA_H_ */
