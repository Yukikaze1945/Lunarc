// Original CUDA adapter for the VFX packed-10 encoding boundary. Colour
// constants come from the existing HDR bridge, not from a second colour policy.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdint>
#include <string>
#include <vector>
#include <mutex>
#include <memory>
#include "include/cuda.h"

#define API extern "C" __declspec(dllexport)
namespace {
thread_local std::string error;
using Program = void*;
struct Contract { float to2020[9], to709[9], m1,m2,c1,c2,c3,scale; };
struct Codec {
    HMODULE cuda{},rtc{},builtins{}; CUcontext context{}; CUmodule module{};
    CUfunction pack{},unpack{}; CUdeviceptr packed{},counts{};
    unsigned w{},h{},hdr{}; Contract colour{};
    uint64_t calls{},clipped{},scalar_bytes{};
    CUresult (CUDAAPI* push)(CUcontext){};
    CUresult (CUDAAPI* pop)(CUcontext*){};
    CUresult (CUDAAPI* alloc)(CUdeviceptr*,size_t){};
    CUresult (CUDAAPI* free)(CUdeviceptr){};
    CUresult (CUDAAPI* zero)(CUdeviceptr,unsigned,size_t){};
    CUresult (CUDAAPI* read)(void*,CUdeviceptr,size_t){};
    CUresult (CUDAAPI* load)(CUmodule*,const void*){};
    CUresult (CUDAAPI* function)(CUfunction*,CUmodule,const char*){};
    CUresult (CUDAAPI* unload)(CUmodule){};
    CUresult (CUDAAPI* launch)(CUfunction,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,CUstream,void**,void**){};
    CUresult (CUDAAPI* sync)(CUstream){};
    ~Codec() {
        CUcontext old{};
        if(context && push && pop && push(context)==CUDA_SUCCESS) {
            if(packed && free) free(packed);
            if(counts && free) free(counts);
            if(module && unload) unload(module);
            pop(&old);
        }
        if(rtc) FreeLibrary(rtc);
        if(builtins) FreeLibrary(builtins);
        if(cuda) FreeLibrary(cuda);
    }
};
int fail(const std::string& s) {error=s;return -1;}
const char* source=R"cuda(
struct Contract { float to2020[9],to709[9],m1,m2,c1,c2,c3,scale; };
__device__ float half_float(unsigned short v) {
 unsigned s=(v&32768u)<<16, e=(v>>10)&31u, m=v&1023u;
 if(!e) return (v&32768u ? -1.f:1.f)*ldexpf((float)m,-24);
 return __uint_as_float(s|((e==31u ? 255u:e+112u)<<23)|(m<<13));
}
__device__ unsigned short float_half(float f) {
 unsigned short h; asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f)); return h;
}
__device__ float pq_encode(float l, Contract c) {
 float p=powf(l,c.m1); return powf((c.c1+c.c2*p)/(1.f+c.c3*p),c.m2);
}
__device__ float pq_decode(float e, Contract c) {
 float p=powf(e,1.f/c.m2); return powf(fmaxf(p-c.c1,0.f)/(c.c2-c.c3*p),1.f/c.m1);
}
extern "C" __global__ void pack10(const unsigned char* input, unsigned pitch,
 unsigned* output,unsigned w,unsigned h,unsigned hdr,Contract c,unsigned* counters) {
 unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h)return;
 const unsigned short* p=(const unsigned short*)(input+y*pitch)+x*4;
 float a[3]={half_float(p[0]),half_float(p[1]),half_float(p[2])}; unsigned v=3u<<30;
 for(unsigned k=0;k<3;k++) {
  float l=hdr ? (c.to2020[k*3]*a[0]+c.to2020[k*3+1]*a[1]+c.to2020[k*3+2]*a[2])*c.scale:a[k];
  if(!isfinite(l)){atomicAdd(counters+1,1u);l=0.f;}
  if(l<0.f||l>1.f)atomicAdd(counters,1u);
  l=fminf(fmaxf(l,0.f),1.f);
  float e=hdr ? pq_encode(l,c):(l<.018f ? 4.5f*l:1.099f*powf(l,.45f)-.099f);
  unsigned q=__float2uint_rn(fminf(fmaxf(e,0.f),1.f)*1023.f);
  v|=q<<(k*10);
 }
 output[y*w+x]=v;
}
extern "C" __global__ void unpack10(const unsigned char* input,unsigned pitch,
 unsigned char* output,unsigned opitch,unsigned w,unsigned h,unsigned hdr,Contract c) {
 unsigned x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x>=w||y>=h)return;
 unsigned v=((const unsigned*)(input+y*pitch))[x]; float a[3];
 for(unsigned k=0;k<3;k++) {
  float e=((v>>(k*10))&1023u)/1023.f;
  a[k]=hdr ? pq_decode(e,c)/c.scale:(e<.081f ? e/4.5f:powf((e+.099f)/1.099f,1.f/.45f));
 }
 unsigned short* p=(unsigned short*)(output+y*opitch)+x*4;
 for(unsigned k=0;k<3;k++)p[k]=float_half(hdr ? c.to709[k*3]*a[0]+c.to709[k*3+1]*a[1]+c.to709[k*3+2]*a[2]:a[k]);
 p[3]=float_half(1.f);
}
)cuda";
}
API const char* vfx_rgb10_last_error(){return error.c_str();}
API void vfx_rgb10_close(Codec* c){delete c;}
API int vfx_rgb10_pointer(Codec* c,uint64_t* ptr,unsigned* pitch) {
 if(!c||!ptr||!pitch)return fail("invalid codec pointer output");*ptr=c->packed;*pitch=c->w*4;return 0;
}
API int vfx_rgb10_open(void* context,unsigned w,unsigned h,unsigned hdr,
 const float* colour,const wchar_t* runtime,Codec** out) {
 if(!out)return fail("null codec output"); *out=nullptr;
 if(!context||!w||!h||w>16384||h>16384||hdr>1||!colour||!runtime)return fail("invalid codec arguments");
 auto c=std::make_unique<Codec>(); c->context=(CUcontext)context;c->w=w;c->h=h;c->hdr=hdr;
 memcpy(&c->colour,colour,sizeof(Contract));
 c->cuda=LoadLibraryW(L"nvcuda.dll");
    c->builtins=LoadLibraryExW((std::wstring(runtime)+L"\\bin\\nvrtc-builtins64_128.dll").c_str(),nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 c->rtc=LoadLibraryExW((std::wstring(runtime)+L"\\bin\\nvrtc64_120_0.dll").c_str(),nullptr,
  LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!c->cuda||!c->rtc)return fail("VFX 1.3 CUDA compiler runtime is missing");
#define BIND(field,symbol) c->field=reinterpret_cast<decltype(c->field)>(GetProcAddress(c->cuda,symbol)); if(!c->field)return fail(symbol " missing")
 BIND(push,"cuCtxPushCurrent_v2");BIND(pop,"cuCtxPopCurrent_v2");BIND(alloc,"cuMemAlloc_v2");
 BIND(free,"cuMemFree_v2");BIND(zero,"cuMemsetD32_v2");BIND(read,"cuMemcpyDtoH_v2");
 BIND(load,"cuModuleLoadData");BIND(function,"cuModuleGetFunction");BIND(unload,"cuModuleUnload");
 BIND(launch,"cuLaunchKernel");BIND(sync,"cuStreamSynchronize");
#undef BIND
 auto create=reinterpret_cast<int(*)(Program*,const char*,const char*,int,const char**,const char**)>(GetProcAddress(c->rtc,"nvrtcCreateProgram"));
 auto compile=reinterpret_cast<int(*)(Program,int,const char**)>(GetProcAddress(c->rtc,"nvrtcCompileProgram"));
 auto size=reinterpret_cast<int(*)(Program,size_t*)>(GetProcAddress(c->rtc,"nvrtcGetPTXSize"));
 auto ptx=reinterpret_cast<int(*)(Program,char*)>(GetProcAddress(c->rtc,"nvrtcGetPTX"));
 auto destroy=reinterpret_cast<int(*)(Program*)>(GetProcAddress(c->rtc,"nvrtcDestroyProgram"));
 auto logsize=reinterpret_cast<int(*)(Program,size_t*)>(GetProcAddress(c->rtc,"nvrtcGetProgramLogSize"));
 auto log=reinterpret_cast<int(*)(Program,char*)>(GetProcAddress(c->rtc,"nvrtcGetProgramLog"));
 if(!create||!compile||!size||!ptx||!destroy||!logsize||!log)return fail("incomplete NVRTC interface");
 Program program{};if(create(&program,source,"original_rgb10.cu",0,nullptr,nullptr))return fail("NVRTC create failed");
 const char* options[]={"--gpu-architecture=compute_80","--std=c++14"};
 if(compile(program,2,options)) {
  size_t n=0;logsize(program,&n);std::vector<char> data(n+1);log(program,data.data());destroy(&program);
  return fail(std::string("RGB10 kernel compilation failed: ")+data.data());
 }
 size_t n=0;int result=size(program,&n);std::vector<char> data(n);
 if(!result)result=ptx(program,data.data());destroy(&program);
 if(result)return fail("NVRTC PTX extraction failed");
 CUcontext old{};if(c->push(c->context)!=CUDA_SUCCESS)return fail("codec context activation failed");
 auto rc=c->load(&c->module,data.data());
 if(rc==CUDA_SUCCESS)rc=c->function(&c->pack,c->module,"pack10");
 if(rc==CUDA_SUCCESS)rc=c->function(&c->unpack,c->module,"unpack10");
 if(rc==CUDA_SUCCESS)rc=c->alloc(&c->packed,size_t(w)*h*4);
 if(rc==CUDA_SUCCESS)rc=c->alloc(&c->counts,8);
 c->pop(&old);
 if(rc!=CUDA_SUCCESS)return fail("CUDA codec setup failed: "+std::to_string(rc));
 *out=c.release();error.clear();return 0;
}
API int vfx_rgb10_pack(Codec* c,uint64_t input,unsigned pitch,uint64_t* output,unsigned* opitch) {
 if(!c||!input||pitch<c->w*8||!output||!opitch)return fail("invalid linear input");
 CUcontext old{};if(c->push(c->context)!=CUDA_SUCCESS)return fail("codec context activation failed");
 auto rc=c->zero(c->counts,0,2);
 void* args[]={&input,&pitch,&c->packed,&c->w,&c->h,&c->hdr,&c->colour,&c->counts};
 if(rc==CUDA_SUCCESS)rc=c->launch(c->pack,(c->w+15)/16,(c->h+15)/16,1,16,16,1,0,nullptr,args,nullptr);
 unsigned counts[2]{};
 if(rc==CUDA_SUCCESS)rc=c->sync(nullptr);
 if(rc==CUDA_SUCCESS){rc=c->read(counts,c->counts,8);c->scalar_bytes+=8;}
 c->pop(&old);
 if(rc!=CUDA_SUCCESS)return fail("CUDA RGB10 pack failed: "+std::to_string(rc));
 if(counts[1])return fail("nonfinite linear colour cannot enter VFX RGB10");
 c->calls++;c->clipped+=counts[0];*output=c->packed;*opitch=c->w*4;error.clear();return 0;
}
API int vfx_rgb10_unpack(Codec* c,uint64_t input,unsigned pitch,uint64_t output,unsigned opitch) {
 if(!c||!input||!output||pitch<c->w*4||opitch<c->w*8)return fail("invalid packed input");
 CUcontext old{};if(c->push(c->context)!=CUDA_SUCCESS)return fail("codec context activation failed");
 void* args[]={&input,&pitch,&output,&opitch,&c->w,&c->h,&c->hdr,&c->colour};
 auto rc=c->launch(c->unpack,(c->w+15)/16,(c->h+15)/16,1,16,16,1,0,nullptr,args,nullptr);
 if(rc==CUDA_SUCCESS)rc=c->sync(nullptr);c->pop(&old);
 if(rc!=CUDA_SUCCESS)return fail("CUDA RGB10 unpack failed: "+std::to_string(rc));
 error.clear();return 0;
}
API int vfx_rgb10_counters(Codec* c,uint64_t* values) {
 if(!c||!values)return fail("invalid codec counters");
 values[0]=c->calls;values[1]=c->clipped;values[2]=c->scalar_bytes;values[3]=0;return 0;
}
