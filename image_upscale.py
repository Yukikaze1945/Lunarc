"""SDR image SR models on CUDA, copying between explicitly owned contexts."""
import ctypes as C
import time
from image_model_store import store,CATALOG

def model_config(settings,kind):
    variant=settings.get('image_model_variant','denoise')
    if kind=='illustration' and variant not in ('denoise','detail'):raise ValueError('未知 IllustrationJaNai V3 版本')
    key='illustration-'+variant if kind=='illustration' else 'animesharp'
    if key not in CATALOG:raise ValueError('未知图片放大模型')
    tile=settings.get('image_model_tile',0)
    if isinstance(tile,bool) or tile not in (0,128,256,512):raise ValueError('图片模型分块尺寸须为整图、128、256 或 512')
    return dict(CATALOG[key],key=key,tile=tile)

class ImageModelEffect:
    wide=False;output_hdr=False
    def __init__(self,settings,kind,size,*,wide=False,hdr=False):
        if wide or hdr:raise ValueError('图片放大模型需要 SDR 图像，不能接在 TrueHDR 后面')
        self.config=model_config(settings,kind);self.kind=kind;self.source_size=tuple(size)
        self.target=tuple(n*self.config['scale'] for n in size)
        self.model=self.input=self.output=None;self.calls=self.device_bytes=0
        self.weight_path=None;self.gpu_ms=0

    def _load(self):
        try:
            import torch
            from spandrel import ModelLoader,ImageModelDescriptor
        except ImportError as error:
            raise ValueError('请先在组件管理中安装 AI 模型运行环境和图片模型组件') from error
        if not torch.cuda.is_available():raise ValueError('图片放大模型需要 CUDA PyTorch')
        self.weight_path=store.resolve(self.config['key'])
        from spandrel.__helpers.canonicalize import canonicalize_state_dict
        if self.weight_path.suffix=='.safetensors':
            from safetensors.torch import load_file
            state=load_file(str(self.weight_path))
        else:state=torch.load(str(self.weight_path),map_location='cpu',weights_only=True)
        descriptor=ModelLoader().load_from_state_dict(canonicalize_state_dict(state))
        if not isinstance(descriptor,ImageModelDescriptor) or descriptor.scale!=self.config['scale'] or descriptor.input_channels!=3 or descriptor.output_channels!=3:
            raise ValueError('导入模型的倍率或 RGB 输入输出不匹配')
        if descriptor.architecture.id!=('DAT' if self.kind=='illustration' else 'RCAN'):
            raise ValueError('导入模型架构不匹配：需要 DAT2／RCAN 正式版本')
        self.model=descriptor.to('cuda').eval()

    def process(self,candidate,frame,reset):
        import torch
        from native_effects import Exchange
        from raft_gpu_flow import _PeerCopy,_check
        from torch_cuda_context import torch_primary_context
        w,h=self.source_size;ow,oh=self.target
        if self.input is None:
            self.input=Exchange(candidate['d3d12_device'],frame.cuda_context,self.source_size)
            self.output=Exchange(candidate['d3d12_device'],frame.cuda_context,self.target)
        color=self.input.color(candidate);output_ptr,output_pitch=self.output.pointer()
        cuda=C.WinDLL('nvcuda.dll')
        cuda.cuCtxGetCurrent.argtypes=[C.POINTER(C.c_void_p)]
        cuda.cuMemcpy3DPeer.argtypes=[C.POINTER(_PeerCopy)]
        started=time.perf_counter()
        with torch_primary_context('cuda'):
            if self.model is None:self._load()
            context=C.c_void_p();_check(cuda.cuCtxGetCurrent(C.byref(context)),'query Torch context')
            raw=torch.empty((h,w,4),dtype=torch.uint8,device='cuda')
            copy=_PeerCopy(srcMemoryType=2,srcDevice=color['device_ptr'],srcContext=frame.cuda_context,srcPitch=color['pitch_bytes'],srcHeight=h,
                dstMemoryType=2,dstDevice=raw.data_ptr(),dstContext=context.value,dstPitch=w*4,dstHeight=h,WidthInBytes=w*4,Height=h,Depth=1)
            _check(cuda.cuMemcpy3DPeer(C.byref(copy)),'image model input device copy')
            tensor=raw[...,:3].permute(2,0,1)[None].float().div_(255)
            with torch.inference_mode():
                tile=self.config['tile'];scale=self.config['scale']
                if not tile:result=self.model(tensor)
                else:
                    # Explicit optional halo tiles: global-attention models can differ from whole-image inference.
                    result=torch.empty((1,3,oh,ow),device='cuda');halo=32
                    for y in range(0,h,tile):
                        for x in range(0,w,tile):
                            top,left=max(0,y-halo),max(0,x-halo);bottom,right=min(h,y+tile+halo),min(w,x+tile+halo)
                            patch=self.model(tensor[:,:,top:bottom,left:right]);ch,cw=min(tile,h-y)*scale,min(tile,w-x)*scale
                            sy,sx=(y-top)*scale,(x-left)*scale
                            result[:,:,y*scale:y*scale+ch,x*scale:x*scale+cw]=patch[:,:,sy:sy+ch,sx:sx+cw]
                            del patch
            rgba=torch.empty((oh,ow,4),dtype=torch.uint8,device='cuda')
            # The inference output is private. Keep the same FP32 operations,
            # reusing its storage instead of allocating full-size RGB copies.
            with torch.inference_mode():result.clamp_(0,1).mul_(255).round_()
            rgba[...,:3]=result[0].permute(1,2,0).to(torch.uint8);rgba[...,3]=255
            torch.cuda.synchronize()
            copy=_PeerCopy(srcMemoryType=2,srcDevice=rgba.data_ptr(),srcContext=context.value,srcPitch=ow*4,srcHeight=oh,
                dstMemoryType=2,dstDevice=output_ptr,dstContext=frame.cuda_context,dstPitch=output_pitch,dstHeight=oh,WidthInBytes=ow*4,Height=oh,Depth=1)
            _check(cuda.cuMemcpy3DPeer(C.byref(copy)),'image model output device copy')
            del raw,tensor,result,rgba
        self.calls+=1;self.device_bytes+=(w*h+ow*oh)*4;self.gpu_ms=(time.perf_counter()-started)*1000
        return self.output.texture(output_ptr,output_pitch)

    def report(self):
        return dict(stage=self.kind,model=self.config['name'],scale=self.config['scale'],tile=self.config['tile'],
            input_size=self.source_size,output_size=self.target,calls=self.calls,host_image_bytes=0,
            device_copy_bytes=self.device_bytes,processing_format='rgba8',precision='float32',
            model_ms=self.gpu_ms,weights=str(self.weight_path) if self.weight_path else None)

    def close(self):
        from gpu_cleanup import release_owned, finish_cleanup
        from torch_cuda_context import torch_primary_context
        errors=[]
        if self.model is not None:
            try:
                with torch_primary_context('cuda'):
                    self.model=None
                    import torch
                    torch.cuda.empty_cache()
            except Exception as error:errors.append(('Torch model',error))
        for name in ('output','input'):release_owned(self,name,errors)
        finish_cleanup(errors)
