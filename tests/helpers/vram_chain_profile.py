"""Short real-chain VRAM profile; --full-resources provides a controlled baseline.

Both use the corrected RGBA8 guidance shader. Legacy exchange/temporal allocation
DLLs may be supplied from a locally saved baseline, never distributed with the app.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import bundle_paths
import gpu_hdr_transfer_export as htx
import hdr_gpu_engine as hdr
import native_effects
import pipeline
from gpu_preview_engine import GpuPreview


class Memory(C.Structure):
    _fields_ = [(key, C.c_uint64) for key in
                ('luid','local_usage','local_budget','nonlocal_usage','nonlocal_budget')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video'); ap.add_argument('settings', type=Path)
    ap.add_argument('--transcode', type=Path); ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--full-resources', action='store_true')
    ap.add_argument('--legacy-resources', type=Path)
    ap.add_argument('--preview', action='store_true'); ap.add_argument('--frames', type=int, default=6)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    if args.full_resources:
        original = hdr.GpuHdrBridge.__init__
        def untrimmed(self, **kwargs):
            kwargs['purpose'] = 'full'
            original(self, **kwargs)
        hdr.GpuHdrBridge.__init__ = untrimmed
    if args.legacy_resources:
        original_require = bundle_paths.require_native
        def require(name, *rest, **kwargs):
            if name in ('gpu_rgba_exchange.dll', 'gpu_temporal_bridge.dll'):
                return str((args.legacy_resources/name).resolve())
            return original_require(name, *rest, **kwargs)
        bundle_paths.require_native = require
    bundle_paths.add_dll_directory(bundle_paths.native_dir())
    query = C.CDLL(str(ROOT/'native/build/memory/gpu_process_memory_probe.dll'))
    query.gpu_process_memory_query.argtypes = [C.c_void_p,C.POINTER(Memory)]
    def sample(device):
        memory = Memory()
        assert query.gpu_process_memory_query(device,C.byref(memory)) == 0
        return {key: int(getattr(memory,key)) for key,_ in Memory._fields_}
    def resources(chain):
        rows, seen = [], set()
        def visit(owner, label, depth=0):
            if owner is None or id(owner) in seen or depth > 5: return
            seen.add(id(owner))
            if isinstance(owner, hdr.GpuHdrBridge):
                rows.append(dict(owner=label,purpose=owner.purpose,bytes=owner.allocated_device_bytes()))
            elif isinstance(owner,native_effects.Exchange):
                fn = getattr(owner.lib,'gpu_rgba_exchange_allocated_bytes',None)
                value = None
                if fn and owner.handle:
                    fn.argtypes,fn.restype=[C.c_void_p],C.c_uint64
                    value=int(fn(owner.handle))
                rows.append(dict(owner=label,bytes=value))
            elif owner.__class__.__name__ == 'GpuTemporalControl':
                rows.append(dict(owner=label,bytes=owner.allocated_device_bytes(),controls=owner.controls.as_dict()))
            for name in ('bridge','decode_bridge','encode_bridge','guide_bridge','input','output',
                         'input_effect','output_effect','truehdr_tail','temporal_control','core','exchange'):
                visit(getattr(owner,name,None),label+'.'+name,depth+1)
            for i,effect in enumerate(getattr(owner,'effects',[])):
                visit(effect,label+'.effects.'+str(i),depth+1)
        visit(chain,'chain')
        return rows
    def adapter():
        return subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
    settings=json.loads(args.settings.read_text('utf-8'))
    result=dict(baseline=args.full_resources,legacy_resources=str(args.legacy_resources or ''),
        preview=args.preview,rows=[],adapter_start_mib=adapter(),
        timing_scope='CPU wall around completed GPU stages; sampling outside timed scope',
        memory_scope='DXGI LOCAL in-process frame-end samples, not global DedicatedUsage')
    live,flow,preview=None,None,None
    try:
        if args.preview:
            settings['temporal_controls']={'mode':'off','scene_detection':True,'history_weight':.6}
            preview=GpuPreview()
            for index in range(args.frames):
                _,_,diag=preview.render(args.video,settings,index,display_edge=640,need_source=True)
                row=preview.executor.submit(lambda:dict(frame=index,diagnostics=diag,
                    memory=sample(preview.live.d3d12_device()),resources=resources(preview.chain))).result()
                result['rows'].append(row)
                print('preview',index,diag['processing_ms'],row['memory']['local_usage'],flush=True)
            preview.close()
            result['after_close']=preview.executor.submit(lambda:sample(None)).result()
        else:
            settings['_frame_range']={'start_frame':0,'end_frame':args.frames}
            source=htx.gtx.probe_source(args.video)
            size=native_effects.processing_size(settings,(source['width'],source['height']))
            live=native_effects.make_live(*size,settings,wide=True)
            flow=native_effects.make_flow_provider(settings,live)
            result['after_model_open']=sample(live.d3d12_device())
            original_process=htx.HdrChain.process
            def process(chain,*items,**kwargs):
                mark=time.perf_counter(); outcome=original_process(chain,*items,**kwargs)
                ms=(time.perf_counter()-mark)*1000
                result['rows'].append(dict(frame=len(result['rows']),process_wall_ms=ms,
                    stages=outcome['stages'],memory=sample(live.d3d12_device()),resources=resources(chain)))
                print('export',len(result['rows']),ms,result['rows'][-1]['memory']['local_usage'],flush=True)
                return outcome
            htx.HdrChain.process=process
            original_export=htx.export_gpu_hdr_transfer
            def export(**kwargs):
                value,report=original_export(**kwargs); result['report']=report
                return value,report
            htx.export_gpu_hdr_transfer=export
            tx=json.loads(args.transcode.read_text('utf-8')) if args.transcode else {}
            tx.update(audio_mode='remove',subtitle_mode='remove',copy_metadata=False,copy_chapters=False,
                      copy_attachments=False,resolution_mode='original')
            pipeline.export_gpu_transfer_stream(video=args.video,live=live,frames=args.frames,
                fps=120,cq=24,with_audio=False,out_path=str(args.output/'output.mkv'),
                transcode_settings=tx,job_settings=settings,flow_provider=flow)
            # The export pipeline consumes/closes the passed live owner.
            result['after_chain_close']=sample(None)
        result['passed']=True
    except Exception:
        result['error']=traceback.format_exc();raise
    finally:
        if preview: preview.shutdown()
        if live: live.close()
        if flow: flow.close_independent_layer()
        result['adapter_end_mib']=adapter()
        (args.output/'result.json').write_text(json.dumps(result,indent=2,default=str),encoding='utf-8')


if __name__ == '__main__': main()
