"""Independent NR sessions in series: each layer consumes the previous output.

No data-consistency solve runs here. The existing multiscale solver remains an
explicit alternative; sequential reconstruction uses the existing colour bridge.
"""
import ctypes as C
import time
from preview_layer_cache import evaluate


def process_sdr(chain, frame, provider, index, discontinuity):
    import gpu_transfer_export as gtx
    from native_effects import Exchange
    out = gtx.Chain.process(chain, frame, provider, index, discontinuity)
    frame = out.get('_processing_frame', frame)
    reset = bool(out.get('_is_discontinuous', discontinuity) or index == 0)
    candidate = out['_output_candidate']
    reused = [out['_dlss_reused']]
    times = [out['stages'].get('dlss_inference', 0.)]
    if not reused[0]:
        chain.layer_calls[0] += 1; chain.layer_reset_flags[0].append(reset)
    if not getattr(chain, 'sequential_exchange', None):
        chain.sequential_exchange = Exchange(candidate['d3d12_device'],frame.cuda_context,
                                             (frame.width,frame.height))
    for number, layer in enumerate(chain.extra_layers, 1):
        source = chain.sequential_exchange.color(candidate)
        ptr, pitch = C.c_uint64(source['device_ptr']), C.c_uint32(source['pitch_bytes'])
        if layer['lanczos']:
            code = chain.b['lanczos'].gpu_lanczos_process(layer['lanczos'],ptr.value,
                C.c_size_t(pitch.value),C.byref(ptr),C.byref(pitch))
            if code: gtx._bridge_error(chain.b['lanczos'],'gpu_lanczos_last_error',code)
        working = dict(format='rgba8_cuda',width=layer['dims'][0],height=layer['dims'][1],
                       cuda_context=frame.cuda_context,device_ptr=ptr.value,pitch_bytes=pitch.value)
        def infer():
            if chain.uses_flow:
                flow = layer['flow_provider'].get_flow_device_rgba(index,working,frame.cuda_context,
                                                                  is_discontinuous=reset)
                mark = time.perf_counter()
                result = layer['live'].process_device_color_flow_resource(working,flow,reset=reset)
            else:
                mark = time.perf_counter()
                result = layer['live'].process_device_color_zero_motion_resource(working,reset=reset)
            return result,(time.perf_counter()-mark)*1000
        (dlss, ms), cached = evaluate(chain,number,infer)
        reused.append(cached); times.append(0. if cached else ms)
        if not cached:
            chain.layer_calls[number] += 1; chain.layer_reset_flags[number].append(reset)
        if tuple(layer['dims']) == (frame.width,frame.height):
            candidate = dict(dlss)
        else:
            lib = chain.b['residual']
            if not layer.get('sequential_residual'):
                handle = C.c_void_p()
                code = lib.gpu_residual_open_scaled(dlss['d3d12_device'],frame.width,frame.height,
                    *layer['dims'],C.byref(layer['tap_plans'].upscale),C.byref(handle))
                if code: gtx._bridge_error(lib,'gpu_residual_last_error',code)
                layer['sequential_residual'] = handle
            texture = C.c_void_p()
            code = lib.gpu_residual_process(layer['sequential_residual'],dlss['d3d12_resource'],
                dlss['d3d12_fence'],dlss['fence_value'],frame.cuda_context,source['device_ptr'],
                C.c_size_t(source['pitch_bytes']),frame.cuda_context,ptr.value,C.c_size_t(pitch.value),
                C.c_float(chain.strength),C.byref(texture))
            if code: gtx._bridge_error(lib,'gpu_residual_last_error',code)
            candidate = dict(d3d12_resource=texture.value,d3d12_device=dlss['d3d12_device'],
                             d3d12_fence=None,fence_value=0,producer_completed=True)
    chain.layer_inference_ms = times
    out['stages']['nr_total_ms'] = sum(times)
    out['layer_inference_reused'] = reused
    controlled = chain._post_process(frame,*out['_full_color'],candidate)
    chain._defer_preview_consumer = False
    result = chain._submit_output(frame,controlled,out['stages'])
    if chain.preview_consumer is not None: out['preview_output'] = result
    out['_output_candidate'] = controlled
    out['sequential_input_provenance'] = ['source']+['previous_layer_output']*len(chain.extra_layers)
    return out


def process_wide(chain, frame, provider, index, discontinuity):
    import gpu_hdr_transfer_export as htx
    first = chain.first
    if first:
        chain._open_stages(frame); chain.first = False
    else:
        chain._assert_metadata_stable()
    if (int(frame.width),int(frame.height)) != chain.src:
        raise htx.HdrExportError('Sequential NR source dimensions changed')
    colour = {key:int(getattr(frame,key)) for key in
              ('color_range','color_primaries','color_transfer','color_space')}
    if chain.sdr10:
        from source_colour import resolve_frame_colour
        colour = resolve_frame_colour(colour,chain.policy.get('source_colour_resolution'))
    stages = {}
    chain._frame_reset = bool(first or discontinuity)
    master,_,proxy,guidance,source_report,proxy_report = chain._primary_colour(frame,colour,stages,first or discontinuity)
    discontinuity = (chain._prefix_discontinuous if chain.input_effect is not None
                     else chain._detect_scene(master,first,discontinuity))
    reset = bool(first or discontinuity)
    chain._frame_reset = reset
    current = master
    records, reused, times = [], [], []
    owners = [dict(live=chain.live,bridge=chain.bridge,flow_provider=provider,dims=chain.working,
                   scale=chain.multi['scales'][0])] + chain.extra_layers
    for number, layer in enumerate(owners):
        bridge = layer['bridge']
        if number:
            layer_master = bridge.import_master(current)
            bridge.downscale()
            proxy,guidance = bridge.proxy()
            proxy_report = bridge.proxy_report()
        else:
            layer_master = master
        timing = {}
        def infer():
            return chain._guidance_and_inference(layer['live'],layer['flow_provider'],proxy,guidance,
                frame,index,reset,discontinuity,timing,counted=chain.layer_motion_calls[number])
        (dlss,flow), cached = evaluate(chain,number,infer)
        reused.append(cached); times.append(float(timing.get('dlss_inference') or 0))
        if not cached:
            chain.layer_calls[number] += 1; chain.layer_reset_flags[number].append(reset)
        mark = time.perf_counter()
        current = bridge.reconstruct(dlss,residual_strength=chain.strength,
            **({'preserve_source_detail':chain.policy.get('preserve_source_detail',True)} if chain.sdr10 else {}))
        timing['sequential_reconstruct'] = (time.perf_counter()-mark)*1000
        records.append(dict(layer=number,session_id=layer['live'].session_id,working_dims=list(layer['dims']),
                            input='previous_layer_output' if number else 'source',
                            parameters=__import__('layer_settings').layer_parameters(layer['live'].settings),
                            reused=cached,stages=timing))
    chain.layer_inference_ms = times
    stages['nr_total_ms'] = sum(times)
    composited = chain._post_process(master,current,stages,frame)
    planes, encode_report = chain._encode(frame,stages,composited)
    preview = chain._consume_preview_output(frame,composited)
    chain.frame_reports.append(dict(frame_idx=index,pts=int(frame.pts),colour_fields_sent=colour,
        source_report=source_report,proxy_report=proxy_report,encode_report=encode_report,
        reconstruction_mode=chain.multi['mode'],fusion_mode_actual='sequential_independent_nr_sessions',
        layers=records,reset=reset,solver=None))
    result = dict(stages=stages,flow=None,layer_inference_reused=reused,nvof_timing=None)
    if chain.preview_consumer is not None:
        result.update(preview_output=preview,_preview_master=chain._preview_source)
    return result
