"""One native render in flight, latest command wins, display-ack backpressure.

Cancellation of an asyncio wrapper cannot cancel CUDA/NGX. Never start a
replacement native request until the previous one has actually returned.
"""
import asyncio
import base64
import json
import time

_render_gates = {}


def render_gate():
    loop = asyncio.get_running_loop()
    # A server has one loop; don't retain gates for closed test/server loops.
    for old in list(_render_gates):
        if old.is_closed():
            del _render_gates[old]
    return _render_gates.setdefault(loop, asyncio.Lock())


class PreviewPlayer:
    def __init__(self, socket, service):
        self.socket, self.service = socket, service
        self.wake = asyncio.Event()
        self.closed = self.playing = self.dirty = False
        self.revision = self.sequence = 0
        self.request_id = None
        self.options = dict(frame=0, view='dlss', split_x=.5, zoom=1.,
                            pan_x=.5, pan_y=.5, display_edge=0)
        self.binary = self.ack_enabled = False
        self.awaiting = None
        self.acks = {}
        self.loop = True
        self.anchor_time = 0.
        self.anchor_frame = 0
        self.play_position = 0
        self.last_sent = None
        self.dropped = self.obsolete = 0

    def command(self, cmd):
        action = cmd.get('action')
        if action == 'presented':
            if cmd.get('sequence') == self.awaiting:
                self.acks = {k: float(cmd[k]) for k in ('decode_ms', 'paint_wait_ms',
                    'receive_to_present_ms', 'presentation_interval_ms')
                    if isinstance(cmd.get(k), (int, float)) and 0 <= cmd[k] <= 60000}
                self.awaiting = None
                self.wake.set()
            return
        if action not in ('play', 'pause', 'seek', 'set_view'):
            return
        self.revision += 1
        self.request_id = cmd.get('request_id')
        self.awaiting = None
        for key in self.options:
            if key in cmd:
                self.options[key] = cmd[key]
        self.options['frame'] = max(0, int(self.options['frame']))
        self.options['display_edge'] = max(0, min(3840, int(self.options['display_edge'])))
        if cmd.get('transport') == 'binary':
            self.binary = True
            self.ack_enabled = bool(cmd.get('ack'))
        self.loop = bool(cmd.get('loop', self.loop))
        if action == 'play':
            self.playing = True
            self.anchor_time, self.anchor_frame = time.perf_counter(), self.options['frame']
            self.play_position = self.anchor_frame
        elif action in ('seek', 'pause'):
            self.playing = False
        elif self.playing and 'frame' in cmd:
            self.anchor_time, self.anchor_frame = time.perf_counter(), self.options['frame']
            self.play_position = self.anchor_frame
        self.dirty = action != 'pause'
        self.wake.set()

    async def _send(self, body):
        await self.socket.send_text(json.dumps(body, separators=(',', ':')))

    async def run(self):
        while not self.closed:
            if not self.dirty and not self.playing or self.awaiting is not None:
                self.wake.clear()
                # A slow renderer must not build a websocket image queue. An
                # unresponsive client eventually stops playback visibly.
                try:
                    await asyncio.wait_for(self.wake.wait(), 5 if self.awaiting is not None else None)
                except asyncio.TimeoutError:
                    self.playing = False
                    self.awaiting = None
                    await self._send(dict(type='error', request_id=self.request_id,
                        message='预览画面确认超时，播放已暂停'))
                continue
            info = self.service.video_info
            if not info:
                self.dirty = self.playing = False
                continue
            fps = float(info.get('fps', 30) or 30)
            total = max(1, int(info.get('frames', 1)))
            step = max(1, round(fps / min(fps, 60)))
            opts = dict(self.options)
            frame = min(total-1, int(opts.pop('frame')))
            revision = self.revision
            service_revision = getattr(self.service, '_preview_revision', 0)
            source = self.service.video_path
            request_id = self.request_id
            self.dirty = False
            started = time.perf_counter()
            async with render_gate():
                if revision != self.revision or self.closed:
                    continue
                future = asyncio.create_task(asyncio.to_thread(self.service.render_preview,
                    frame_idx=frame, view=opts['view'], split_x=float(opts['split_x']),
                    zoom=float(opts['zoom']), pan=(float(opts['pan_x']), float(opts['pan_y'])),
                    preview_mode=self.playing, display_edge=opts['display_edge']))
                try:
                    data, message = await asyncio.shield(future)
                except asyncio.CancelledError:
                    # Keep the global slot occupied through native completion.
                    await future
                    raise
                except Exception as error:
                    data, message = None, str(error)
                diagnostics = self.service.preview_diagnostics()
            if (self.closed or revision != self.revision or source != self.service.video_path or
                    service_revision != getattr(self.service, '_preview_revision', 0)):
                self.obsolete += 1
                if not self.closed and revision == self.revision:
                    self.dirty = True
                continue
            if not data:
                self.playing = False
                await self._send(dict(type='error', frame=frame, request_id=request_id, message=message))
                continue
            self.sequence += 1
            if self.last_sent is not None and frame > self.last_sent:
                self.dropped += max(0, frame-self.last_sent-step)
            self.last_sent = frame
            prep = time.perf_counter()
            body = dict(type='frame', sequence=self.sequence, request_id=request_id,
                frame=frame, total=total, source=source, preview=dict(diagnostics,
                    render_request_ms=(prep-started)*1000, stale_results_discarded=self.obsolete,
                    dropped_frames=self.dropped, client_presentation=self.acks,
                    playback_strategy='latest_completed_frame', status='playing' if self.playing else 'still'))
            if self.binary:
                body.update(encoding='jpeg_binary', jpeg_bytes=len(data))
            else:
                body['image'] = 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')
            body['preview']['wire_prepare_ms'] = (time.perf_counter()-prep)*1000
            send_start = time.perf_counter()
            if self.ack_enabled:
                self.awaiting = self.sequence
            # Single writer preserves the header/JPEG pair order.
            await self._send(body)
            if self.binary:
                await self.socket.send_bytes(data)
            send_ms = (time.perf_counter()-send_start)*1000
            await self._send(dict(type='frame_timing', sequence=self.sequence,
                                 ws_send_ms=send_ms, request_id=request_id))
            if self.playing and revision == self.revision:
                wall = self.anchor_frame + int((time.perf_counter()-self.anchor_time)*fps)
                if not self.loop and frame == total-1:
                    self.playing = False
                    await self._send(dict(type='ended', frame=frame, total=total, request_id=request_id))
                else:
                    # Keep the clock position unwrapped. Comparing a wrapped
                    # frame index with elapsed absolute frames repeats images
                    # after each loop, especially with fast display ACKs.
                    next_frame = max(self.play_position+step, wall)
                    self.play_position = next_frame
                    self.options['frame'] = next_frame % total if self.loop else min(total-1, next_frame)
                deadline = started + 1/min(fps, 60)
                while not self.closed and revision == self.revision:
                    remaining = deadline-time.perf_counter()
                    if remaining <= 0:
                        break
                    self.wake.clear()
                    try:
                        await asyncio.wait_for(self.wake.wait(), remaining)
                    except asyncio.TimeoutError:
                        break
