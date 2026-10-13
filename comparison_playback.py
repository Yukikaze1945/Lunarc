"""Continuous, bounded file decode for the comparison display (no AI pipeline)."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import queue
import re
import struct
import subprocess
import threading
import time

import cv2
import numpy as np

from comparison_media import MediaComparisonError, linear_rgb_to_display_bgr


class DisplayDecoder:
    def __init__(self, service, media, at, max_width=1280):
        self.media, self.service, self.start_time = media, service, at
        self.backend = 'software'
        self.fallback_reason = None
        self.decoded_frames = 0
        self.jpeg_ms = self.jpeg_calls = 0
        self.stopped = threading.Event()
        self.frames = queue.Queue(maxsize=2)
        self.timestamps = queue.Queue(maxsize=16)
        self.errors = deque(maxlen=8)
        self.previous = self.following = None
        self.eof = False
        meta = media.metadata
        self.width = min(int(meta['width']), max_width)
        self.height = max(1, round(int(meta['height']) * self.width / int(meta['width'])))
        # CUDA 4:2:0 resampling needs even display dimensions.
        self.width = max(2, self.width - self.width % 2)
        self.height = max(2, self.height - self.height % 2)
        self.hdr = bool(meta['is_hdr'])
        codec, depth = meta.get('codec'), int(meta.get('bit_depth') or 8)
        supported = (codec in ('h264','hevc','av1','vp9','mpeg2video','vc1')
                     and meta.get('pixel_format') in ('yuv420p','yuv420p10le','nv12','p010le')
                     and not (codec == 'h264' and depth > 8))
        if not supported:
            self.fallback_reason = '此编码或像素格式不适用当前 NVIDIA 解码路径'
        self._start_decode(supported)

    def _start_decode(self, cuda):
        meta, service, media = self.media.metadata, self.service, self.media
        self.backend = 'nvdec' if cuda else 'software'
        pixel_format = 'gbrpf32le' if self.hdr else 'bgr24'
        colour = service._zscale_filter(meta)
        if not self.hdr:
            colour = colour.replace('format=gbrpf32le', 'format=gbrp,format=bgr24')
        if cuda:
            download_format = 'p010le' if int(meta.get('bit_depth') or 8) > 8 else 'nv12'
            resize = (f'scale_cuda={self.width}:{self.height}:interp_algo=bicubic:'
                      f'format={download_format}:passthrough=0,hwdownload,format={download_format},')
        else:
            resize = f'scale={self.width}:{self.height}:flags=bicubic,'
        filters = resize + colour + ',showinfo'
        source_fps = float(meta.get('frame_rate') or 30)
        self.display_fps = source_fps / max(1, math.ceil(source_fps / 60))
        if source_fps > 60:
            # Only the continuous display is capped. Select retains actual PTS;
            # frame stepping still sees every source frame through get_timeline.
            filters = f"select='not(mod(n,{math.ceil(source_fps / 60)}))'," + filters
        self.frame_bytes = self.width * self.height * (12 if self.hdr else 3)
        acceleration = ['-hwaccel','cuda','-hwaccel_output_format','cuda'] if cuda else []
        command = [service.ffmpeg, '-hide_banner', '-v', 'info', '-nostdin',
            *acceleration, '-threads', '4', '-filter_threads', '1', '-copyts',
            '-ss', f"{self.start_time + media.origin_pts - float(meta['start_time']):.9f}", '-i', media.path,
            '-map', '0:v:0', '-an', '-sn', '-vf', filters,
            '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', pixel_format, 'pipe:1']
        self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.threads = [threading.Thread(target=self._stderr, daemon=True),
                        threading.Thread(target=self._stdout, daemon=True)]
        for thread in self.threads:
            thread.start()

    def _put(self, target, value):
        while not self.stopped.is_set():
            try:
                target.put(value, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def _stderr(self):
        for raw in iter(self.process.stderr.readline, b''):
            if self.stopped.is_set():
                break
            line = raw.decode('utf-8', 'replace')
            if 'showinfo' in line:
                match = re.search(r'\bn:\s*\d+.*?pts_time:\s*([-+\d.eE]+)', line)
                if match:
                    if not self._put(self.timestamps, float(match.group(1))):
                        break
            elif any(word in line.lower() for word in ('error','invalid','failed','cannot','impossible')):
                self.errors.append(line.strip())

    def _stdout(self):
        try:
            while not self.stopped.is_set():
                chunks, remaining = [], self.frame_bytes
                while remaining and not self.stopped.is_set():
                    chunk = self.process.stdout.read(remaining)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if remaining:
                    break
                deadline = time.monotonic() + 15
                while not self.stopped.is_set():
                    try:
                        pts = self.timestamps.get(timeout=.1)
                        break
                    except queue.Empty:
                        if time.monotonic() >= deadline:
                            raise MediaComparisonError('连续解码未返回帧时间戳')
                else:
                    break
                self.decoded_frames += 1
                if not self._put(self.frames, (pts, b''.join(chunks))):
                    break
        except Exception as error:
            self._put(self.frames, error)
        finally:
            self._put(self.frames, None)

    def _read(self):
        deadline = time.monotonic() + 15
        while not self.stopped.is_set():
            try:
                value = self.frames.get(timeout=.1)
                if isinstance(value, Exception):
                    raise value
                if value is None:
                    self.eof = True
                    code = self.process.wait(timeout=2)
                    if code:
                        if self.backend == 'nvdec' and self.decoded_frames == 0:
                            # One bounded startup fallback, surfaced in diagnostics.
                            # Never switch decoding underneath already displayed PTS.
                            self.fallback_reason = '\n'.join(self.errors) or 'NVIDIA 解码启动失败'
                            self.close()
                            self.stopped = threading.Event()
                            self.frames = queue.Queue(maxsize=2)
                            self.timestamps = queue.Queue(maxsize=16)
                            self.errors.clear(); self.eof = False
                            self._start_decode(False)
                            return self._read()
                        raise MediaComparisonError('连续视频解码失败：' + '\n'.join(self.errors))
                return value
            except queue.Empty:
                if time.monotonic() >= deadline:
                    raise MediaComparisonError('连续视频解码超时')
        return None

    def at(self, relative_time):
        if self.following is None and not self.eof:
            self.following = self._read()
        target = relative_time + self.media.origin_pts
        while self.following and self.following[0] <= target:
            self.previous, self.following = self.following, self._read()
        candidates = [value for value in (self.previous, self.following) if value]
        if not candidates:
            raise MediaComparisonError('视频未返回可播放画面')
        selected = min(candidates, key=lambda value: abs(value[0] - target))
        pts, payload = selected
        # Repeated source frames (e.g. 24 fps compared with 48 fps) reuse JPEG.
        if getattr(self, '_jpeg_pts', None) != pts:
            mark = time.perf_counter()
            if self.hdr:
                planes = np.frombuffer(payload, '<f4').reshape(3, self.height, self.width)
                rgb = np.stack((planes[2], planes[0], planes[1]), axis=-1)
                bgr = linear_rgb_to_display_bgr(rgb, hdr=True)
            else:
                bgr = np.frombuffer(payload, np.uint8).reshape(self.height, self.width, 3)
            ok, encoded = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise MediaComparisonError('无法编码连续显示帧')
            self._jpeg_pts, self._jpeg = pts, encoded.tobytes()
            self.jpeg_ms += (time.perf_counter()-mark)*1000
            self.jpeg_calls += 1
        return self._jpeg, pts - self.media.origin_pts, pts

    def diagnostics(self):
        return dict(decoder=self.backend, gpu_scale=self.backend=='nvdec',
                    fallback_reason=self.fallback_reason, display_size=[self.width,self.height],
                    display_fps=self.display_fps, decoded_frames=self.decoded_frames,
                    jpeg_calls=self.jpeg_calls, jpeg_mean_ms=self.jpeg_ms/max(1,self.jpeg_calls),
                    display_mapping='HDR to SDR' if self.hdr else 'SDR')

    def close(self):
        self.stopped.set()
        if self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for thread in self.threads:
            thread.join(timeout=1)
        self.process.stdout.close()
        self.process.stderr.close()


class ComparisonPlayback:
    def __init__(self, service, session, at, rate):
        self.medias = service._pair(session)
        self.duration = max(float(media.metadata['duration_seconds']) for media in self.medias)
        if not math.isfinite(at) or not 0 <= at <= self.duration:
            raise MediaComparisonError('播放起点超出视频范围')
        if not math.isfinite(rate) or not .25 <= rate <= 2:
            raise MediaComparisonError('播放速度必须为 0.25–2')
        self.rate = rate
        self.fps = min(60., max(float(media.metadata.get('frame_rate') or 30) for media in self.medias))
        self.decoders = []
        self.workers = ThreadPoolExecutor(max_workers=2,thread_name_prefix='ComparisonDecode')
        try:
            for media in self.medias:
                # Keep the shorter side's last actual frame at its EOF.
                start = min(at, max(0., float(media.metadata['duration_seconds']) -
                    1 / float(media.metadata.get('frame_rate') or 30)))
                self.decoders.append(DisplayDecoder(service, media, start))
            self.fps = max(decoder.display_fps for decoder in self.decoders)
        except Exception:
            self.close()
            raise

    def packet(self, at):
        pending = [self.workers.submit(decoder.at,at) for decoder in self.decoders]
        frames = [result.result() for result in pending]
        header = json.dumps({'time': at, 'original': {'size': len(frames[0][0]),
            'actualTime': frames[0][1], 'actualPts': frames[0][2]},
            'exported': {'size': len(frames[1][0]), 'actualTime': frames[1][1],
            'actualPts': frames[1][2]}, 'playback': self.diagnostics()}).encode('utf-8')
        return struct.pack('!I', len(header)) + header + frames[0][0] + frames[1][0]

    def close(self):
        for decoder in self.decoders:
            decoder.close()
        self.workers.shutdown(wait=True,cancel_futures=True)

    def diagnostics(self):
        return dict(original=self.decoders[0].diagnostics(),exported=self.decoders[1].diagnostics(),
                    display_fps=self.fps)
