"""Continuous, bounded file decode for the comparison display (no AI pipeline)."""
from collections import deque
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
        self.media = media
        self.stopped = threading.Event()
        self.frames = queue.Queue(maxsize=2)
        self.timestamps = queue.Queue(maxsize=16)
        self.errors = deque(maxlen=8)
        self.previous = self.following = None
        self.eof = False
        meta = media.metadata
        self.width = min(int(meta['width']), max_width)
        self.height = max(1, round(int(meta['height']) * self.width / int(meta['width'])))
        self.hdr = bool(meta['is_hdr'])
        pixel_format = 'gbrpf32le' if self.hdr else 'bgr24'
        colour = service._zscale_filter(meta)
        if not self.hdr:
            colour = colour.replace('format=gbrpf32le', 'format=gbrp,format=bgr24')
        filters = f'scale={self.width}:{self.height}:flags=bicubic,' + colour + ',showinfo'
        source_fps = float(meta.get('frame_rate') or 30)
        self.display_fps = source_fps / max(1, math.ceil(source_fps / 60))
        if source_fps > 60:
            # Only the continuous display is capped. Select retains actual PTS;
            # frame stepping still sees every source frame through get_timeline.
            filters = f"select='not(mod(n,{math.ceil(source_fps / 60)}))'," + filters
        self.frame_bytes = self.width * self.height * (12 if self.hdr else 3)
        command = [service.ffmpeg, '-hide_banner', '-v', 'info', '-nostdin',
            '-hwaccel', 'auto', '-threads', '4', '-filter_threads', '1', '-copyts',
            '-ss', f"{at + media.origin_pts - float(meta['start_time']):.9f}", '-i', media.path,
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
            elif 'error' in line.lower() or 'invalid' in line.lower():
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
        return self._jpeg, pts - self.media.origin_pts, pts

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
        frames = [decoder.at(at) for decoder in self.decoders]
        header = json.dumps({'time': at, 'original': {'size': len(frames[0][0]),
            'actualTime': frames[0][1], 'actualPts': frames[0][2]},
            'exported': {'size': len(frames[1][0]), 'actualTime': frames[1][1],
            'actualPts': frames[1][2]}}).encode('utf-8')
        return struct.pack('!I', len(header)) + header + frames[0][0] + frames[1][0]

    def close(self):
        for decoder in self.decoders:
            decoder.close()
