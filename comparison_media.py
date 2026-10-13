"""Read-only media comparison service for the original and exported files.

The service never invokes the enhancement pipeline.  Each frame is selected from
the file's local PTS neighborhood with ffprobe, then decoded by CPU FFmpeg.  The
HTTP layer can return ``get_frame(...)["jpeg"]`` and forward the remaining scalar
fields as response headers.  Paths are passed as individual subprocess arguments;
no shell is involved.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import subprocess
import threading
import uuid

import cv2
import numpy as np


class MediaComparisonError(ValueError):
    """An input media file or comparison request cannot be served."""


class MediaComparisonEOF(MediaComparisonError):
    """The requested shared timestamp lies beyond one side's media."""


@dataclass
class _Media:
    path: str
    metadata: dict
    origin_pts: float


def linear_rgb_to_display_bgr(rgb: np.ndarray, hdr: bool = False) -> np.ndarray:
    """Fixed display-only Rec.709 OETF mapping used for both comparison sides.

    HDR frames are linear Rec.709 relative to an 80-nit reference white.  The
    fixed luminance mapping matches the project's existing uncalibrated HDR
    preview rule; it is not an HDR monitor output or a color-accuracy claim.
    """
    values = np.asarray(rgb, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError("linear RGB image must have shape HxWx3")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    if hdr:
        from hdr_display_mapping import hdr_to_sdr_linear
        values = hdr_to_sdr_linear(values)
    values = np.clip(values, 0.0, 1.0)
    encoded = np.where(values < .018, 4.5 * values,
                       1.099 * np.power(values, .45) - .099)
    rgb8 = np.rint(np.clip(encoded, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.ascontiguousarray(rgb8[..., ::-1])


def _rate(value) -> float | None:
    try:
        numerator, denominator = str(value).split("/", 1)
        result = float(numerator) / float(denominator)
        return result if math.isfinite(result) and result > 0 else None
    except (ValueError, TypeError, ZeroDivisionError):
        return None


def _finite(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


class MediaComparisonService:
    """Manage short-lived local file pairs and PTS-aligned display frames."""

    def __init__(self, ffprobe: str | None = None, ffmpeg: str | None = None,
                 runner=None, max_sessions: int = 8):
        if ffprobe is None or ffmpeg is None:
            try:
                import pipeline
                ffprobe = ffprobe or pipeline._bundle_ffprobe()
                ffmpeg = ffmpeg or pipeline._bundle_ffmpeg()
            except Exception as exc:
                raise RuntimeError(f"Bundled FFmpeg tools are unavailable: {exc}") from exc
        if not ffprobe or not ffmpeg:
            raise RuntimeError("ffprobe and ffmpeg are required for video comparison")
        self.ffprobe = str(ffprobe)
        self.ffmpeg = str(ffmpeg)
        self._run = runner or subprocess.run
        self.max_sessions = max(1, int(max_sessions))
        self._sessions: OrderedDict[str, tuple[_Media, _Media]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _local_path(value) -> str:
        if not isinstance(value, (str, os.PathLike)):
            raise MediaComparisonError("媒体路径必须是本机文件路径")
        raw = os.fspath(value).strip()
        if not raw or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", raw):
            raise MediaComparisonError("仅支持显式选择的本机媒体文件")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise MediaComparisonError("媒体路径必须是绝对路径")
        try:
            resolved = str(path.resolve(strict=True))
        except (OSError, RuntimeError) as exc:
            raise MediaComparisonError(f"媒体文件不存在：{raw}") from exc
        if not os.path.isfile(resolved):
            raise MediaComparisonError(f"媒体文件不是普通文件：{raw}")
        return resolved

    def _command(self, args, timeout: float, binary: bool = False):
        try:
            result = self._run(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, shell=False, check=False,
                               timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise MediaComparisonError("媒体工具处理超时") from exc
        except OSError as exc:
            raise MediaComparisonError(f"无法启动媒体工具：{exc}") from exc
        if result.returncode:
            stderr = result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else str(result.stderr or "")
            detail = stderr.strip()[-1200:]
            raise MediaComparisonError("媒体工具无法读取文件" + (f"：{detail}" if detail else ""))
        if binary:
            return result.stdout
        try:
            payload = result.stdout.decode("utf-8", "replace") if isinstance(result.stdout, bytes) else result.stdout
            return __import__("json").loads(payload)
        except Exception as exc:
            raise MediaComparisonError("媒体工具返回了无效的 JSON") from exc

    def _probe_metadata(self, path: str) -> dict:
        data = self._command([self.ffprobe, "-v", "error", "-show_streams",
            "-show_format", "-of", "json", path], timeout=20)
        streams = data.get("streams") or []
        video = next((s for s in streams if s.get("codec_type") == "video"), None)
        if not video:
            raise MediaComparisonError(f"没有可解码的视频流：{os.path.basename(path)}")
        width, height = int(video.get("width") or 0), int(video.get("height") or 0)
        if width <= 0 or height <= 0 or width * height > 64_000_000:
            raise MediaComparisonError("视频分辨率缺失或超出支持范围")
        fmt = data.get("format") or {}
        duration = _finite(video.get("duration")) or _finite(fmt.get("duration"))
        if duration is None or duration <= 0:
            raise MediaComparisonError("无法确定视频时长，不能建立同步时间轴")
        fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
        pixel_format = str(video.get("pix_fmt") or "unknown")
        try:
            bits = int(video.get("bits_per_raw_sample") or 0)
        except (TypeError, ValueError):
            bits = 0
        if bits <= 0:
            match = re.search(r"(?:p|yuv\d+p)(9|10|12|14|16)(?:le|be)?", pixel_format)
            bits = int(match.group(1)) if match else (8 if pixel_format != "unknown" else 0)
        transfer = str(video.get("color_transfer") or "")
        primaries = str(video.get("color_primaries") or "")
        matrix = str(video.get("color_space") or "")
        is_hdr = transfer in ("smpte2084", "arib-std-b67")
        audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
        stream_start = _finite(video.get("start_time"))
        format_start = _finite(fmt.get("start_time"))
        fallback_transfer = transfer if transfer in (
            "bt709", "smpte170m", "bt2020-10", "bt2020-12",
            "smpte2084", "arib-std-b67", "linear") else "bt709"
        fallback_primaries = primaries if primaries in (
            "bt709", "bt2020", "smpte170m", "smpte432", "bt470bg",
            "bt470m", "smpte431", "film", "jedec-p22") else (
                "bt2020" if is_hdr else "bt709")
        fallback_matrix = matrix if matrix in (
            "bt709", "bt2020nc", "bt2020ncl", "bt2020c", "bt2020cl",
            "smpte170m", "smpte240m", "gbr", "fcc", "bt470bg") else (
                "bt2020nc" if is_hdr else "bt709")
        source_range = video.get("color_range")
        decode_range = source_range if source_range in ("tv", "pc") else "tv"
        return {
            "path": path, "name": os.path.basename(path),
            "width": width, "height": height,
            "codec": video.get("codec_name"), "profile": video.get("profile"),
            "pixel_format": pixel_format, "bit_depth": bits,
            "frame_rate": fps, "time_base": video.get("time_base"),
            "duration_seconds": duration,
            "start_time": stream_start if stream_start is not None else (format_start or 0.0),
            "color_range": video.get("color_range") or "unknown",
            "color_matrix": matrix or "unknown", "color_transfer": transfer or "unknown",
            "color_primaries": primaries or "unknown", "is_hdr": is_hdr,
            "display_assumption": ("HDR linear Rec.709, fixed display tone map (uncalibrated)"
                if is_hdr else ("BT.709 transfer/primaries assumed where tags are missing"
                    if not transfer or not primaries else "SDR converted to linear Rec.709 for display")),
            "decode_transfer": fallback_transfer,
            "decode_primaries": fallback_primaries,
            "decode_matrix": fallback_matrix,
            "decode_range": decode_range,
            "format": fmt.get("format_name"),
            "file_size_bytes": int(_finite(fmt.get("size")) or 0),
            "audio": ({"index": int(audio.get("index", -1)),
                       "codec": audio.get("codec_name"),
                       "channels": audio.get("channels"),
                       "sample_rate": audio.get("sample_rate")} if audio else None),
        }

    def _frame_window(self, media: _Media, target_pts: float) -> list[dict]:
        fps = media.metadata.get("frame_rate") or 30.0
        margin = max(.35, min(2.0, 4.0 / fps))
        start = max(float(media.metadata["start_time"]), target_pts - margin)
        duration = max(.5, min(2.0, 8.0 / fps))
        # An interval duration starts at the seeked keyframe, which may be far
        # before `start`. Use an absolute end so long GOP/VFR sources include the
        # requested neighborhood instead of returning twelve unrelated old frames.
        intervals = [f"{start:.9f}%{target_pts + duration:.9f}"]
        last_error = None
        for interval in intervals:
            result = self._command([self.ffprobe, "-v", "error", "-select_streams", "v:0",
                "-read_intervals", interval, "-show_frames", "-show_entries",
                "frame=best_effort_timestamp_time,pts_time,pkt_duration_time,key_frame",
                "-of", "json", media.path], timeout=15)
            rows = []
            for frame in result.get("frames") or []:
                pts = _finite(frame.get("best_effort_timestamp_time"))
                if pts is None:
                    pts = _finite(frame.get("pts_time"))
                if pts is None:
                    continue
                rows.append({"pts": pts,
                    "duration": _finite(frame.get("pkt_duration_time")),
                    "key_frame": bool(int(frame.get("key_frame") or 0))})
            if rows:
                return sorted({row["pts"]: row for row in rows}.values(), key=lambda row: row["pts"])
            last_error = "ffprobe did not find frames in the requested local time interval"
        raise MediaComparisonEOF(last_error or "视频在请求时间没有可用帧")

    @staticmethod
    def _select_frame(media: _Media, relative_time: float, frames: list[dict]) -> dict:
        duration = float(media.metadata["duration_seconds"])
        if relative_time < 0 or relative_time > duration + 1e-6:
            raise MediaComparisonEOF("请求时间超出视频时长")
        target_pts = media.origin_pts + relative_time
        chosen = min(frames, key=lambda row: abs(row["pts"] - target_pts))
        previous = max((row for row in frames if row["pts"] < chosen["pts"]),
                       key=lambda row: row["pts"], default=None)
        following = min((row for row in frames if row["pts"] > chosen["pts"]),
                        key=lambda row: row["pts"], default=None)
        return {"requested_time": relative_time,
            "requested_pts": target_pts, "actual_pts": chosen["pts"],
            "actual_time": chosen["pts"] - media.origin_pts,
            "delta_seconds": chosen["pts"] - target_pts,
            "frame_duration": chosen["duration"], "key_frame": chosen["key_frame"],
            "previous_time": previous["pts"] - media.origin_pts if previous else None,
            "next_time": following["pts"] - media.origin_pts if following else None,
            "window_frames": len(frames)}

    def _origin(self, path: str, metadata: dict) -> float:
        start = float(metadata["start_time"])
        rows = self._command([self.ffprobe, "-v", "error", "-select_streams", "v:0",
            "-read_intervals", f"{start:.9f}%+#3", "-show_frames", "-show_entries",
            "frame=best_effort_timestamp_time,pts_time", "-of", "json", path], timeout=15)
        pts_values = []
        for frame in rows.get("frames") or []:
            pts = _finite(frame.get("best_effort_timestamp_time"))
            if pts is None:
                pts = _finite(frame.get("pts_time"))
            if pts is not None:
                pts_values.append(pts)
        if not pts_values:
            raise MediaComparisonError(f"无法读取首帧 PTS：{metadata['name']}")
        return min(pts_values)

    def open_pair(self, original: str, exported: str, original_offset_seconds=0., duration_seconds=None) -> dict:
        paths = [self._local_path(original), self._local_path(exported)]
        medias = []
        # Metadata is intentionally freshly probed for every open, not read from a
        # stale application cache. Only the initial three timestamps are inspected.
        for path in paths:
            metadata = self._probe_metadata(path)
            medias.append(_Media(path, metadata, self._origin(path, metadata)))
        offset = _finite(original_offset_seconds)
        if offset is None or offset < 0 or offset >= medias[0].metadata['duration_seconds']:
            raise MediaComparisonError('原片对比起点无效')
        if duration_seconds is not None:
            duration = _finite(duration_seconds)
            if duration is None or duration <= 0 or offset + duration > medias[0].metadata['duration_seconds'] + .1:
                raise MediaComparisonError('原片对比时长无效')
            medias[0].metadata['duration_seconds'] = duration
            medias[1].metadata['duration_seconds'] = min(duration, medias[1].metadata['duration_seconds'])
        medias[0].origin_pts += offset
        session = uuid.uuid4().hex
        with self._lock:
            self._sessions[session] = tuple(medias)
            self._sessions.move_to_end(session)
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        return {"session": session, "original": dict(medias[0].metadata),
                "exported": dict(medias[1].metadata),
                "duration_seconds": max(float(m.metadata["duration_seconds"]) for m in medias),
                "alignment": "each side uses nearest decoded presentation timestamp to the same relative time",
                "audio_default_side": "original",
                "hdr_display_note": "HDR uses one fixed, uncalibrated HDR-to-SDR display mapping on both sides."}

    def _pair(self, session: str) -> tuple[_Media, _Media]:
        with self._lock:
            pair = self._sessions.get(str(session))
            if pair is not None:
                self._sessions.move_to_end(str(session))
                return pair
        raise MediaComparisonError("对比会话不存在或已过期，请重新选择文件")

    @staticmethod
    def _side_index(side: str) -> int:
        if side == "original":
            return 0
        if side == "exported":
            return 1
        raise MediaComparisonError("side 必须为 original 或 exported")

    @staticmethod
    def _zscale_filter(metadata: dict) -> str:
        # Missing SDR tags are explicitly assumed BT.709 for display. HDR keeps the
        # actual probed transfer/primaries/matrix instead of silently treating PQ as SDR.
        # zimg uses the display-referred BT.1886 EOTF for its BT.709-to-linear
        # default. The project's SDR preview uses the BT.709 OETF pair instead.
        # Decode SDR into encoded Rec.709 RGB, then invert that pair explicitly
        # below; otherwise re-encoding with our OETF darkens the comparison.
        target_transfer = "linear" if metadata["is_hdr"] else "bt709"
        return ("zscale="
            f"primariesin={metadata['decode_primaries']}:"
            f"transferin={metadata['decode_transfer']}:"
            f"matrixin={metadata['decode_matrix']}:rangein={metadata['decode_range']}:"
            f"primaries=bt709:transfer={target_transfer}:matrix=gbr:range=full:npl=80,"
            "format=gbrpf32le")

    def _decode_linear(self, media: _Media, actual_pts: float) -> np.ndarray:
        meta = media.metadata
        # seek_timestamp makes -ss use the absolute presentation timestamp chosen by
        # ffprobe. Input seeking with transcoding retains FFmpeg's accurate seek step.
        cmd = [self.ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
            "-seek_timestamp", "1", "-ss", f"{actual_pts:.9f}", "-i", media.path,
            "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-vf",
            self._zscale_filter(meta), "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "pipe:1"]
        payload = self._command(cmd, timeout=45, binary=True)
        width, height = int(meta["width"]), int(meta["height"])
        expected = width * height * 3 * 4
        if len(payload) != expected:
            raise MediaComparisonEOF(
                f"FFmpeg 未输出请求帧：收到 {len(payload)} bytes，预期 {expected}")
        planes = np.frombuffer(payload, dtype="<f4").reshape(3, height, width)
        # gbrpf32le plane order is G, B, R.
        rgb = np.stack((planes[2], planes[0], planes[1]), axis=-1)
        if not meta["is_hdr"]:
            rgb = np.where(rgb < .081, rgb / 4.5,
                np.power(np.maximum((rgb + .099) / 1.099, 0), 1 / .45))
        return rgb

    def get_frame(self, session: str, time_seconds: float, side: str) -> dict:
        index = self._side_index(side)
        relative_time = _finite(time_seconds)
        if relative_time is None or relative_time < 0:
            raise MediaComparisonError("time 必须是有限的非负秒数")
        media = self._pair(session)[index]
        frames = self._frame_window(media, media.origin_pts + relative_time)
        selected = self._select_frame(media, relative_time, frames)
        linear = self._decode_linear(media, selected["actual_pts"])
        bgr = linear_rgb_to_display_bgr(linear, hdr=bool(media.metadata["is_hdr"]))
        ok, encoded = cv2.imencode(".jpg", bgr,
            [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise MediaComparisonError("无法编码对比显示帧")
        return {"jpeg": encoded.tobytes(), "side": side,
                "metadata": dict(media.metadata), **selected,
                "display_mapping": "fixed_hdr_to_sdr_uncalibrated" if media.metadata["is_hdr"] else "linear_rec709_to_sdr_display"}

    def get_timeline(self, session: str, time_seconds: float) -> dict:
        time_value = _finite(time_seconds)
        if time_value is None or time_value < 0:
            raise MediaComparisonError("time 必须是有限的非负秒数")
        pair = self._pair(session)
        result = {"requested_time": time_value, "sides": {}}
        for label, media in zip(("original", "exported"), pair):
            if time_value > float(media.metadata["duration_seconds"]) + 1e-6:
                result["sides"][label] = {"eof": True,
                    "duration_seconds": media.metadata["duration_seconds"]}
                continue
            frames = self._frame_window(media, media.origin_pts + time_value)
            result["sides"][label] = {"eof": False,
                **self._select_frame(media, time_value, frames)}
        return result

    def audio_source(self, session: str, side: str = "original") -> dict:
        media = self._pair(session)[self._side_index(side)]
        audio = media.metadata.get("audio")
        if not audio:
            raise MediaComparisonError(f"{side} 文件没有音轨")
        # The HTTP adapter streams this one selected stream. The page only attaches
        # one audio_source at once to prevent doubled audio.
        return {"path": media.path, "stream_index": audio["index"],
                "codec": audio["codec"], "side": side,
                "origin_pts": media.origin_pts,
                "duration_seconds": media.metadata['duration_seconds']}

    def close_session(self, session: str) -> bool:
        with self._lock:
            return self._sessions.pop(str(session), None) is not None
