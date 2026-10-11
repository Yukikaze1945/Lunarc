// tests/helpers/media_probe.js
import { execSync } from 'node:child_process';
import fs from 'node:fs';

export function probeMedia(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`Media file does not exist: ${filePath}`);
  }
  const cmd = `ffprobe -v error -show_entries format=duration,size,format_name,bit_rate -show_entries stream=index,codec_name,codec_type,width,height,pix_fmt,color_range,r_frame_rate,avg_frame_rate,duration,nb_read_packets -count_packets -of json "${filePath}"`;
  const raw = execSync(cmd, { encoding: 'utf-8' });
  return JSON.parse(raw);
}

export function verifyVideoIntegrity(filePath, expected = {}) {
  const errors = [];
  if (!fs.existsSync(filePath)) {
    return { ok: false, errors: [`File does not exist: ${filePath}`], details: null };
  }

  const stat = fs.statSync(filePath);
  if (stat.size <= 1024) {
    errors.push(`File size is too small: ${stat.size} bytes (<= 1024 bytes)`);
  }

  let probe;
  try {
    probe = probeMedia(filePath);
  } catch (err) {
    return { ok: false, errors: [`ffprobe parse error: ${err.message}`], details: null };
  }

  const video = probe.streams?.find((s) => s.codec_type === 'video');
  const audio = probe.streams?.find((s) => s.codec_type === 'audio');

  if (!video) {
    errors.push('Output has NO video stream');
  }

  const duration = parseFloat(probe.format?.duration || video?.duration || '0');
  const frames = parseInt(video?.nb_read_packets || '0', 10);
  const width = video?.width || 0;
  const height = video?.height || 0;
  const codec = video?.codec_name || '';

  // 1. Duration check
  if (expected.minDuration !== undefined && duration < expected.minDuration) {
    errors.push(`Duration too short: ${duration.toFixed(2)}s < expected min ${expected.minDuration}s`);
  }
  if (expected.maxDuration !== undefined && duration > expected.maxDuration) {
    errors.push(`Duration too long: ${duration.toFixed(2)}s > expected max ${expected.maxDuration}s`);
  }

  // 2. Frame count check
  if (expected.minFrames !== undefined && frames < expected.minFrames) {
    errors.push(`Frame count too low: ${frames} < expected min ${expected.minFrames} (silent frame drop or truncation!)`);
  }
  if (expected.maxFrames !== undefined && frames > expected.maxFrames) {
    errors.push(`Frame count too high: ${frames} > expected max ${expected.maxFrames}`);
  }

  // 3. Resolution check
  if (expected.width !== undefined && width !== expected.width) {
    errors.push(`Width mismatch: ${width} != expected ${expected.width}`);
  }
  if (expected.height !== undefined && height !== expected.height) {
    errors.push(`Height mismatch: ${height} != expected ${expected.height}`);
  }

  // 4. Codec check
  if (expected.codec && codec.toLowerCase() !== expected.codec.toLowerCase()) {
    errors.push(`Codec mismatch: ${codec} != expected ${expected.codec}`);
  }

  // 5. Audio check
  if (expected.requireAudio && !audio) {
    errors.push('Expected audio stream, but none was found in output');
  }

  return {
    ok: errors.length === 0,
    errors,
    details: {
      filePath,
      sizeBytes: stat.size,
      duration,
      frames,
      width,
      height,
      codec,
      hasAudio: !!audio,
      audioCodec: audio?.codec_name,
    },
  };
}
