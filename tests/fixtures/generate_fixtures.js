// tests/fixtures/generate_fixtures.js
import { execSync } from 'node:child_process';
import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const FIXTURES_DIR = __dirname;
export const FIXTURE_120FPS_MKV = path.join(FIXTURES_DIR, 'fixture_120fps.mkv');
// The bounded clip the GPU Transfer export test renders: six frames at the same
// extent and rate the frozen device-chain probe generates for itself, and no audio
// track. Six frames is the whole job — the render queue exports the clip it was
// handed, so the fixture is what keeps a real export short.
export const FIXTURE_GPU_TRANSFER_SHORT = path.join(FIXTURES_DIR, 'fixture_gpu_transfer_short.mkv');

export function probeMedia(filePath) {
  if (!fs.existsSync(filePath)) {
    throw new Error(`File does not exist: ${filePath}`);
  }
  const cmd = `ffprobe -v error -show_entries format=duration,size,format_name -show_entries stream=index,codec_name,codec_type,width,height,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_read_packets -count_packets -of json "${filePath}"`;
  const raw = execSync(cmd, { encoding: 'utf-8' });
  return JSON.parse(raw);
}

export function generateFixture120fps(outputPath = FIXTURE_120FPS_MKV, force = false) {  if (!force && fs.existsSync(outputPath)) {
    try {
      const probe = probeMedia(outputPath);
      const videoStream = probe.streams?.find((s) => s.codec_type === 'video');
      const audioStream = probe.streams?.find((s) => s.codec_type === 'audio');
      const frames = parseInt(videoStream?.nb_read_packets || '0', 10);
      const duration = parseFloat(probe.format?.duration || '0');
      if (frames >= 595 && frames <= 605 && duration >= 4.9 && duration <= 5.1 && audioStream) {
        console.log(`[Fixtures] Valid fixture already exists at: ${outputPath} (${frames} frames, ${duration.toFixed(2)}s)`);
        return { outputPath, frames, duration, probe };
      }
    } catch (e) {
      console.warn(`[Fixtures] Existing fixture validation failed (${e.message}), recreating...`);
    }
  }

  console.log(`[Fixtures] Generating fixture_120fps.mkv: 1280x720, 120fps, 5s, 600 frames with audio...`);
  const cmd = `ffmpeg -y -f lavfi -i testsrc2=size=1280x720:rate=120:duration=5 -f lavfi -i sine=frequency=1000:duration=5 -c:v libx264 -preset ultrafast -pix_fmt yuv420p -c:a aac -shortest "${outputPath}"`;
  execSync(cmd, { stdio: 'inherit' });

  // Self-verification with ffprobe
  const probe = probeMedia(outputPath);
  const videoStream = probe.streams?.find((s) => s.codec_type === 'video');
  const audioStream = probe.streams?.find((s) => s.codec_type === 'audio');

  if (!videoStream) {
    throw new Error('Fixture verification FAILED: missing video stream!');
  }
  if (!audioStream) {
    throw new Error('Fixture verification FAILED: missing audio stream!');
  }

  const width = videoStream.width;
  const height = videoStream.height;
  const frames = parseInt(videoStream.nb_read_packets || '0', 10);
  const duration = parseFloat(probe.format?.duration || '0');
  const rFps = videoStream.r_frame_rate;

  console.log(`[Fixtures] Generated metadata: ${width}x${height}, ${rFps} fps, ${frames} frames, ${duration.toFixed(2)}s`);

  if (width !== 1280 || height !== 720) {
    throw new Error(`Fixture verification FAILED: resolution ${width}x${height} != 1280x720`);
  }
  if (frames < 595 || frames > 605) {
    throw new Error(`Fixture verification FAILED: frame count ${frames} not in [595, 605]`);
  }
  if (duration < 4.9 || duration > 5.1) {
    throw new Error(`Fixture verification FAILED: duration ${duration}s not in [4.9, 5.1]`);
  }

  console.log(`[Fixtures] Verified OK: ${outputPath}`);
  return { outputPath, frames, duration, probe };
}

/**
 * The six-frame clip the GPU Transfer browser test exports. Deliberately a separate
 * fixture rather than a slice of fixture_120fps.mkv: the render queue exports every
 * frame it is handed, so a clip that is already short is what bounds a real export.
 */
export function generateFixtureGpuTransferShort(outputPath = FIXTURE_GPU_TRANSFER_SHORT,
                                                force = false) {
  if (!force && fs.existsSync(outputPath)) {
    try {
      const probe = probeMedia(outputPath);
      const videoStream = probe.streams?.find((s) => s.codec_type === 'video');
      const frames = parseInt(videoStream?.nb_read_packets || '0', 10);
      if (videoStream?.width === 1280 && videoStream?.height === 720 &&
          videoStream?.pix_fmt === 'yuv420p' && frames >= 5 && frames <= 7 &&
          !probe.streams?.find((s) => s.codec_type === 'audio')) {
        console.log(`[Fixtures] Valid short fixture already exists: ${outputPath} (${frames} frames)`);
        return { outputPath, frames, probe };
      }
    } catch (e) {
      console.warn(`[Fixtures] Short fixture validation failed (${e.message}), recreating...`);
    }
  }

  console.log(`[Fixtures] Generating fixture_gpu_transfer_short.mkv: 1280x720, 24fps, 6 frames, silent`);
  const cmd = `ffmpeg -y -f lavfi -i testsrc2=size=1280x720:rate=24:duration=0.25 -frames:v 6 -c:v libx264 -preset veryfast -pix_fmt yuv420p -an "${outputPath}"`;
  execSync(cmd, { stdio: 'inherit' });

  const probe = probeMedia(outputPath);
  const videoStream = probe.streams?.find((s) => s.codec_type === 'video');
  if (!videoStream) {
    throw new Error('Short fixture verification FAILED: missing video stream');
  }
  const frames = parseInt(videoStream.nb_read_packets || '0', 10);
  if (videoStream.width !== 1280 || videoStream.height !== 720) {
    throw new Error(`Short fixture verification FAILED: ${videoStream.width}x${videoStream.height}`);
  }
  if (videoStream.pix_fmt !== 'yuv420p') {
    throw new Error(`Short fixture verification FAILED: pix_fmt=${videoStream.pix_fmt}, the device chain takes 8-bit 4:2:0`);
  }
  if (frames < 5 || frames > 7) {
    throw new Error(`Short fixture verification FAILED: ${frames} frames is outside the 3-8 window`);
  }
  console.log(`[Fixtures] Verified OK: ${outputPath} (${frames} frames, yuv420p, silent)`);
  return { outputPath, frames, probe };
}

// Run directly if invoked as script
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(__filename)) {
  try {
    generateFixture120fps();
    generateFixtureGpuTransferShort();
    console.log('[Fixtures] Success!');
  } catch (err) {
    console.error('[Fixtures] Generation error:', err);
    process.exit(1);
  }
}
