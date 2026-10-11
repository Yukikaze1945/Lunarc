/**
 * The working size the delivery page prints is a second implementation of
 * pipeline.resolve_working_dims - in another language, for another process - and
 * the export gate jobs the size the Python resolver produced. These rows are the
 * Python output, transcribed: every percentage the slider can reach, on sources
 * whose axes do and do not halve cleanly. They are the 75%-on-4K class of bug
 * (2880x1624 against 2880x1616): a mirror is only a mirror while the numbers
 * still agree, so the numbers are what is asked here.
 *
 *   node --test ../tests/node/
 */
import { test } from 'node:test'
import assert from 'node:assert/strict'

import {
  WORKING_PERCENT_MAX,
  WORKING_PERCENT_MIN,
  resolveWorkingDims,
  workingPercent,
} from '../../frontend/src/services/workingScale.ts'

// [src_w, src_h, percent, working_w, working_h]
const GOLDEN: [number, number, number, number, number][] = [
  [3840, 2160, 25, 960, 544],
  [3840, 2160, 26, 1000, 560],
  [3840, 2160, 33, 1264, 712],
  [3840, 2160, 40, 1536, 864],
  [3840, 2160, 50, 1920, 1080],
  [3840, 2160, 63, 2416, 1360],
  [3840, 2160, 67, 2576, 1448],
  [3840, 2160, 75, 2880, 1616],
  [3840, 2160, 90, 3456, 1944],
  [3840, 2160, 99, 3800, 2136],
  [3840, 2160, 100, 3840, 2160],
  [1920, 1080, 25, 480, 272],
  [1920, 1080, 26, 496, 280],
  [1920, 1080, 33, 632, 360],
  [1920, 1080, 40, 768, 432],
  [1920, 1080, 50, 960, 544],
  [1920, 1080, 63, 1208, 680],
  [1920, 1080, 67, 1288, 720],
  [1920, 1080, 75, 1440, 808],
  [1920, 1080, 90, 1728, 976],
  [1920, 1080, 99, 1904, 1072],
  [1920, 1080, 100, 1920, 1080],
  [1280, 720, 25, 320, 176],
  [1280, 720, 26, 336, 184],
  [1280, 720, 33, 424, 240],
  [1280, 720, 40, 512, 288],
  [1280, 720, 50, 640, 360],
  [1280, 720, 63, 808, 456],
  [1280, 720, 67, 856, 480],
  [1280, 720, 75, 960, 544],
  [1280, 720, 90, 1152, 648],
  [1280, 720, 99, 1264, 712],
  [1280, 720, 100, 1280, 720],
  [2560, 1440, 25, 640, 360],
  [2560, 1440, 26, 664, 376],
  [2560, 1440, 33, 848, 472],
  [2560, 1440, 40, 1024, 576],
  [2560, 1440, 50, 1280, 720],
  [2560, 1440, 63, 1616, 904],
  [2560, 1440, 67, 1712, 968],
  [2560, 1440, 75, 1920, 1080],
  [2560, 1440, 90, 2304, 1296],
  [2560, 1440, 99, 2536, 1424],
  [2560, 1440, 100, 2560, 1440],
  // An odd source: 8-alignment rounds its quarter axis away from a clean quarter,
  // which is why the width a percentage needs is a property of the source.
  [3842, 2162, 25, 960, 544],
  [3842, 2162, 26, 1000, 560],
  [3842, 2162, 33, 1264, 712],
  [3842, 2162, 40, 1536, 864],
  [3842, 2162, 50, 1920, 1080],
  [3842, 2162, 63, 2424, 1360],
  [3842, 2162, 67, 2576, 1448],
  [3842, 2162, 75, 2880, 1624],
  [3842, 2162, 90, 3456, 1944],
  [3842, 2162, 99, 3800, 2144],
  [3842, 2162, 100, 3842, 2162],
  // The small source the tap window is worst on: a quarter of 80 is 20, and the
  // resolver's own rounding puts it at 16.
  [80, 80, 25, 16, 16],
  [80, 80, 26, 24, 24],
  [80, 80, 33, 24, 24],
  [80, 80, 40, 32, 32],
  [80, 80, 50, 40, 40],
  [80, 80, 63, 48, 48],
  [80, 80, 67, 56, 56],
  [80, 80, 75, 64, 64],
  [80, 80, 90, 72, 72],
  [80, 80, 99, 80, 80],
  [80, 80, 100, 80, 80],
  [720, 720, 25, 176, 176],
  [720, 720, 26, 184, 184],
  [720, 720, 33, 240, 240],
  [720, 720, 40, 288, 288],
  [720, 720, 50, 360, 360],
  [720, 720, 63, 456, 456],
  [720, 720, 67, 480, 480],
  [720, 720, 75, 544, 544],
  [720, 720, 90, 648, 648],
  [720, 720, 99, 712, 712],
  [720, 720, 100, 720, 720],
  [4096, 2160, 25, 1024, 544],
  [4096, 2160, 26, 1064, 560],
  [4096, 2160, 33, 1352, 712],
  [4096, 2160, 40, 1640, 864],
  [4096, 2160, 50, 2048, 1080],
  [4096, 2160, 63, 2584, 1360],
  [4096, 2160, 67, 2744, 1448],
  [4096, 2160, 75, 3072, 1616],
  [4096, 2160, 90, 3688, 1944],
  [4096, 2160, 99, 4056, 2136],
  [4096, 2160, 100, 4096, 2160],
]

test('the page resolves every percentage to the size the backend resolves', () => {
  for (const [srcW, srcH, percent, wantW, wantH] of GOLDEN) {
    assert.deepEqual(
      resolveWorkingDims(srcW, srcH, percent / 100),
      [wantW, wantH],
      `${srcW}x${srcH} at ${percent}%`,
    )
  }
})

test('the resolver keeps the two clauses the Python rule has', () => {
  // 100% is the source rounded down to even, not the source times a ratio: an odd
  // axis is the difference between the two, and the encoder will not take the odd.
  assert.deepEqual(resolveWorkingDims(3841, 2161, 1.0), [3840, 2160])
  // Anything at or below the quarter floor resolves at 16 rather than to nothing.
  assert.deepEqual(resolveWorkingDims(40, 40, 0.25), [16, 16])
  assert.deepEqual(resolveWorkingDims(60, 60, 0.0), [16, 16])
  // And a ratio above the range is clipped, the same way np.clip clips it.
  assert.deepEqual(resolveWorkingDims(1920, 1080, 1.5), [1920, 1080])
})

test('the served range is whole percentages and nothing else', () => {
  assert.equal(WORKING_PERCENT_MIN, 25)
  assert.equal(WORKING_PERCENT_MAX, 100)
  for (let percent = 25; percent <= 100; percent += 1) {
    // 0.67 * 100 is 67.00000000000001 in binary floating point; a percentage the
    // slider produced (value / 100) has to read back as that percentage.
    assert.equal(workingPercent(percent / 100), percent, percent)
  }
  assert.equal(workingPercent(1 / 3), null)
  assert.equal(workingPercent(0.24), null)
  assert.equal(workingPercent(1.01), null)
  assert.equal(workingPercent(Number.NaN), null)
})
