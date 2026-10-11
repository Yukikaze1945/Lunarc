// The working size a scale resolves to, mirroring pipeline.resolve_working_dims
// clause for clause: the same clip to 0.25..1.0, the same "100% is the source,
// rounded down to even" branch, the same 8-pixel rounding and 16-pixel floor.
// The delivery page shows this number and the export gate jobs the one the
// resolver produced, so a second rounding rule here would advertise a size the
// chain then refuses (2880x1624 against 2880x1616 at 75% was exactly that).

// Match Python's round() in pipeline.resolve_working_dims at exact half steps.
export const roundToEven = (value: number): number => {
  const lower = Math.floor(value)
  if (value - lower === 0.5) return lower % 2 === 0 ? lower : lower + 1
  return Math.round(value)
}

// Python's int(round(v)) - 1 for odd, which on the integer extents this is called
// with is the same as rounding down to even.
const even = (value: number): number => {
  const rounded = Math.round(value)
  return Math.max(2, rounded % 2 === 0 ? rounded : rounded - 1)
}

export const resolveWorkingDims = (
  srcW: number, srcH: number, scale: number,
): [number, number] => {
  const s = Math.min(1.0, Math.max(0.25, scale))
  if (Math.abs(s - 1.0) < 1e-4) return [even(srcW), even(srcH)]
  const dim = (src: number): number => Math.max(16, roundToEven((src * s) / 8) * 8)
  return [dim(srcW), dim(srcH)]
}

// The served range, as whole percentages: gpu_transfer_export.WORKING_PERCENT_MIN
// / _MAX. The slider steps over it and the gate accepts all of it, so a percentage
// outside these two numbers is a control that lies.
export const WORKING_PERCENT_MIN = 25
export const WORKING_PERCENT_MAX = 100

export const workingPercent = (scale: number): number | null => {
  if (!Number.isFinite(scale)) return null
  const percent = Math.round(scale * 100)
  if (Math.abs(scale * 100 - percent) > 1e-6) return null
  return percent >= WORKING_PERCENT_MIN && percent <= WORKING_PERCENT_MAX
    ? percent
    : null
}
