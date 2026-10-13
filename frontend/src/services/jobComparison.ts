import type {RenderJob} from '../types'

export function jobComparison(job:Pick<RenderJob,'input_path'|'output_path'|'frame_range'|'source_fps'>) {
  const range=job.frame_range,fps=job.source_fps
  return {original:job.input_path,exported:job.output_path,
    ...(range&&fps&&fps>0?{original_offset_seconds:range.start_frame/fps,
      duration_seconds:(range.end_frame-range.start_frame)/fps}:{})}
}
