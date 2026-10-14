import assert from 'node:assert/strict'
import {test} from 'node:test'
import {normalizeExportOptions, exportOptions} from '../../frontend/src/services/exportOptions.ts'
import {droppedMediaPath} from '../../frontend/src/services/mediaDrop.ts'

const settings: any = {export_backend:'gpu_transfer', experimental_hdr_scrgb:false}
const tx: any = {codec:'h264_nvenc',container:'.webm',pixel_format:'nv16',profile:'high',color_preset:'custom',resolution_mode:'custom_size',audio_mode:'copy',tune:'uhq',b_ref_mode:'each',weighted_prediction:false,copy_attachments:true,extra_args:'',custom_color_transfer:'smpte2084'}
test('GPU SDR restores valid persisted selections before queuing',()=>{
  const next=normalizeExportOptions(tx,settings,null)
  assert.equal(next.container,'.mkv');assert.equal(next.pixel_format,'auto')
  assert.equal(next.color_preset,'preserve');assert.equal(next.resolution_mode,'custom_size')
  assert.equal(next.custom_color_transfer,'');assert.equal(next.tune,'hq')
  assert.equal(normalizeExportOptions(next,settings,null),next)
})
test('HDR and SDR10 constrain codec/container/profile without dropping track copying',()=>{
  const sdr: any = {is_hdr:false,bit_depth:10,pix_fmt:'yuv420p10le',color_transfer:'bt709',color_space:'bt709',color_primaries:'bt709',color_range:'tv'}
  for(const [config,video] of [[{...settings,experimental_hdr_scrgb:true},null],[settings,sdr]]) {
    const next=normalizeExportOptions({...tx,extra_args:'-bsf:v remove'},config,video)
    assert.equal(next.codec,'hevc_nvenc');assert.equal(next.container,'.mkv')
    assert.equal(next.profile,'auto');assert.equal(next.pixel_format,'auto');assert.equal(next.extra_args,'')
    assert.equal(next.audio_mode,'copy');assert.equal(next.copy_attachments,true)
    assert.deepEqual(exportOptions(config,video,next).codec,['hevc_nvenc','av1_nvenc'])
  }
})

test('wide AV1 stays selected and uses Main at 10-bit instead of HEVC Main10',()=>{
  for(const settings of [{export_backend:'gpu_transfer',experimental_hdr_scrgb:true},{export_backend:'gpu_transfer',truehdr_enabled:true}]) {
    const next=normalizeExportOptions({...tx,codec:'av1_nvenc',profile:'main10',pixel_format:'p010le',container:'.mkv'},settings as any,null)
    assert.equal(next.codec,'av1_nvenc');assert.equal(next.profile,'auto')
    assert.deepEqual(exportOptions(settings as any,null,next).profile,['auto','main'])
    const main=normalizeExportOptions({...next,profile:'main'},settings as any,null)
    assert.equal(main.profile,'main');assert.equal(main.pixel_format,'p010le')
  }
})
test('WebM audio and non-MKV attachments cannot carry incompatible requests',()=>{
  const next=normalizeExportOptions({...tx,codec:'av1_nvenc'},settings,null)
  assert.equal(next.container,'.webm');assert.equal(next.audio_mode,'opus')
  assert.equal(next.copy_attachments,false);assert.equal(next.b_ref_mode,'auto')
})
test('media drop selects first supported real file and skips text or synthetic paths',()=>{
  const files:any[]=[{name:'note.txt',path:'note'},{name:'cover.PNG',path:''},{name:'电影.mkv',path:'D:\\电影.mkv'},{name:'image.jpg',path:'D:\\image.jpg'}]
  assert.equal(droppedMediaPath(files,file=>(file as any).path),'D:\\电影.mkv')
  assert.equal(droppedMediaPath([{name:'照片.webp'}] as any,()=> 'D:\\照片.webp'),'D:\\照片.webp')
  assert.equal(droppedMediaPath([{name:'not-media.exe'}] as any,()=> 'any'),null)
})
