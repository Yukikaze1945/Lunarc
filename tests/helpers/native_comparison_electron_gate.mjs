// Actual packaged Electron, actual final video files; never calls AI processing.
import {_electron as electron} from '@playwright/test'
import fs from 'node:fs/promises'
import path from 'node:path'
import {ROOT_DIR,pythonExe} from './paths.js'
const out=path.resolve(process.argv[2]);await fs.mkdir(out,{recursive:false})
const executable=path.resolve(process.argv[3]);const rows=[],errors=[]
const compareSource=await fs.readFile(path.join(ROOT_DIR,'frontend/src/pages/ComparePage.tsx'),'utf8')
const compareCss=await fs.readFile(path.join(ROOT_DIR,'frontend/src/pages/compare.css'),'utf8')
if(/eyebrow|LUNARC\s*\/\s*MEDIA REVIEW/i.test(compareSource+compareCss))throw new Error('Comparison page contains an eyebrow label')
for(const [,value] of compareCss.matchAll(/border-radius\s*:\s*([\d.]+)px/gi))
 if(Number(value)>2)throw new Error(`Comparison CSS radius exceeds 2px: ${value}px`)
if(await fetch('http://127.0.0.1:18923/api/system').then(()=>true).catch(()=>false))throw new Error('Existing backend on 18923; refusing to attach')
const app=await electron.launch({executablePath:executable,cwd:path.dirname(executable),
 args:[`--user-data-dir=${path.join(out,'profile')}`,'--no-sandbox'],
 env:{...process.env,PYTHON_PATH:pythonExe(),E2E_TEST:'1'}})
let page,stage='launch'
try{
 page=await app.firstWindow({timeout:90000});page.on('pageerror',e=>errors.push(String(e)));page.on('close',()=>errors.push(`window closed during ${stage}`))
 await page.setViewportSize({width:1440,height:900})
 app.on('close',()=>errors.push(`Electron closed during ${stage}`))
 stage='backend-ready'
 await page.waitForFunction(async()=>{try{const r=await fetch('http://127.0.0.1:18923/api/system',{headers:{'X-DLSS-Token':window.electronAPI?.localToken||''}});return r.ok&&(await r.json()).backend_ready}catch{return false}},null,{timeout:60000})
 const pairs=[
  ['sdr_720_1080','bench_out/native_backends_20261010/pipeline1/source720.mkv','bench_out/native_backends_20261010/pipeline1/nr_vsr.mkv'],
  ['sdr10_4k','bench_out/sdr10_20261010/source6.mkv','bench_out/sdr10_20261010/service/service_sdr10_hevc.mkv'],
  ['hdr10','bench_out/optimization_20261009/hdr_cut_source_v3.mkv','bench_out/optimization_20261009/hdr_triple_baseline/final.mkv'],
 ]
 for(const [name,original,exported] of pairs){
  stage=`${name}: open pair`
  const paths={original:path.join(ROOT_DIR,original),exported:path.join(ROOT_DIR,exported)}
  await fs.stat(paths.original);await fs.stat(paths.exported)
  await page.evaluate(pair=>window.dispatchEvent(new CustomEvent('lunarc:compare',{detail:pair})),paths)
  const compare=page.locator('.compare-page')
  await page.waitForFunction(({originalName,exportedName})=>{
   const root=document.querySelector('.compare-page')
   const names=root?.querySelector('.compare-filebar')?.textContent||''
   const imgs=[...root?.querySelectorAll('img')||[]]
   return names.includes(originalName)&&names.includes(exportedName)&&imgs.length===2&&
    imgs.every(x=>x.complete&&x.naturalWidth>0&&x.src.startsWith('blob:'))&&
    !root?.querySelector('[aria-label="前进一帧"]')?.disabled
  },{originalName:path.basename(paths.original),exportedName:path.basename(paths.exported)},{timeout:60000})
  if(await page.getByRole('button',{name:'视频对比'}).getAttribute('aria-current')!=='page')
   throw new Error(`${name}: compare event did not leave the app on the comparison page`)
  if(name==='sdr_720_1080'){
   const theme=page.getByRole('button',{name:/切换浅色模式|切换深色模式/})
   if(await theme.getAttribute('aria-label')==='切换深色模式')await theme.click()
   await page.screenshot({path:path.join(out,'compare_dark_1440x900.png')})
   await page.getByRole('button',{name:'切换浅色模式'}).click()
   await page.screenshot({path:path.join(out,'compare_light_1440x900.png')})
   await page.getByRole('button',{name:'切换深色模式'}).click()
   const radii=await compare.evaluate(root=>Array.from(root.querySelectorAll('*')).map(el=>getComputedStyle(el).borderTopLeftRadius)
    .filter(value=>value.endsWith('px')).map(parseFloat))
   if(Math.max(0,...radii)>2)throw new Error(`Rendered comparison element has a radius above 2px: ${Math.max(...radii)}px`)
   if(await compare.locator('.eyebrow').count())throw new Error('Rendered comparison page contains an eyebrow label')
  }
  await page.screenshot({path:path.join(out,name+'_side.png')})
  stage=`${name}: wipe mode`
  const initial=await page.locator('.compare-filebar').innerText()
  await compare.getByRole('button',{name:'中央滑块',exact:true}).click()
  const divider=compare.getByRole('button',{name:'拖动对比滑块'});const box=await divider.boundingBox()
  await page.mouse.move(box.x+box.width/2,box.y+box.height/2);await page.mouse.down();await page.mouse.move(box.x+140,box.y+box.height/2);await page.mouse.up()
  const split=await divider.evaluate(x=>x.style.left)
  if(split==='50%')throw new Error('Wipe divider did not move')
  await page.screenshot({path:path.join(out,name+'_wipe.png')})
  stage=`${name}: A/B and zoom`
  await compare.getByRole('button',{name:'A / B',exact:true}).click()
  await compare.getByRole('button',{name:'B · 导出',exact:true}).click()
  await compare.getByRole('button',{name:'200%',exact:true}).click()
  await page.screenshot({path:path.join(out,name+'_ab_200.png')})
  stage=`${name}: frame step and playback setup`
  await compare.getByRole('button',{name:'适配',exact:true}).click()
  await compare.getByRole('button',{name:'前进一帧'}).click()
  await page.waitForFunction(()=>Number(document.querySelector('.compare-page [aria-label="同步时间轴"]')?.value)>0,null,{timeout:60000})
  await page.waitForFunction(()=>!document.querySelector('.compare-page [aria-label="前进一帧"]')?.disabled,null,{timeout:60000})
  const stepped=await compare.getByRole('slider',{name:'同步时间轴'}).inputValue()
  if(Number(stepped)<=0)throw new Error('Timestamp frame step did not advance')
  await compare.getByRole('combobox',{name:'播放速度'}).selectOption('0.5')
  const audioRate=await compare.locator('.compare-transport audio').evaluate(x=>x.playbackRate)
  if(audioRate!==.5)throw new Error('Audio rate differs from selected clock')
  const timeline=compare.getByRole('slider',{name:'同步时间轴'})
  const timelineBox=await timeline.boundingBox()
  await timeline.click({position:{x:2,y:timelineBox.height/2}})
  const playbackStart=await timeline.inputValue()
  const audio=compare.locator('.compare-transport audio')
  const audioStart=await audio.evaluate(x=>{
   x.__gateEvents=[]
   for(const type of ['playing','timeupdate','ended'])x.addEventListener(type,()=>x.__gateEvents.push({type,time:x.currentTime}),{once:type==='playing'})
   return x.currentTime
  })
  const playBox=await compare.getByRole('button',{name:'播放',exact:true}).boundingBox()
  const playTarget=await page.evaluate(({x,y})=>{const e=document.elementFromPoint(x,y);return e?{tag:e.tagName,aria:e.getAttribute('aria-label'),text:e.textContent?.trim().slice(0,80),className:String(e.className)}:null},{x:playBox.x+playBox.width/2,y:playBox.y+playBox.height/2})
  stage=`${name}: playback click target ${JSON.stringify(playTarget)}`
  await compare.getByRole('button',{name:'播放',exact:true}).click()
  if(await page.getByRole('button',{name:'视频对比'}).getAttribute('aria-current')!=='page')
   throw new Error(`${name}: playback click navigated away from the comparison page`)
  // Short real clips can end between visual-frame clock ticks. Check
  // actual timeline movement or real media playback events and time progress.
  await page.waitForFunction(({timelineBefore,audioBefore})=>{
   const root=document.querySelector('.compare-page')
   const timeline=Number(root?.querySelector('[aria-label="同步时间轴"]')?.value)
   const audio=root?.querySelector('.compare-transport audio')
   const events=audio?.__gateEvents||[]
   const progressed=Number(audio?.currentTime)>Number(audioBefore)+.001||
    events.some(event=>Number(event.time)>Number(audioBefore)+.001)
   return timeline>Number(timelineBefore)||
    Boolean(progressed&&events.some(event=>['playing','timeupdate','ended'].includes(event.type)))
  },{timelineBefore:playbackStart,audioBefore:audioStart},{timeout:5000})
  stage=`${name}: pause`
  const pause=compare.getByRole('button',{name:'暂停',exact:true})
  if(await pause.count())await pause.click()
  const audioCount=await compare.locator('audio').count()
  if(audioCount!==1)throw new Error('More than one audio player')
  rows.push({name,paths,initial,split,stepped,audioRate,audioStart,
   playbackStart,audioEnd:await audio.evaluate(x=>x.currentTime),audioEvents:await audio.evaluate(x=>x.__gateEvents),audioCount,
   hdrNote:await compare.locator('.compare-hdr-note').count()})
  await compare.getByRole('button',{name:'左右并排',exact:true}).click()
 }
 stage='missing-file check'
 await page.evaluate(original=>window.dispatchEvent(new CustomEvent('lunarc:compare',{detail:{original,exported:original+'.missing'}})),path.join(ROOT_DIR,pairs[0][1]))
 await page.waitForFunction(()=>document.querySelector('.compare-footer [role=status]')?.textContent?.match(/不存在|missing|找不到/),null,{timeout:30000})
 await page.screenshot({path:path.join(out,'missing_file.png')});rows.push({name:'missing_file',status:await page.locator('.compare-footer').innerText()})
 if(errors.length)throw new Error(errors.join('\n'))
 await fs.writeFile(path.join(out,'report.json'),JSON.stringify({status:'PASS',actualElectron:true,executable,rows,errors},null,2))
}catch(e){if(page&&!page.isClosed()){try{await page.screenshot({path:path.join(out,'failure.png')});await fs.writeFile(path.join(out,'failure.html'),await page.content())}catch{}}await fs.writeFile(path.join(out,'report.json'),JSON.stringify({status:'FAIL',stage,error:String(e),rows,errors},null,2));throw e}
finally{await app.close()}
