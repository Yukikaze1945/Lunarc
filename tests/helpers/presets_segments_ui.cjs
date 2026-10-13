// Isolated browser checks; no real queue, settings, or GPU sessions are touched.
const path=require('node:path'),fs=require('node:fs'),http=require('node:http'),assert=require('node:assert/strict')
const root=path.resolve(__dirname,'../..'),dist=path.join(root,'frontend/dist'),out=path.join(root,'build/segment-preview-check')
const {chromium}=require(path.join(root,'frontend/node_modules/playwright'))
fs.mkdirSync(out,{recursive:true})
const server=http.createServer((req,res)=>{const file=path.join(dist,decodeURIComponent(req.url.split('?')[0]==='/'?'/index.html':req.url.split('?')[0]));
  if(!file.startsWith(dist)||!fs.existsSync(file)){res.writeHead(404);res.end();return}
  res.setHeader('Content-Type',file.endsWith('.js')?'application/javascript':file.endsWith('.css')?'text/css':'text/html');res.end(fs.readFileSync(file))})
;(async()=>{
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
  const browser=await chromium.launch({headless:true})
  try{
    const page=await browser.newPage({viewport:{width:1600,height:1000}}),errors=[]
    page.on('pageerror',error=>errors.push(error.message))
    await page.addInitScript(()=>{
      localStorage.setItem('lunarc.guide.v1','done')
      class Socket{static OPEN=1;readyState=1;constructor(){setTimeout(()=>this.onopen?.(),0)}send(){}close(){this.readyState=3}addEventListener(){}removeEventListener(){}}
      window.WebSocket=Socket
    })
    let settings={neural_mode:'nr',style:0,intensity:1,local_tone:1,local_struct:1,skin_struct:1,use_auto_mask:1,
      guidance_mode:0,working_scale:1,export_backend:'gpu_transfer',render_pipeline:['nr','vsr'],vsr_output_mode:'scale',vsr_scale:2}
    const video={path:'C:\\fixture.mkv',filename:'fixture.mkv',width:1920,height:1080,frames:1200,fps:30,duration_seconds:40,is_hdr:false}
    let segments=[],added=null
    await page.route('http://127.0.0.1:18923/**',async route=>{
      const request=route.request(),url=new URL(request.url());let data={}
      if(url.pathname==='/api/system')data={gpu:'RTX',backend_ready:true,adapter_memory_used_gb:4.5,adapter_memory_total_gb:16}
      if(url.pathname==='/api/video/current')data={video}
      if(url.pathname==='/api/settings'){if(request.method()==='POST')settings={...settings,...request.postDataJSON()};data=settings}
      if(url.pathname==='/api/queue')data={jobs:[],active_id:null,queue_running:false}
      if(url.pathname==='/api/flow/providers')data={providers:[]}
      if(url.pathname==='/api/preview/segments/list')data={segments}
      if(url.pathname==='/api/preview/segments'){
        added=request.postDataJSON();data={key:'one',job_id:'abc',frame_range:added.frame_range,output_path:'C:\\cache.mkv',status:'running',progress:25,ready:false,
          original_offset_seconds:added.frame_range.start_frame/30,duration_seconds:(added.frame_range.end_frame-added.frame_range.start_frame)/30};segments=[data]
      }
      await route.fulfill({json:data,headers:{'Access-Control-Allow-Origin':'*'}})
    })
    await page.goto(`http://127.0.0.1:${server.address().port}`)
    await page.getByLabel('RTX VSR 倍率预设',{exact:true}).waitFor()
    await page.getByRole('button',{name:'收起 RTX VSR 参数',exact:true}).click()
    assert.equal(await page.getByLabel('RTX VSR 倍率预设',{exact:true}).isVisible(),false)
    await page.getByRole('button',{name:'展开 RTX VSR 参数',exact:true}).click()
    await page.getByRole('button',{name:'保存预设',exact:true}).click()
    await page.getByLabel('保存预设名称').fill('工作台测试')
    await page.locator('.preset-save-form').getByRole('button',{name:'保存',exact:true}).click()
    await page.getByRole('button',{name:'预设管理',exact:true}).click()
    await page.locator('.preset-entry summary').click()
    assert.match(await page.locator('.preset-entry table').innerText(),/渲染顺序/)
    await page.screenshot({path:path.join(out,'presets.png')})
    await page.getByRole('button',{name:'重建工作台',exact:true}).click()
    await page.getByLabel('选区导出',{exact:true}).check()
    await page.getByLabel('起始帧',{exact:true}).fill('91')
    await page.getByLabel('结束帧',{exact:true}).click()
    await page.getByLabel('结束帧',{exact:true}).fill('150')
    await page.getByRole('button',{name:'预渲染选区',exact:true}).click()
    await page.waitForFunction(()=>document.body.textContent.includes('预渲染 25%'),{},{timeout:4000})
    assert.deepEqual(added.frame_range,{start_frame:90,end_frame:150})
    assert.equal(await page.locator('.segment-cached').count(),0)
    segments=[{...segments[0],ready:true,status:'completed',progress:100}]
    await page.getByRole('button',{name:'回看结果',exact:true}).waitFor({timeout:8000})
    assert.equal(await page.locator('.segment-cached').count(),1)
    await page.screenshot({path:path.join(out,'workbench.png')})
    await page.getByLabel('选择工作台预设',{exact:true}).selectOption('__default__')
    await page.waitForFunction(()=>!document.querySelector('.segment-cached'))
    assert.equal(errors.length,0,errors.join('\n'))
    fs.writeFileSync(path.join(out,'ui-proof.json'),JSON.stringify({passed:true,checks:['preset save and default diff','stage folding','range request','no green for unfinished cache','green only completed','settings invalidate cache'],errors},null,2))
    console.log('PASS preset/selection UI')
  }finally{await browser.close();await new Promise(resolve=>server.close(resolve))}
})().catch(error=>{console.error(error);process.exitCode=1})
