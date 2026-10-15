const {chromium} = require('../../frontend/node_modules/playwright-core')
const fs = require('node:fs')
const path = require('node:path')
const http = require('node:http')
const assert = require('node:assert/strict')
const root=path.resolve(__dirname,'../../frontend/dist')
const video={path:'D:\\素材\\第 2 层.mkv',filename:'第 2 层.mkv',width:1920,height:1080,
  fps:24,frames:240,duration_seconds:10,pix_fmt:'yuv420p',bit_depth:8,is_hdr:false,media_type:'video'}
const settings={preset:1,style:0,intensity:1,local_tone:1,local_struct:1,skin_struct:1,
  working_scale:1,reconstruction_mode:'single',guidance_mode:1,flow_engine:'nvof',
  flow_quality:'1F',render_pipeline:['nr','dlaa','vsr'],experimental_hdr_scrgb:false}
const mime={'.js':'application/javascript','.css':'text/css','.html':'text/html','.svg':'image/svg+xml'}
const server=http.createServer((req,res)=>{
  const requested=path.resolve(root,'.'+new URL(req.url,'http://localhost').pathname)
  const file=requested.startsWith(root+path.sep)&&fs.existsSync(requested)&&fs.statSync(requested).isFile()?requested:path.join(root,'index.html')
  res.setHeader('Content-Type',mime[path.extname(file)]||'application/octet-stream');res.end(fs.readFileSync(file))
})
async function run(){
  fs.mkdirSync(path.resolve(__dirname,'../../build'),{recursive:true})
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve))
  const browser=await chromium.launch({headless:true})
  try{
    const context=await browser.newContext({viewport:{width:1600,height:1000}})
    const page=await context.newPage(),errors=[],writes=[]
    page.on('pageerror',error=>errors.push(error.message))
    await page.addInitScript(()=>{
      window.__languageCalls=[]
      window.electronAPI={isElectron:true,getLanguage:async()=> 'zh-CN',
        setLanguage:async language=>window.__languageCalls.push(language),
        getComponents:async()=>location.search.includes('setup=1')?{requiresSetup:true,components:[],busy:null,phase:''}:null,onComponents:()=>()=>{},getBackendStartup:async()=>null}
    })
    await page.route('http://127.0.0.1:18923/**',async route=>{
      const req=route.request(),url=new URL(req.url());let result={}
      if(url.pathname==='/api/settings'){
        if(req.method()==='POST'){writes.push(req.postDataJSON());Object.assign(settings,req.postDataJSON())}
        result=settings
      }else if(url.pathname==='/api/video/current')result={video}
      else if(url.pathname==='/api/system')result={gpu:'Mock GPU',backend_ready:true,vram_total_gb:16}
      else if(url.pathname==='/api/queue')result={jobs:[],active_id:null,queue_running:false}
      else if(url.pathname==='/api/flow/providers')result={providers:[],provider_contracts:{}}
      else if(url.pathname.startsWith('/api/runtime/dlssnr'))result={active:null,versions:[],sources:[],catalog:[],cache_dir:'D:\\cache'}
      else if(url.pathname==='/api/preview/memory')result={suspended:true,releasing:false}
      else if(url.pathname.startsWith('/api/diagnostics/logs'))result={logs:[]}
      else if(url.pathname.startsWith('/api/presets'))result={presets:[]}
      else if(url.pathname==='/api/image/models')result={models:[]}
      else if(url.pathname==='/api/environment')result={checks:[],ready:true}
      await route.fulfill({status:200,contentType:'application/json',headers:{'Access-Control-Allow-Origin':'*'},body:JSON.stringify(result)})
    })
    await page.goto(`http://127.0.0.1:${server.address().port}`)
    await page.getByRole('button',{name:'导出与队列',exact:true}).click()
    const name=page.getByPlaceholder('输出视频文件名',{exact:true})
    await name.fill('my-output-unchanged')
    const snapshot=JSON.stringify(settings),writeCount=writes.length
    const cases=[['en','Export & queue','Output video file name','Preferences','Software updates'],
      ['ja','書き出しとキュー','出力動画のファイル名','ソフトウェア設定','ソフトウェア更新'],
      ['ru','Экспорт и очередь','Имя выходного видео','Настройки','Обновление программы'],
      ['zh-CN','导出与队列','输出视频文件名','软件设置','软件更新']]
    for(const [locale,nav,placeholder,preferences] of cases){
      await page.locator('.header-actions .language-picker select').selectOption(locale)
      await page.getByRole('button',{name:nav,exact:true}).waitFor()
      assert.equal(await page.getByPlaceholder(placeholder,{exact:true}).inputValue(),'my-output-unchanged')
      assert.equal(await page.locator('html').getAttribute('lang'),locale)
      await page.getByRole('button',{name:preferences,exact:true}).click()
      assert.equal(await page.locator('.settings-grid .language-picker select').inputValue(),locale)
      await page.getByRole('button',{name:nav,exact:true}).click()
      if(locale==='ru'||locale==='ja')await page.screenshot({path:path.resolve(__dirname,`../../build/localization-${locale}.png`)})
    }
    assert.equal(JSON.stringify(settings),snapshot,'language must not alter render settings')
    assert.equal(writes.length,writeCount,'language must not POST settings')
    await page.locator('.header-actions .language-picker select').selectOption('en')
    await page.getByRole('button',{name:'Reconstruction workbench',exact:true}).click()
    await page.locator('.qi-advanced > summary').click()
    await page.getByPlaceholder('Search settings',{exact:true}).fill('guidance')
    assert.ok(await page.getByRole('option',{name:/Guidance & optical flow/}).count())
    await page.getByRole('button',{name:'Export & queue',exact:true}).click()
    await page.setViewportSize({width:1000,height:800})
    await page.evaluate(()=>{document.documentElement.dataset.windowOverlay='true'})
    await page.locator('.header-actions .language-picker select').selectOption('ru')
    const tabs=await page.locator('.header-tabs').boundingBox()
    const actions=await page.locator('.header-actions').boundingBox()
    assert.ok(tabs.x+tabs.width<=actions.x,'translated header must not overlap at minimum window width')
    assert.ok(actions.x+actions.width<=1000-138,'header must leave native window controls accessible')
    await page.locator('.header-actions .language-picker select').selectOption('en')
    await page.setViewportSize({width:1600,height:1000})
    await page.reload()
    await page.getByRole('button',{name:'Export & queue',exact:true}).click()
    assert.equal(await page.locator('html').getAttribute('lang'),'en')
    await page.waitForFunction(name=>document.querySelector('.dlss-material strong')?.textContent===name,video.filename)
    assert.equal(await page.locator('.dlss-material strong').textContent(),video.filename)
    await page.screenshot({path:path.resolve(__dirname,'../../build/localization-en.png'),fullPage:true})
    await page.goto(`http://127.0.0.1:${server.address().port}/?setup=1`)
    await page.getByText('First use & component setup',{exact:true}).waitFor()
    await page.locator('.project-header .language-picker select').selectOption('ja')
    await page.getByText('初回使用とコンポーネント導入',{exact:true}).waitFor()
    assert.deepEqual(errors,[])
    console.log('PASS: four languages; persistent preference; memoized export view; preserved media/draft/settings; translated search; no page errors')
  }finally{await browser.close()}
}
run().catch(error=>{console.error(error);process.exitCode=1}).finally(()=>server.close())
