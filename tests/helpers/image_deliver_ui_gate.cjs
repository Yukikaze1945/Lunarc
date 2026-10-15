// Browser check with isolated API responses. Actual GPU output has a separate gate.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const {chromium}=require(path.join(root,'frontend/node_modules/playwright'));
const source=process.argv[2],out=path.join(root,'build/image-export-gate/ui');
fs.mkdirSync(out,{recursive:true});
(async()=>{
  const browser=await chromium.launch({headless:true});
  try{
    const page=await browser.newPage({viewport:{width:1440,height:1000}});
    const seeks=[];
    await page.routeWebSocket('ws://127.0.0.1:18923/**',ws=>{ws.onMessage(message=>{
      const command=JSON.parse(message.toString());
      if(command.action==='seek'){
        seeks.push(command);
        ws.send(JSON.stringify({type:'frame',frame:0,request_id:command.request_id,
          image:'data:image/svg+xml;charset=utf-8,'+encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="2048" height="1024"><defs><linearGradient id="g" x2="0" y2="1"><stop stop-color="red"/><stop offset="1" stop-color="blue"/></linearGradient></defs><rect width="2048" height="1024" fill="url(#g)"/><rect x="1024" width="80" height="1024" fill="lime"/><rect width="80" height="1024" fill="orange"/></svg>'),preview:{settings_current:true}}));
      }
    });});
    let settings={render_pipeline:['nr','dlaa'],style:0,guidance_mode:0};
    const video={path:source,filename:path.basename(source),width:1536,height:2752,
      frames:1,fps:1,duration_seconds:1,media_type:'image',has_alpha:false};
    let exported=null,memoryPaused=false,releases=0;
    await page.route('http://127.0.0.1:18923/**',async route=>{
      const request=route.request(),url=new URL(request.url());
      const headers={'Access-Control-Allow-Origin':'*','Access-Control-Allow-Headers':'*','Access-Control-Allow-Methods':'*'};
      if(request.method()==='OPTIONS')return route.fulfill({status:204,headers});
      if(url.pathname==='/api/image/thumbnail')return route.fulfill({status:200,headers,contentType:'image/png',body:fs.readFileSync(source)});
      let data={};
      if(url.pathname==='/api/video/current')data={video,loaded:true};
      else if(url.pathname==='/api/settings'){if(request.method()==='POST'){settings={...settings,...request.postDataJSON()};data={settings,preview_revision:1}}else data=settings;}
      else if(url.pathname==='/api/queue')data={jobs:[],is_running:false,active_job_id:null};
      else if(url.pathname==='/api/system')data={};
      else if(url.pathname==='/api/preview/memory/release'){memoryPaused=true;releases++;data={suspended:true,releasing:false};}
      else if(url.pathname==='/api/preview/memory/resume'){memoryPaused=false;data={suspended:false,releasing:false};}
      else if(url.pathname==='/api/preview/memory')data={suspended:memoryPaused,releasing:false};
      else if(url.pathname==='/api/gpu/memory')data={memory:null};
      else if(url.pathname==='/api/image/models')data={busy:null,models:{animesharp:{installed:true},'illustration-denoise':{installed:true}}};
      else if(url.pathname==='/api/image/formats')data={formats:{png:true,jpeg:true,webp:true,avif:true,heic:true}};
      else if(url.pathname==='/api/image/export'){
        exported=request.postDataJSON();
        data={status:'ok',image:{path:exported.output_path,width:1536,height:2752,format:exported.format}};
      }
      await route.fulfill({status:200,headers,contentType:'application/json',body:JSON.stringify(data)});
    });
    await page.goto('http://127.0.0.1:5179');
    await page.getByRole('button',{name:'导出图片',exact:true}).click();
    await page.getByRole('heading',{name:'图片导出'}).waitFor();
    await page.getByRole('button',{name:'导出与队列',exact:true}).click();
    assert.equal(await page.locator('.image-deliver').isVisible(),false);
    await page.getByRole('button',{name:'图片导出',exact:true}).click();
    assert.equal(await page.locator('.image-deliver').isVisible(),true);
    await page.locator('.image-deliver-canvas canvas').waitFor();
    assert.equal(await page.getByRole('button',{name:'PNG',exact:true}).getAttribute('aria-pressed'),'true');
    await page.getByRole('button',{name:'JPEG',exact:true}).click();
    await page.getByRole('slider',{name:'图片质量'}).fill('88');
    assert.equal(await page.getByRole('checkbox',{name:'保留透明度'}).count(),0);
    await page.getByRole('button',{name:'PNG',exact:true}).click();
    await page.screenshot({path:path.join(out,'image-deliver.png'),fullPage:true});
    await page.getByRole('button',{name:'保存图片',exact:true}).click();
    await page.locator('.image-deliver-saved').waitFor();
    assert.equal(exported.format,'png');
    assert.equal(exported.path,source);
    assert.equal(exported.preserve_alpha,true);
    assert.ok(exported.output_path.endsWith('_lunarc.png'));
    for(const format of ['avif','heic']){
      await page.getByRole('button',{name:format.toUpperCase(),exact:true}).click();
      await page.getByRole('button',{name:'保存图片',exact:true}).click();
      await page.locator('.image-deliver-saved').waitFor();
      assert.equal(exported.format,format);assert.ok(exported.output_path.endsWith('.'+format));
    }
    const exportCanvas=await page.locator('.image-deliver-canvas').boundingBox();
    await page.mouse.move(exportCanvas.x+exportCanvas.width/2,exportCanvas.y+exportCanvas.height/2);
    await page.keyboard.down('Alt');await page.mouse.wheel(0,-600);await page.keyboard.up('Alt');
    await page.waitForFunction(()=>document.querySelector('.image-fit').textContent==='200%');
    const exportBefore=await page.locator('.image-deliver-canvas canvas').evaluate(c=>c.toDataURL());
    await page.mouse.down();await page.mouse.move(exportCanvas.x+exportCanvas.width/2,exportCanvas.y+100,{steps:12});await page.mouse.up();
    assert.notEqual(await page.locator('.image-deliver-canvas canvas').evaluate(c=>c.toDataURL()),exportBefore,'left drag moves the image');
    const exportDragged=await page.locator('.image-deliver-canvas canvas').evaluate(c=>c.toDataURL());
    await page.mouse.wheel(0,-300);await page.waitForTimeout(100);
    assert.notEqual(await page.locator('.image-deliver-canvas canvas').evaluate(c=>c.toDataURL()),exportDragged,'ordinary wheel scrolls a zoomed image');
    assert.equal(await page.locator('.image-fit').innerText(),'200%');
    await page.locator('.image-fit').click();assert.equal(await page.locator('.image-fit').innerText(),'100%');
    await page.getByRole('button',{name:'工作台',exact:true}).click();
    await page.locator('.dlss-monitor canvas').waitFor();await page.waitForTimeout(150);
    const canvas=await page.locator('.dlss-monitor').boundingBox(),before=seeks.length;
    assert.equal(seeks.at(-1).view,'dual');assert.equal(seeks.at(-1).zoom,1);assert.equal(seeks.at(-1).display_edge,0);
    await page.mouse.move(canvas.x+canvas.width*.7,canvas.y+canvas.height/2);
    await page.keyboard.down('Alt');await page.mouse.wheel(0,-1200);await page.keyboard.up('Alt');
    await page.waitForFunction(()=>Number(document.querySelector('[aria-label="预览缩放"]').value)===4);
    const snapshot=()=>page.locator('.dlss-monitor canvas').evaluate(c=>c.toDataURL());
    const zoomed=await snapshot();
    await page.mouse.down();await page.mouse.move(canvas.x+canvas.width*.7,canvas.y+100,{steps:12});await page.mouse.up();
    assert.notEqual(await snapshot(),zoomed,'workbench left drag moves vertically');
    const dragged=await snapshot();await page.mouse.wheel(0,-300);await page.waitForTimeout(100);
    assert.notEqual(await snapshot(),dragged,'workbench wheel pans vertically without zooming');
    assert.equal(await page.getByRole('combobox',{name:'预览缩放'}).inputValue(),'4');
    const marker=await page.locator('.dlss-monitor .split-marker').getAttribute('style');
    await page.mouse.move(canvas.x+canvas.width/2,canvas.y+canvas.height/2);await page.mouse.down();
    await page.mouse.move(canvas.x+canvas.width*.65,canvas.y+canvas.height/2);await page.mouse.up();
    assert.notEqual(await page.locator('.dlss-monitor .split-marker').getAttribute('style'),marker);
    await page.mouse.move(canvas.x+canvas.width*.8,canvas.y+canvas.height/2);await page.mouse.down();
    await page.mouse.move(canvas.x+canvas.width*.8+2500,canvas.y+canvas.height/2);await page.mouse.up();
    const atEdge=await snapshot();
    await page.mouse.move(canvas.x+canvas.width*.8,canvas.y+canvas.height/2);await page.mouse.down();
    await page.mouse.move(canvas.x+canvas.width*.8-50,canvas.y+canvas.height/2);await page.mouse.up();
    assert.notEqual(await snapshot(),atEdge,'panning reverses immediately from the image edge');
    await page.waitForTimeout(150);assert.equal(seeks.length,before,'image view gestures should never request a new render');
    await page.locator('[aria-label="添加图片放大模型"]').scrollIntoViewIfNeeded();
    const inspector=await page.locator('.quality-inspector').boundingBox();
    for(const button of await page.locator('.render-add button').all()){
      const bounds=await button.boundingBox();
      assert.ok(bounds.x>=inspector.x&&bounds.x+bounds.width<=inspector.x+inspector.width,'all add buttons must fit inside the inspector');
    }
    assert.equal(await page.locator('[aria-label="添加图片放大模型"] button').count(),2);
    await page.locator('.render-pipeline').screenshot({path:path.join(out,'render-models-visible.png')});
    await page.locator('.render-add button').filter({hasText:'RTX VSR'}).click();
    await page.locator('.render-add button').filter({hasText:'RTX VSR'}).click();
    await page.getByRole('combobox',{name:'RTX VSR 档位',exact:true}).nth(0).selectOption('1');
    await page.getByRole('combobox',{name:'RTX VSR 档位',exact:true}).nth(1).selectOption('4');
    assert.equal(await page.getByRole('combobox',{name:'RTX VSR 档位',exact:true}).nth(0).inputValue(),'1');
    assert.equal(await page.getByRole('combobox',{name:'RTX VSR 档位',exact:true}).nth(1).inputValue(),'4');
    await page.getByRole('button',{name:'停用 RTX VSR',exact:true}).nth(1).click();
    assert.equal(await page.getByRole('button',{name:'停用 RTX VSR',exact:true}).count(),1);
    await page.locator('.render-add button').filter({hasText:'IllustrationJaNai V3'}).click();
    await page.locator('.render-add button').filter({hasText:'AnimeSharp V4'}).click();
    assert.equal(await page.getByRole('combobox',{name:'IllustrationJaNai V3 版本'}).inputValue(),'denoise');
    assert.equal(await page.getByRole('combobox',{name:'图片模型处理方式',exact:true}).count(),2);
    const rows=await page.locator('.render-stage-head strong').allTextContents();
    assert.equal(rows.filter(n=>n==='RTX VSR').length,2);
    assert.ok(rows.includes('IllustrationJaNai V3')&&rows.includes('AnimeSharp V4'));
    await page.getByRole('button',{name:'释放预览显存',exact:true}).click();
    await page.getByRole('button',{name:'恢复预览',exact:true}).waitFor();
    assert.equal(releases,1);const pausedCount=seeks.length;
    await page.getByRole('combobox',{name:'RTX VSR 档位',exact:true}).nth(0).selectOption('2');
    await page.waitForTimeout(150);
    assert.equal(seeks.length,pausedCount,'parameter writes after manual release must not reload models');
    await page.getByRole('button',{name:'恢复预览',exact:true}).click();
    await page.getByRole('button',{name:'释放预览显存',exact:true}).waitFor();
    await page.waitForTimeout(100);
    assert.ok(seeks.length>pausedCount,'resume explicitly requests the current frame');
    fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,scope:'isolated UI and request body',exported},null,2));
    console.log('Image export page controls and save request passed');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
