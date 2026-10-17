// Isolated settings panel; all backend calls are mocked. Never starts Electron or a GPU backend.
const path=require('node:path');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const {pathToFileURL}=require('node:url');
const {createRequire}=require('node:module');
const root=path.resolve(__dirname,'../..');
const frontend=path.join(root,'frontend');
const req=createRequire(path.join(frontend,'package.json'));
const {chromium}=req('playwright');
(async()=>{
  const {createServer}=await import(pathToFileURL(req.resolve('vite')).href);
  const {default:react}=await import(pathToFileURL(req.resolve('@vitejs/plugin-react')).href);
  const {default:tailwind}=await import(pathToFileURL(req.resolve('@tailwindcss/vite')).href);
  const server=await createServer({root:frontend,configFile:false,plugins:[react(),tailwind(),{
    name:'mcp-panel-fixture',configureServer(vite){vite.middlewares.use('/__mcp-preview',async(request,response)=>{
      const html=await vite.transformIndexHtml('/__mcp-preview',`<!doctype html><html><head></head><body>
        <div class="workspace-page" style="max-width:1100px;margin:40px auto"><div id="root"></div></div>
        <script type="module">import React from 'react';import {createRoot} from 'react-dom/client';
        import {McpConnectionPanel} from '/src/components/McpConnectionPanel.tsx';
        import '/src/index.css';import '/src/App.css';
        createRoot(document.getElementById('root')).render(React.createElement(McpConnectionPanel));</script>
        </body></html>`);
      response.setHeader('Content-Type','text/html');response.end(html);
    })}}],server:{host:'127.0.0.1',port:0}});
  let browser;
  try{
    await server.listen();
    const origin=`http://127.0.0.1:${server.httpServer.address().port}`;
    browser=await chromium.launch({headless:true,args:['--disable-gpu']});
    const context=await browser.newContext({locale:'zh-CN',viewport:{width:1280,height:800}});
    await context.grantPermissions(['clipboard-read','clipboard-write'],{origin});
    const page=await context.newPage();let enabled=false,fail=false,requests=0;
    const errors=[];page.on('pageerror',error=>errors.push(String(error)));
    await page.addInitScript(()=>localStorage.setItem('dlss_local_token','test-session'));
    await page.route('http://127.0.0.1:18923/**',async route=>{
      assert.equal(new URL(route.request().url()).pathname,'/api/mcp');
      const cors={'Access-Control-Allow-Origin':origin,'Access-Control-Allow-Headers':'Content-Type,X-DLSS-Token',
        'Access-Control-Allow-Methods':'GET,POST,OPTIONS'};
      if(route.request().method()==='OPTIONS')return route.fulfill({status:204,headers:cors});
      assert.equal(route.request().headers()['x-dlss-token'],'test-session');requests++;
      if(route.request().method()==='POST'){
        if(fail)return route.fulfill({status:409,headers:cors,json:{detail:'permission save failed'}});
        enabled=route.request().postDataJSON().enabled;
      }
      return route.fulfill({headers:cors,json:{enabled,transport:'streamable-http',url:'http://127.0.0.1:18923/mcp',
        prompt:'Add a Streamable HTTP MCP server named "lunarc" at http://127.0.0.1:18923/mcp to my user-level MCP configuration on this PC. If you cannot configure MCP servers directly, show me where to add it manually.',
        config:{mcpServers:{lunarc:{url:'http://127.0.0.1:18923/mcp'}}}}});
    });
    await page.goto(origin+'/__mcp-preview');
    const toggle=page.getByRole('switch',{name:'允许 AI 智能体控制 Lunarc'});
    await toggle.waitFor();await page.waitForFunction(()=>!document.querySelector('[role=switch]').disabled);
    assert(await page.getByRole('button',{name:'复制提示词',exact:true}).isDisabled());
    await toggle.click();await page.getByRole('status').filter({hasText:'AI 智能体控制已开启'}).waitFor();
    assert(enabled);
    await page.getByRole('button',{name:'复制提示词',exact:true}).click();
    await page.getByText('提示词已复制',{exact:true}).waitFor();
    const copiedPrompt=await page.evaluate(()=>navigator.clipboard.readText());
    assert(copiedPrompt.includes('Streamable HTTP'));
    assert(copiedPrompt.includes('http://127.0.0.1:18923/mcp'));
    await page.getByRole('button',{name:'查看工具',exact:true}).click();
    await page.locator('.mcp-tool-row').first().waitFor();
    assert.equal(await page.locator('.mcp-tool-row').count(),13);
    assert.equal(await page.locator('.mcp-tool-row code').allTextContents().then(names=>names.includes('render_video')),true);
    assert.equal(await page.locator('.mcp-tools-list').innerText().then(text=>text.includes('mcpServers')),false);
    const out=path.join(root,'runtime','validation','mcp-connection');fs.mkdirSync(out,{recursive:true});
    await page.screenshot({path:path.join(out,'settings.png')});
    fail=true;await toggle.click();await page.getByRole('alert').filter({hasText:'permission save failed'}).waitFor();
    assert.equal(await toggle.getAttribute('aria-checked'),'true');
    fail=false;await toggle.click();await page.getByRole('status').filter({hasText:'AI 智能体控制已关闭'}).waitFor();
    assert(!enabled);assert.deepEqual(errors,[]);assert.equal(requests,4);
    console.log('MCP settings UI passed: permission, short HTTP prompt, 13 tool descriptions and failed-save rollback; backend mocked.');
  }finally{if(browser)await browser.close();await server.close();}
})().catch(error=>{console.error(error);process.exitCode=1});
