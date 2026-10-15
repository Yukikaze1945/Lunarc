const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const {createRequire} = require('node:module')
const ts = require('../../frontend/node_modules/typescript')
const filename = path.resolve(__dirname, '../../frontend/src/i18n/translate.ts')
const exportsObject = {}
const compiled = ts.transpileModule(fs.readFileSync(filename,'utf8'), {
  compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022,esModuleInterop:true},
}).outputText
new Function('require','exports',compiled)(createRequire(filename),exportsObject)
const {translateText,translateVisible,isLocale} = exportsObject
const catalogs = ['en','ja','ru'].map(locale=>require(`../../frontend/src/i18n/${locale}.json`))

test('all catalogs cover the same keys and preserve template arguments',()=>{
  const keys=Object.keys(catalogs[0]).sort()
  assert.ok(keys.length>1200)
  for(const catalog of catalogs){
    assert.deepEqual(Object.keys(catalog).sort(),keys)
    for(const [source,target] of Object.entries(catalog)){
      assert.ok(target.trim(),source)
      assert.ok(!target.includes('\ufeff'),source)
      const slots=text=>[...new Set(text.match(/\{\d+\}/g)||[])].sort()
      assert.deepEqual(slots(target),slots(source),source)
    }
  }
})
test('locale validation and Chinese fallback',()=>{
  for(const locale of ['zh-CN','en','ja','ru'])assert.ok(isLocale(locale))
  for(const locale of [undefined,null,{},'de','en-US','__proto__'])assert.ok(!isLocale(locale))
  assert.equal(translateText('渲染顺序','zh-CN'),'渲染顺序')
  assert.equal(translateText('unknown diagnostics 0x887a0005','en'),'unknown diagnostics 0x887a0005')
})
test('exact labels and runtime templates, including punctuation and multiple slots',()=>{
  assert.equal(translateText('渲染顺序','en'),'Processing order')
  assert.equal(translateText('画面风格','ja'),'画面スタイル')
  assert.equal(translateText('软件设置','ru'),'Настройки')
  assert.equal(translateText('第 2 层','en'),'Layer 2')
  assert.equal(translateText('请求失败 (400)','en'),'Request failed (400)')
  assert.equal(translateText('预览失败：GPU device lost','en'),'Preview failed: GPU device lost')
  assert.equal(translateText('实际尺寸 1920×1080 · 处理 30 FPS · 播放 25 FPS · 跳过 2 帧','en'),
    'Actual 1920×1080 · Processing 30 FPS · Playback 25 FPS · Skipped 2 frames')
})
test('visible translation leaves settings objects, paths and technical values intact',()=>{
  const settings={style:0,codec:'av1_nvenc',render_pipeline:['nr','truehdr']}
  assert.equal(translateVisible(settings,'en'),settings)
  for(const value of ['C:\\素材\\第 2 层.mkv','\\\\NAS\\素材\\预览失败：a.mkv','https://example.com/渲染顺序','av1_nvenc','Main10'])
    assert.equal(translateText(value,'en'),value)
  assert.deepEqual(translateVisible(['导出',42,null],'en'),['Export',42,null])
})
