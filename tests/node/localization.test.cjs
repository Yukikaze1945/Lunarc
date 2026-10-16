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
const {translateText,translateVisible,isLocale,isLanguagePreference,resolveLocale} = exportsObject
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

test('automatic language resolves system preferences and keeps manual overrides',()=>{
  for(const [languages,expected] of [
    [['zh-CN'],'zh-CN'],[['zh-TW'],'zh-CN'],[['en-GB'],'en'],[['ja-JP'],'ja'],
    [['ru_RU'],'ru'],[['de-DE','ja-JP','en-US'],'ja'],[['de-DE'],'en'],[[],'en'],
  ])assert.equal(resolveLocale('auto',languages),expected)
  assert.equal(resolveLocale('ja',['en-US']),'ja')
  assert.ok(isLanguagePreference('auto'))
  assert.ok(!isLanguagePreference('de'))
})

function languageStore(saved,native){
  const filename=path.resolve(__dirname,'../../frontend/src/i18n/locale.ts')
  const source=ts.transpileModule(fs.readFileSync(filename,'utf8'),{
    compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022},
  }).outputText
  const values=new Map(saved===undefined?[]:[['lunarc.language.v1',saved]])
  const calls=[],events={},exports={}
  const navigator={languages:['en-US'],language:'en-US'}
  const window={electronAPI:{getLanguage:async()=>native,setLanguage:async value=>calls.push(value)},
    addEventListener:(name,callback)=>{events[name]=callback}}
  const storage={getItem:key=>values.get(key)??null,setItem:(key,value)=>values.set(key,value)}
  const document={documentElement:{dataset:{},lang:''}}
  new Function('require','exports','localStorage','navigator','window','document',source)(
    name=>name==='react'?{useSyncExternalStore:()=>{}}:exportsObject,exports,storage,navigator,window,document)
  return {exports,values,calls,events,navigator,document}
}

test('fresh installs detect native system language without saving a fixed locale',async()=>{
  const store=languageStore(undefined,{preference:'auto',systemLanguages:['ja-JP']})
  await store.exports.initializeLocale()
  assert.equal(store.exports.getLocale(),'ja')
  assert.equal(store.exports.getLanguagePreference(),'auto')
  assert.equal(store.values.has('lunarc.language.v1'),false)
  assert.deepEqual(store.calls,['auto'])
  store.exports.setLocale('ru')
  assert.equal(store.exports.getLocale(),'ru')
  assert.equal(store.values.get('lunarc.language.v1'),'ru')
  store.exports.setLocale('auto')
  assert.equal(store.exports.getLocale(),'ja')
  assert.equal(store.values.get('lunarc.language.v1'),'auto')
  store.navigator.languages=['zh-CN'];store.events.languagechange()
  assert.equal(store.exports.getLocale(),'zh-CN')
})

test('saved choices override detection and auto stays auto after restart',async()=>{
  for(const [stored,native,expected,preference] of [
    ['ru',{preference:'auto',systemLanguages:['ja-JP']},'ru','ru'],
    [undefined,{preference:'ja',systemLanguages:['en-US']},'ja','ja'],
    ['auto',{preference:'ru',systemLanguages:['en-US']},'en','auto'],
    ['auto',{preference:'auto',systemLanguages:['ru-RU']},'ru','auto'],
    ['invalid',{preference:'auto',systemLanguages:['de-DE']},'en','auto'],
  ]){
    const store=languageStore(stored,native)
    await store.exports.initializeLocale()
    assert.equal(store.exports.getLocale(),expected)
    assert.equal(store.exports.getLanguagePreference(),preference)
  }
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
