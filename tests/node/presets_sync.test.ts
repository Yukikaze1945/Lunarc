import test from 'node:test'
import assert from 'node:assert/strict'

test('preset migration, MCP refresh and targeted deletion preserve saved data',async()=>{
  const storage=new Map<string,string>([
    ['lunarc.presets.v1',JSON.stringify([{name:'Legacy',settings:{intensity:1.2}}])],
    ['dlss_local_token','test-session'],
  ])
  Object.assign(globalThis,{
    localStorage:{getItem:(key:string)=>storage.get(key)??null,setItem:(key:string,value:string)=>storage.set(key,value)},
    window:new EventTarget(),
  })
  let server:any[]=[{name:'From MCP',settings:{intensity:1.4}}],fail=false
  const originalFetch=globalThis.fetch
  globalThis.fetch=async(_url,options:any={})=>{
    assert.equal(options.headers['X-DLSS-Token'],'test-session')
    if(fail)return new Response(JSON.stringify({detail:'save failed'}),{status:409})
    const path=String(_url),body=options.body?JSON.parse(options.body):{}
    if(path.endsWith('/import'))for(const preset of body.presets)if(!server.some(p=>p.name===preset.name))server.push(preset)
    if(path.endsWith('/delete'))server=server.filter(p=>p.name!==body.name)
    if(path.endsWith('/save'))server=[...server.filter(p=>p.name!==body.name),{name:body.name,settings:body.settings}]
    return new Response(JSON.stringify({presets:server,revision:1}))
  }
  try{
    const presets=await import('../../frontend/src/services/presets.ts')
    await presets.initializePresets()
    assert.deepEqual(presets.presetSnapshot().map(p=>p.name),['From MCP','Legacy'])
    assert.ok(storage.get('lunarc.presets.migration-backup.v1')?.includes('Legacy'))
    // A new MCP preset created after the UI snapshot must survive deleting a visible row.
    server.push({name:'Created Later',settings:{intensity:.8}})
    await presets.deletePreset('Legacy')
    assert.deepEqual(presets.presetSnapshot().map(p=>p.name),['From MCP','Created Later'])
    await presets.importPresets([{name:'From MCP',settings:{intensity:.1} as any}])
    assert.equal(presets.presetSnapshot().find(p=>p.name==='From MCP')?.settings.intensity,1.4)
    const before=storage.get('lunarc.presets.v1')
    fail=true
    await assert.rejects(presets.savePreset('Rejected',presets.DEFAULT_PRESET),/save failed/)
    assert.equal(storage.get('lunarc.presets.v1'),before)
  }finally{globalThis.fetch=originalFetch}
})
