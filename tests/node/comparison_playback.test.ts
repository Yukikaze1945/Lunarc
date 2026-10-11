import {test} from 'node:test'
import assert from 'node:assert/strict'
import {decodeComparisonPacket} from '../../frontend/src/services/comparisonPlayback.ts'

function packet(header:any,a=new Uint8Array([255,216,1]),b=new Uint8Array([255,216,2])) {
  const text=new TextEncoder().encode(JSON.stringify(header))
  const output=new Uint8Array(4+text.length+a.length+b.length)
  new DataView(output.buffer).setUint32(0,text.length)
  output.set(text,4);output.set(a,4+text.length);output.set(b,4+text.length+a.length)
  return output.buffer
}
const header={time:1,original:{size:3,actualTime:1,actualPts:3},exported:{size:3,actualTime:.999,actualPts:5.999}}
test('both display frames retain their own actual timestamps on a shared clock',async()=>{
  const pair=decodeComparisonPacket(packet(header))
  assert.equal(pair.time,1);assert.equal(pair.original.actualPts,3)
  assert.equal(pair.exported.actualTime,.999)
  assert.deepEqual([...new Uint8Array(await pair.original.blob.arrayBuffer())],[255,216,1])
  assert.deepEqual([...new Uint8Array(await pair.exported.blob.arrayBuffer())],[255,216,2])
})
test('bad frame sizes and incomplete packets are refused before display',()=>{
  assert.throws(()=>decodeComparisonPacket(new ArrayBuffer(3)))
  assert.throws(()=>decodeComparisonPacket(packet({...header,original:{...header.original,size:4}})))
  assert.throws(()=>decodeComparisonPacket(packet({...header,exported:{...header.exported,actualPts:null}})))
})
