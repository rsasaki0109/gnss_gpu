const test=require("node:test"),assert=require("node:assert/strict"),fs=require("node:fs");
const {gunzipSync}=require("node:zlib"),{createHash}=require("node:crypto");
const modules=import("../../docs/assets/simulation/recorded-strength.mjs");
const folder=__dirname+"/../../docs/assets/plateau/";
const fixture=()=>({version:1,receiver:"trimble",week:2032,unit:"dB-Hz",unitStatus:"inferred-rtkconv",cadence:.1,codes:{G:"S1C"},source:{sha256:"raw-sha"},signals:{G01:{code:"S1C",samples:[[100,40.125],[100.1,39.25],[100.301,38.5]]}}});
const pose=(tow,antenna=0,week=2032)=>({tow,antenna,week});

test("recorded validation preserves raw values and rejects wrong bands, reordered or fake zero samples",async()=>{
  const {validateRecorded}=await modules,data=fixture(),before=structuredClone(data);assert.equal(validateRecorded(data,"trimble"),data);assert.deepEqual(data,before);
  for(const change of [d=>d.signals.G01.code="S2W",d=>d.signals.G01.samples.reverse(),d=>d.signals.G01.samples[0][1]=0,d=>d.signals.G01.samples[0][1]=NaN,d=>d.unitStatus="calibrated",d=>d.receiver="ublox"]){const invalid=fixture();change(invalid);assert.throws(()=>validateRecorded(invalid,"trimble"));}
});
test("nearest observations use the original GPS clock within 2 ms without interpolation or stale carry",async()=>{
  const {nearestRecorded}=await modules,data=fixture();
  assert.equal(nearestRecorded(data,"G01",pose(100)).value,40.125);
  const before=structuredClone(data),near=nearestRecorded(data,"G01",pose(100.3));assert.equal(near.value,38.5);assert.ok(Math.abs(near.delta-.001)<1e-10);
  assert.equal(nearestRecorded(data,"G01",pose(100.2)),null);assert.equal(nearestRecorded(data,"G02",pose(100)),null);
  assert.equal(nearestRecorded(data,"G01",pose(100,1)),null);assert.throws(()=>nearestRecorded(data,"G01",pose(100),.1),RangeError);assert.deepEqual(data,before);
});
test("recorded window retains actual gap timestamps, uses scene-week offsets and rejects virtual heights",async()=>{
  const {recordedWindow,nearestRecorded}=await modules,data=fixture();
  assert.deepEqual(recordedWindow(data,"G01",pose(100.3)).samples,data.signals.G01.samples);
  assert.equal(recordedWindow(data,"G01",pose(100,20)),null);
  data.signals.G01.samples=[[604799.9,44],[604800.001,43]];
  assert.equal(nearestRecorded(data,"G01",pose(0,0,2033)).value,43);
  assert.equal(recordedWindow(data,"G01",pose(0,0,2033)).samples.length,2);
});
test("recorded CSV preserves code, raw precision, unit inference and source hash without modeled substitutions",async()=>{
  const {recordedCSV}=await modules,data=fixture();
  const csv=recordedCSV(data,"G01",pose(100.3));assert.ok(csv.includes("2032,100.301,trimble,G01,S1C,38.5,inferred-rtkconv,raw-sha"));
  assert.equal(recordedCSV(data,"G01",pose(100,20)).trim().split("\r\n").length,1);
});
test("packaged first-band observations match checksums, exact sample counts and the current scene",async()=>{
  const {validateRecorded}=await modules,m=JSON.parse(fs.readFileSync(folder+"strength-manifest.json")),scene=fs.readFileSync(folder+"odaiba-scene.json.gz");
  assert.equal(m.sceneSha256,createHash("sha256").update(scene).digest("hex"));
  for(const [receiver,entry]of Object.entries(m.receivers)){
    const bytes=fs.readFileSync(folder+entry.file);assert.equal(bytes.length,entry.bytes);assert.equal(createHash("sha256").update(bytes).digest("hex"),entry.sha256);
    const data=validateRecorded(JSON.parse(gunzipSync(bytes)),receiver);assert.equal(Object.values(data.signals).reduce((n,s)=>n+s.samples.length,0),entry.samples);
    assert.equal(data.unitStatus,"inferred-rtkconv");assert.equal(data.source.sha256,entry.source.sha256);
    assert.deepEqual([...new Set(Object.keys(data.signals).map(id=>id[0]))].sort(),["C","E","G","J","R"]);
    assert.equal(data.codes.C,"S1I");assert.equal(data.codes.E,receiver==="trimble"?"S1X":"S1C");
  }
});
