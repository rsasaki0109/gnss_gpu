const test = require("node:test");
const assert = require("node:assert/strict");
const modulePromise = import("../../docs/assets/simulation/sky-plot.mjs");
const close = (actual,expected) => assert.ok(Math.abs(actual-expected)<1e-12,`${actual} != ${expected}`);

test("sky projection puts geodetic north up, east right and the zenith at the center",async()=>{
  const {skyPosition}=await modulePromise;
  for(const [azimuth,expected] of [[0,[0,-1]],[90,[1,0]],[180,[0,1]],[270,[-1,0]]])skyPosition(azimuth,0).forEach((v,i)=>close(v,expected[i]));
  for(const azimuth of [0,45,90,164.84832473851634,359.9])skyPosition(azimuth,90).forEach(v=>close(v,0));
});

test("radial distance represents zenith angle linearly and azimuth wraps without changing the point",async()=>{
  const {skyPosition}=await modulePromise;
  for(const elevation of [0,5,10,30,60,89.9,90]){
    const point=skyPosition(37,elevation);close(Math.hypot(...point),(90-elevation)/90);
    skyPosition(397,elevation).forEach((v,i)=>close(v,point[i]));skyPosition(-323,elevation).forEach((v,i)=>close(v,point[i]));
  }
});

test("invalid or non-finite directions fail rather than being clamped into a plausible sky",async()=>{
  const {skyPosition}=await modulePromise;
  for(const angles of [[NaN,10],[Infinity,10],[1,NaN],[1,Infinity],[0,-.001],[0,90.001]])assert.throws(()=>skyPosition(...angles),RangeError);
});

test("real GPS directions retain full precision, correct marker positions and stable non-overlapping labels",async()=>{
  const {skyLayout}=await modulePromise;
  const source=[{id:"G05",azimuth:137.460799211976,elevation:18.798666036484814,status:"LOS"},{id:"G10",azimuth:315.7179928416634,elevation:12.452388211398953,status:"BLOCKED"},{id:"G12",azimuth:164.84832473851634,elevation:14.665448266301897,status:"NLOS"},{id:"G13",azimuth:63.97761074979172,elevation:33.515611871435624,status:"LOS"},{id:"G15",azimuth:30.692736311260944,elevation:65.11905875426068,status:"LOS"},{id:"G20",azimuth:311.31286048025675,elevation:38.79655569594936,status:"BLOCKED"},{id:"G21",azimuth:260.389706563763,elevation:32.79132861559597,status:"BLOCKED"},{id:"G24",azimuth:236.9821360980908,elevation:78.63051536544495,status:"LOS"}];
  const before=structuredClone(source),layout=skyLayout(source);
  assert.deepEqual(source,before);assert.deepEqual(skyLayout(source.toReversed()),layout);
  for(let i=0;i<layout.length;i++){
    const point=layout[i],original=source.find(s=>s.id===point.id);
    assert.equal(point.azimuth,original.azimuth);assert.equal(point.elevation,original.elevation);
    close(Math.hypot(point.x-150,point.y-150),110*(90-original.elevation)/90);
    close((Math.atan2(point.x-150,150-point.y)*180/Math.PI+360)%360,original.azimuth);
    assert.ok(point.labelX>=32&&point.labelX<=268&&point.labelY>=32&&point.labelY<=268);
    for(const other of layout.slice(i+1))assert.ok(Math.abs(point.labelX-other.labelX)>=28||Math.abs(point.labelY-other.labelY)>=16,`${point.id} label overlaps ${other.id}`);
  }
});

test("crowded labels can move while coincident markers preserve the physical directions",async()=>{
  const {skyLayout}=await modulePromise,source=Array.from({length:8},(_,i)=>({id:`G0${i+1}`,azimuth:0,elevation:45,status:"LOS"})),layout=skyLayout(source);
  assert.ok(new Set(layout.map(p=>`${p.labelX},${p.labelY}`)).size>=4);
  for(const point of layout){close(point.x,150);close(point.y,95);assert.ok(Number.isFinite(point.labelX)&&Number.isFinite(point.labelY));}
  assert.deepEqual(skyLayout([]),[]);
});
