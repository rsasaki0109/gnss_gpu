const {test}=require("node:test");
const assert=require("node:assert/strict");
const model=import("../../docs/assets/forest/model.mjs");
const appearance=import("../../docs/assets/forest/appearance.mjs");

test("woodland detail is deterministic and leaves RF inputs and traces untouched",async()=>{
  const {generateForest,forestReceiver,traceForest}=await model,{forestAppearance}=await appearance;
  const forest=generateForest({count:4,stand:"mixed"}),saved=structuredClone(forest),rx=forestReceiver(600),before=traceForest(forest,rx,[.3,.2,1]);
  const mesh=forestAppearance(forest);
  assert.deepEqual(mesh,forestAppearance(forest));assert.deepEqual(forest,saved);
  assert.deepEqual(traceForest(forest,rx,[.3,.2,1]),before);
  assert.ok(mesh.every(Number.isFinite));
});

test("leaf-off removes broadleaf cards but preserves evergreen appearance",async()=>{
  const {generateForest}=await model,{forestAppearance}=await appearance;
  const mesh=(stand,season)=>forestAppearance(generateForest({count:2,stand,season}));
  assert.ok(mesh("broadleaf","leaf-off").length<mesh("broadleaf","leaf-on").length);
  assert.deepEqual(mesh("conifer","leaf-off"),mesh("conifer","leaf-on"));
});

test("dense woodland buffers stay bounded with valid normals, materials and atlas coordinates",async()=>{
  const {generateForest}=await model,{forestAppearance,APPEARANCE_STRIDE:s}=await appearance;
  for(const stand of ["broadleaf","conifer","mixed"]){
    const mesh=forestAppearance(generateForest({count:128,stand}));
    assert.ok(mesh.length/s<800000);assert.ok(mesh.byteLength<44*1024*1024);
    for(let i=0;i<mesh.length;i+=s){
      assert.ok(Math.abs(Math.hypot(mesh[i+3],mesh[i+4],mesh[i+5])-1)<2e-6);
      assert.ok(mesh[i+12]>=0&&mesh[i+12]<=1&&mesh[i+13]>=0&&mesh[i+13]<=1);
    }
  }
});
