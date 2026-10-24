const {test}=require("node:test");
const assert=require("node:assert/strict");
const causes=import("../../docs/assets/forest/tree-causes.mjs");
const model=import("../../docs/assets/forest/model.mjs");
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-8*Math.max(1,Math.abs(b)),`${a} != ${b}`);

test("per-tree sums reproduce both attenuation models across species, profiles and seasons",async()=>{
  const {attributeForestTrees}=await causes,{generateForest,traceForest}=await model;
  for(const stand of ["broadleaf","conifer","mixed"])for(const season of ["leaf-on","leaf-off"])for(const densityProfile of ["uniform","radial-height-v1"])for(const attenuationModel of ["linear-v1","itu-extinction-v1"]){
    const forest=generateForest({count:6,stand,season}),tree=forest.trees[0],options={densityProfile,attenuationModel,foliage:attenuationModel==="linear-v1"?.7:1,specificLoss:.3};
    const trace=traceForest(forest,[tree.x-12,tree.y,tree.crown.center[2]],[1,0,0],options),saved=structuredClone(trace),result=attributeForestTrees([{name:"direct",trace}],options);
    close(result.totalCanopyLossDb,trace.canopyLoss);close(result.expectedCanopyLossDb,trace.canopyLoss);assert.deepEqual(trace,saved);
    assert.equal(result.woodIntersections,trace.wood.length);assert.ok(result.rows.every(r=>Number.isFinite(r.crownLossDb)&&r.crownLossDb>=0));
  }
});

test("overlapping crown lengths are unioned within a tree while their density losses add",async()=>{
  const {attributeForestTrees}=await causes;
  const trace={crowns:[{tree:"T1",enter:1,exit:5,density:1},{tree:"T1",enter:3,exit:7,density:2}],wood:[],canopyLoss:2.4};
  const result=attributeForestTrees([{name:"direct",trace}],{specificLoss:.2});
  close(result.rows[0].crownPathM,6);close(result.rows[0].crownLossDb,2.4);
});

test("two ground legs count distinct traversals and never assign a dB loss to wood",async()=>{
  const {attributeForestTrees}=await causes;
  const a={crowns:[{tree:"T1",enter:0,exit:2,density:1}],wood:[],canopyLoss:.4};
  const b={crowns:[{tree:"T1",enter:0,exit:3,density:1}],wood:[{tree:"T2",part:"branch",enter:4,exit:4.2}],canopyLoss:.6};
  const result=attributeForestTrees([{name:"receiver",trace:a},{name:"satellite",trace:b}],{});
  close(result.totalCanopyLossDb,1);close(result.rows[0].crownPathM,5);assert.deepEqual(result.rows[0].legs,["receiver","satellite"]);
  assert.equal(result.rows[1].crownLossDb,0);assert.equal(result.rows[1].wood.length,1);
});

test("unavailable ground paths remain unavailable, while direct blockage is retained",async()=>{
  const {selectedTreeCauses}=await causes;
  const signal={trace:{crowns:[],wood:[],canopyLoss:0,directRelativeDb:null}};
  const direct=selectedTreeCauses(signal,null,{},"direct"),ground=selectedTreeCauses(signal,null,{},"ground");
  assert.equal(direct.available,true);assert.equal(direct.directBlocked,true);assert.equal(ground.available,false);assert.deepEqual(ground.rows,[]);
  assert.throws(()=>selectedTreeCauses(signal,null,{},"bad"),RangeError);
});

test("invalid model settings fail rather than reporting NaN or negative attribution",async()=>{
  const {attributeForestTrees}=await causes;
  for(const options of [{specificLoss:-1},{specificLoss:NaN},{foliage:3},{densityProfile:"unknown"},{attenuationModel:"itu-extinction-v1",foliage:.5}])assert.throws(()=>attributeForestTrees([],options),RangeError);
  assert.throws(()=>attributeForestTrees([{name:"direct",trace:{crowns:[],wood:[],canopyLoss:1}}]),/completed trace/);
});
