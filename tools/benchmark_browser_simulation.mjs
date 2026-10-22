#!/usr/bin/env node
// CPU compute benchmark. Rendering, transfer and native/GPU engines are excluded.
// A baseline Git revision runs in separate processes in ABBA order. Every
// snapshot is checked at full Number precision, outside the timed region.
import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {execFileSync} from "node:child_process";
import {existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import {basename, dirname, join, resolve} from "node:path";
import {performance} from "node:perf_hooks";
import {fileURLToPath, pathToFileURL} from "node:url";
import {deserialize, serialize} from "node:v8";
import {gunzipSync} from "node:zlib";

const script=fileURLToPath(import.meta.url),root=resolve(dirname(script),"..");
const settings=[0,2000,4000,6000,8000,12409].flatMap(epoch=>[{epoch},{epoch,antenna:20},{epoch,mask:30}]);
const sha=bytes=>createHash("sha256").update(bytes).digest("hex");
const safePath=value=>{
  const path=resolve(value);
  if(path.split(/[\\/]/).some(part=>part.toLowerCase()===".claude"))throw new Error("Excluded path");
  return path;
};
const median=values=>{
  const sorted=[...values].sort((a,b)=>a-b),i=Math.floor(sorted.length/2);
  return sorted.length%2?sorted[i]:(sorted[i-1]+sorted[i])/2;
};

async function worker(config) {
  const [{CitySimulation},{decodeMesh}]=await Promise.all([
    import(pathToFileURL(config.module)),import(pathToFileURL(join(dirname(config.module),"mesh.mjs"))),
  ]);
  const scene=JSON.parse(gunzipSync(readFileSync(config.scene)));
  const bytes=gunzipSync(readFileSync(config.mesh));
  const geometry=decodeMesh(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength));
  const started=performance.now(),simulation=new CitySimulation(scene,geometry),indexMs=performance.now()-started;
  let expected;
  if(existsSync(config.expected))expected=deserialize(readFileSync(config.expected));
  else {
    expected=settings.map(value=>simulation.compute(value));
    writeFileSync(config.expected,serialize(expected)); // Retains signed zero.
  }
  const times=settings.map(()=>[]);let checks=0;
  for(let round=-2;round<config.rounds;round++)for(let step=0;step<settings.length;step++){
    const i=(step+round+2)%settings.length,start=performance.now(),actual=simulation.compute(settings[i]),milliseconds=performance.now()-start;
    assert.deepStrictEqual(actual,expected[i],JSON.stringify(settings[i]));
    checks++;
    if(round>=0)times[i].push(milliseconds);
  }
  process.stdout.write(JSON.stringify({indexMs,times,checks,reflectionBackend:simulation.bvh.reflectionBackend??"js",reflectionSimd:simulation.bvh.reflectionKernel?.simd??false,kernelScans:simulation.bvh.reflectionKernel?.scans??0,kernelFallbacks:simulation.bvh.reflectionKernel?.fallbacks??0}));
}

async function main() {
  const args=process.argv.slice(2);
  if(args[0]==="--worker"){
    if(args.length!==2)throw new Error("Invalid worker configuration");
    await worker(JSON.parse(readFileSync(safePath(args[1]),"utf8")));return;
  }
  if(args.includes("--help")){
    console.log("Usage: node tools/benchmark_browser_simulation.mjs [--baseline GIT_REF] [--module CITY_FILE] [--rounds N]");
    console.log("Default: 5 rounds over 18 Odaiba settings, after 2 warmup rounds.");
    console.log("With --baseline, source files are read from Git without changing the checkout.");
    console.log("Both versions use the current full-resolution assets. Output is JSON; no timing threshold is enforced.");
    return;
  }
  let baseline=null,module=join(root,"docs/assets/simulation/city.mjs"),rounds=5;
  for(let i=0;i<args.length;i++){
    const key=args[i],value=args[++i];
    if(value===undefined)throw new Error("Missing value for "+key);
    if(key==="--baseline")baseline=value;
    else if(key==="--module")module=safePath(value);
    else if(key==="--rounds")rounds=Number(value);
    else throw new Error("Unknown option: "+key);
  }
  if(!Number.isInteger(rounds)||rounds<1||rounds>50)throw new Error("Rounds must be an integer from 1 to 50");
  module=safePath(module);
  const folder=join(root,"docs/assets/plateau"),manifest=JSON.parse(readFileSync(join(folder,"manifest.json"),"utf8"));
  for(const [kind,name] of [["mesh","odaiba-mesh.bin.gz"],["scene","odaiba-scene.json.gz"]]){
    assert.equal(manifest[kind].file,name);assert.equal(sha(readFileSync(join(folder,name))),manifest[kind].sha256);
  }
  const temporaryBase=realpathSync(safePath(tmpdir())),scratch=mkdtempSync(join(temporaryBase,"gnss-browser-bench-"));
  try {
    let revision=null,baselineModule=null;
    if(baseline){
      if(baseline.startsWith("-"))throw new Error("Invalid baseline revision");
      revision=execFileSync("git",["rev-parse","--verify","--end-of-options",baseline+"^{commit}"],{cwd:root,encoding:"utf8",windowsHide:true}).trim();
      assert.match(revision,/^[0-9a-f]{40,64}$/);
      const directory=join(scratch,"baseline");mkdirSync(directory);
      const files=execFileSync("git",["ls-tree","--name-only",revision+":docs/assets/simulation"],{cwd:root,encoding:"utf8",windowsHide:true}).trim().split(/\r?\n/).filter(name=>name.endsWith(".mjs")&&!name.includes("/")&&!name.includes("\\"));
      for(const name of files){
        writeFileSync(join(directory,name),execFileSync("git",["show",revision+":docs/assets/simulation/"+name],{cwd:root,maxBuffer:2*1024*1024,windowsHide:true}));
      }
      baselineModule=join(directory,"city.mjs");
    }
    const order=baseline?["baseline","current","current","baseline"]:["current"];
    const runs={baseline:[],current:[]},expected=join(scratch,"expected.bin");
    for(let i=0;i<order.length;i++){
      const label=order[i],config={module:label==="baseline"?baselineModule:module,scene:join(folder,"odaiba-scene.json.gz"),mesh:join(folder,"odaiba-mesh.bin.gz"),rounds,expected};
      const file=join(scratch,"worker-"+i+".json");writeFileSync(file,JSON.stringify(config));
      console.error("Measuring "+label+" ("+(i+1)+"/"+order.length+")");
      const result=execFileSync(process.execPath,[script,"--worker",file],{encoding:"utf8",stdio:["ignore","pipe","inherit"],maxBuffer:2*1024*1024,windowsHide:true});
      runs[label].push(JSON.parse(result));
    }
    const cases=settings.map((value,i)=>{
      const currentMs=median(runs.current.flatMap(run=>run.times[i]));
      const baselineMs=baseline?median(runs.baseline.flatMap(run=>run.times[i])):null;
      return {settings:value,baselineMs,currentMs,speedup:baseline?baselineMs/currentMs:null};
    });
    const currentTotalMs=cases.reduce((sum,item)=>sum+item.currentMs,0),baselineTotalMs=baseline?cases.reduce((sum,item)=>sum+item.baselineMs,0):null;
    const sources=Object.fromEntries(readdirSync(dirname(module)).filter(name=>name.endsWith(".mjs")).sort().map(name=>[name,sha(readFileSync(join(dirname(module),name)))]));
    const report={
      schema:2,engine:process.version,scope:"CPU CitySimulation.compute only; excludes rendering, transfer and native/GPU engines",
      baselineRevision:revision,currentSources:sources,meshSha256:manifest.mesh.sha256,sceneSha256:manifest.scene.sha256,
      roundsPerProcess:rounds,warmupRounds:2,processOrder:order,cases:cases.length,
      exactReference:baseline?"baseline snapshot including signed zero":"current snapshot (determinism only)",
      exactChecks:Object.values(runs).flat().reduce((sum,run)=>sum+run.checks,0),
      baselineTotalMs,currentTotalMs,speedup:baseline?baselineTotalMs/currentTotalMs:null,
      timeReductionPercent:baseline?100*(1-currentTotalMs/baselineTotalMs):null,
      backends:Object.fromEntries(Object.entries(runs).filter(([,value])=>value.length).map(([key,value])=>[key,value.map(({reflectionBackend,reflectionSimd,kernelScans,kernelFallbacks})=>({reflectionBackend,reflectionSimd,kernelScans,kernelFallbacks}))])),
      indexMs:Object.fromEntries(Object.entries(runs).filter(([,value])=>value.length).map(([key,value])=>[key,value.map(run=>run.indexMs)])),
      results:cases,
    };
    process.stdout.write(JSON.stringify(report,null,2)+"\n");
  } finally {
    // Remove only the fresh temporary directory created by this invocation.
    const target=realpathSync(scratch);
    if(dirname(target)!==temporaryBase||!basename(target).startsWith("gnss-browser-bench-"))throw new Error("Temporary directory guard failed");
    rmSync(target,{recursive:true,force:true,maxRetries:3,retryDelay:50});
  }
}
main().catch(error=>{console.error(error);process.exitCode=1;});
