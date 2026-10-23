// Small deterministic synthetic campaign; no real-data accuracy claim.
// Run: node tools/benchmark_forest_positioning.mjs > report.json
import {readFileSync} from "node:fs";
import {gunzipSync} from "node:zlib";
import {createHash} from "node:crypto";
import {generateForest,forestReceiver} from "../docs/assets/forest/model.mjs";
import {generateObservations,solvePosition} from "../docs/assets/forest/positioning.mjs";
import {selectNavigation,receivedSatellite,geodetic,enuRotation,dot,sub,norm} from "../docs/assets/simulation/orbit.mjs";
const base=new URL("../docs/assets/plateau/",import.meta.url),manifest=JSON.parse(readFileSync(new URL("manifest.json",base))),bytes=readFileSync(new URL(manifest.scene.file,base));
const sha256=createHash("sha256").update(bytes).digest("hex");
if(bytes.length!==manifest.scene.bytes||sha256!==manifest.scene.sha256)throw new Error("Orbit source checksum mismatch");
const scene=JSON.parse(gunzipSync(bytes)),forest=generateForest(),cases=[];
for(const seconds of [0,300,600,900]){
  const receiver=forestReceiver(seconds),ecef=scene.origin.map((v,i)=>v+scene.rotation.reduce((sum,axis,j)=>sum+axis[i]*receiver[j],0)),llh=geodetic(ecef),orientation=enuRotation(llh.lat,llh.lon),tow=scene.track[0][0]+seconds;
  const signals=[];
  for(const record of selectNavigation(scene.navigation.filter(r=>r.id[0]==="G"),tow,scene.week)){
    const state=receivedSatellite(record,tow,ecef,scene.week),delta=sub(state.position,ecef),elevation=Math.asin(dot(orientation[2],delta)/norm(delta))*180/Math.PI;
    if(elevation>=10)signals.push({id:record.id,sourceENU:scene.rotation.map(axis=>dot(axis,sub(state.position,scene.origin)))});
  }
  for(const seed of [7,19]){
    const generated=generateObservations(forest,receiver,signals,{seed}),prior=[receiver[0]+7.37,receiver[1]-5.13,0];
    for(const mapBias of [0,1]){
      const start=performance.now();
      try{
        const solution=solvePosition(generated.observations,prior,forest,{mapBias});
        cases.push({seconds,seed,mapBias,observations:generated.observations.length,results:solution.results.map(r=>({...r,horizontalErrorM:r.receiverENU?Math.hypot(r.receiverENU[0]-receiver[0],r.receiverENU[1]-receiver[1]):null})),elapsedMs:performance.now()-start});
      }catch(error){cases.push({seconds,seed,mapBias,observations:generated.observations.length,error:error.message});}
    }
  }
}
console.log(JSON.stringify({schema:"forest-positioning-campaign-v1",source:{sha256,week:scene.week,startTow:scene.track[0][0]},assumptions:"Synthetic single-arrival ranges, 0.3 m Gaussian noise, known receiver height, perturbed prior, local search. Zero map bias is an inverse crime. No measured-data or GPU performance claim. All cases retained.",cases},null,2));
