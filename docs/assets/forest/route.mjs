import {forestReceiver,traceForest} from "./model.mjs?v=density-route-1";
import {DENSITY_PROFILES} from "./density.mjs?v=density-route-1";
import {forestSignals} from "./signals.mjs?v=density-route-1";
import {GroundReflectionModel} from "./ground-reflections.mjs?v=density-route-1";
import {generateObservations,solvePosition,METHODS} from "./positioning.mjs?v=density-route-1";
export function routeTimes(stepSeconds=300){
  if(![60,120,300].includes(stepSeconds))throw new RangeError("Route step must be 60, 120 or 300 seconds");
  return Array.from({length:1200/stepSeconds+1},(_,i)=>i*stepSeconds);
}
export const ROUTE_TIMES=Object.freeze(routeTimes());
export function routeExperiment({forest,orbitScene,context,seed=7,sigma=.3,mapBias=0,specificLoss=.2,foliage=1,stepSeconds=300},onProgress=()=>{}){
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff||!Number.isFinite(sigma)||sigma<.05||sigma>3||!Number.isFinite(mapBias)||Math.abs(mapBias)>3||!Number.isFinite(specificLoss)||specificLoss<0||specificLoss>2||!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid route experiment settings");
  const times=routeTimes(stepSeconds),model=new GroundReflectionModel(forest),cases=[];
  for(const [index,seconds]of times.entries()){
    onProgress({route:true,index,total:times.length,seconds,phase:"propagation"});
    const receiver=forestReceiver(seconds),signals=forestSignals(orbitScene,receiver,seconds),epochSeed=(seed+Math.imul(seconds,0x9e3779b9))>>>0;
    const generated=generateObservations(forest,receiver,signals,{seed:epochSeed,sigma}),counts=new Map(generated.truth.associations.map(a=>[a.id,a.available]));
    const channels=signals.map(signal=>{
      const maximum=Math.hypot(...signal.sourceENU.map((v,i)=>v-receiver[i])),trace=traceForest(forest,receiver,signal.directionENU,{specificLoss,foliage,maximum,densityProfile:"radial-height-v1"});
      const uniformOptical=trace.crowns.reduce((s,c)=>s+(c.exit-c.enter)*c.density,0)*foliage,uniformLoss=specificLoss*uniformOptical,structuredLoss=trace.canopyLoss,blocked=trace.ground!==null||trace.wood.length>0;
      const feasiblePaths=signal.id.startsWith("G")?(counts.get(signal.id)??0):model.ranges(receiver,signal.sourceENU).length;
      return {id:signal.id,azimuth:signal.azimuth,elevation:signal.elevation,sourceENU:signal.sourceENU,directionENU:signal.directionENU,geometry:trace.classification,canopyPathM:trace.canopyPath,feasiblePaths,directBlocked:blocked,
        profiles:{uniform:{canopyLossDb:uniformLoss,directRelativeDb:blocked?null:-uniformLoss},"radial-height-v1":{canopyLossDb:structuredLoss,directRelativeDb:blocked?null:-structuredLoss}}};
    });
    const prior=[receiver[0]+7.37,receiver[1]-5.13,0];let positioning;
    if(generated.observations.length<3)positioning={status:"insufficient-observations",results:METHODS.map(method=>({method,status:"insufficient-observations",horizontalErrorM:null,clockErrorM:null}))};
    else{
      try{
        const solution=solvePosition(generated.observations,prior,forest,{mapBias,onProgress:p=>onProgress({route:true,index,total:times.length,seconds,phase:"positioning",...p})});
        positioning={status:"completed",...solution,results:solution.results.map(r=>({...r,horizontalErrorM:r.receiverENU?Math.hypot(r.receiverENU[0]-receiver[0],r.receiverENU[1]-receiver[1]):null,clockErrorM:r.clock===undefined?null:r.clock-generated.truth.clock}))};
      }catch(error){positioning={status:"error",error:error.message,results:METHODS.map(method=>({method,status:"error",horizontalErrorM:null,clockErrorM:null}))};}
    }
    cases.push({seconds,tow:orbitScene.startTow+seconds,receiverENU:receiver,epochSeed,channels,...generated,prior,positioning});
  }
  return {schema:"synthetic-forest-route-v1",simulationOnly:true,context:{week:orbitScene.week,startTow:orbitScene.startTow,source:context.source,anchorECEF:orbitScene.origin,rotation:orbitScene.rotation},scene:forest,
    settings:{seed,sigma,mapBias,specificLoss,foliage,elevationMaskDeg:10,times,stepSeconds,propagationSystems:"GECJR",positioningSystems:"G"},densityProfiles:DENSITY_PROFILES,cases,
    assumptions:{sources:"broadcast endpoints evaluated at direct flight time and held fixed; no reflected-flight-time orbit iteration",paths:"direct plus finite-terrain single specular ground paths; complete wood/terrain occlusion; finite synthetic patch only",density:"uncalibrated smooth radial/height surrogate; equal total density mass to uniform; analytic cubic chord integration; overlapping tree densities add",propagation:"both profiles share geometry and complete direct rays; no coherent RF, receiver tracking, C/N0 or calibrated wood power",positioning:"fixed-sigma, artificial single-arrival GPS baseline; RF attenuation is NOT used to alter ranges, sigma, tracking or satellite selection, so density-only changes cannot establish positioning gains",initialization:"synthetic truth-offset prior [+7.37,-5.13] m; height aided, bounded local search, no global guarantee",noise:"deterministic epoch-time-derived seeds; common epochs share random draws across resolutions; no temporal receiver-noise model",limitations:"matched-map inverse crime; failures and unavailable channels retained; no real-data accuracy claim"}};
}
