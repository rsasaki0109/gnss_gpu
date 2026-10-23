import {checkDensityProfile} from "./density.mjs?v=density-route-1";
// Finite, single-bounce terrain paths. No receiver/tracking or coherent RF model.
import { traceForest,terrainDistance,ForestOcclusion } from "./model.mjs?v=density-route-1";
import { C } from "../simulation/orbit.mjs?v=glonass-1";
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0),sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length=a=>Math.hypot(...a),finite=a=>Array.isArray(a)&&a.length===3&&a.every(Number.isFinite);
export const GROUND_MODEL_VERSION="finite-terrain-reflection-v1";
export const ENDPOINT_TOLERANCE_M=1e-6;

// Rebuild after changing forest geometry; generated forests are otherwise static.
export class GroundReflectionModel {
  constructor(forest){
    this.forest=forest;this.volumes={...forest,terrain:[]};this.occlusion=new ForestOcclusion(forest);
    this.planes=forest.terrain.flatMap(([a,b,c],triangle)=>{
      if(![a,b,c].every(finite))throw new RangeError("Invalid ground triangle");
      const e=sub(b,a),f=sub(c,a),raw=cross(e,f),size=length(raw);
      if(size<1e-12)return [];
      const n=raw.map(v=>v/size*(raw[2]<0?-1:1));
      if(n[2]<1e-12)return []; // Ground has an upward-facing reflecting side.
      return [{a,e,f,n,triangle,ee:dot(e,e),ef:dot(e,f),ff:dot(f,f),areaSquared:dot(raw,raw)}];
    });
  }

  paths(receiver,source,{specificLoss=.2,foliage=1,bounceLoss=6,densityProfile="uniform"}={}){
    return this.#query(receiver,source,{specificLoss,foliage,bounceLoss,densityProfile},true);
  }

  // Exact feasible ranges for a likelihood that does not consume canopy/RF
  // diagnostics. Both complete legs retain the same wood and terrain checks.
  ranges(receiver,source){
    const result=this.#query(receiver,source,{specificLoss:.2,foliage:1,bounceLoss:6},false),directDelta=sub(source,receiver);
    const direction=directDelta.map(v=>v/result.directRange),ranges=[];
    if(!this.occlusion.blocked(receiver,direction,result.directRange))ranges.push(result.directRange);
    return ranges.concat(result.paths.map(p=>p.totalRange));
  }

  #query(receiver,source,{specificLoss,foliage,bounceLoss,densityProfile="uniform"},details){
    checkDensityProfile(densityProfile);
    if(!finite(receiver)||!finite(source)||!Number.isFinite(length(sub(source,receiver)))||length(sub(source,receiver))<ENDPOINT_TOLERANCE_M)throw new RangeError("Invalid reflection endpoints");
    if(!Number.isFinite(specificLoss)||specificLoss<0||specificLoss>2||!Number.isFinite(foliage)||foliage<0||foliage>2||!Number.isFinite(bounceLoss)||bounceLoss<0||bounceLoss>60)throw new RangeError("Invalid reflection assumptions");
    const paths=[],blocked=[],accepted=[],directDelta=sub(source,receiver),directRange=length(directDelta);
    for(const plane of this.planes){
      const {a,e,f,n,ee,ef,ff,areaSquared,triangle}=plane,dr=dot(sub(receiver,a),n),ds=dot(sub(source,a),n);
      if(dr<=ENDPOINT_TOLERANCE_M||ds<=ENDPOINT_TOLERANCE_M)continue;
      // Intersect receiver -> image(source) with this finite triangle.
      const image=source.map((v,i)=>v-2*ds*n[i]),t=dr/(dr+ds),point=receiver.map((v,i)=>v+t*(image[i]-v));
      const relative=sub(point,a),re=dot(relative,e),rf=dot(relative,f),u=(re*ff-rf*ef)/areaSquared,v=(rf*ee-re*ef)/areaSquared;
      if(u< -1e-10||v< -1e-10||u+v>1+1e-10)continue;
      // Shared coplanar edges describe one physical path, not two replicas.
      if(accepted.some(p=>length(sub(p.point,point))<1e-7&&length(sub(p.normal,n))<1e-9))continue;
      accepted.push({point,normal:n});
      const down=sub(point,receiver),up=sub(source,point),receiverRange=length(down),sourceRange=length(up);
      if(receiverRange<=ENDPOINT_TOLERANCE_M||sourceRange<=ENDPOINT_TOLERANCE_M)continue;
      const receiverDirection=down.map(v=>v/receiverRange),sourceDirection=up.map(v=>v/sourceRange);
      // Stable excess avoids subtracting two almost equal orbital ranges.
      const extra=receiverRange+(dot(down,down)-2*dot(directDelta,down))/(sourceRange+directRange),totalRange=directRange+extra;
      if(!details){
        if(this.occlusion.blocked(receiver,receiverDirection,receiverRange,{groundMaximum:receiverRange-ENDPOINT_TOLERANCE_M,terrainDirection:receiverDirection})||this.occlusion.blocked(point,sourceDirection,sourceRange,{groundMaximum:sourceRange-ENDPOINT_TOLERANCE_M,terrainDirection:sourceDirection}))continue;
        paths.push({triangle,totalRange,extra});continue;
      }
      // Exclude only the reflecting endpoint from terrain occlusion. Volume/wood
      // integrals still include the complete modeled segments, without an offset.
      const receiverGround=terrainDistance(receiver,receiverDirection,this.forest.terrain,receiverRange-ENDPOINT_TOLERANCE_M);
      const sourceGround=terrainDistance(point,sourceDirection,this.forest.terrain,sourceRange-ENDPOINT_TOLERANCE_M);
      const receiverLeg=traceForest(this.volumes,receiver,receiverDirection,{specificLoss,foliage,densityProfile,maximum:receiverRange});
      const sourceLeg=traceForest(this.volumes,point,sourceDirection,{specificLoss,foliage,densityProfile,maximum:sourceRange});
      const reason=receiverGround!==null||sourceGround!==null?"terrain":receiverLeg.wood.length||sourceLeg.wood.length?"wood":null;
      const canopyLoss=receiverLeg.canopyLoss+sourceLeg.canopyLoss;
      const result={triangle,point,normal:n,receiverRange,sourceRange,totalRange,extra,delayNs:extra/C*1e9,
        canopyLoss,relativePowerDb:reason?null:-bounceLoss-canopyLoss-20*Math.log10(totalRange/directRange),
        legs:{receiver:receiverLeg,satellite:sourceLeg},occlusion:{receiverGround,sourceGround},reason};
      (reason?blocked:paths).push(result);
    }
    const order=(a,b)=>a.extra-b.extra||a.triangle-b.triangle;paths.sort(order);blocked.sort(order);
    return {model:GROUND_MODEL_VERSION,options:{specificLoss,foliage,bounceLoss,densityProfile,completeSegments:true,endpointToleranceM:ENDPOINT_TOLERANCE_M},directRange,paths,blocked};
  }
}
