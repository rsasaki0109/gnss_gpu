import {checkAttenuation,extinctionMaterial,extinctionLoss} from "./attenuation.mjs?v=extinction-reference-1";
import {checkDensityProfile,crownIntegral} from "./density.mjs?v=extinction-reference-1";
// Synthetic ENU forest. Geometry and integrals use binary64; mesh is display only.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const point=(o,d,t)=>o.map((v,i)=>v+t*d[i]);
const unit=a=>{const n=Math.hypot(...a);if(!Number.isFinite(n)||n===0)throw new RangeError("Invalid direction");return a.map(v=>v/n);};
export const MODEL_VERSION="synthetic-forest-v2";
export function randomSequence(seed) {
  if(!Number.isInteger(seed)||seed<0||seed>0xffffffff)throw new RangeError("Invalid seed");
  let value=seed;
  return ()=>{value=(value+0x6d2b79f5)>>>0;let t=value;t=Math.imul(t^(t>>>15),t|1);t^=t+Math.imul(t^(t>>>7),t|61);return ((t^(t>>>14))>>>0)/4294967296;};
}
export const trailX=y=>9*Math.sin(y/28);
const height=(x,y)=>2.6*Math.sin(x/35)+1.8*Math.sin(y/30);
// Same piecewise planar surface as the displayed ground triangles.
export function terrainHeight(x,y) {
  if(!Number.isFinite(x)||!Number.isFinite(y)||Math.abs(x)>80||Math.abs(y)>80)throw new RangeError("Outside forest patch");
  const gx=Math.min(15,Math.floor((x+80)/10)),gy=Math.min(15,Math.floor((y+80)/10));
  const x0=gx*10-80,y0=gy*10-80,u=(x-x0)/10,v=(y-y0)/10;
  const a=height(x0,y0),b=height(x0+10,y0),c=height(x0,y0+10),d=height(x0+10,y0+10);
  return u+v<=1?a+u*(b-a)+v*(c-a):d+(1-u)*(c-d)+(1-v)*(b-d);
}
export function forestReceiver(seconds) {
  if(!Number.isFinite(seconds)||seconds<0||seconds>1200)throw new RangeError("Invalid forest time");
  const y=-65+130*seconds/1200,x=trailX(y);
  return [x,y,terrainHeight(x,y)+1.6];
}
// Growth forms, not calibrated botanical species. The component ellipsoids
// below are shared by rendering and RF; with canopies, crown is a size/shadow proxy.
export const PHENOLOGY_ASSUMPTION="synthetic deciduous broadleaf and evergreen conifer growth forms; leaf-off removes broadleaf leaves while residual crown attenuation depends on the RF model; wood and evergreen foliage remain; component volumes are authoritative; densities are not mass-matched across tree types; not calibrated species/season coefficients";
export const FOREST_TYPES=Object.freeze(["broadleaf","conifer","mixed"]);
export const FOREST_SEASONS=Object.freeze(["leaf-on","leaf-off"]);
export const treeCrowns=tree=>tree.canopies??[tree.crown];
export const leafRetention=tree=>tree.leafRetention??1;
export function generateForest({seed=20261023,count=64,stand="broadleaf",season="leaf-on"}={}) {
  if(!FOREST_TYPES.includes(stand)||!FOREST_SEASONS.includes(season))throw new RangeError("Invalid stand or season");
  if(!Number.isInteger(count)||count<1||count>128)throw new RangeError("Invalid tree count");
  const random=randomSequence(seed),trees=[],terrain=[];
  for(let y=-80;y<80;y+=10)for(let x=-80;x<80;x+=10){
    const a=[x,y,height(x,y)],b=[x+10,y,height(x+10,y)],c=[x,y+10,height(x,y+10)],d=[x+10,y+10,height(x+10,y+10)];
    terrain.push([a,b,c],[b,d,c]);
  }
  for(let attempts=0;trees.length<count&&attempts<100000;attempts++){
    const x=(random()-.5)*144,y=(random()-.5)*144;
    if(Math.abs(x-trailX(y))<5||trees.some(t=>Math.hypot(t.x-x,t.y-y)<5))continue;
    const base=terrainHeight(x,y),h=13+random()*12,r=3.4+random()*2.8,depth=Math.min(5+random()*4,(h-4)/2);
    const trunk={a:[x,y,base],b:[x,y,base+h-1],radius:.16+random()*.22};
    const crown={center:[x,y,base+h-depth],radii:[r,r*(.85+random()*.3),depth]};
    const branches=Array.from({length:5},(_,j)=>{
      const angle=random()*Math.PI*2,z=base+h*(.44+j*.09),reach=r*(.55+random()*.25);
      return {a:[x,y,z],b:[x+Math.cos(angle)*reach,y+Math.sin(angle)*reach,z+1.4],radius:.07+random()*.045};
    });
    trees.push({id:`T${String(trees.length+1).padStart(3,"0")}`,x,y,height:h,trunk,branches,crown,density:.75+random()*.5});
  }
  if(trees.length!==count)throw new Error("Could not place requested trees");
  // Transform after placement: stand/season/count comparisons preserve tree
  // locations and random draws. No seasonal change to trunk or branch geometry.
  for(const [index,tree]of trees.entries()){
    tree.species=stand==="mixed"?(index%2?"conifer":"broadleaf"):stand;
    tree.leafRetention=season==="leaf-off"&&tree.species==="broadleaf"?0:1;
    if(tree.species!=="conifer")continue;
    const base=tree.trunk.a[2],bottom=base+4,step=(tree.height-4)/5,r=tree.crown.radii[0]*.72;
    tree.crown={center:[tree.x,tree.y,(bottom+base+tree.height)/2],radii:[r,r,(tree.height-4)/2]};
    // Five vertically disjoint bough envelopes: their density is never counted
    // twice within one tree. Each tapering tier has an analytic ellipsoid chord.
    tree.canopies=Array.from({length:5},(_,j)=>({center:[tree.x,tree.y,bottom+step*(j+.5)],radii:[r*(1-.18*j),r*(1-.18*j),step/2]}));
    const original=tree.branches;
    tree.branches=tree.canopies.flatMap((c,j)=>Array.from({length:3},(_,k)=>{
      const old=original[j],angle=Math.atan2(old.b[1]-tree.y,old.b[0]-tree.x)+k*2*Math.PI/3,z=c.center[2]-.25*step;
      return {a:[tree.x,tree.y,z],b:[tree.x+.85*c.radii[0]*Math.cos(angle),tree.y+.85*c.radii[1]*Math.sin(angle),z+.3*step],radius:old.radius};
    }));
  }
  return {version:MODEL_VERSION,seed,count,stand,season,bounds:[[-80,-80,-6],[80,80,36]],trees,terrain};
}
function checkedRay(origin,direction,maximum) {
  if(!Array.isArray(origin)||origin.length!==3||!origin.every(Number.isFinite)||!Array.isArray(direction)||direction.length!==3||!direction.every(Number.isFinite)||Math.abs(Math.hypot(...direction)-1)>1e-12||!Number.isFinite(maximum)||maximum<=0)throw new RangeError("Expected finite unit ray");
}
function quadraticInterval(a,b,c) {
  if(a<1e-15)return c<=0?[-Infinity,Infinity]:null;
  const disc=b*b-4*a*c;
  if(disc<=0)return null;
  // Avoid loss of the smaller root when |b| is large.
  const q=-.5*(b+Math.sign(b||1)*Math.sqrt(disc)),t0=q/a,t1=c/q;
  return [Math.min(t0,t1),Math.max(t0,t1)];
}
export function ellipsoidInterval(origin,direction,shape,maximum=400) {
  checkedRay(origin,direction,maximum);
  if(!shape.center?.every(Number.isFinite)||shape.center.length!==3||!shape.radii?.every(r=>Number.isFinite(r)&&r>0)||shape.radii.length!==3)throw new RangeError("Invalid crown");
  const o=sub(origin,shape.center).map((v,i)=>v/shape.radii[i]),d=direction.map((v,i)=>v/shape.radii[i]);
  const span=quadraticInterval(dot(d,d),2*dot(o,d),dot(o,o)-1);
  if(!span)return null;
  const lo=Math.max(0,span[0]),hi=Math.min(maximum,span[1]);return hi>lo?[lo,hi]:null;
}
export function cylinderInterval(origin,direction,cylinder,maximum=400) {
  checkedRay(origin,direction,maximum);
  return preparedCylinderInterval(origin,direction,prepareCylinder(cylinder),maximum);
}
function prepareCylinder(cylinder) {
  if(!cylinder.a?.every(Number.isFinite)||cylinder.a.length!==3||!cylinder.b?.every(Number.isFinite)||cylinder.b.length!==3||!Number.isFinite(cylinder.radius)||cylinder.radius<=0)throw new RangeError("Invalid wood cylinder");
  const axis=sub(cylinder.b,cylinder.a),length=Math.hypot(...axis);
  if(length===0)throw new RangeError("Zero-length wood cylinder");
  return {a:cylinder.a,n:axis.map(v=>v/length),length,radiusSquared:cylinder.radius**2};
}
function preparedCylinderInterval(origin,direction,{a,n,length,radiusSquared},maximum) {
  // Keep the original subtraction, multiply and left-to-right accumulation
  // order (including the initial zero); avoid per-cylinder temporary vectors.
  const ox=origin[0]-a[0],oy=origin[1]-a[1],oz=origin[2]-a[2];
  const z=0+ox*n[0]+oy*n[1]+oz*n[2],dz=0+direction[0]*n[0]+direction[1]*n[1]+direction[2]*n[2];
  const rx=ox-z*n[0],ry=oy-z*n[1],rz=oz-z*n[2];
  const dx=direction[0]-dz*n[0],dy=direction[1]-dz*n[1],dzr=direction[2]-dz*n[2];
  const span=quadraticInterval(0+dx*dx+dy*dy+dzr*dzr,2*(0+rx*dx+ry*dy+rz*dzr),0+rx*rx+ry*ry+rz*rz-radiusSquared);
  if(!span)return null;
  let lo=Math.max(0,span[0]),hi=Math.min(maximum,span[1]);
  if(Math.abs(dz)<1e-14){if(z<0||z>length)return null;}
  else {const a=-z/dz,b=(length-z)/dz;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));}
  return hi>lo?[lo,hi]:null;
}
export function terrainDistance(origin,direction,triangles,maximum) {
  return terrainDistanceImpl(origin,direction,triangles,maximum,false);
}
function terrainDistanceImpl(origin,direction,triangles,maximum,prepared,any=false) {
  let nearest=maximum;
  for(const triangle of triangles){
    const a=prepared?triangle.a:triangle[0],e=prepared?triangle.e:sub(triangle[1],a),f=prepared?triangle.f:sub(triangle[2],a);
    const p=[direction[1]*f[2]-direction[2]*f[1],direction[2]*f[0]-direction[0]*f[2],direction[0]*f[1]-direction[1]*f[0]],det=dot(e,p);
    if(Math.abs(det)<1e-12)continue;
    const s=sub(origin,a),u=dot(s,p)/det;if(u<0||u>1)continue;
    const q=[s[1]*e[2]-s[2]*e[1],s[2]*e[0]-s[0]*e[2],s[0]*e[1]-s[1]*e[0]],v=dot(direction,q)/det;
    if(v<0||u+v>1)continue;
    const t=dot(f,q)/det;
    if(t>1e-6&&t<nearest){
      if(any)return t; // Boolean callers need one blocker, not the nearest.
      nearest=t;
    }
  }
  return nearest<maximum?nearest:null;
}
// Static-scene Boolean occlusion only. Rebuild after changing geometry. This
// shares the binary64 intersection kernels with the detailed propagation model;
// crowns attenuate but do not block, so their density integrals are unnecessary.
export class ForestOcclusion {
  constructor(forest){
    this.terrain=forest.terrain.map(([a,b,c])=>{
      if(![a,b,c].every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)))throw new RangeError("Invalid ground triangle");
      return {a,e:sub(b,a),f:sub(c,a)};
    });
    this.wood=forest.trees.flatMap(t=>[t.trunk,...t.branches].map(prepareCylinder));
  }
  blocked(origin,rawDirection,maximum,{groundMaximum=maximum,terrainDirection}={}){
    const direction=unit(rawDirection);
    checkedRay(origin,direction,maximum);
    if(!Number.isFinite(groundMaximum)||groundMaximum<=0||groundMaximum>maximum)throw new RangeError("Invalid terrain endpoint bound");
    // Detailed ground reflection checks terrain with its original geometric ray
    // but normalizes volume rays in traceForest. Preserve both rounding paths.
    const groundDirection=terrainDirection??direction;checkedRay(origin,groundDirection,groundMaximum);
    if(terrainDistanceImpl(origin,groundDirection,this.terrain,groundMaximum,true,true)!==null)return true;
    return this.wood.some(shape=>preparedCylinderInterval(origin,direction,shape,maximum)!==null);
  }
}
export function traceForest(forest,origin,rawDirection,{specificLoss=.2,foliage=1,maximum=400,densityProfile="uniform",attenuationModel="linear-v1",materialPreset="maple-pine"}={}) {
  if(!Number.isFinite(specificLoss)||specificLoss<0||specificLoss>2||!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid attenuation assumptions");
  checkDensityProfile(densityProfile);checkAttenuation(attenuationModel,materialPreset,foliage);
  const reference=attenuationModel==="itu-extinction-v1";
  const direction=unit(rawDirection);checkedRay(origin,direction,maximum);
  const ground=terrainDistance(origin,direction,forest.terrain,maximum),limit=ground??maximum,crowns=[],wood=[];
  for(const tree of forest.trees){
    if(reference||leafRetention(tree)>0)for(const [component,shape]of treeCrowns(tree).entries()){
      const span=ellipsoidInterval(origin,direction,shape,limit);
      if(!span)continue;
      const density=tree.density*(reference?1:leafRetention(tree)),crown={tree:tree.id,enter:span[0],exit:span[1],density};
      if(reference)crown.material=extinctionMaterial(tree,materialPreset);
      if(tree.canopies)crown.component=component;
      if(densityProfile!=="uniform")crown.densityIntegral=crownIntegral(origin,direction,shape,span,density,densityProfile);
      crowns.push(crown);
    }
    for(const [index,shape] of [tree.trunk,...tree.branches].entries()){
      const span=cylinderInterval(origin,direction,shape,limit);
      if(span)wood.push({tree:tree.id,part:index===0?"trunk":"branch",enter:span[0],exit:span[1]});
    }
  }
  crowns.sort((a,b)=>a.enter-b.enter);wood.sort((a,b)=>a.enter-b.enter);
  const intervals=crowns.map(c=>[c.enter,c.exit]).sort((a,b)=>a[0]-b[0]);
  let canopyPath=0,end=-Infinity;
  for(const [a,b] of intervals){canopyPath+=Math.max(0,b-Math.max(a,end));end=Math.max(end,b);}
  // Densities add in overlapping crowns; union length is reported separately.
  const opticalPath=(densityProfile==="uniform"?crowns.reduce((sum,c)=>sum+(c.exit-c.enter)*c.density,0):crowns.reduce((sum,c)=>sum+c.densityIntegral,0))*foliage,canopyLoss=reference?extinctionLoss(crowns,densityProfile):specificLoss*opticalPath;
  return {direction,crowns,wood,ground,canopyPath,opticalPath,canopyLoss,...(reference?{attenuation:{model:attenuationModel,materialPreset,uniformLossDb:extinctionLoss(crowns,"uniform")}}:{}),
    classification:ground!==null?"TERRAIN":wood.length?"WOOD":opticalPath>0?"CANOPY":"CLEAR",
    directRelativeDb:ground!==null||wood.length?null:-canopyLoss};
}
export function forestMesh(forest,{foliage=1}={}) {
  if(!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid display foliage");
  const data=[],random=randomSequence((forest.seed??0)>>>0),tri=(a,b,c,color,normals)=>{
    const u=sub(b,a),v=sub(c,a),flat=unit([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]);
    [a,b,c].forEach((p,i)=>data.push(...p,...(normals?.[i]??flat),...color));
  };
  // Render subdivision stays on the same planar terrain. Soft sun shadows are
  // appearance only; GNSS occlusion continues to use the original triangles.
  const floorColors=new Map(),floorColor=p=>{
    const key=`${p[0]},${p[1]}`;if(floorColors.has(key))return floorColors.get(key);
    let shade=1;
    for(const tree of forest.trees){
      const {center,radii}=tree.crown,h=center[2]-p[2];
      const x=(p[0]-center[0]+.375*h)/(radii[0]+.375*radii[2]),y=(p[1]-center[1]-.625*h)/(radii[1]+.625*radii[2]);
      shade*=1-.42*leafRetention(tree)*Math.min(foliage,1)*Math.exp(-1.6*(x*x+y*y));
    }
    const variation=.94+.06*Math.sin(p[0]*.61+p[1]*.37),color=[.30,.33,.21].map(v=>v*variation*(.45+.55*shade));
    floorColors.set(key,color);return color;
  };
  const groundTriangle=(a,b,c)=>{
    // Each subdivided face is contained in one original triangle, including
    // arbitrary imported terrain; interpolation cannot invent a new surface.
    const n=8,point=(u,v)=>a.map((x,i)=>x+u*(b[i]-x)/n+v*(c[i]-x)/n);
    for(let i=0;i<n;i++)for(let j=0;j<n-i;j++){
      const p=point(i,j),q=point(i+1,j),r=point(i,j+1),color=[p,q,r].map(floorColor);
      const e=sub(q,p),f=sub(r,p),normal=unit([e[1]*f[2]-e[2]*f[1],e[2]*f[0]-e[0]*f[2],e[0]*f[1]-e[1]*f[0]]);
      [p,q,r].forEach((v,k)=>data.push(...v,...normal,...color[k]));
      if(i+j<n-1){const s=point(i+1,j+1);[q,s,r].forEach(v=>data.push(...v,...normal,...floorColor(v)));}
    }
  };
  for(const [a,b,c] of forest.terrain)groundTriangle(a,b,c);
  for(let y=-68;y<68;y+=1){
    const road=(y,side)=>{const x=trailX(y)+side*(1.6+.12*Math.sin(y*.7));return [x,y,terrainHeight(x,y)+.035];};
    tri(road(y,-1),road(y,1),road(y+1,-1),[.48,.43,.32]);tri(road(y,1),road(y+1,1),road(y+1,-1),[.48,.43,.32]);
  }
  for(const tree of forest.trees){
    for(const shape of [tree.trunk,...tree.branches]){
      const n=unit(sub(shape.b,shape.a)),axis=Math.abs(n[2])<.9?[0,0,1]:[1,0,0];
      const u=unit([n[1]*axis[2]-n[2]*axis[1],n[2]*axis[0]-n[0]*axis[2],n[0]*axis[1]-n[1]*axis[0]]),v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];
      const radial=j=>u.map((x,i)=>Math.cos(j*Math.PI/5)*x+Math.sin(j*Math.PI/5)*v[i]),ring=(center,j)=>center.map((x,i)=>x+shape.radius*radial(j)[i]);
      for(let j=0;j<10;j++){
        const a=ring(shape.a,j),b=ring(shape.a,j+1),c=ring(shape.b,j),d=ring(shape.b,j+1),na=radial(j),nb=radial(j+1);
        tri(a,b,c,[.32,.23,.15],[na,nb,na]);tri(b,d,c,[.32,.23,.15],[nb,nb,na]);tri(shape.a,b,a,[.32,.23,.15]);tri(shape.b,c,d,[.32,.23,.15]);
      }
    }
    // Consume the same random sequence even with foliage hidden. Display lobes
    // are contained in the physical density ellipsoid; they are not new RF volumes.
    const lobes=[{center:[0,0,0],radius:.58},{center:[0,0,.55],radius:.43},{center:[0,0,-.5],radius:.42}];
    for(let i=0;i<18;i++){
      const z=2*random()-1,angle=2*Math.PI*random(),radius=.46+.12*random(),r=Math.sqrt(1-z*z);
      const center=[r*Math.cos(angle)*radius,r*Math.sin(angle)*radius,z*radius];
      lobes.push({center,radius:Math.min(.32+.09*random(),.98-radius)});
    }
    const tint=random();
    if(foliage===0||leafRetention(tree)===0)continue;
    const components=tree.canopies?tree.canopies.map(crown=>({crown,lobes:[{center:[0,0,0],radius:1}]})):[{crown:tree.crown,lobes}];
    for(const {crown,lobes:clusters}of components)for(const lobe of clusters){
      const center=crown.center.map((v,i)=>v+lobe.center[i]*crown.radii[i]),radii=crown.radii.map(v=>v*lobe.radius);
      const sample=(i,j)=>{
        const u=[Math.sin(i*Math.PI/4)*Math.cos(j*Math.PI/4),Math.sin(i*Math.PI/4)*Math.sin(j*Math.PI/4),Math.cos(i*Math.PI/4)];
        return {p:center.map((v,k)=>v+u[k]*radii[k]),n:unit(u.map((v,k)=>v/radii[k]))};
      };
      const color=[.12+.055*tint,.31+.08*tint+.025*Math.min(foliage,1),.105+.035*tint].map(v=>v*(.80+.20*(lobe.center[2]+.6)/1.2));
      const face=(a,b,c)=>tri(a.p,b.p,c.p,color,[a.n,b.n,c.n]);
      for(let i=0;i<4;i++)for(let j=0;j<8;j++){
        if(i>0)face(sample(i,j),sample(i+1,j),sample(i,j+1));
        if(i<3)face(sample(i,j+1),sample(i+1,j),sample(i+1,j+1));
      }
    }
  }
  // Low ground cover is decorative, not an uncalibrated new attenuation term.
  for(let i=0;i<1250;i++){
    const fern=i<500,tree=forest.trees.length?forest.trees[Math.floor(random()*forest.trees.length)]:null;
    const x=fern&&tree?tree.x+(random()-.5)*11:(random()-.5)*152,y=fern&&tree?tree.y+(random()-.5)*11:(random()-.5)*152,h=.22+.45*random();
    if(Math.abs(x)>77||Math.abs(y)>77||Math.abs(x-trailX(y))<2.3)continue;
    const z=terrainHeight(x,y)+.025,angle=random()*Math.PI,color=[.21,.30+.07*random(),.105];
    if(fern){
      for(let j=0;j<4;j++){
        const a=angle+j*Math.PI/2,along=[Math.cos(a),Math.sin(a)],side=[-along[1],along[0]];
        for(let k=1;k<=4;k++){
          const t=k/5,reach=h*t,stem=[x+along[0]*reach,y+along[1]*reach,z+h*(1.6*t-.8*t*t)],width=h*.32*(1-t);
          for(const sign of [-1,1]){
            const base=[stem[0]-along[0]*.08,stem[1]-along[1]*.08,stem[2]-.04],tip=[stem[0]+side[0]*sign*width,stem[1]+side[1]*sign*width,stem[2]+.025],end=[stem[0]+along[0]*.09,stem[1]+along[1]*.09,stem[2]+.015];
            tri(base,tip,end,color);
          }
        }
      }
    }else{
      for(let j=0;j<7;j++){
        const a=angle+j*2.4,dx=Math.cos(a),dy=Math.sin(a),height=h*(.5+.5*random()),base=[x,y,z],left=[x-.018*dy,y+.018*dx,z],bend=[x+.12*dx,y+.12*dy,z+height*.65],tip=[x+.24*dx,y+.24*dy,z+height];
        tri(base,left,bend,color);tri(left,bend,tip,color);
      }
    }
  }
  return new Float32Array(data); // Visualization only, never read by traceForest.
}

export const rayPoint=point;
