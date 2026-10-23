// Synthetic ENU forest. Geometry and integrals use binary64; mesh is display only.
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const point=(o,d,t)=>o.map((v,i)=>v+t*d[i]);
const unit=a=>{const n=Math.hypot(...a);if(!Number.isFinite(n)||n===0)throw new RangeError("Invalid direction");return a.map(v=>v/n);};
export const MODEL_VERSION="synthetic-forest-v1";
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
export function generateForest({seed=20261023,count=64}={}) {
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
  return {version:MODEL_VERSION,seed,count,bounds:[[-80,-80,-6],[80,80,36]],trees,terrain};
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
  if(!cylinder.a?.every(Number.isFinite)||cylinder.a.length!==3||!cylinder.b?.every(Number.isFinite)||cylinder.b.length!==3||!Number.isFinite(cylinder.radius)||cylinder.radius<=0)throw new RangeError("Invalid wood cylinder");
  const axis=sub(cylinder.b,cylinder.a),length=Math.hypot(...axis);
  if(length===0)throw new RangeError("Zero-length wood cylinder");
  const n=axis.map(v=>v/length),o=sub(origin,cylinder.a),z=dot(o,n),dz=dot(direction,n);
  const radialO=o.map((v,i)=>v-z*n[i]),radialD=direction.map((v,i)=>v-dz*n[i]);
  const span=quadraticInterval(dot(radialD,radialD),2*dot(radialO,radialD),dot(radialO,radialO)-cylinder.radius**2);
  if(!span)return null;
  let lo=Math.max(0,span[0]),hi=Math.min(maximum,span[1]);
  if(Math.abs(dz)<1e-14){if(z<0||z>length)return null;}
  else {const a=-z/dz,b=(length-z)/dz;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));}
  return hi>lo?[lo,hi]:null;
}
export function terrainDistance(origin,direction,triangles,maximum) {
  let nearest=maximum;
  for(const [a,b,c] of triangles){
    const e=sub(b,a),f=sub(c,a),p=[direction[1]*f[2]-direction[2]*f[1],direction[2]*f[0]-direction[0]*f[2],direction[0]*f[1]-direction[1]*f[0]],det=dot(e,p);
    if(Math.abs(det)<1e-12)continue;
    const s=sub(origin,a),u=dot(s,p)/det;if(u<0||u>1)continue;
    const q=[s[1]*e[2]-s[2]*e[1],s[2]*e[0]-s[0]*e[2],s[0]*e[1]-s[1]*e[0]],v=dot(direction,q)/det;
    if(v<0||u+v>1)continue;
    const t=dot(f,q)/det;if(t>1e-6&&t<nearest)nearest=t;
  }
  return nearest<maximum?nearest:null;
}
export function traceForest(forest,origin,rawDirection,{specificLoss=.2,foliage=1,maximum=400}={}) {
  if(!Number.isFinite(specificLoss)||specificLoss<0||specificLoss>2||!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid attenuation assumptions");
  const direction=unit(rawDirection);checkedRay(origin,direction,maximum);
  const ground=terrainDistance(origin,direction,forest.terrain,maximum),limit=ground??maximum,crowns=[],wood=[];
  for(const tree of forest.trees){
    const span=ellipsoidInterval(origin,direction,tree.crown,limit);
    if(span)crowns.push({tree:tree.id,enter:span[0],exit:span[1],density:tree.density});
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
  const opticalPath=crowns.reduce((sum,c)=>sum+(c.exit-c.enter)*c.density,0)*foliage,canopyLoss=specificLoss*opticalPath;
  return {direction,crowns,wood,ground,canopyPath,opticalPath,canopyLoss,
    classification:ground!==null?"TERRAIN":wood.length?"WOOD":opticalPath>0?"CANOPY":"CLEAR",
    directRelativeDb:ground!==null||wood.length?null:-canopyLoss};
}
export function forestMesh(forest,{foliage=1}={}) {
  if(!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid display foliage");
  const data=[],tri=(a,b,c,color)=>{
    const u=sub(b,a),v=sub(c,a),n=unit([u[1]*v[2]-u[2]*v[1],u[2]*v[0]-u[0]*v[2],u[0]*v[1]-u[1]*v[0]]);
    for(const p of [a,b,c])data.push(...p,...n,...color);
  };
  for(const [a,b,c] of forest.terrain)tri(a,b,c,[.18,.25,.19]);
  for(let y=-68;y<68;y+=2){
    const road=(y,side)=>{const x=trailX(y)+side*2;return [x,y,terrainHeight(x,y)+.035];};
    tri(road(y,-1),road(y,1),road(y+2,-1),[.48,.43,.32]);tri(road(y,1),road(y+2,1),road(y+2,-1),[.48,.43,.32]);
  }
  for(const tree of forest.trees){
    for(const shape of [tree.trunk,...tree.branches]){
      const n=unit(sub(shape.b,shape.a)),axis=Math.abs(n[2])<.9?[0,0,1]:[1,0,0];
      const u=unit([n[1]*axis[2]-n[2]*axis[1],n[2]*axis[0]-n[0]*axis[2],n[0]*axis[1]-n[1]*axis[0]]),v=[n[1]*u[2]-n[2]*u[1],n[2]*u[0]-n[0]*u[2],n[0]*u[1]-n[1]*u[0]];
      const ring=(center,j)=>center.map((x,i)=>x+shape.radius*(Math.cos(j*Math.PI/4)*u[i]+Math.sin(j*Math.PI/4)*v[i]));
      for(let j=0;j<8;j++){const a=ring(shape.a,j),b=ring(shape.a,j+1),c=ring(shape.b,j),d=ring(shape.b,j+1);tri(a,b,c,[.32,.23,.15]);tri(b,d,c,[.32,.23,.15]);tri(shape.a,b,a,[.32,.23,.15]);tri(shape.b,c,d,[.32,.23,.15]);}
    }
    if(foliage===0)continue;
    const crown=tree.crown,p=(i,j)=>crown.center.map((c,k)=>c+crown.radii[k]*[Math.sin(i*Math.PI/8)*Math.cos(j*Math.PI/7),Math.sin(i*Math.PI/8)*Math.sin(j*Math.PI/7),Math.cos(i*Math.PI/8)][k]);
    const color=[.10+.03*Math.min(foliage,1),.24+tree.density*.065+.06*Math.min(foliage,1),.15+.04*Math.min(foliage,1)];
    for(let i=0;i<8;i++)for(let j=0;j<14;j++){
      if(i>0)tri(p(i,j),p(i+1,j),p(i,j+1),color);
      if(i<7)tri(p(i,j+1),p(i+1,j),p(i+1,j+1),color);
    }
  }
  return new Float32Array(data); // Visualization only, never read by traceForest.
}
export const rayPoint=point;
