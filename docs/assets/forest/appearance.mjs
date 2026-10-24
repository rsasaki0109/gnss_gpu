import { forestMesh, randomSequence, terrainHeight, trailX, treeCrowns, leafRetention } from "./model.mjs?v=extinction-reference-1";

// Display detail only. Never fed to the ray integrator or positioning worker.
// position, normal, linear albedo, material/wind/occlusion, atlas UV.
// Materials: 0 ground, 1 bark, 2 broadleaf, 2.25 needles, 3 stone,
// 4 cut wood, 5 geometric herbs, 6 litter. Needle coverage has its own cutoff.
export const APPEARANCE_STRIDE = 14;
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mul=(a,s)=>a.map(v=>v*s);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>mul(a,1/Math.hypot(...a));

export function forestAppearance(forest,{foliage=1}={}) {
  if(!Number.isFinite(foliage)||foliage<0||foliage>2)throw new RangeError("Invalid display foliage");
  const base=forestMesh(forest,{foliage:0}),data=[],rng=randomSequence(((forest.seed??0)^0x73bca541)>>>0);
  const vertex=(p,n,c,m,uv=[0,0])=>data.push(...p,...n,...c,...m,...uv);
  const tri=(a,b,c,color,material=[1,0,1])=>{
    const normal=unit(cross(sub(b,a),sub(c,a)));
    for(const p of [a,b,c])vertex(p,normal,color,material);
  };
  // Reuse exactly the physical ground and primary wood surface. Replace the
  // old lobe foliage with botanical alpha-cutout branch clusters below.
  for(let i=0;i<base.length;i+=9){
    const p=Array.from(base.subarray(i,i+3)),n=Array.from(base.subarray(i+3,i+6)),c=Array.from(base.subarray(i+6,i+9));
    const ground=p[2]<terrainHeight(p[0],p[1])+.09;
    const wood=!ground&&c[0]>c[1]*1.25;
    const path=ground&&Math.abs(p[0]-trailX(p[1]))<1.85;
    const color=ground?(path?[.25,.185,.112]:[.105,.125,.057]):wood?[.18,.125,.071]:[.12,.19,.045];
    vertex(p,n,color,[ground?0:wood?1:5,wood?0:ground?0:.18,1]);
  }
  const tube=(a,b,ra,rb,color=[.18,.12,.066],sides=6,cap=false)=>{
    const axis=unit(sub(b,a)),u=unit(cross(axis,Math.abs(axis[2])>.9?[1,0,0]:[0,0,1])),v=cross(axis,u);
    const radial=j=>add(mul(u,Math.cos(j*2*Math.PI/sides)),mul(v,Math.sin(j*2*Math.PI/sides)));
    for(let j=0;j<sides;j++){
      const n=radial(j),next=radial(j+1),p=add(a,mul(n,ra)),q=add(a,mul(next,ra)),r=add(b,mul(n,rb)),s=add(b,mul(next,rb));
      for(const [point,normal]of [[p,n],[q,next],[r,n],[q,next],[s,next],[r,n]])vertex(point,normal,color,[1,0,.85]);
      if(cap)tri(b,r,s,[.48,.34,.18],[4,0,1]);
    }
  };
  const card=(center,u,v,color,tile,wind,ao)=>{
    const n=unit(cross(u,v)),p=add(sub(center,u),mul(v,-1)),q=add(add(center,u),mul(v,-1)),r=add(sub(center,u),v),s=add(add(center,u),v);
    const x=tile%2*.5,y=Math.floor(tile/2)*.5;
    for(const [point,uv]of [[p,[x,y]],[q,[x+.5,y]],[r,[x,y+.5]],[q,[x+.5,y]],[s,[x+.5,y+.5]],[r,[x,y+.5]]])vertex(point,n,color,[tile===3?2.25:2,wind,ao],uv);
  };
  for(const tree of forest.trees){
    const crowns=treeCrowns(tree),baseZ=tree.trunk.a[2],r=tree.trunk.radius;
    // Root flares anchor the stem in the ground instead of ending in a stick.
    for(let j=0;j<6;j++){
      const angle=j*Math.PI/3+rng()*.3,length=.65+rng()*1.2,x=tree.x+Math.cos(angle)*length,y=tree.y+Math.sin(angle)*length;
      tube([tree.x,tree.y,baseZ+.55],[x,y,terrainHeight(x,y)+.035],r*.6,.025);
    }
    // Tapered scaffold branches and secondary twigs; also visible in winter.
    for(let j=0;j<12;j++){
      const crown=crowns[j%crowns.length],angle=j*2.39996+rng()*.5;
      const reach=.55+rng()*.23;
      const start=[tree.x,tree.y,crown.center[2]-crown.radii[2]*.66];
      const end=[crown.center[0]+Math.cos(angle)*crown.radii[0]*reach,crown.center[1]+Math.sin(angle)*crown.radii[1]*reach,crown.center[2]+(rng()-.3)*crown.radii[2]];
      tube(start,end,.07+rng()*.035,.012);
      for(const side of [-1,1])tube(add(start,mul(sub(end,start),.65)),add(end,[Math.cos(angle+side*.65)*.6,Math.sin(angle+side*.65)*.6,.5]),.025,.005);
    }
    // Always consume the same random draws in leaf-on/off: ground details and
    // neighboring trees do not move when a season control is changed.
    for(const crown of crowns){
      const evergreen=tree.species==="conifer",count=evergreen?36:360;
      for(let j=0;j<count;j++){
        const angle=rng()*Math.PI*2,z=2*rng()-1,rad=Math.cbrt(rng())*.77,xy=Math.sqrt(1-z*z);
        const center=add(crown.center,[Math.cos(angle)*xy*rad*crown.radii[0],Math.sin(angle)*xy*rad*crown.radii[1],z*rad*crown.radii[2]]);
        const az=rng()*Math.PI*2,tilt=(rng()-.5)*1.7,size=evergreen?.80+rng()*.35:.78+rng()*.40;
        const u=[Math.cos(az)*size,Math.sin(az)*size,0];
        const v=evergreen?[-Math.sin(az)*size,Math.cos(az)*size,-(.25+Math.abs(tilt)*.45)*size]:[-Math.sin(az)*Math.sin(tilt)*size,Math.cos(az)*Math.sin(tilt)*size,Math.cos(tilt)*size];
        const tint=.72+rng()*.5,tile=evergreen?3:Math.floor(rng()*3);
        if(foliage>0&&leafRetention(tree)>0)card(center,u,v,evergreen?[.14*tint,.23*tint,.18*tint]:[.16*tint,.25*tint,.19*tint],tile,.35+.65*(z+1)/2,.58+.28*(z+1)/2);
      }
    }
  }
  // Knee-high understory is clustered at stem bases and path margins rather
  // than evenly spaced. These small ornamental plants carry no new RF volume.
  for(let j=0;j<1600;j++){
    const tree=forest.trees[Math.floor(rng()*forest.trees.length)],angle=rng()*Math.PI*2,radius=1.3+rng()*5;
    const y=j%3?tree.y+Math.sin(angle)*radius:(rng()-.5)*140;
    const x=j%3?tree.x+Math.cos(angle)*radius:trailX(y)+(j%2?1:-1)*(2.0+rng()*2.4);
    if(Math.abs(x)>77||Math.abs(y)>77||Math.abs(x-trailX(y))<1.9)continue;
    const h=.26+rng()*.48,z=terrainHeight(x,y),az=rng()*Math.PI*2;
    for(let k=0;k<2;k++){
      const a=az+k*Math.PI/2;
      card([x,y,z+h*.7],[Math.cos(a)*h,Math.sin(a)*h,0],[0,0,h*.7],[.12,.22,.13],2,.12,.75);
    }
  }
  // Fallen wood with exposed end grain, scattered stones and curled litter.
  for(let j=0;j<42;j++){
    const tree=forest.trees[Math.floor(rng()*forest.trees.length)],angle=rng()*Math.PI*2,length=1.2+rng()*3;
    const x=tree.x+(rng()-.5)*7,y=tree.y+(rng()-.5)*7;
    if(Math.abs(x)>73||Math.abs(y)>73||Math.abs(x-trailX(y))<3)continue;
    const end=[x+Math.cos(angle)*length,y+Math.sin(angle)*length];
    if(Math.abs(end[0])>78||Math.abs(end[1])>78)continue;
    tube([x,y,terrainHeight(x,y)+.12],[...end,terrainHeight(...end)+.12],.09+rng()*.11,.08,[.105,.077,.041],7,true);
  }
  for(let j=0;j<1900;j++){
    const x=(rng()-.5)*152,y=(rng()-.5)*152,angle=rng()*Math.PI*2,path=Math.abs(x-trailX(y))<1.7;
    const z=terrainHeight(x,y)+.015,size=(path?.025:.06)+rng()*(path?.10:.19);
    if(j<280){
      const h=size*(.35+rng()*.65),a=[x-size,y-size*.6,z],b=[x+size,y-size*.5,z],c=[x+size*.5,y+size,z],d=[x-size*.6,y+size*.5,z],top=[x,y,z+h];
      const tint=.11+rng()*.12,color=[tint*.87,tint,tint*.72];
      for(const [p,q]of [[a,b],[b,c],[c,d],[d,a]])tri(p,q,top,color,[3,0,.88]);
    }else if(!path){
      const u=[Math.cos(angle)*size,Math.sin(angle)*size,0],v=[-Math.sin(angle)*size*.42,Math.cos(angle)*size*.42,size*.22];
      const color=[.14+rng()*.13,.095+rng()*.06,.028];
      tri(sub([x,y,z],u),add([x,y,z],v),add([x,y,z],u),color,[6,0,.82]);
      tri(sub([x,y,z],u),add([x,y,z],u),sub([x,y,z],v),color,[6,0,.82]);
    }
  }
  return new Float32Array(data);
}

// A botanical branch atlas is generated locally; no downloaded texture, asset
// service or network request. The same alpha test is used in the shadow pass.
export function createLeafAtlas(document) {
  const canvas=document.createElement("canvas");canvas.width=canvas.height=1024;
  const ctx=canvas.getContext("2d"),rng=randomSequence(71831);
  const leaf=(x,y,length,width,angle,tint)=>{
    ctx.save();ctx.translate(x,y);ctx.rotate(angle);
    const gradient=ctx.createLinearGradient(-width,0,width,length);
    gradient.addColorStop(0,`rgb(${46+tint},${82+tint},${40+tint})`);gradient.addColorStop(.48,`rgb(${90+tint},${125+tint},${62+tint})`);gradient.addColorStop(1,`rgb(${36+tint},${69+tint},${32+tint})`);
    ctx.fillStyle=gradient;ctx.beginPath();ctx.moveTo(0,0);ctx.bezierCurveTo(-width,length*.2,-width*.7,length*.78,0,length);ctx.bezierCurveTo(width*.75,length*.8,width,length*.24,0,0);ctx.fill();
    ctx.strokeStyle="#d1d68a65";ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(0,2);ctx.lineTo(0,length*.92);ctx.stroke();
    ctx.lineWidth=.7;for(let i=1;i<5;i++){ctx.beginPath();ctx.moveTo(0,length*i/6);ctx.lineTo(-width*.55,length*(i+.8)/6);ctx.moveTo(0,length*i/6);ctx.lineTo(width*.55,length*(i+.8)/6);ctx.stroke();}ctx.restore();
  };
  for(let tile=0;tile<4;tile++){
    ctx.save();ctx.translate(tile%2*512,(1-Math.floor(tile/2))*512);
    ctx.beginPath();ctx.rect(0,0,512,512);ctx.clip();
    ctx.strokeStyle="#71613b";ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(246,35);ctx.bezierCurveTo(275,150,225,350,268,477);ctx.stroke();
    if(tile<3){
      for(let i=0;i<7;i++){
        const y=85+i*50,side=i%2?-1:1,x=253+side*(13+rng()*10),length=100+rng()*60;
        ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(253,y-12);ctx.lineTo(x+side*45,y+23);ctx.stroke();
        leaf(x+side*40,y+14,length,36+rng()*18,-side*(.85+rng()*.4),Math.floor(rng()*24));
      }
      leaf(248,60,100,35,-.15,12);
    }else{
      for(let k=0;k<8;k++){
        const y=75+k*49;
        for(const side of [-1,1])for(let i=0;i<13;i++){
          const x=253+side*i*9,py=y+i*3;
          ctx.strokeStyle=i%3?"#4a713b":"#749152";ctx.lineWidth=4.5;ctx.beginPath();ctx.moveTo(x,py);ctx.lineTo(x+side*(25+rng()*15),py+40+rng()*26);ctx.stroke();
        }
      }
    }
    ctx.restore();
  }
  return canvas;
}
