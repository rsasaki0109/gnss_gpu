// Display projection only. All path endpoints come from the RF scene/trace.
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
export const SIGNAL_COLORS=Object.freeze({CLEAR:"#91f8da",CANOPY:"#ffcf68",WOOD:"#ff7984",TERRAIN:"#bac7e1"});

export function forestCameraFrame(camera,width,height){
  const {yaw,pitch,zoom}=camera,c=Math.cos(yaw),s=Math.sin(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch);
  const right=[c,s,0],up=[-s*sp,c*sp,cp],forward=[s*cp,-c*cp,sp];
  // An observer behind the antenna can see the incoming ray bundle. An eye
  // placed at the antenna projects each straight incoming ray to one pixel.
  const eye=camera.follow?[camera.target[0]+s*6,camera.target[1]-c*6,camera.target[2]+1]:camera.target.map((v,i)=>v+forward[i]*180);
  const extentRight=80*(Math.abs(right[0])+Math.abs(right[1])),extentUp=80*(Math.abs(up[0])+Math.abs(up[1]))+31*Math.abs(up[2]);
  const scale=.9*Math.min(width/(2*extentRight),height/(2*extentUp))*zoom,lens=zoom/Math.tan(Math.PI/5.2);
  return {width,height,right,up,forward,eye,perspective:camera.follow,
    scale:camera.follow?[lens*height/width,lens,-1/700]:[2*scale/width,2*scale/height,-1/700],
    skyScale:[lens*height/width,lens,1],
    matrix:new Float32Array([right[0],up[0],forward[0],right[1],up[1],forward[1],right[2],up[2],forward[2]])};
}
const viewPoint=(point,frame)=>{const p=point.map((v,i)=>v-frame.eye[i]);return [dot(p,frame.right),dot(p,frame.up),-dot(p,frame.forward)];};
const screenPoint=(p,f)=>[f.width/2+p[0]*f.scale[0]*f.width/2/(f.perspective?p[2]:1),f.height/2-p[1]*f.scale[1]*f.height/2/(f.perspective?p[2]:1)];
export function projectForestPoint(point,frame){
  const p=viewPoint(point,frame);if(frame.perspective&&p[2]<.08)return null;
  return screenPoint(p,frame);
}
export function projectForestSegment(start,end,frame){
  let a=viewPoint(start,frame),b=viewPoint(end,frame);
  if(frame.perspective){
    if(a[2]<.08&&b[2]<.08)return null;
    if(a[2]<.08)a=mix(a,b,(.08-a[2])/(b[2]-a[2]));
    if(b[2]<.08)b=mix(b,a,(.08-b[2])/(a[2]-b[2]));
  }
  a=screenPoint(a,frame);b=screenPoint(b,frame);
  // Liang–Barsky clipping keeps near-plane intersections from generating
  // enormous SVG coordinates, including rays that leave the field of view.
  const d=[b[0]-a[0],b[1]-a[1]];let lower=0,upper=1;
  for(const [p,q]of [[-d[0],a[0]-8],[d[0],frame.width-8-a[0]],[-d[1],a[1]-8],[d[1],frame.height-8-a[1]]]){
    if(Math.abs(p)<1e-12){if(q<0)return null;continue;}
    const t=q/p;if(p<0)lower=Math.max(lower,t);else upper=Math.min(upper,t);
    if(lower>upper)return null;
  }
  const clipped=[mix(a,b,lower),mix(a,b,upper)];
  return Math.hypot(clipped[1][0]-clipped[0][0],clipped[1][1]-clipped[0][1])>.5?clipped:null;
}

export function forestSignalOverlay(receiver,signals,selected,ground,frame,mode="all",reflectionTrail=[]){
  const lines=[],labels=[];
  const point=(origin,direction,t)=>origin.map((v,i)=>v+direction[i]*t);
  const line=(a,b,satellite,kind,color,active,dashed=false)=>{
    const projected=projectForestSegment(a,b,frame);
    if(projected)lines.push({a:projected[0],b:projected[1],satellite,kind,color,active,dashed});
    return projected;
  };
  if(mode!=="off")for(let i=1;i<reflectionTrail.length;i++){
    if(reflectionTrail[i-1]&&reflectionTrail[i])line(reflectionTrail[i-1],reflectionTrail[i],selected,"reflection-trail",SIGNAL_COLORS.CANOPY,false,true);
  }
  if(mode!=="off")for(const signal of signals){
    const active=signal.id===selected;if(mode==="selected"&&!active)continue;
    const trace=signal.trace,blocked=trace.classification==="WOOD"||trace.classification==="TERRAIN",color=SIGNAL_COLORS[trace.classification];
    const path=line(receiver,point(receiver,trace.direction,160),signal.id,"direct",color,active,blocked);
    if(path)labels.push({id:signal.id,point:path[1],color,active,blocked,loss:trace.canopyLoss});
    if(active){
      for(const crown of trace.crowns)line(point(receiver,trace.direction,crown.enter),point(receiver,trace.direction,crown.exit),signal.id,"canopy",SIGNAL_COLORS.CANOPY,true);
      for(const wood of trace.wood)line(point(receiver,trace.direction,wood.enter),point(receiver,trace.direction,wood.exit),signal.id,"wood",SIGNAL_COLORS.WOOD,true);
    }
  }
  if(ground&&mode!=="off"){
    line(receiver,ground.point,selected,"reflection",SIGNAL_COLORS.CANOPY,true);
    line(ground.point,point(ground.point,ground.legs.satellite.direction,160),selected,"reflection",SIGNAL_COLORS.CANOPY,true);
  }
  return {lines,labels,receiver:mode==="off"?null:projectForestPoint(receiver,frame),reflection:ground&&mode!=="off"?projectForestPoint(ground.point,frame):null};
}

export function drawForestSignalOverlay(svg,packet,width,height){
  if(!svg)return;
  svg.setAttribute("viewBox",`0 0 ${width} ${height}`);svg.replaceChildren();
  const el=(tag,attrs,text)=>{const node=svg.ownerDocument.createElementNS("http://www.w3.org/2000/svg",tag);for(const [k,v]of Object.entries(attrs))node.setAttribute(k,String(v));if(text)node.textContent=text;svg.append(node);return node;};
  // Context first, highlighted signal last. These are explicitly an X-ray
  // overlay: blocked geometric alternatives are dashed, not valid arrivals.
  const ordered=packet.lines.toSorted((a,b)=>Number(a.active)-Number(b.active));
  for(const path of ordered){
    const d=`M${path.a.join(",")}L${path.b.join(",")}`,width=path.kind==="canopy"||path.kind==="wood"?5:path.active?3:1.5;
    el("path",{d,fill:"none",stroke:"#071a22","stroke-width":width+2,opacity:path.active?.85:.55});
    el("path",{d,fill:"none",stroke:path.color,"stroke-width":width,"stroke-linecap":"round","stroke-dasharray":path.kind==="reflection-trail"?"2 5":path.dashed?"8 6":"none",opacity:path.active?1:.65,"data-signal":path.satellite,"data-path-kind":path.kind});
    if(path.kind==="direct"&&!path.dashed){
      const dx=path.a[0]-path.b[0],dy=path.a[1]-path.b[1],length=Math.hypot(dx,dy);
      if(length>35){const x=path.a[0]+(path.b[0]-path.a[0])*.57,y=path.a[1]+(path.b[1]-path.a[1])*.57,ux=dx/length,uy=dy/length;
        el("path",{d:`M${x+ux*7},${y+uy*7}L${x-ux*5-uy*4},${y-uy*5+ux*4}L${x-ux*5+uy*4},${y-uy*5-ux*4}Z`,fill:path.color,opacity:.95});}
    }
  }
  const occupied=[];
  for(const label of packet.labels.toSorted((a,b)=>Number(b.active)-Number(a.active))){
    const text=label.active?`${label.id} · ${label.blocked?"blocked":`${label.loss.toFixed(1)} dB`}`:label.id;
    const boxWidth=label.active?118:44,x=Math.max(8,Math.min(width-boxWidth-8,label.point[0]-boxWidth/2));let y=Math.max(12,Math.min(height-95,label.point[1]-12));
    if(x<218&&y<95)y=96;
    for(let tries=0;tries<8&&occupied.some(p=>x<p.x+p.width+4&&x+boxWidth+4>p.x&&Math.abs(p.y-y)<22);tries++)y=y+24<height-95?y+24:Math.max(12,y-48);
    occupied.push({x,y,width:boxWidth});
    el("rect",{x,y:y-12,width:boxWidth,height:22,rx:5,fill:"#071c25",stroke:label.color,"stroke-opacity":label.active?.9:.4,"fill-opacity":.9,"data-signal-select":label.id,tabindex:0,role:"button","aria-label":`Select ${label.id}`,"aria-pressed":String(label.active),"pointer-events":"all"});
    el("text",{x:x+7,y:y+3,fill:label.color,"font-size":11,"font-family":"ui-monospace,monospace","font-weight":label.active?700:500,"data-signal-label":label.id},text);
  }
  const marker=(p,color,text)=>{if(!p||p[0]<0||p[0]>width||p[1]<0||p[1]>height)return;el("circle",{cx:p[0],cy:p[1],r:8,fill:"#0b2930",stroke:color,"stroke-width":2});el("circle",{cx:p[0],cy:p[1],r:3,fill:color});el("text",{x:p[0]+12,y:p[1]+5,fill:color,stroke:"#071c25","stroke-width":3,"paint-order":"stroke","font-size":12,"font-weight":700},text);};
  marker(packet.receiver,"#c7fff0","RX");marker(packet.reflection,"#ffcf68","Ground bounce");
  svg.dataset.signals=String(packet.labels.length);svg.dataset.segments=String(packet.lines.length);
}
