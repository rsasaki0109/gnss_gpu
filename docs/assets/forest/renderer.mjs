import { forestMesh, rayPoint } from "./model.mjs?v=ground-1";
export class ForestRenderer {
  constructor(canvas,forest){
    this.canvas=canvas;this.camera={yaw:-.7,pitch:.85,zoom:1,target:[0,0,5]};
    const gl=canvas.getContext("webgl2",{alpha:false,antialias:true});if(!gl)throw new Error("WebGL 2 is required for the forest view");this.gl=gl;
    const compile=(type,source)=>{const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s;};
    const v=compile(gl.VERTEX_SHADER,`#version 300 es
      in vec3 a_position;in vec3 a_normal;in vec3 a_color;uniform mat3 u_view;uniform vec3 u_scale;uniform vec3 u_target;uniform float u_lit;out vec3 color;
      void main(){vec3 p=u_view*(a_position-u_target);gl_Position=vec4(p*u_scale,1.);float light=.45+.55*abs(dot(a_normal,normalize(vec3(.3,-.5,.8))));color=a_color*mix(1.,light,u_lit);}`);
    const f=compile(gl.FRAGMENT_SHADER,`#version 300 es
      precision highp float;in vec3 color;out vec4 result;void main(){result=vec4(color,1.);}`);
    this.program=gl.createProgram();gl.attachShader(this.program,v);gl.attachShader(this.program,f);gl.linkProgram(this.program);
    if(!gl.getProgramParameter(this.program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(this.program));gl.deleteShader(v);gl.deleteShader(f);
    this.locations=["position","normal","color"].map(n=>gl.getAttribLocation(this.program,"a_"+n));
    this.uniforms=Object.fromEntries(["view","scale","target","lit"].map(n=>[n,gl.getUniformLocation(this.program,"u_"+n)]));
    this.foliage=1;this.mesh=this.upload(forestMesh(forest),gl.TRIANGLES);this.lines=this.upload(new Float32Array(),gl.LINES);this.forest=forest;
    let drag;
    canvas.addEventListener("pointerdown",e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,...this.camera};});
    canvas.addEventListener("pointermove",e=>{if(drag){this.camera.yaw=drag.yaw+(e.clientX-drag.x)*.007;this.camera.pitch=Math.max(.12,Math.min(1.5,drag.pitch+(e.clientY-drag.y)*.007));this.redraw();}});
    const end=()=>{drag=null;};canvas.addEventListener("pointerup",end);canvas.addEventListener("pointercancel",end);canvas.addEventListener("lostpointercapture",end);
    canvas.addEventListener("wheel",e=>{e.preventDefault();this.zoom(Math.exp(-e.deltaY*.001));},{passive:false});
  }
  upload(data,mode){const gl=this.gl,vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);this.locations.forEach((location,i)=>{gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,3,gl.FLOAT,false,36,i*12);});return {vao,buffer,count:data.length/9,mode};}
  setForest(forest){const gl=this.gl;gl.deleteBuffer(this.mesh.buffer);gl.deleteVertexArray(this.mesh.vao);this.mesh=this.upload(forestMesh(forest,{foliage:this.foliage}),gl.TRIANGLES);this.forest=forest;}
  setFoliage(value){if(this.foliage!==value){this.foliage=value;this.setForest(this.forest);}}
  zoom(factor){this.camera.zoom=Math.max(.65,Math.min(4,this.camera.zoom*factor));this.redraw();}
  redraw(){if(this.last)this.draw(...this.last);}
  focusReflection(path,receiver){this.camera.target=receiver.map((v,i)=>(v+path.point[i])/2);this.camera.zoom=3;this.redraw();}
  draw(receiver,signals,selected,ground){
    this.last=[receiver,signals,selected,ground];const gl=this.gl,ratio=Math.min(devicePixelRatio||1,2),w=this.canvas.clientWidth,h=this.canvas.clientHeight;
    if(this.canvas.width!==Math.round(w*ratio)||this.canvas.height!==Math.round(h*ratio)){this.canvas.width=Math.round(w*ratio);this.canvas.height=Math.round(h*ratio);}
    gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(.045,.082,.09,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(this.program);
    gl.uniform3fv(this.uniforms.target,new Float32Array(this.camera.target));
    const {yaw,pitch,zoom}=this.camera,c=Math.cos(yaw),s=Math.sin(yaw),sp=Math.sin(pitch),cp=Math.cos(pitch),right=[c,s,0],up=[-s*sp,c*sp,cp],forward=[s*cp,-c*cp,sp];
    gl.uniformMatrix3fv(this.uniforms.view,false,new Float32Array([right[0],up[0],forward[0],right[1],up[1],forward[1],right[2],up[2],forward[2]]));
    const extentRight=80*(Math.abs(right[0])+Math.abs(right[1])),extentUp=80*(Math.abs(up[0])+Math.abs(up[1]))+31*Math.abs(up[2]);
    const scale=.9*Math.min(w/(2*extentRight),h/(2*extentUp))*zoom;gl.uniform3f(this.uniforms.scale,2*scale/w,2*scale/h,-1/700);gl.uniform1f(this.uniforms.lit,1);gl.bindVertexArray(this.mesh.vao);gl.drawArrays(this.mesh.mode,0,this.mesh.count);
    const data=[],line=(a,b,color)=>{for(const p of [a,b])data.push(...p,0,0,0,...color);};
    line(receiver,receiver.map((v,i)=>v+(i===2?4:0)),[.65,1,.85]);
    const signal=signals.find(s=>s.id===selected);
    if(signal){const d=signal.trace.direction;line(receiver,rayPoint(receiver,d,130),[.45,.7,.78]);for(const crown of signal.trace.crowns)line(rayPoint(receiver,d,crown.enter),rayPoint(receiver,d,crown.exit),[1,.72,.26]);for(const wood of signal.trace.wood)line(rayPoint(receiver,d,wood.enter),rayPoint(receiver,d,wood.exit),[1,.3,.3]);}
    if(ground){line(receiver,ground.point,[1,.76,.3]);line(ground.point,rayPoint(ground.point,ground.legs.satellite.direction,130),[1,.76,.3]);for(const axis of [[1.2,0,0],[0,1.2,0]])line(ground.point.map((v,i)=>v-axis[i]),ground.point.map((v,i)=>v+axis[i]),[1,.88,.6]);}
    this.canvas.dataset.reflection=String(Boolean(ground));
    gl.bindBuffer(gl.ARRAY_BUFFER,this.lines.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);this.lines.count=data.length/9;
    gl.disable(gl.DEPTH_TEST);gl.uniform1f(this.uniforms.lit,0);gl.bindVertexArray(this.lines.vao);gl.drawArrays(gl.LINES,0,this.lines.count);
  }
}
