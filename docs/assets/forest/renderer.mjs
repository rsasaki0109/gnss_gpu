import { forestCameraFrame, forestSignalOverlay, drawForestSignalOverlay } from "./signal-overlay.mjs?v=live-history-1";
import { drawTreeCauseOutline } from "./tree-causes.mjs?v=tree-causes-1";
import { forestAppearance, createLeafAtlas, APPEARANCE_STRIDE } from "./appearance.mjs?v=woodland-1";
const vs=`#version 300 es
in vec3 a_position;in vec3 a_normal;in vec3 a_color;in vec3 a_surface;in vec2 a_uv;
uniform mat3 u_view;uniform vec3 u_scale;uniform vec3 u_eye;uniform float u_perspective;uniform float u_time;uniform float u_shadowPass;
out vec3 world;out vec3 normal;out vec3 color;out vec3 surface;out vec2 uv;out vec3 shadowCoord;out float depth;
vec3 lightProjection(vec3 p){vec3 d=p-vec3(0.,0.,8.),l=normalize(vec3(.48,-.38,.78)),r=normalize(cross(vec3(0.,0.,1.),l));return vec3(dot(r,d)/116.,dot(cross(l,r),d)/116.,-dot(l,d)/150.);}
void main(){world=a_position;
if(a_surface.x>1.5&&a_surface.x<2.5){float wind=(sin(u_time*.85+world.x*.21+world.y*.14)+.35*sin(u_time*1.8+world.y*.7))*.045*a_surface.y;world.xy+=vec2(wind,wind*.6);}
normal=a_normal;color=a_color;surface=a_surface;uv=a_uv;shadowCoord=lightProjection(world)*.5+.5;
if(u_shadowPass>.5){gl_Position=vec4(lightProjection(world),1.);depth=0.;return;}
vec3 p=u_view*(world-u_eye);depth=-p.z;float near=.08,far=650.;
gl_Position=u_perspective>.5?vec4(p.xy*u_scale.xy,((far+near)*p.z+2.*far*near)/(near-far),-p.z):vec4(p.xy*u_scale.xy,-p.z/450.-.5,1.);}`;
const fs=`#version 300 es
precision highp float;
in vec3 world;in vec3 normal;in vec3 color;in vec3 surface;in vec2 uv;in vec3 shadowCoord;in float depth;
uniform sampler2D u_atlas;uniform sampler2D u_shadow;uniform float u_shadowPass;uniform float u_lit;uniform float u_perspective;uniform vec3 u_eye;uniform float u_overcast;out vec4 result;
float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
float noise(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);}
float fbm(vec2 p){return .57*noise(p)+.28*noise(p*2.03)+.15*noise(p*4.17);}
vec3 tone(vec3 x){return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
float visibility(vec3 n,vec3 l){if(any(lessThan(shadowCoord,vec3(0)))||any(greaterThan(shadowCoord,vec3(1))))return 1.;float bias=max(.002,.003*(1.-abs(dot(n,l)))),sum=0.;for(int x=-1;x<=1;x++)for(int y=-1;y<=1;y++)sum+=step(shadowCoord.z-bias,texture(u_shadow,shadowCoord.xy+vec2(x,y)/1536.).r);return sum/9.;}
void main(){
if(u_lit<.5){result=vec4(color,1.);return;}
bool leaf=surface.x>1.5&&surface.x<2.5,needle=leaf&&surface.x>2.1;vec4 texel=leaf?texture(u_atlas,uv):vec4(1.);float cutoff=needle?.065:.42;if(leaf&&texel.a<cutoff)discard;
if(u_shadowPass>.5){result=vec4(1.);return;}
vec3 n=normalize(normal);if(!gl_FrontFacing)n=-n;vec3 l=normalize(vec3(.48,-.38,.78)),view=normalize(u_eye-world);float shadow=visibility(n,l),grain=noise(world.xy*35.+world.z*13.);vec3 albedo=color;
if(surface.x<.5){float path=1.-smoothstep(1.15,2.1,abs(world.x-9.*sin(world.y/28.))+.28*noise(world.xy*3.));float moss=smoothstep(.45,.72,fbm(world.xy*.6));albedo=mix(vec3(.095,.070,.032),vec3(.030,.055,.026),moss);albedo=mix(albedo,vec3(.24,.18,.105),path);albedo*=.70+.40*fbm(world.xy*5.)+.13*grain;vec2 tile=world.xy*7.,cell=floor(tile),q=fract(tile)-.5;float seed=hash(cell),angle=seed*6.28318;
q=mat2(cos(angle),-sin(angle),sin(angle),cos(angle))*q;
float edge=dot(q/vec2(.43,.19),q/vec2(.43,.19)),aa=max(fwidth(edge),.07),litter=(1.-smoothstep(1.-aa,1.+aa,edge))*step(.15,seed)*(1.-path*.95);
vec3 deadLeaf=mix(vec3(.045,.028,.012),vec3(.155,.082,.026),seed);deadLeaf+=vec3(.025,.015,.006)*(1.-smoothstep(.006,.020,abs(q.y)));
albedo=mix(albedo,deadLeaf,litter);n=normalize(n+vec3((noise(world.xy*40.)-.5)*.18,(grain-.5)*.18,0.));}
else if(surface.x<1.5){float ridges=fbm(vec2((world.x+world.y)*24.,world.z*.55)),cracks=smoothstep(.25,.37,ridges);albedo*=.48+.85*cracks+.25*noise(world.xy*95.+world.z*.6);float moss=smoothstep(.68,.88,noise(world.xy*2.8))*max(n.z,0.);albedo=mix(albedo,vec3(.08,.14,.024),moss*.65);}
else if(leaf){albedo*=texel.rgb*(needle?1.5:2.15);}
else if(surface.x<3.5){albedo*=.65+.6*fbm(world.xy*11.);albedo=mix(albedo,vec3(.11,.16,.052),smoothstep(.6,.8,noise(world.xy*8.))*max(n.z,0.));}
else if(surface.x<4.5){albedo*=.75+.25*sin(length(world.xy-floor(world.xy))*60.);}
else{albedo*=.75+.4*grain;}
float lambert=max(dot(n,l),0.);if(leaf)lambert=.30+.58*abs(dot(n,l));float hemi=.7+.3*n.z;
vec3 ambient=vec3(.21,.27,.26)*hemi*surface.z,direct=vec3(1.65,1.55,1.28)*lambert*mix(.10,1.,shadow);ambient=mix(ambient,vec3(.52,.57,.55)*surface.z,u_overcast);direct*=mix(1.,.18,u_overcast);vec3 lit=albedo*(ambient+direct);
if(leaf)lit+=albedo*vec3(.55,.72,.18)*pow(max(dot(-view,l),0.),5.)*shadow;
float spec=pow(max(dot(n,normalize(l+view)),0.),leaf?28.:12.);lit+=vec3(.10,.09,.065)*spec*shadow*(leaf?.48:.12);
float distance=max(depth-12.,0.),haze=u_perspective>.5?1.-exp(-distance*.0085):.065;
vec3 atmospheric=vec3(.30,.365,.32)+vec3(.11,.085,.035)*pow(max(dot(-view,l),0.),5.);
result=vec4(pow(tone(mix(lit,atmospheric,min(haze,.82))),vec3(1./2.2)),leaf?smoothstep(cutoff,needle?.24:.82,texel.a):1.);}`;
const skyVS=`#version 300 es
out vec2 uv;void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);uv=p*2.-1.;gl_Position=vec4(uv,.9999,1.);}`;
const skyFS=`#version 300 es
precision highp float;in vec2 uv;uniform mat3 u_view;uniform vec3 u_scale;uniform float u_overcast;out vec4 result;
float hash(float p){return fract(sin(p*127.1)*43758.5453);}float noise(float p){float i=floor(p),f=fract(p);return mix(hash(i),hash(i+1.),smoothstep(0.,1.,f));}
void main(){vec3 ray=transpose(u_view)*normalize(vec3(uv/u_scale.xy,-1.));float h=max(ray.z,0.);vec3 c=mix(vec3(.75,.78,.68),vec3(.32,.51,.62),pow(h,.45));vec3 sun=normalize(vec3(.48,-.38,.78));c+=vec3(.28,.19,.07)*pow(max(dot(ray,sun),0.),45.);float a=atan(ray.y,ray.x),tree=.010+.017*noise(a*65.)+.005*noise(a*150.);c=mix(c,vec3(.52,.59,.50),1.-smoothstep(tree-.01,tree+.009,ray.z));result=vec4(mix(c,vec3(.57,.64,.62),u_overcast*.85),1.);}`;

export class ForestRenderer {
  constructor(canvas,forest){
    this.canvas=canvas;this.overlay=canvas.closest(".stage")?.querySelector("#forest-rays");this.signalMode="all";this.reflectionTrail=[];this.highlightedTree=null;this.camera={yaw:-.18,pitch:-.08,zoom:1,target:[0,0,5],follow:true};this.lighting="sunlit";
    const gl=canvas.getContext("webgl2",{alpha:false,antialias:true});if(!gl)throw new Error("WebGL 2 is required for the forest view");this.gl=gl;
    const program=(vertex,fragment)=>{
      const compile=(type,source)=>{const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(shader));return shader;};
      const v=compile(gl.VERTEX_SHADER,vertex),f=compile(gl.FRAGMENT_SHADER,fragment),p=gl.createProgram();gl.attachShader(p,v);gl.attachShader(p,f);gl.linkProgram(p);gl.deleteShader(v);gl.deleteShader(f);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p;
    };
    this.program=program(vs,fs);this.skyProgram=program(skyVS,skyFS);this.skyVAO=gl.createVertexArray();
    this.locations=["position","normal","color","surface","uv"].map(n=>gl.getAttribLocation(this.program,"a_"+n));
    this.uniforms=Object.fromEntries(["view","scale","eye","lit","perspective","time","shadowPass","atlas","shadow","overcast"].map(n=>[n,gl.getUniformLocation(this.program,"u_"+n)]));
    this.skyUniforms=Object.fromEntries(["view","scale","overcast"].map(n=>[n,gl.getUniformLocation(this.skyProgram,"u_"+n)]));
    this.atlas=gl.createTexture();gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.atlas);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,createLeafAtlas(canvas.ownerDocument));gl.generateMipmap(gl.TEXTURE_2D);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR_MIPMAP_LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    this.shadowTexture=gl.createTexture();gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.shadowTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.DEPTH_COMPONENT24,1536,1536,0,gl.DEPTH_COMPONENT,gl.UNSIGNED_INT,null);for(const [p,v]of [[gl.TEXTURE_MIN_FILTER,gl.NEAREST],[gl.TEXTURE_MAG_FILTER,gl.NEAREST],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_2D,p,v);
    this.shadowFramebuffer=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,this.shadowFramebuffer);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.DEPTH_ATTACHMENT,gl.TEXTURE_2D,this.shadowTexture,0);gl.drawBuffers([gl.NONE]);gl.readBuffer(gl.NONE);if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw new Error("Forest shadow framebuffer is incomplete");gl.bindFramebuffer(gl.FRAMEBUFFER,null);
    this.foliage=1;this.forest=forest;this.mesh=this.upload(forestAppearance(forest),gl.TRIANGLES);this.lines=this.upload(new Float32Array(),gl.LINES);this.shadowDirty=true;this.reducedMotion=matchMedia("(prefers-reduced-motion: reduce)").matches;
    let drag;
    canvas.addEventListener("pointerdown",e=>{canvas.setPointerCapture(e.pointerId);drag={x:e.clientX,y:e.clientY,...this.camera};});
    canvas.addEventListener("pointermove",e=>{if(drag){this.camera.yaw=drag.yaw+(e.clientX-drag.x)*.007;this.camera.pitch=Math.max(this.camera.follow?-1.4:.12,Math.min(1.5,drag.pitch+(e.clientY-drag.y)*.007));this.redraw();}});
    const end=()=>{drag=null;};canvas.addEventListener("pointerup",end);canvas.addEventListener("pointercancel",end);canvas.addEventListener("lostpointercapture",end);
    canvas.addEventListener("wheel",e=>{e.preventDefault();this.zoom(Math.exp(-e.deltaY*.001));},{passive:false});
  }
  upload(data,mode){const gl=this.gl,vao=gl.createVertexArray(),buffer=gl.createBuffer();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,data,gl.STATIC_DRAW);let offset=0;[3,3,3,3,2].forEach((size,i)=>{const location=this.locations[i];if(location>=0){gl.enableVertexAttribArray(location);gl.vertexAttribPointer(location,size,gl.FLOAT,false,APPEARANCE_STRIDE*4,offset*4);}offset+=size;});return {vao,buffer,count:data.length/APPEARANCE_STRIDE,mode};}
  setForest(forest){const gl=this.gl;gl.deleteBuffer(this.mesh.buffer);gl.deleteVertexArray(this.mesh.vao);this.mesh=this.upload(forestAppearance(forest,{foliage:this.foliage}),gl.TRIANGLES);this.forest=forest;this.shadowDirty=true;}
  setFoliage(value){if(this.foliage!==value){this.foliage=value;this.setForest(this.forest);}}
  zoom(factor){this.camera.zoom=Math.max(.65,Math.min(4,this.camera.zoom*factor));this.redraw();}
  redraw(){if(this.last)this.draw(...this.last);}
  setHighlightedTree(tree){this.highlightedTree=tree;}
  focusTree(tree){Object.assign(this.camera,{follow:false,pitch:.65,zoom:3.1,target:[tree.x,tree.y,tree.trunk.a[2]+tree.height/2]});}
  setReflectionTrail(points){this.reflectionTrail=points;}
  setSignalMode(value){this.signalMode=value;this.redraw();}
  focusSignal(signal){if(!signal)return;const d=signal.trace.direction;Object.assign(this.camera,{yaw:Math.atan2(-d[0],d[1]),pitch:Math.max(-1.4,-Math.asin(d[2])),zoom:1,follow:true});this.redraw();}
  setAtmosphere(value){this.lighting=value;this.redraw();}
  trailView(){Object.assign(this.camera,{yaw:0,pitch:-.04,zoom:1,follow:true});this.redraw();}
  resetView(){Object.assign(this.camera,{yaw:-.7,pitch:.62,zoom:1.35,target:[0,0,5],follow:false});this.redraw();}
  focusReflection(path,receiver){if(this.camera.follow)this.camera.pitch=.65;this.camera.follow=false;this.camera.target=receiver.map((v,i)=>(v+path.point[i])/2);this.camera.zoom=3;this.redraw();}
  draw(receiver,signals,selected,ground){
    this.last=[receiver,signals,selected,ground];if(this.camera.follow)this.camera.target=receiver.slice();const gl=this.gl,ratio=Math.min(devicePixelRatio||1,1.5),w=this.canvas.clientWidth,h=this.canvas.clientHeight;if(!w||!h)return;
    if(this.canvas.width!==Math.round(w*ratio)||this.canvas.height!==Math.round(h*ratio)){this.canvas.width=Math.round(w*ratio);this.canvas.height=Math.round(h*ratio);}
    gl.useProgram(this.program);gl.uniform1i(this.uniforms.atlas,0);gl.uniform1i(this.uniforms.shadow,1);gl.uniform1f(this.uniforms.lit,1);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,this.atlas);
    if(this.shadowDirty){gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,null);gl.bindFramebuffer(gl.FRAMEBUFFER,this.shadowFramebuffer);gl.viewport(0,0,1536,1536);gl.clear(gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.uniform1f(this.uniforms.shadowPass,1);gl.uniform1f(this.uniforms.time,0);gl.bindVertexArray(this.mesh.vao);gl.drawArrays(gl.TRIANGLES,0,this.mesh.count);gl.bindFramebuffer(gl.FRAMEBUFFER,null);this.shadowDirty=false;}
    gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,this.shadowTexture);gl.viewport(0,0,this.canvas.width,this.canvas.height);gl.clearColor(.65,.72,.65,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.disable(gl.DEPTH_TEST);
    const frame=forestCameraFrame(this.camera,w,h),view=frame.matrix,scales=frame.scale;
    gl.useProgram(this.skyProgram);gl.uniform1f(this.skyUniforms.overcast,this.lighting==="overcast"?1:0);gl.uniformMatrix3fv(this.skyUniforms.view,false,view);gl.uniform3fv(this.skyUniforms.scale,frame.skyScale);gl.bindVertexArray(this.skyVAO);gl.drawArrays(gl.TRIANGLES,0,3);
    gl.enable(gl.DEPTH_TEST);gl.useProgram(this.program);gl.uniform1f(this.uniforms.overcast,this.lighting==="overcast"?1:0);gl.uniform1f(this.uniforms.shadowPass,0);gl.uniform1f(this.uniforms.time,this.reducedMotion?0:performance.now()/1000);gl.uniformMatrix3fv(this.uniforms.view,false,view);gl.uniform3fv(this.uniforms.scale,scales);gl.uniform3fv(this.uniforms.eye,frame.eye);gl.uniform1f(this.uniforms.perspective,this.camera.follow?1:0);gl.enable(gl.SAMPLE_ALPHA_TO_COVERAGE);gl.bindVertexArray(this.mesh.vao);gl.drawArrays(gl.TRIANGLES,0,this.mesh.count);gl.disable(gl.SAMPLE_ALPHA_TO_COVERAGE);
    const packet=forestSignalOverlay(receiver,signals,selected,ground,frame,this.signalMode,this.reflectionTrail);
    drawForestSignalOverlay(this.overlay,packet,w,h);
    if(this.overlay)drawTreeCauseOutline(this.overlay,this.highlightedTree,frame,this.signalMode!=="off");
    this.canvas.dataset.signalMode=this.signalMode;this.canvas.dataset.signalRays=String(packet.labels.length);
    this.canvas.dataset.reflection=String(Boolean(ground));this.canvas.dataset.view=this.camera.follow?"trail":"overview";this.canvas.dataset.vertices=String(this.mesh.count);this.canvas.dataset.appearance="woodland-1";this.canvas.dataset.lighting=this.lighting;
    const stage=this.canvas.closest(".stage");if(stage){stage.querySelector(".stage-label span").textContent=this.camera.follow?"Drag to look around · Scroll to zoom":"Drag to orbit · Scroll to zoom";stage.querySelector("#forest-trail-view").setAttribute("aria-pressed",String(this.camera.follow));}

  }
}
