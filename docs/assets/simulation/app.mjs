import { C, SCENES, ROUTE, RECEIVER_HEIGHT, add, sub, mul, contains, buildScene } from "./model.mjs";

const $ = id => document.getElementById(id);
const canvas = $("scene"), ctx = canvas.getContext("2d");
const settings = { preset: "canyon", heightScale: 1, rotation: 0, progress: 0.35, receiver: null };
const camera = { yaw: -0.63, pitch: 0.72, zoom: 1 };
const colors = { LOS: "#85e4bc", NLOS: "#efbc79", BLOCKED: "#ec8f91" };
let snapshot = null, selected = null, showReflections = true, playing = false, dirty = true;
let width = 800, height = 565, scale = 1, worker, busy = false, pending = null;
let revision = 0, nextId = 0, lastPainted = 0, previousTime = 0, lastRequest = 0;
let toastTimer;

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

function rangePaint(input) {
  const fill = 100 * (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min));
  input.style.setProperty("--fill", `${fill}%`);
}

function syncControls() {
  $("height-out").textContent = `${settings.heightScale.toFixed(2)}×`;
  $("rotation-out").textContent = `${settings.rotation}°`;
  $("progress-out").textContent = settings.receiver ? "Placed manually" : `${Math.round(settings.progress * 100)}%`;
  $("progress").value = settings.progress * 100;
  for (const id of ["height", "rotation", "progress"]) rangePaint($(id));
}

function setPlaying(value) {
  playing = value;
  $("play").setAttribute("aria-pressed", String(value));
  $("play-icon").textContent = value ? "Ⅱ" : "▶";
  $("play-label").textContent = value ? "Pause" : "Play route";
  if (value && settings.receiver) {
    settings.receiver = null;
    revision++;
    requestTrace();
  }
  syncControls();
}

function send(item) {
  busy = true;
  worker.postMessage(item);
}

function requestTrace() {
  if (!worker) return;
  const item = { id: ++nextId, revision, settings: { ...settings, receiver: settings.receiver?.slice() ?? null } };
  if (busy) pending = item;
  else send(item);
}

function updateReadout() {
  $("los-count").textContent = snapshot.counts.LOS;
  $("nlos-count").textContent = snapshot.counts.NLOS;
  $("blocked-count").textContent = snapshot.counts.BLOCKED;
  $("scene-name").textContent = SCENES[snapshot.preset].name;
  $("scene-description").textContent = SCENES[snapshot.preset].description;
  if ($("status-text").textContent !== "Live geometry") $("status-text").textContent = "Live geometry";
  canvas.dataset.ready = "true";
  canvas.dataset.preset = snapshot.preset;
  canvas.dataset.rotation = snapshot.rotation;
  canvas.dataset.height = snapshot.heightScale;
  canvas.dataset.receiver = snapshot.receiver.join(",");
  if (!selected) selected = (snapshot.satellites.find(s => s.status === "NLOS") ?? snapshot.satellites.find(s => s.directHit) ?? snapshot.satellites[0]).id;
  for (const satellite of snapshot.satellites) {
    const button = document.querySelector(`[data-satellite="${satellite.id}"]`);
    button.setAttribute("aria-pressed", String(satellite.id === selected));
    button.style.color = colors[satellite.status];
    button.querySelector(".elevation").textContent = `${Math.round(satellite.elevation)}°`;
    button.querySelector(".path-tag").textContent = satellite.status === "NLOS" ? "REFLECTED" : satellite.status === "BLOCKED" ? "BLOCKED" : "LOS";
    button.querySelector(".path-tag").style.color = colors[satellite.status];
    button.setAttribute("aria-label", `${satellite.id}, elevation ${Math.round(satellite.elevation)} degrees, ${satellite.status === "NLOS" ? "reflected only" : satellite.status === "BLOCKED" ? "no modeled path" : "direct line of sight"}`);
  }
  const signal = snapshot.satellites.find(s => s.id === selected);
  $("selected-id").textContent = signal.id;
  $("channel-status").textContent = signal.status === "LOS" ? "A direct signal path reaches the receiver." : signal.status === "NLOS" ? "The direct path is blocked. A wall reflection reaches the receiver." : "No clear direct or single-bounce path reaches the receiver.";
  $("channel-status").style.color = colors[signal.status];
  $("direct-value").textContent = signal.directHit ? "Blocked" : "Clear";
  $("path-count").textContent = signal.reflectionCount;
  $("excess-value").textContent = signal.reflection ? `+${signal.reflection.extra.toFixed(2)} m` : "—";
  $("delay-value").textContent = signal.reflection ? `+${(signal.reflection.extra / C * 1e9).toFixed(1)} ns` : "—";
  dirty = true;
}

// Orthographic camera. Geometry and tracing stay in local east/north/up metres.
function project(point) {
  const c = Math.cos(camera.yaw), s = Math.sin(camera.yaw);
  const x = point[0] * c - point[1] * s;
  const y = point[0] * s + point[1] * c;
  return [width * 0.5 + x * scale, height * 0.63 + (y * Math.sin(camera.pitch) - point[2] * Math.cos(camera.pitch)) * scale];
}

function polygon(points, fill, stroke = null, lineWidth = 1) {
  ctx.beginPath();
  points.forEach((point, i) => { const p = project(point); if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
  ctx.closePath();
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke(); }
}

function line(points, color, lineWidth = 1, dash = []) {
  ctx.beginPath();
  points.forEach((point, i) => { const p = project(point); if (i) ctx.lineTo(...p); else ctx.moveTo(...p); });
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash);
  ctx.stroke();
  ctx.setLineDash([]);
}

function marker(point, color, radius = 3) {
  const p = project(point);
  ctx.beginPath(); ctx.arc(...p, radius, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill();
}

function building(box, highlight) {
  const [x0, y0] = box.min, [x1, y1, h] = box.max;
  const corners = [[x0,y0,0],[x1,y0,0],[x1,y1,0],[x0,y1,0],[x0,y0,h],[x1,y0,h],[x1,y1,h],[x0,y1,h]];
  const visible = [
    { indices: [0,1,5,4], normal: [0,-1], brightness: 0.75 },
    { indices: [1,2,6,5], normal: [1,0], brightness: 0.9 },
    { indices: [2,3,7,6], normal: [0,1], brightness: 0.8 },
    { indices: [3,0,4,7], normal: [-1,0], brightness: 0.66 },
  ];
  const edge = highlight ? `${highlight}b0` : "#6a879b35";
  const base = 47 + box.tone * 13;
  for (const face of visible) {
    if (face.normal[0] * Math.sin(camera.yaw) + face.normal[1] * Math.cos(camera.yaw) < 0) continue;
    const b = base * face.brightness;
    polygon(face.indices.map(i => corners[i]), `rgb(${b},${b+16},${b+27})`, edge, highlight ? 1.1 : 0.7);
  }
  polygon([corners[4],corners[5],corners[6],corners[7]], `rgb(${base+13},${base+29},${base+40})`, edge, highlight ? 1.1 : 0.7);
  if (h > 38) {
    const inset = 4;
    polygon([[x0+inset,y0+inset,h+0.1],[x1-inset,y0+inset,h+0.1],[x1-inset,y1-inset,h+0.1],[x0+inset,y1-inset,h+0.1]], "#172a343b", "#8da7b01a", 0.6);
  }
}

function rayEnd(start, direction) {
  const distance = Math.min(390, 230 / Math.max(direction[2], 0.1));
  return add(start, mul(direction, distance));
}

function draw() {
  dirty = false;
  ctx.clearRect(0, 0, width, height);
  scale = Math.min(width / 560, height / 430) * camera.zoom;
  polygon([[-240,-166,0],[240,-166,0],[240,166,0],[-240,166,0]], "#152631", "#3450636b");
  for (let x = -240; x <= 240; x += 20) line([[x,-166,0],[x,166,0]], "#43627825", 0.7);
  for (let y = -160; y <= 160; y += 20) line([[-240,y,0],[240,y,0]], "#43627825", 0.7);
  polygon([[-238,-12,0],[238,-12,0],[238,12,0],[-238,12,0]], "#25384470");
  polygon([[-12,-164,0],[12,-164,0],[12,164,0],[-12,164,0]], "#25384470");
  line(ROUTE.map(p => [p[0],p[1],0.15]), "#8cbbef65", 1.5, [4,5]);
  if (!snapshot) return;
  const chosen = snapshot.satellites.find(s => s.id === selected);
  for (const box of snapshot.buildings) {
    const h = box.max[2], dx = h * 0.27, dy = -h * 0.18;
    polygon([[box.min[0],box.min[1],0],[box.max[0],box.min[1],0],[box.max[0]+dx,box.max[1]+dy,0],[box.min[0]+dx,box.max[1]+dy,0]], "#050d1350");
  }
  for (const satellite of snapshot.satellites) line([snapshot.receiver, rayEnd(snapshot.receiver, satellite.direction)], `${colors[satellite.status]}20`, 0.7, satellite.directHit ? [3,5] : []);
  const sorted = snapshot.buildings.slice().sort((a,b) => {
    const depth = box => (box.min[0]+box.max[0])*Math.sin(camera.yaw)+(box.min[1]+box.max[1])*Math.cos(camera.yaw);
    return depth(a) - depth(b);
  });
  for (const box of sorted) {
    const highlight = chosen?.reflection?.buildingId === box.id ? colors.NLOS : chosen?.directHit?.buildingId === box.id ? colors.BLOCKED : null;
    building(box, highlight);
  }
  for (const satellite of snapshot.satellites) {
    const active = satellite.id === selected, end = rayEnd(snapshot.receiver, satellite.direction);
    if (satellite.directHit) {
      line([snapshot.receiver,satellite.directHit.point], active ? "#ec8f91e0" : "#ec8f913c", active ? 1.8 : 0.8, [4,4]);
      if (active) { line([satellite.directHit.point,end], "#ec8f9150", 1, [3,6]); marker(satellite.directHit.point, "#ffd7d8", 3.3); }
    } else line([snapshot.receiver,end], active ? "#85e4bcf0" : "#85e4bc42", active ? 1.8 : 0.8);
    if (showReflections && satellite.reflection) {
      const q = satellite.reflection.point, direction = sub(satellite.source, q), size = Math.hypot(...direction);
      line([snapshot.receiver,q,rayEnd(q,mul(direction,1/size))], active ? "#efbc79f5" : "#efbc7929", active ? 2 : 0.7);
      if (active) marker(q, "#ffddac", 3.2);
    }
    if (active) {
      const p = project(end);
      const x = Math.max(55,Math.min(width-66,p[0])), y = Math.max(93,Math.min(height-58,p[1]));
      ctx.fillStyle = "#0e1b25e8"; ctx.strokeStyle = `${colors[satellite.status]}88`; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.roundRect(x-23,y-12,46,24,5); ctx.fill(); ctx.stroke();
      ctx.fillStyle = colors[satellite.status]; ctx.font = "10px system-ui"; ctx.textAlign = "center"; ctx.fillText(satellite.id,x,y+3.5);
    }
  }
  const receiver = project(snapshot.receiver);
  const glow = ctx.createRadialGradient(...receiver,2,...receiver,24);
  glow.addColorStop(0,"#8cbbef55");glow.addColorStop(1,"#8cbbef00");
  ctx.fillStyle=glow;ctx.fillRect(receiver[0]-24,receiver[1]-24,48,48);
  marker(snapshot.receiver,"#a9d6ff",4);
  ctx.strokeStyle="#c6e5ff";ctx.lineWidth=1;ctx.beginPath();ctx.arc(...receiver,8,0,Math.PI*2);ctx.stroke();
  ctx.fillStyle="#d7eaff";ctx.font="bold 9px system-ui";ctx.textAlign="left";ctx.fillText("RX",receiver[0]+13,receiver[1]+3);
  // A scale in scene metres, independent of decorative satellite endpoint length.
  const sy=height-48,sx=23, span=50*scale;
  ctx.strokeStyle="#a0b3bf88";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(sx,sy-3);ctx.lineTo(sx,sy+3);ctx.moveTo(sx,sy);ctx.lineTo(sx+span,sy);ctx.moveTo(sx+span,sy-3);ctx.lineTo(sx+span,sy+3);ctx.stroke();
  ctx.fillStyle="#8ca3b3";ctx.font="8px system-ui";ctx.textAlign="left";ctx.fillText("50 m",sx,sy-8);
}

for (const id of ["G05","G12","G19","E11","E24","E31","R04","R11","J02","J07"]) {
  const button = document.createElement("button");
  button.type = "button"; button.className = "satellite"; button.dataset.satellite = id;
  button.setAttribute("aria-pressed","false");
  button.innerHTML = `<span class="satellite-id"><i class="satellite-dot"></i>${id}</span><span class="satellite-meta"><span class="elevation">—</span><span class="path-tag">—</span></span>`;
  button.addEventListener("click", () => { if (snapshot) { selected=id; updateReadout(); } });
  $("satellites").append(button);
}

$("preset").addEventListener("change",event=>{settings.preset=event.target.value;settings.receiver=null;revision++;syncControls();requestTrace();});
for (const [id,key] of [["height","heightScale"],["rotation","rotation"]]) $(id).addEventListener("input",event=>{settings[key]=Number(event.target.value);revision++;syncControls();requestTrace();});
$("progress").addEventListener("input",event=>{settings.progress=Number(event.target.value)/100;settings.receiver=null;revision++;syncControls();requestTrace();});
$("reflections").addEventListener("change",event=>{showReflections=event.target.checked;dirty=true;});
$("play").addEventListener("click",()=>setPlaying(!playing));
$("reset").addEventListener("click",()=>{
  setPlaying(false);Object.assign(settings,{preset:"canyon",heightScale:1,rotation:0,progress:0.35,receiver:null});
  $("preset").value="canyon";$("height").value="1";$("rotation").value="0";$("reflections").checked=true;
  showReflections=true;selected=null;revision++;syncControls();requestTrace();
});
$("zoom-in").addEventListener("click",()=>{camera.zoom=Math.min(2.1,camera.zoom*1.16);dirty=true;});
$("zoom-out").addEventListener("click",()=>{camera.zoom=Math.max(0.55,camera.zoom/1.16);dirty=true;});
$("reset-view").addEventListener("click",()=>{Object.assign(camera,{yaw:-0.63,pitch:0.72,zoom:1});dirty=true;});

let pointer = null;
canvas.addEventListener("pointerdown",event=>{
  if (event.button!==0) return;
  pointer={id:event.pointerId,x:event.clientX,y:event.clientY,lastX:event.clientX,lastY:event.clientY,moved:false};
  canvas.setPointerCapture(event.pointerId);
});
canvas.addEventListener("pointermove",event=>{
  if (!pointer || event.pointerId!==pointer.id) return;
  if (Math.hypot(event.clientX-pointer.x,event.clientY-pointer.y)>5) pointer.moved=true;
  if (pointer.moved) {
    camera.yaw+=(event.clientX-pointer.lastX)*0.007;
    camera.pitch=Math.max(0.32,Math.min(1.16,camera.pitch+(event.clientY-pointer.lastY)*0.005));
    dirty=true;
  }
  pointer.lastX=event.clientX;pointer.lastY=event.clientY;
});
canvas.addEventListener("pointerup",event=>{
  if (!pointer || event.pointerId!==pointer.id) return;
  const clicked=!pointer.moved;pointer=null;
  if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  if (!clicked || !snapshot) return;
  const bounds=canvas.getBoundingClientRect();
  const localX=(event.clientX-bounds.left-width*0.5)/scale;
  const localY=(event.clientY-bounds.top-height*0.63)/(scale*Math.sin(camera.pitch));
  const point=[localX*Math.cos(camera.yaw)+localY*Math.sin(camera.yaw),-localX*Math.sin(camera.yaw)+localY*Math.cos(camera.yaw),RECEIVER_HEIGHT];
  if (Math.abs(point[0])>230 || Math.abs(point[1])>170 || buildScene(settings.preset,settings.heightScale).some(box=>contains(point,box,0.2))) {
    toast("Place the receiver on a street or in the plaza.");return;
  }
  setPlaying(false);settings.receiver=point;revision++;syncControls();requestTrace();
});
canvas.addEventListener("pointercancel",()=>{pointer=null;});
canvas.addEventListener("wheel",event=>{event.preventDefault();camera.zoom=Math.max(0.55,Math.min(2.1,camera.zoom*Math.exp(-event.deltaY*0.001)));dirty=true;},{passive:false});
canvas.addEventListener("keydown",event=>{
  if (event.code==="Space") { event.preventDefault();if(snapshot)setPlaying(!playing); }
  if (event.code==="ArrowLeft"||event.code==="ArrowRight") { event.preventDefault();camera.yaw+=(event.code==="ArrowLeft"?-1:1)*0.12;dirty=true; }
  if (event.code==="ArrowUp"||event.code==="ArrowDown") {event.preventDefault();camera.pitch=Math.max(0.32,Math.min(1.16,camera.pitch+(event.code==="ArrowUp"?1:-1)*0.07));dirty=true;}
});
document.addEventListener("visibilitychange",()=>{if(document.hidden)setPlaying(false);});

new ResizeObserver(()=>{
  const bounds=canvas.getBoundingClientRect(),ratio=Math.min(window.devicePixelRatio||1,2);
  width=bounds.width;height=bounds.height;canvas.width=Math.round(width*ratio);canvas.height=Math.round(height*ratio);
  ctx.setTransform(ratio,0,0,ratio,0,0);dirty=true;
}).observe($("stage"));

function frame(time) {
  const dt=previousTime?Math.min(time-previousTime,100):0;previousTime=time;
  if (playing) {
    settings.progress=(settings.progress+dt/85000)%1;
    if(time-lastRequest>=100){lastRequest=time;syncControls();requestTrace();}
  }
  if(dirty)draw();
  requestAnimationFrame(frame);
}
syncControls();requestAnimationFrame(frame);
try {
  worker=new Worker(new URL("./worker.mjs",import.meta.url),{type:"module"});
  // Discard an old scene after a control change; keep animation frames flowing.
  worker.onmessage=({data})=>{
    busy=false;
    if(data.revision===revision&&data.id>lastPainted){
      if(data.error){$("status-text").textContent="Check the scene";toast(data.error);setPlaying(false);}
      else{snapshot=data.snapshot;lastPainted=data.id;$("play").disabled=false;updateReadout();}
    }
    if(pending){const item=pending;pending=null;send(item);}
  };
  worker.onerror=()=>{$("status-text").textContent="Simulation unavailable";setPlaying(false);$("play").disabled=true;toast("The simulation could not load. Reload the page to try again.");};
  requestTrace();
} catch {
  $("status-text").textContent="Simulation unavailable";
  toast("This demo needs a browser with JavaScript modules and Web Workers.");
}
