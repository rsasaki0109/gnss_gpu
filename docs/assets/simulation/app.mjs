import { C } from "./orbit.mjs";
import { decodeMesh } from "./mesh.mjs";
import { CityRenderer } from "./renderer.mjs?v=surface-1";
import { RouteNavigator, stepEpoch } from "./route-navigation.mjs?v=route-1";
import { clampViewPitch } from "./view-geometry.mjs";

const $ = id => document.getElementById(id);
const canvas = $("scene"), settings = { epoch: 8000, antenna: 0, mask: 10 };
const colors = { LOS: "#85e4bc", NLOS: "#efbc79", BLOCKED: "#ec8f91" };
let scene, renderer, worker, snapshot, selected, busy = true, pending, revision = 0, nextId = 0;
let playing = false, dirty = true, previousTime = 0, playbackEpoch = settings.epoch, showReflections = true;
let routeNavigator, controlsReady = false;

function status(message, error = false) {
  $("status-text").textContent = message; $("trace-status").classList.toggle("error", error);
}
function fail(error) {
  controlsReady = false; canvas.dataset.error = "true"; setPlaying(false); clearSurfaceFocus(); status("Unable to simulate", true); syncRoute(); syncFocusUI();
  $("toast").textContent = error.message; $("toast").hidden = false;
  for (const input of document.querySelectorAll("[data-load-control]")) input.disabled = true;
}
function gpst(tow) { return new Date((scene.week * 604800 + tow) * 1000 + Date.UTC(1980, 0, 6)).toISOString().slice(11, 23); }
function syncControls() {
  $("antenna-out").textContent = `+${settings.antenna.toFixed(1)} m`; $("mask-out").textContent = `${settings.mask}°`;
  $("progress").value = settings.epoch;
  $("progress-out").textContent = scene ? `${gpst(scene.track[settings.epoch][0])} GPST` : "Loading…";
  $("preset").value = [...$("preset").options].some(option => option.value === String(settings.epoch)) ? String(settings.epoch) : "";
  for (const id of ["antenna", "mask", "progress"]) {
    const input = $(id), fill = 100 * (Number(input.value) - Number(input.min)) / (Number(input.max) - Number(input.min));
    input.style.setProperty("--fill", `${fill}%`);
  }
  syncRoute();
}
function syncRoute() {
  if (!scene || !routeNavigator) return;
  const ready = currentTrace();
  const label = `Pose ${(settings.epoch + 1).toLocaleString("en-US")} of ${scene.track.length.toLocaleString("en-US")} · ${gpst(scene.track[settings.epoch][0])} GPST`;
  $("route-selection").textContent = label;
  $("route-trace").textContent = !controlsReady ? canvas.dataset.error ? "Simulation unavailable" : "Preparing full-mesh tracer…" : ready ? "Trace matches selected pose" : "Tracing selected pose…";
  $("route-trace").dataset.ready = String(ready);
  for (const button of document.querySelectorAll("[data-route-step]")) {
    button.disabled = !controlsReady || stepEpoch(settings.epoch, Number(button.dataset.routeStep), scene.track.length) === settings.epoch;
  }
  routeNavigator.update(settings.epoch, ready, controlsReady, label);
  syncFocusUI();
}
function currentTrace() { return Boolean(controlsReady && snapshot?.epoch === settings.epoch && snapshot.antenna === settings.antenna && snapshot.mask === settings.mask); }
function seekEpoch(epoch) {
  if (!controlsReady) return;
  setPlaying(false); settings.epoch = stepEpoch(epoch, 0, scene.track.length);
  revision++; syncControls(); requestTrace();
}
function setPlaying(value) {
  if (value) clearSurfaceFocus();
  playing = value; playbackEpoch = settings.epoch; $("play").setAttribute("aria-pressed", String(value));
  $("play-icon").textContent = value ? "Ⅱ" : "▶"; $("play-label").textContent = value ? "Pause" : "Play · 10×";
}
function send(item) { busy = true; worker.postMessage(item); }
function updateMap(stats) {
  canvas.dataset.mapState = stats.state; canvas.dataset.mapTiles = stats.ready; canvas.dataset.mapTotal = stats.total;
  $("map-credit").hidden = stats.state === "off";
  const text = stats.state === "off" ? "Map off" : stats.state === "ready" ? "OpenStreetMap · live basemap" : stats.state === "partial" ? "Map partly unavailable · toggle to retry" : stats.state === "unavailable" ? "Map unavailable · simulation active" : "Loading OpenStreetMap…";
  if ($("map-status").textContent !== text) $("map-status").textContent = text;
  dirty = true;
}
function requestTrace() {
  if (!worker) return;
  clearSurfaceFocus();
  const item = { type: "trace", id: ++nextId, revision, settings: { ...settings } };
  status("Tracing full mesh…"); if (busy) pending = item; else send(item);
}
function updateReadout(milliseconds) {
  for (const name of ["LOS", "NLOS", "BLOCKED"]) $(`${name.toLowerCase()}-count`).textContent = snapshot.counts[name];
  $("scene-description").textContent = `${snapshot.latitude.toFixed(6)}° N  ${snapshot.longitude.toFixed(6)}° E`;
  $("osm-location").href = `https://www.openstreetmap.org/?mlat=${snapshot.latitude.toFixed(6)}&mlon=${snapshot.longitude.toFixed(6)}#map=17/${snapshot.latitude.toFixed(6)}/${snapshot.longitude.toFixed(6)}`;
  $("epoch-readout").textContent = `${snapshot.epoch + 1} / ${scene.track.length.toLocaleString("en-US")} · ${gpst(snapshot.tow)} GPST`;
  $("gps-readout").textContent = `${snapshot.satellites.length} GPS above ${snapshot.mask}° · ${Math.round(milliseconds)} ms / epoch`;
  status("Live · PLATEAU geometry");
  Object.assign(canvas.dataset, { ready: "true", epoch: snapshot.epoch, antenna: snapshot.antenna, mask: snapshot.mask, triangles: scene.stats.triangles, receiver: snapshot.receiver.join(",") });
  if (!snapshot.satellites.some(s => s.id === selected)) selected = (snapshot.satellites.find(s => s.status === "NLOS") ?? snapshot.satellites[0])?.id;
  const buttons = snapshot.satellites.map(signal => {
    const button = document.createElement("button"); button.type = "button"; button.className = "satellite"; button.dataset.satellite = signal.id;
    button.setAttribute("aria-pressed", String(signal.id === selected));
    button.setAttribute("aria-label", `${signal.id}, elevation ${signal.elevation.toFixed(1)} degrees, ${signal.status}`); button.style.color = colors[signal.status];
    const name = document.createElement("span"); name.className = "sat-id"; name.textContent = signal.id;
    const elevation = document.createElement("span"); elevation.className = "elevation"; elevation.textContent = `${signal.elevation.toFixed(1)}°`;
    const tag = document.createElement("span"); tag.className = "path-tag"; tag.textContent = signal.status === "NLOS" ? "REFLECTED" : signal.status;
    button.append(name, elevation, tag); button.addEventListener("click", () => { if (selected !== signal.id) clearSurfaceFocus(); selected = signal.id; inspect(); dirty = true; }); return button;
  });
  $("satellites").replaceChildren(...buttons);
  $("spp-all").textContent = snapshot.spp.all ? `${snapshot.spp.all.horizontal.toFixed(2)} m` : "Unavailable";
  $("spp-los").textContent = snapshot.spp.los ? `${snapshot.spp.los.horizontal.toFixed(2)} m` : "Unavailable";
  $("spp-count").textContent = `${snapshot.spp.all?.satellites ?? 0} usable paths · 4 required, with full rank`;
  inspect(); syncRoute(); dirty = true;
}
function inspect() {
  for (const button of document.querySelectorAll("[data-satellite]")) button.setAttribute("aria-pressed", String(button.dataset.satellite === selected));
  const signal = snapshot.satellites.find(s => s.id === selected); $("selected-id").textContent = signal?.id ?? "—";
  syncFocusUI();
  if (!signal) { $("channel-status").textContent = "No GPS satellite above this mask."; return; }
  $("channel-status").textContent = signal.status === "LOS" ? "A direct path reaches the recorded receiver position." : signal.status === "NLOS" ? "Direct path blocked. A single reflection reaches the receiver." : "No unobstructed direct or single-bounce path in this mesh.";
  $("channel-status").style.color = colors[signal.status];
  $("direct-value").textContent = signal.directHit ? `Blocked at ${signal.directHit.distance.toFixed(1)} m` : "Clear within loaded mesh";
  $("path-count").textContent = signal.reflectionCount;
  $("excess-value").textContent = signal.reflection ? `+${signal.reflection.extra.toFixed(3)} m` : "—";
  $("delay-value").textContent = signal.reflection ? `+${(signal.reflection.extra / C * 1e9).toFixed(2)} ns` : "—";
  const hit = renderer?.focus?.kind === "blockage" ? signal.directHit : signal.reflection ?? signal.directHit, object = hit ? scene.objects[hit.object] : null;
  $("feature-id").textContent = object ? `${object.id} · LoD ${object.lod} · triangle ${hit.triangle}` : "No intersected feature";
  $("feature-id").title = object?.source ?? "";
}
function syncFocusUI() {
  const signal = snapshot?.satellites.find(s => s.id === selected), focus = renderer?.focus;
  $("focus-reflection").disabled = !currentTrace() || !signal?.reflection;
  $("focus-blockage").disabled = !currentTrace() || !signal?.directHit;
  $("focus-reflection").setAttribute("aria-pressed", String(focus?.kind === "reflection"));
  $("focus-blockage").setAttribute("aria-pressed", String(focus?.kind === "blockage"));
  $("focus-exit").hidden = !focus;
  $("scene-name").textContent = focus ? `${focus.satellite} · ${focus.kind === "reflection" ? "Reflecting surface" : "Obstructing surface"}` : "Tokyo · Odaiba";
  $("overview").setAttribute("aria-pressed", String(Boolean(renderer?.camera.overview)));
  updateViewCaption();
}
function updateViewCaption() {
  const text = renderer?.focus ? `PLATEAU 2022 / ${renderer.camera.pitch < 0 ? "UNDERSIDE VIEW" : "SURFACE VIEW"}` : "PLATEAU / 2022";
  if ($("view-caption").textContent !== text) $("view-caption").textContent = text;
}
function clearSurfaceFocus() {
  if (!renderer?.focus) return;
  renderer.clearFocus(); inspect(); dirty = true;
}
function focusSurface(kind) {
  if (!currentTrace()) return;
  if (renderer.focus?.kind === kind) { clearSurfaceFocus(); return; }
  const signal = snapshot.satellites.find(s => s.id === selected);
  setPlaying(false);
  if (!renderer.focusSurface(snapshot, signal, kind)) return;
  if (kind === "reflection") { showReflections = true; $("reflections").checked = true; }
  inspect(); dirty = true; canvas.scrollIntoView({ block: "center" }); canvas.focus({ preventScroll: true });
}
$("focus-reflection").addEventListener("click", () => focusSurface("reflection"));
$("focus-blockage").addEventListener("click", () => focusSurface("blockage"));
$("focus-exit").addEventListener("click", () => { clearSurfaceFocus(); canvas.focus({ preventScroll: true }); });
for (const id of ["antenna", "mask"]) $(id).addEventListener("input", event => { settings[id] = Number(event.target.value); revision++; syncControls(); requestTrace(); });
$("progress").addEventListener("input", event => seekEpoch(Number(event.target.value)));
$("preset").addEventListener("change", event => seekEpoch(Number(event.target.value)));
for (const button of document.querySelectorAll("[data-route-step]")) button.addEventListener("click", () => seekEpoch(settings.epoch + Number(button.dataset.routeStep)));
$("play").addEventListener("click", () => setPlaying(!playing));
$("reflections").addEventListener("change", event => { showReflections = event.target.checked; if (!showReflections && renderer?.focus?.kind === "reflection") clearSurfaceFocus(); dirty = true; });
$("basemap").addEventListener("change", event => { renderer?.basemap.setEnabled(event.target.checked); dirty = true; });
$("reset").addEventListener("click", () => {
  setPlaying(false); Object.assign(settings, { epoch: 8000, antenna: 0, mask: 10 });
  $("antenna").value = 0; $("mask").value = 10; $("preset").value = "8000";
  resetCamera(); revision++; syncControls(); requestTrace();
});
function resetCamera() { clearSurfaceFocus(); if (renderer) Object.assign(renderer.camera, { yaw: -.63, pitch: .82, zoom: 1, overview: false, span: 740 }); syncFocusUI(); dirty = true; }
$("reset-view").addEventListener("click", resetCamera);
$("zoom-in").addEventListener("click", () => { if (renderer) renderer.camera.zoom = Math.min(6, renderer.camera.zoom * 1.25); dirty = true; });
$("zoom-out").addEventListener("click", () => { if (renderer) renderer.camera.zoom = Math.max(.3, renderer.camera.zoom / 1.25); dirty = true; });
$("overview").addEventListener("click", () => {
  if (!renderer) return; const overview = renderer.focus ? true : !renderer.camera.overview;
  clearSurfaceFocus(); renderer.camera.overview = overview; renderer.camera.zoom = 1;
  $("overview").setAttribute("aria-pressed", String(renderer.camera.overview)); dirty = true;
});
let drag;
canvas.addEventListener("pointerdown", event => {
  if (!renderer) return; canvas.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, y: event.clientY, yaw: renderer.camera.yaw, pitch: renderer.camera.pitch };
});
canvas.addEventListener("pointermove", event => {
  if (!drag) return; renderer.camera.yaw = drag.yaw + (event.clientX - drag.x) * .006;
  renderer.camera.pitch = clampViewPitch(drag.pitch + (event.clientY - drag.y) * .005, Boolean(renderer.focus)); dirty = true;
});
for (const type of ["pointerup", "pointercancel"]) canvas.addEventListener(type, () => { drag = null; });
canvas.addEventListener("wheel", event => {
  event.preventDefault(); if (!renderer) return;
  renderer.camera.zoom = Math.max(.3, Math.min(6, renderer.camera.zoom * Math.exp(-event.deltaY * .001))); dirty = true;
}, { passive: false });
canvas.addEventListener("keydown", event => {
  if (!renderer) return;
  if (event.code === "Space") { event.preventDefault(); setPlaying(!playing); }
  if (event.key.startsWith("Arrow")) {
    event.preventDefault(); renderer.camera.yaw += event.key === "ArrowLeft" ? -.1 : event.key === "ArrowRight" ? .1 : 0;
    const direction = event.key === "ArrowUp" ? .1 : event.key === "ArrowDown" ? -.1 : 0;
    renderer.camera.pitch = clampViewPitch(renderer.camera.pitch + direction, Boolean(renderer.focus), direction); dirty = true;
  }
});
new ResizeObserver(() => { renderer?.resize(); dirty = true; }).observe(canvas);
canvas.addEventListener("webglcontextlost", event => { event.preventDefault(); fail(new Error("The 3D graphics context was lost. Reload this page to restore the view.")); });
document.addEventListener("visibilitychange", () => { if (document.hidden) setPlaying(false); });

async function asset(entry) {
  const response = await fetch(new URL(`../plateau/${entry.file}`, import.meta.url));
  if (!response.ok) throw new Error(`Unable to load ${entry.file}: HTTP ${response.status}`);
  const bytes = await response.arrayBuffer();
  const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", bytes))].map(v => v.toString(16).padStart(2, "0")).join("");
  if (bytes.byteLength !== entry.bytes || hash !== entry.sha256) throw new Error(`Data integrity check failed: ${entry.file}`);
  if (!("DecompressionStream" in window)) throw new Error("This browser needs gzip DecompressionStream support. Please update your browser.");
  return new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
async function initialize() {
  status("Loading verified PLATEAU data · 4.9 MB…");
  const response = await fetch(new URL("../plateau/manifest.json", import.meta.url));
  if (!response.ok) throw new Error(`Unable to load data manifest: HTTP ${response.status}`);
  const manifest = await response.json();
  const [meshBuffer, sceneBuffer] = await Promise.all([asset(manifest.mesh), asset(manifest.scene)]);
  scene = JSON.parse(new TextDecoder().decode(sceneBuffer));
  renderer = new CityRenderer(canvas, $("annotations"), scene, decodeMesh(meshBuffer), updateMap);
  routeNavigator = new RouteNavigator($("route"), scene, seekEpoch, epoch => {
    $("route-preview").textContent = epoch === null ? "Click or tap the route to select a recorded pose." : `Preview · ${gpst(scene.track[epoch][0])} GPST · pose ${(epoch + 1).toLocaleString("en-US")}`;
  });
  $("progress").max = scene.track.length - 1;
  $("mesh-readout").textContent = `${scene.stats.triangles.toLocaleString("en-US")} triangles · ${scene.provenance.files.length} CityGML tiles`;
  $("geometry-readout").textContent = `${scene.stats.buildings.toLocaleString("en-US")} building components · ${scene.stats.bridges} bridge components · ${scene.stats.holes} interior rings preserved`;
  $("repair-readout").textContent = `${scene.stats.topology_repaired_faces} source face repaired with GEOS linework; ${scene.stats.inserted_boundary_vertices} boundary-intersection vertices inserted. No mesh decimation or coordinate quantization.`;
  status("Indexing all 320,932 triangles…");
  worker = new Worker(new URL("./city-worker.mjs", import.meta.url), { type: "module" });
  worker.onerror = event => fail(new Error(event.message || "The simulation worker stopped."));
  worker.onmessage = ({ data }) => {
    if (data.type === "error") { fail(new Error(data.error)); return; }
    if (data.type === "ready") { busy = false; controlsReady = true; for (const input of document.querySelectorAll("[data-load-control]")) input.disabled = false; syncRoute(); requestTrace(); return; }
    busy = false;
    if (data.revision === revision) { clearSurfaceFocus(); snapshot = data.snapshot; canvas.dataset.traceId = data.id; updateReadout(data.milliseconds); }
    if (pending) { const item = pending; pending = null; send(item); }
  };
  worker.postMessage({ type: "init", scene, buffer: meshBuffer }, [meshBuffer]); syncControls();
}
function frame(time) {
  const elapsed = previousTime ? Math.min(time - previousTime, 250) : 0; previousTime = time;
  if (playing && scene) {
    playbackEpoch += elapsed / 10; // 10 Hz recorded route, played at 10× wall time.
    const epoch = Math.min(scene.track.length - 1, Math.floor(playbackEpoch));
    if (epoch !== settings.epoch) { settings.epoch = epoch; syncControls(); requestTrace(); }
    if (epoch === scene.track.length - 1) setPlaying(false);
  }
  if (dirty && renderer) { renderer.draw(snapshot, selected, showReflections); updateViewCaption(); dirty = false; }
  requestAnimationFrame(frame);
}
syncControls(); requestAnimationFrame(frame); initialize().catch(fail);
