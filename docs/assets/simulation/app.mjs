import { C } from "./orbit.mjs";
import { decodeMesh } from "./mesh.mjs";
import { CityRenderer } from "./renderer.mjs?v=surface-1";
import { RouteNavigator, stepEpoch } from "./route-navigation.mjs?v=route-1";
import { clampViewPitch } from "./view-geometry.mjs";
import { DEFAULT_SCENE, readSceneLink, writeSceneLink } from "./scene-link.mjs";
import { captureReference, compareReceiver } from "./comparison.mjs";
import { SkyPlot } from "./sky-plot.mjs?v=arrival-1";
import { ReflectionGeometry } from "./reflection-geometry.mjs";
import { RangeFit } from "./range-fit.mjs";
import { PositionFit } from "./position-fit.mjs";

const $ = id => document.getElementById(id);
const canvas = $("scene"), settings = { epoch: 8000, antenna: 0, mask: 10 };
const colors = { LOS: "#85e4bc", NLOS: "#efbc79", BLOCKED: "#ec8f91" };
let scene, renderer, worker, snapshot, selected, busy = true, pending, revision = 0, nextId = 0;
let playing = false, dirty = true, previousTime = 0, playbackEpoch = settings.epoch, showReflections = true;
let routeNavigator, controlsReady = false;
let linkAssets, pendingReplay;
let reference, referenceURL, comparisonReference, comparisonSnapshot;
const skyPlot = new SkyPlot($("sky-plot"), id => { setPlaying(false); selectSatellite(id); });
const reflectionPanel = new ReflectionGeometry($("reflection-panel"));
const rangePanel = new RangeFit($("range-panel"), id => { setPlaying(false); selectSatellite(id); });
const positionOverlay = new PositionFit($("position-fit"), $("fit-legend"));

function status(message, error = false) {
  $("status-text").textContent = message; $("trace-status").classList.toggle("error", error);
}
function fail(error) {
  controlsReady = false; canvas.dataset.error = "true"; setPlaying(false); clearSurfaceFocus(); status("Unable to simulate", true); syncRoute(); syncFocusUI();
  $("toast").textContent = error.message; $("toast").hidden = false;
  for (const input of document.querySelectorAll("[data-load-control]")) input.disabled = true;
  updateShare();
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
  if (value) { clearSurfaceFocus(); pendingReplay = null; $("share-panel").open = false; $("compare-panel").open = false; $("reflection-panel").open = false; $("range-panel").open = false; }
  playing = value; playbackEpoch = settings.epoch; $("play").setAttribute("aria-pressed", String(value));
  $("play-icon").textContent = value ? "Ⅱ" : "▶"; $("play-label").textContent = value ? "Pause" : "Play · 10×";
  updateSky();
  updateReflectionGeometry();
  updateRangeFit();
}
function send(item) { busy = true; worker.postMessage(item); }
function updateMap(stats) {
  canvas.dataset.mapState = stats.state; canvas.dataset.mapTiles = stats.ready; canvas.dataset.mapTotal = stats.total;
  $("map-credit").hidden = stats.state === "off";
  const text = stats.state === "off" ? "Map off" : stats.state === "ready" ? "OpenStreetMap · live basemap" : stats.state === "partial" ? "Map partly unavailable · toggle to retry" : stats.state === "unavailable" ? "Map unavailable · simulation active" : "Loading OpenStreetMap…";
  if ($("map-status").textContent !== text) $("map-status").textContent = text;
  dirty = true;
}
function requestTrace(replay = false) {
  if (!worker) return;
  if (!replay) pendingReplay = null;
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
    button.append(name, elevation, tag); button.addEventListener("click", () => selectSatellite(signal.id)); return button;
  });
  $("satellites").replaceChildren(...buttons);
  $("spp-all").textContent = snapshot.spp.all ? `${snapshot.spp.all.horizontal.toFixed(2)} m` : "Unavailable";
  $("spp-los").textContent = snapshot.spp.los ? `${snapshot.spp.los.horizontal.toFixed(2)} m` : "Unavailable";
  $("spp-count").textContent = `${snapshot.satellites.filter(signal => signal.status !== "BLOCKED").length} usable paths · ideal ranges`;
  inspect(); syncRoute(); dirty = true;
}
function selectSatellite(id) {
  pendingReplay = null; if (selected !== id) clearSurfaceFocus(); selected = id; inspect(); dirty = true;
}
function inspect() {
  for (const button of document.querySelectorAll("[data-satellite]")) button.setAttribute("aria-pressed", String(button.dataset.satellite === selected));
  const signal = snapshot.satellites.find(s => s.id === selected); $("selected-id").textContent = signal?.id ?? "—";
  syncFocusUI();
  if (!signal) { $("channel-status").textContent = "No GPS satellite above this mask."; return; }
  $("channel-status").textContent = signal.status === "LOS" ? "Direct LOS" : signal.status === "NLOS" ? "Reflected arrival · direct blocked" : "No modeled path";
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
  if (!currentTrace() || canvas.dataset.error || focus) updatePositionFit(null);
  $("focus-reflection").disabled = !currentTrace() || !signal?.reflection;
  $("focus-blockage").disabled = !currentTrace() || !signal?.directHit;
  $("focus-reflection").setAttribute("aria-pressed", String(focus?.kind === "reflection"));
  $("focus-blockage").setAttribute("aria-pressed", String(focus?.kind === "blockage"));
  $("focus-exit").hidden = !focus;
  $("scene-name").textContent = focus ? `${focus.satellite} · ${focus.kind === "reflection" ? "Reflecting surface" : "Obstructing surface"}` : "Tokyo · Odaiba";
  $("overview").setAttribute("aria-pressed", String(Boolean(renderer?.camera.overview)));
  updateViewCaption();
  updateShare();
  updateComparison();
  updateSky();
  updateReflectionGeometry();
  updateRangeFit();
}
function updateReflectionGeometry() {
  reflectionPanel.update(snapshot, selected, {ready:currentTrace(),error:Boolean(canvas.dataset.error)});
}
$("reflection-panel").addEventListener("toggle", () => { if ($("reflection-panel").open && playing) setPlaying(false); });
function updateRangeFit() {
  rangePanel.update(snapshot, selected, { ready: currentTrace(), error: Boolean(canvas.dataset.error) });
}
function updatePositionFit(view) {
  positionOverlay.update(snapshot, view, {ready:currentTrace(),error:Boolean(canvas.dataset.error),focused:Boolean(renderer?.focus)});
}
$("range-panel").addEventListener("toggle", () => { if ($("range-panel").open && playing) setPlaying(false); });
function updateSky() {
  skyPlot.update(snapshot, selected, {scene, ready:currentTrace(), error:Boolean(canvas.dataset.error), playing, time:snapshot?gpst(snapshot.tow):""});
  $("sky-pause").hidden = !playing; $("sky-pause").disabled = !controlsReady;
}
$("sky-pause").addEventListener("click", () => setPlaying(false));
$("sky-panel").addEventListener("toggle", () => { if ($("sky-panel").open && playing) setPlaying(false); });
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
  pendingReplay = null;
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
function sceneNotice(message, warning = false) {
  $("scene-link-notice").hidden = !message; $("scene-link-notice").dataset.warning = String(warning);
  $("scene-link-notice-text").textContent = message;
}
function shareStatus(message) { if ($("share-status").textContent !== message) $("share-status").textContent = message; }
function sceneURL() {
  const camera = renderer.camera;
  return writeSceneLink(location.href, { epoch: snapshot.epoch, antenna: snapshot.antenna, mask: snapshot.mask, satellite: selected, view: renderer.focus?.kind ?? (camera.overview ? "overview" : "receiver"), yaw: camera.yaw, pitch: camera.pitch, zoom: camera.zoom, basemap: $("basemap").checked, reflections: showReflections }, linkAssets);
}
function updateShare() {
  const ready = currentTrace() && Boolean(linkAssets), input = $("scene-link"), open = $("open-scene-link");
  $("copy-scene-link").disabled = !ready; input.disabled = !ready;
  open.setAttribute("aria-disabled", String(!ready)); open.tabIndex = ready ? 0 : -1;
  if (!ready) {
    input.value = ""; open.removeAttribute("href");
    shareStatus(canvas.dataset.error ? "Scene link unavailable." : "Waiting for the selected pose to finish tracing…"); return;
  }
  if (!$("share-panel").open) return;
  const link = sceneURL();
  if (input.value !== link) {
    input.value = link; open.href = link;
    shareStatus(`Link ready · pose ${(snapshot.epoch + 1).toLocaleString("en-US")} · ${selected || "no satellite selected"}`);
  }
}
function updateComparison() {
  const ready = currentTrace(), panel = $("compare-panel"), back = $("return-reference");
  $("pin-reference").disabled = !ready; $("clear-reference").disabled = !reference;
  $("pin-reference").textContent = reference ? "Replace pinned reference" : "Pin completed scene";
  back.hidden = !reference || settings.epoch === reference.epoch; back.disabled = !controlsReady;
  if (!panel.open) return;
  $("reference-info").hidden = !reference;
  if (reference) {
    $("reference-label").textContent = `Pinned · pose ${(reference.epoch + 1).toLocaleString("en-US")} · ${gpst(reference.tow)} GPST · +${reference.antenna.toFixed(1)} m · mask ${reference.mask}°`;
    $("reference-link").href = referenceURL;
  }
  $("comparison-results").hidden = true;
  const message = (state, text) => { panel.dataset.state = state; $("comparison-status").textContent = text; };
  if (!controlsReady) { message("unavailable", canvas.dataset.error ? "Comparison unavailable." : "Preparing the full-mesh tracer…"); return; }
  if (!ready) { message("pending", "Waiting for the selected pose and receiver settings to finish tracing…"); return; }
  if (!reference) { message("empty", "Pin a completed scene, then change height or elevation mask at the same pose."); return; }
  const result = compareReceiver(reference, snapshot);
  if (result.state !== "ready") { message("different-pose", "Different recorded pose. Return to the pinned pose to compare your current receiver settings, or replace the reference."); return; }
  message("ready", `Same recorded pose · ${result.changes} satellite path / mask ${result.changes === 1 ? "change" : "changes"} · current − pinned`);
  $("comparison-results").hidden = false;
  if (comparisonReference === reference && comparisonSnapshot === snapshot) return;
  comparisonReference = reference; comparisonSnapshot = snapshot;
  $("comparison-conditions").textContent = `Pinned: +${reference.antenna.toFixed(1)} m / ${reference.mask}° mask → Current: +${snapshot.antenna.toFixed(1)} m / ${snapshot.mask}° mask`;
  const cell = (row, text, tag = "td") => { const element = document.createElement(tag); element.textContent = text; row.append(element); return element; };
  const labels = { LOS: "Direct LOS", NLOS: "Reflected only", BLOCKED: "No path", bias: "Geometric bias²" };
  const value = (v, metric, delta = false) => v === null ? delta ? "—" : "Unavailable" : `${delta && v > 0 ? "+" : ""}${metric === "bias" ? v.toFixed(2) : v}${metric === "bias" ? " m" : ""}`;
  $("comparison-metrics").replaceChildren(...result.metrics.map(metric => {
    const row = document.createElement("tr"); row.dataset.metric = metric.key;
    cell(row, labels[metric.key], "th").scope = "row";
    cell(row, value(metric.before, metric.key)); cell(row, value(metric.after, metric.key)); cell(row, value(metric.delta, metric.key, true)); return row;
  }));
  const path = (row, label) => { const element = cell(row, label); element.dataset.path = label; return element; };
  const excess = v => v === null ? "—" : `${v.toFixed(3)} m`;
  $("comparison-signals").replaceChildren(...result.signals.map(signal => {
    const row = document.createElement("tr"); row.dataset.signal = signal.id; row.dataset.changed = String(signal.before !== signal.after);
    cell(row, signal.id, "th").scope = "row"; path(row, signal.before); path(row, signal.after); cell(row, excess(signal.beforeExcess)); cell(row, excess(signal.afterExcess)); return row;
  }));
}
$("compare-panel").addEventListener("toggle", () => { if ($("compare-panel").open) { setPlaying(false); updateComparison(); } });
$("pin-reference").addEventListener("click", () => {
  if (!currentTrace()) return;
  setPlaying(false); reference = captureReference(snapshot); referenceURL = sceneURL(); updateComparison();
});
$("clear-reference").addEventListener("click", () => {
  reference = referenceURL = comparisonReference = comparisonSnapshot = null; $("reference-link").removeAttribute("href"); updateComparison();
});
$("return-reference").addEventListener("click", () => { if (reference) seekEpoch(reference.epoch); });
function restoreSceneLink(resetEmpty = false) {
  let state, warning = "";
  try { state = readSceneLink(location.hash, linkAssets); }
  catch (error) { state = { ...DEFAULT_SCENE }; warning = `${error.message} Showing the default scene.`; }
  if (!state) { if (resetEmpty && !location.hash) state = { ...DEFAULT_SCENE }; else return false; }
  setPlaying(false); resetCamera();
  Object.assign(settings, { epoch: state.epoch, antenna: state.antenna, mask: state.mask });
  $("antenna").value = state.antenna; $("mask").value = state.mask;
  selected = state.satellite || undefined; showReflections = state.reflections;
  $("reflections").checked = state.reflections; $("basemap").checked = state.basemap; renderer.basemap.setEnabled(state.basemap);
  pendingReplay = { state, warning }; sceneNotice(warning, Boolean(warning)); syncControls();
  if (currentTrace() && !busy && !pending) applySceneView();
  else { revision++; requestTrace(true); }
  return true;
}
function applySceneView() {
  if (!pendingReplay || !currentTrace()) return;
  const { state, warning: initialWarning } = pendingReplay; pendingReplay = null;
  const signal = snapshot.satellites.find(s => s.id === state.satellite), focused = ["reflection", "blockage"].includes(state.view);
  if (!snapshot.satellites.some(s => s.id === selected)) selected = (snapshot.satellites.find(s => s.status === "NLOS") ?? snapshot.satellites[0])?.id;
  let warning = initialWarning;
  if (state.satellite && !signal) warning = `${state.satellite} is unavailable at this epoch and mask. Showing ${selected || "the receiver"}.`;
  else if (focused && !(state.view === "reflection" ? signal?.reflection : signal?.directHit)) warning = `${state.satellite} has no ${state.view === "reflection" ? "reflecting" : "obstructing"} surface here. Showing the receiver view.`;
  if (!warning) {
    if (focused) renderer.focusSurface(snapshot, signal, state.view);
    Object.assign(renderer.camera, { yaw: state.yaw, pitch: state.pitch, zoom: state.zoom, overview: state.view === "overview" });
  }
  inspect(); syncControls(); dirty = true;
  sceneNotice(warning || "Scene link restored. Playback is paused.", Boolean(warning));
  if (!warning) canvas.scrollIntoView({ block: "center" });
}
$("share-panel").addEventListener("toggle", () => { if ($("share-panel").open) { setPlaying(false); updateShare(); } });
$("scene-link").addEventListener("focus", event => event.target.select());
$("copy-scene-link").addEventListener("click", async () => {
  const input = $("scene-link"), link = input.value;
  if (!currentTrace() || !link) return;
  try {
    if (!navigator.clipboard?.writeText) throw new Error("Clipboard unavailable");
    await navigator.clipboard.writeText(link);
    if (input.value === link) shareStatus("Link copied.");
  } catch {
    if (input.value === link) { input.focus(); input.select(); shareStatus("Clipboard unavailable. Select and copy the link manually."); }
  }
});
$("dismiss-scene-notice").addEventListener("click", () => sceneNotice(""));
window.addEventListener("hashchange", () => { if (controlsReady && linkAssets) restoreSceneLink(true); });
for (const id of ["antenna", "mask"]) $(id).addEventListener("input", event => { settings[id] = Number(event.target.value); revision++; syncControls(); requestTrace(); });
$("progress").addEventListener("input", event => seekEpoch(Number(event.target.value)));
$("preset").addEventListener("change", event => seekEpoch(Number(event.target.value)));
for (const button of document.querySelectorAll("[data-route-step]")) button.addEventListener("click", () => seekEpoch(settings.epoch + Number(button.dataset.routeStep)));
$("play").addEventListener("click", () => setPlaying(!playing));
$("reflections").addEventListener("change", event => { pendingReplay = null; showReflections = event.target.checked; if (!showReflections && renderer?.focus?.kind === "reflection") clearSurfaceFocus(); dirty = true; });
$("basemap").addEventListener("change", event => { pendingReplay = null; renderer?.basemap.setEnabled(event.target.checked); dirty = true; });
$("reset").addEventListener("click", () => {
  setPlaying(false); Object.assign(settings, { epoch: 8000, antenna: 0, mask: 10 });
  $("antenna").value = 0; $("mask").value = 10; $("preset").value = "8000";
  resetCamera(); revision++; syncControls(); requestTrace();
});
function resetCamera() { pendingReplay = null; clearSurfaceFocus(); if (renderer) Object.assign(renderer.camera, { yaw: -.63, pitch: .82, zoom: 1, overview: false, span: 740 }); syncFocusUI(); dirty = true; }
$("reset-view").addEventListener("click", resetCamera);
$("zoom-in").addEventListener("click", () => { pendingReplay = null; if (renderer) renderer.camera.zoom = Math.min(6, renderer.camera.zoom * 1.25); dirty = true; });
$("zoom-out").addEventListener("click", () => { pendingReplay = null; if (renderer) renderer.camera.zoom = Math.max(.3, renderer.camera.zoom / 1.25); dirty = true; });
$("overview").addEventListener("click", () => {
  if (!renderer) return; pendingReplay = null; const overview = renderer.focus ? true : !renderer.camera.overview;
  clearSurfaceFocus(); renderer.camera.overview = overview; renderer.camera.zoom = 1;
  $("overview").setAttribute("aria-pressed", String(renderer.camera.overview)); dirty = true;
});
let drag;
canvas.addEventListener("pointerdown", event => {
  if (!renderer) return; pendingReplay = null; canvas.setPointerCapture(event.pointerId);
  drag = { x: event.clientX, y: event.clientY, yaw: renderer.camera.yaw, pitch: renderer.camera.pitch };
});
canvas.addEventListener("pointermove", event => {
  if (!drag) return; renderer.camera.yaw = drag.yaw + (event.clientX - drag.x) * .006;
  renderer.camera.pitch = clampViewPitch(drag.pitch + (event.clientY - drag.y) * .005, Boolean(renderer.focus)); dirty = true;
});
for (const type of ["pointerup", "pointercancel"]) canvas.addEventListener(type, () => { drag = null; });
canvas.addEventListener("wheel", event => {
  event.preventDefault(); if (!renderer) return; pendingReplay = null;
  renderer.camera.zoom = Math.max(.3, Math.min(6, renderer.camera.zoom * Math.exp(-event.deltaY * .001))); dirty = true;
}, { passive: false });
canvas.addEventListener("keydown", event => {
  if (!renderer) return;
  pendingReplay = null;
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
  linkAssets = { mesh: manifest.mesh.sha256, data: manifest.scene.sha256, epochs: scene.track.length, satellites: [...new Set(scene.navigation.map(record => record.id))] };
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
    if (data.type === "ready") { busy = false; controlsReady = true; for (const input of document.querySelectorAll("[data-load-control]")) input.disabled = false; syncRoute(); if (!restoreSceneLink()) requestTrace(); return; }
    busy = false;
    if (data.revision === revision) { clearSurfaceFocus(); snapshot = data.snapshot; canvas.dataset.traceId = data.id; updateReadout(data.milliseconds); applySceneView(); }
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
  if (dirty && renderer) {
    renderer.draw(snapshot, selected, showReflections);
    updatePositionFit({width:renderer.width,height:renderer.height,scale:renderer.scale,right:renderer.right,up:renderer.up,target:renderer.camera.target});
    updateViewCaption(); updateShare(); dirty = false;
  }
  requestAnimationFrame(frame);
}
syncControls(); requestAnimationFrame(frame); initialize().catch(fail);
