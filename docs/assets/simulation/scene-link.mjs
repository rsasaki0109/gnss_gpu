import { GNSS_SYSTEMS, validateSystems } from "./orbit.mjs?v=glonass-1";
// Scene links preserve original input values; camera values affect drawing only.
export const DEFAULT_SCENE = Object.freeze({ epoch: 8000, antenna: 0, mask: 10, systems: GNSS_SYSTEMS, satellite: "", view: "receiver", yaw: -.63, pitch: .82, zoom: 1, basemap: true, reflections: true });
const FIELDS = ["scene", "epoch", "height", "mask", "sat", "view", "yaw", "pitch", "zoom", "map", "reflections", "systems", "mesh", "data"];
const NUMBER = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?$/i;
const VIEWS = new Set(["receiver", "overview", "reflection", "blockage"]);

export function readSceneLink(hash, assets) {
  if (!hash || hash === "#scene") return null;
  if (hash.length > 4096) throw new Error("Scene link is too long.");
  const fields = new URLSearchParams(hash.replace(/^#/, ""));
  if (!fields.has("scene")) return null;
  if (fields.get("scene") !== "1") throw new Error("Scene link version is unsupported.");
  if ([...fields.keys()].some(key => !FIELDS.includes(key)) || FIELDS.some(key => key !== "systems" && fields.getAll(key).length !== 1) || fields.getAll("systems").length > 1) throw new Error("Scene link has missing, duplicate or unknown fields.");
  if (fields.get("mesh") !== assets.mesh || fields.get("data") !== assets.data) throw new Error("Scene link uses a different dataset.");
  const number = key => {
    const text = fields.get(key), value = Number(text);
    if (!NUMBER.test(text) || !Number.isFinite(value)) throw new Error("Scene link contains an invalid number.");
    return value;
  };
  const state = { systems: validateSystems(fields.get("systems") ?? "G"), epoch: number("epoch"), antenna: number("height"), mask: number("mask"), satellite: fields.get("sat"), view: fields.get("view"), yaw: number("yaw"), pitch: number("pitch"), zoom: number("zoom"), basemap: fields.get("map") === "1", reflections: fields.get("reflections") === "1" };
  if (!Number.isInteger(state.epoch) || state.epoch < 0 || state.epoch >= assets.epochs || state.antenna < 0 || state.antenna > 20 || !Number.isInteger(state.antenna * 2) || !Number.isInteger(state.mask) || state.mask < 5 || state.mask > 30) throw new Error("Scene link receiver settings are outside the recorded controls.");
  if (state.satellite && (!/^[GECJR]\d{2}$/.test(state.satellite) || !assets.satellites.includes(state.satellite))) throw new Error("Scene link requests an unknown satellite.");
  const focused = state.view === "reflection" || state.view === "blockage";
  if (!VIEWS.has(state.view) || state.zoom < .3 || state.zoom > 6 || Math.abs(state.pitch) < .15 || Math.abs(state.pitch) > 1.48 || (!focused && state.pitch < 0) || (focused && !state.satellite)) throw new Error("Scene link view is invalid.");
  if (!["0", "1"].includes(fields.get("map")) || !["0", "1"].includes(fields.get("reflections")) || (state.view === "reflection" && !state.reflections)) throw new Error("Scene link display options are inconsistent.");
  return state;
}

export function writeSceneLink(base, state, assets) {
  const url = new URL(base);
  if (!["https:", "http:"].includes(url.protocol)) throw new Error("Scene links require HTTP or HTTPS.");
  url.username = ""; url.password = ""; url.search = "";
  // Number-to-string conversion preserves binary64 values without rounding.
  const number = value => Object.is(value, -0) ? "-0" : String(value);
  url.hash = new URLSearchParams({ scene: "1", epoch: number(state.epoch), height: number(state.antenna), mask: number(state.mask), sat: state.satellite || "", view: state.view, yaw: number(state.yaw), pitch: number(state.pitch), zoom: number(state.zoom), map: state.basemap ? "1" : "0", reflections: state.reflections ? "1" : "0", systems: validateSystems(state.systems ?? GNSS_SYSTEMS), mesh: assets.mesh, data: assets.data }).toString();
  return url.href;
}
