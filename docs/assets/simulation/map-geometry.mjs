// WGS84 / Web Mercator helpers for display. Signal geometry is unchanged.
import { geodetic, localPoint } from "./orbit.mjs";

const MAX_LAT = 85.0511287798066;
const EARTH_CIRCUMFERENCE = 2 * Math.PI * 6378137;
const radians = degrees => degrees * Math.PI / 180;
const degrees = value => value * 180 / Math.PI;
const clamp = (value, lo, hi) => Math.max(lo, Math.min(hi, value));

export function lonLatToTile(lon, lat, zoom) {
  const n = 2 ** zoom, latitude = radians(clamp(lat, -MAX_LAT, MAX_LAT));
  return [(lon + 180) / 360 * n, (1 - Math.asinh(Math.tan(latitude)) / Math.PI) * n / 2];
}

export function tileToLonLat(x, y, zoom) {
  const n = 2 ** zoom;
  return [x / n * 360 - 180, degrees(Math.atan(Math.sinh(Math.PI * (1 - 2 * y / n))))];
}

export function lonLatToLocal(lon, lat, height, scene) {
  const phi = radians(lat), lambda = radians(lon);
  const e2 = (1 / 298.257223563) * (2 - 1 / 298.257223563);
  const n = 6378137 / Math.sqrt(1 - e2 * Math.sin(phi) ** 2);
  return localPoint([(n + height) * Math.cos(phi) * Math.cos(lambda), (n + height) * Math.cos(phi) * Math.sin(lambda), (n * (1 - e2) + height) * Math.sin(phi)], scene);
}

export function localToLonLat(point, scene) {
  const ecef = scene.origin.map((value, i) => value + scene.rotation.reduce((sum, axis, j) => sum + axis[i] * point[j], 0));
  const llh = geodetic(ecef);
  return [degrees(llh.lon), degrees(llh.lat)];
}

export function groundFootprint(camera, width, height, scale, planeZ) {
  const c = Math.cos(camera.yaw), s = Math.sin(camera.yaw);
  const sp = Math.sin(camera.pitch), cp = Math.cos(camera.pitch);
  return [[0,0], [width,0], [width,height], [0,height]].map(([px, py]) => {
    const right = (px - width / 2) / scale;
    const up = (height * .56 - py) / scale;
    const forward = (cp * (planeZ - camera.target[2]) - up) / sp;
    return [camera.target[0] + c * right + s * forward, camera.target[1] - s * right + c * forward, planeZ];
  });
}

// Bound the on-screen tile set; oblique/overview views use a coarser display map.
export function visibleTiles(footprint, scene, scale, budget = 32) {
  if (!Number.isInteger(budget) || budget < 1 || !(scale > 0)) throw new RangeError("Invalid map viewport");
  const corners = footprint.map(point => localToLonLat(point, scene));
  const west = Math.min(...corners.map(p => p[0])), east = Math.max(...corners.map(p => p[0]));
  const south = Math.min(...corners.map(p => p[1])), north = Math.max(...corners.map(p => p[1]));
  let zoom = clamp(Math.round(Math.log2(EARTH_CIRCUMFERENCE * Math.cos(radians((north + south) / 2)) * scale / 256)), 0, 19);
  let x0, x1, y0, y1;
  for (;;) {
    const a = lonLatToTile(west, north, zoom), b = lonLatToTile(east, south, zoom), n = 2 ** zoom;
    x0 = clamp(Math.floor(a[0]), 0, n - 1); x1 = clamp(Math.floor(b[0]), 0, n - 1);
    y0 = clamp(Math.floor(a[1]), 0, n - 1); y1 = clamp(Math.floor(b[1]), 0, n - 1);
    if ((x1 - x0 + 1) * (y1 - y0 + 1) <= budget || zoom === 0) break;
    zoom--;
  }
  const tiles = [];
  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) tiles.push({ x, y, z: zoom, key: `${zoom}/${x}/${y}` });
  return tiles;
}

// Subdivision follows geographic curvature. Float32 is exclusively a GPU buffer.
export function tileMesh(tile, scene, height, divisions = 8) {
  const data = new Float32Array(divisions * divisions * 6 * 5);
  let offset = 0;
  for (let j = 0; j < divisions; j++) for (let i = 0; i < divisions; i++) {
    for (const [u, v] of [[i,j], [i,j+1], [i+1,j+1], [i,j], [i+1,j+1], [i+1,j]]) {
      const uv = [u / divisions, v / divisions];
      const [lon, lat] = tileToLonLat(tile.x + uv[0], tile.y + uv[1], tile.z);
      data.set([...lonLatToLocal(lon, lat, height, scene), ...uv], offset); offset += 5;
    }
  }
  return data;
}
