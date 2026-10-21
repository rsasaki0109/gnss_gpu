// Browser geometry model: opaque prisms, direct paths and one specular bounce.
// All geometry uses JavaScript Number (binary64); no RF power or position claim.

export const C = 299792458;
export const EPS = 1e-7;
export const RECEIVER_HEIGHT = 1.8;
// Notional GNSS-scale sources; the sky is illustrative, not broadcast ephemeris.
export const SOURCE_RANGE = 20_200_000;
export const SCENES = {
  canyon: { name: "Street canyon", description: "Tall blocks, narrow streets", height: 1 },
  plaza: { name: "Open plaza", description: "A clearing in the city", height: 0.8 },
  lowrise: { name: "Low-rise district", description: "The same streets, a lower skyline", height: 0.35 },
};

export const add = (a, b) => a.map((v, i) => v + b[i]);
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const mul = (a, k) => a.map(v => v * k);
export const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
export const length = a => Math.hypot(...a);
const finitePoint = p => Array.isArray(p) && p.length === 3 && p.every(Number.isFinite);

// Slab intersection with a finite segment. A face/edge touch counts as blocked.
export function segmentBox(from, to, box) {
  if (!finitePoint(from) || !finitePoint(to)) throw new TypeError("Finite 3D points required");
  const delta = sub(to, from);
  let enter = 0, exit = 1, normal = [0, 0, 0];
  for (let axis = 0; axis < 3; axis++) {
    if (delta[axis] === 0) {
      if (from[axis] < box.min[axis] || from[axis] > box.max[axis]) return null;
      continue;
    }
    let near = (box.min[axis] - from[axis]) / delta[axis];
    let far = (box.max[axis] - from[axis]) / delta[axis];
    if (near > far) [near, far] = [far, near];
    if (near > enter) {
      enter = near;
      normal = [0, 0, 0];
      normal[axis] = delta[axis] > 0 ? -1 : 1;
    }
    exit = Math.min(exit, far);
    if (enter > exit) return null;
  }
  return { enter, exit, normal, point: add(from, mul(delta, enter)), distance: length(delta) * enter };
}

export function blocked(from, to, buildings) {
  const size = length(sub(to, from));
  if (size <= EPS) return false;
  const margin = EPS / size;
  return buildings.some(box => {
    const hit = segmentBox(from, to, box);
    return hit && hit.exit > margin && hit.enter < 1 - margin;
  });
}

export function contains(point, box, padding = 0) {
  return point.every((v, i) => v >= box.min[i] - padding && v <= box.max[i] + padding);
}

export function faces(box) {
  return [0, 1, 2].flatMap(axis => [-1, 1].map(sign => {
    const normal = [0, 0, 0];
    normal[axis] = sign;
    return { axis, sign, normal, plane: sign === 1 ? box.max[axis] : box.min[axis] };
  }));
}

export function specularPath(receiver, source, box, face, buildings) {
  const sideRx = face.sign * (receiver[face.axis] - face.plane);
  const sideSource = face.sign * (source[face.axis] - face.plane);
  if (sideRx <= EPS || sideSource <= EPS) return null;
  const mirror = receiver.slice();
  mirror[face.axis] = 2 * face.plane - receiver[face.axis];
  const direction = sub(mirror, source);
  const denominator = direction[face.axis];
  if (denominator === 0) return null;
  const t = (face.plane - source[face.axis]) / denominator;
  if (t <= 0 || t >= 1) return null;
  const point = add(source, mul(direction, t));
  point[face.axis] = face.plane;
  for (let axis = 0; axis < 3; axis++) {
    if (axis !== face.axis && (point[axis] < box.min[axis] - EPS || point[axis] > box.max[axis] + EPS)) return null;
  }
  if (blocked(receiver, point, buildings) || blocked(point, source, buildings)) return null;
  const total = length(sub(receiver, point)) + length(sub(source, point));
  const extra = total - length(sub(source, receiver));
  if (extra < -EPS) throw new Error("Reflected path violated the triangle inequality");
  return { point, buildingId: box.id, distance: total, extra: Math.max(0, extra), normal: face.normal };
}

function random(seed) {
  let state = seed >>> 0;
  return () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function buildScene(preset = "canyon", heightScale = 1) {
  if (!Object.hasOwn(SCENES, preset) || !Number.isFinite(heightScale) || heightScale < 0.5 || heightScale > 2) {
    throw new RangeError("Unknown scene or building height outside 0.5–2.0");
  }
  const rand = random(260311);
  const buildings = [];
  for (const x of [-173, -105, -37, 37, 105, 173]) {
    for (const y of [-105, -37, 37, 105]) {
      const w = 39 + rand() * 13, d = 38 + rand() * 11;
      const h = (29 + rand() * 77) * SCENES[preset].height * heightScale;
      const tone = rand();
      if (preset === "plaza" && Math.abs(x) < 70 && Math.abs(y) < 70) continue;
      buildings.push({ id: buildings.length, min: [x - w / 2, y - d / 2, 0], max: [x + w / 2, y + d / 2, h], tone });
    }
  }
  return buildings;
}

export const ROUTE = [[-216, 0, RECEIVER_HEIGHT], [-70, 0, RECEIVER_HEIGHT], [0, 0, RECEIVER_HEIGHT], [0, 154, RECEIVER_HEIGHT]];

export function receiverAt(progress) {
  if (!Number.isFinite(progress) || progress < 0 || progress > 1) throw new RangeError("Route progress must be 0–1");
  const distances = ROUTE.slice(1).map((p, i) => length(sub(p, ROUTE[i])));
  let remaining = progress * distances.reduce((a, b) => a + b, 0);
  for (let i = 0; i < distances.length; i++) {
    if (remaining <= distances[i] || i === distances.length - 1) return add(ROUTE[i], mul(sub(ROUTE[i + 1], ROUTE[i]), remaining / distances[i]));
    remaining -= distances[i];
  }
  return ROUTE.at(-1).slice();
}

const SKY = [
  ["G05", 26, 18], ["G12", 71, 28], ["G19", 116, 36],
  ["E11", 141, 44], ["E24", 190, 52], ["E31", 215, 60],
  ["R04", 264, 68], ["R11", 289, 34], ["J02", 330, 58], ["J07", 7, 74],
];

export function constellation(rotation = 0) {
  if (!Number.isFinite(rotation) || rotation < 0 || rotation > 360) throw new RangeError("Sky rotation must be 0–360 degrees");
  return SKY.map(([id, azimuth, elevation]) => {
    azimuth = (azimuth + rotation) % 360;
    const az = azimuth * Math.PI / 180, el = elevation * Math.PI / 180;
    return { id, azimuth, elevation, direction: [Math.sin(az) * Math.cos(el), Math.cos(az) * Math.cos(el), Math.sin(el)] };
  });
}

export function simulate({ preset = "canyon", heightScale = 1, rotation = 0, progress = 0.35, receiver = null } = {}) {
  const buildings = buildScene(preset, heightScale);
  const rx = receiver === null ? receiverAt(progress) : receiver.slice();
  if (!finitePoint(rx) || rx[2] !== RECEIVER_HEIGHT || Math.abs(rx[0]) > 230 || Math.abs(rx[1]) > 170) throw new RangeError("Receiver is outside the scene");
  if (buildings.some(box => contains(rx, box, 0.2))) throw new RangeError("Place the receiver on a street or in the plaza");
  const satellites = constellation(rotation).map(satellite => {
    const source = add(rx, mul(satellite.direction, SOURCE_RANGE));
    let directHit = null;
    for (const box of buildings) {
      const hit = segmentBox(rx, source, box);
      if (hit && (!directHit || hit.enter < directHit.enter)) directHit = { ...hit, buildingId: box.id };
    }
    const reflections = [];
    for (const box of buildings) {
      for (const face of faces(box)) {
        const path = specularPath(rx, source, box, face, buildings);
        if (path) reflections.push(path);
      }
    }
    reflections.sort((a, b) => a.distance - b.distance || a.buildingId - b.buildingId);
    const reflection = reflections[0] ?? null;
    const status = !directHit ? "LOS" : reflection ? "NLOS" : "BLOCKED";
    return { ...satellite, source, directHit, reflection, reflectionCount: reflections.length, status };
  });
  const counts = { LOS: 0, NLOS: 0, BLOCKED: 0 };
  satellites.forEach(satellite => counts[satellite.status]++);
  return { preset, heightScale, rotation, progress, receiver: rx, buildings, satellites, counts };
}
