// GPS broadcast orbits (IS-GPS-200), all state computation in binary64.
export const C = 299792458;
export const WEEK = 604800;
export const OMEGA_E = 7.2921151467e-5;
const MU = 3.986005e14;
const RELATIVITY = -4.442807633e-10;
export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const add = (a, b) => a.map((v, i) => v + b[i]);
export const mul = (a, k) => a.map(v => v * k);
export const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
export const norm = a => Math.hypot(...a);
const weekDelta = t => t > WEEK / 2 ? t - WEEK : t < -WEEK / 2 ? t + WEEK : t;

// Keep GPS week and seconds-of-week separate to retain sub-microsecond time.
export function orbit(record, time, week = record.parameters[18]) {
  const p = record.parameters;
  if (p.length < 26 || !p.every(Number.isFinite) || !(p[5] >= 0 && p[5] < 1) || p[7] < 1000 || !Number.isFinite(time)) throw new RangeError("Invalid broadcast orbit");
  const tk = weekDelta((week - p[18]) * WEEK + time - p[8]);
  const a = p[7] ** 2, motion = Math.sqrt(MU / a ** 3) + p[2];
  const mean = p[3] + motion * tk;
  let eccentric = mean, converged = false;
  for (let i = 0; i < 20; i++) {
    const step = (eccentric - p[5] * Math.sin(eccentric) - mean) / (1 - p[5] * Math.cos(eccentric));
    eccentric -= step;
    if (Math.abs(step) < 1e-14) { converged = true; break; }
  }
  if (!converged) throw new Error("Kepler solution did not converge");
  const sinE = Math.sin(eccentric), cosE = Math.cos(eccentric), denominator = 1 - p[5] * cosE;
  const anomaly = Math.atan2(Math.sqrt(1 - p[5] ** 2) * sinE / denominator, (cosE - p[5]) / denominator);
  const phi = anomaly + p[14], sin2 = Math.sin(2 * phi), cos2 = Math.cos(2 * phi);
  const u = phi + p[4] * cos2 + p[6] * sin2;
  const radius = a * denominator + p[13] * cos2 + p[1] * sin2;
  const inclination = p[12] + p[16] * tk + p[9] * cos2 + p[11] * sin2;
  const x = radius * Math.cos(u), y = radius * Math.sin(u);
  const node = p[10] + (p[15] - OMEGA_E) * tk - OMEGA_E * p[8];
  const position = [x * Math.cos(node) - y * Math.cos(inclination) * Math.sin(node), x * Math.sin(node) + y * Math.cos(inclination) * Math.cos(node), y * Math.sin(inclination)];
  const dt = weekDelta((week - Math.floor(record.toc / WEEK)) * WEEK + time - record.toc % WEEK), f = record.clock;
  const clock = f[0] + f[1] * dt + f[2] * dt * dt + RELATIVITY * p[5] * p[7] * sinE - p[22];
  return { position, clock };
}

export function selectNavigation(records, time, week) {
  const selected = new Map();
  for (const record of records) {
    const p = record.parameters, toe = (p[18] - week) * WEEK + p[8];
    const age = Math.abs(time - toe), fit = (p[25] > 1 ? p[25] : 4) * 1800;
    let transmission = (p[18] - week) * WEEK + p[24];
    if (transmission - toe > WEEK / 2) transmission -= WEEK;
    if (transmission - toe < -WEEK / 2) transmission += WEEK;
    if (p[21] !== 0 || age > fit || transmission > time + 1e-6) continue;
    const previous = selected.get(record.id);
    if (!previous || age < previous.age || (age === previous.age && record.toc > previous.record.toc)) selected.set(record.id, { record, age });
  }
  return [...selected.values()].map(item => item.record).sort((a, b) => a.id.localeCompare(b.id));
}

export function receivedSatellite(record, time, receiver, week = record.parameters[18]) {
  let tau = 0.075, state;
  for (let i = 0; i < 8; i++) {
    const raw = orbit(record, time - tau, week), theta = OMEGA_E * tau;
    const position = [Math.cos(theta) * raw.position[0] + Math.sin(theta) * raw.position[1], -Math.sin(theta) * raw.position[0] + Math.cos(theta) * raw.position[1], raw.position[2]];
    state = { position, clock: raw.clock, transitTime: tau };
    const next = norm(sub(position, receiver)) / C;
    if (Math.abs(next - tau) < 1e-13) return state;
    tau = next;
  }
  throw new Error("Signal transit time did not converge");
}

export function geodetic(ecef) {
  const [x, y, z] = ecef, horizontal = Math.hypot(x, y);
  const e2 = (1 / 298.257223563) * (2 - 1 / 298.257223563);
  let lat = Math.atan2(z, horizontal * (1 - e2));
  for (let i = 0; i < 12; i++) {
    const n = 6378137 / Math.sqrt(1 - e2 * Math.sin(lat) ** 2);
    const next = Math.atan2(z + e2 * n * Math.sin(lat), horizontal);
    if (Math.abs(next - lat) < 1e-15) { lat = next; break; }
    lat = next;
  }
  const lon = Math.atan2(y, x), n = 6378137 / Math.sqrt(1 - e2 * Math.sin(lat) ** 2);
  return { lat, lon, height: horizontal / Math.cos(lat) - n };
}

export function enuRotation(lat, lon) {
  return [[-Math.sin(lon), Math.cos(lon), 0], [-Math.sin(lat) * Math.cos(lon), -Math.sin(lat) * Math.sin(lon), Math.cos(lat)], [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)]];
}

export function localPoint(point, scene) {
  const delta = sub(point, scene.origin);
  return scene.rotation.map(axis => dot(axis, delta));
}

// Householder QR avoids forming normal equations for weak satellite geometry.
export function leastSquares(rows, values) {
  if (rows.length < 4) return null;
  const a = rows.map(row => row.slice()), b = values.slice();
  for (let k = 0; k < 4; k++) {
    const size = Math.hypot(...a.slice(k).map(row => row[k]));
    if (size < 1e-11) return null;
    const alpha = a[k][k] >= 0 ? -size : size;
    const v = a.slice(k).map(row => row[k]); v[0] -= alpha;
    const vv = dot(v, v);
    for (let j = k; j < 4; j++) {
      let product = 0;
      for (let i = k; i < a.length; i++) product += v[i - k] * a[i][j];
      for (let i = k; i < a.length; i++) a[i][j] -= 2 * v[i - k] * product / vv;
    }
    let product = 0;
    for (let i = k; i < a.length; i++) product += v[i - k] * b[i];
    for (let i = k; i < a.length; i++) b[i] -= 2 * v[i - k] * product / vv;
  }
  const solution = [0, 0, 0, 0];
  for (let i = 3; i >= 0; i--) {
    let residual = b[i];
    for (let j = i + 1; j < 4; j++) residual -= a[i][j] * solution[j];
    if (Math.abs(a[i][i]) < 1e-11) return null;
    solution[i] = residual / a[i][i];
  }
  return solution.every(Number.isFinite) ? solution : null;
}

export function geometricSPP(receiver, satellites) {
  const usable = satellites.filter(satellite => satellite.status !== "BLOCKED");
  if (usable.length < 4) return null;
  const ranges = usable.map(s => norm(sub(s.source, receiver)) + (s.directHit ? s.reflection.extra : 0));
  let position = receiver.slice(), clock = 0;
  for (let i = 0; i < 12; i++) {
    const rows = [], residual = [];
    for (let j = 0; j < usable.length; j++) {
      const delta = sub(position, usable[j].source), distance = norm(delta);
      rows.push([...mul(delta, 1 / distance), 1]);
      residual.push(ranges[j] - distance - clock);
    }
    const step = leastSquares(rows, residual);
    if (!step) return null;
    position = add(position, step.slice(0, 3)); clock += step[3];
    if (norm(step.slice(0, 3)) < 1e-5 && Math.abs(step[3]) < 1e-5) {
      const offset = sub(position, receiver);
      return { position, offset, clock, horizontal: Math.hypot(offset[0], offset[1]), spatial: norm(offset), satellites: usable.length };
    }
    if (norm(sub(position, receiver)) > 100000) return null;
  }
  return null;
}
