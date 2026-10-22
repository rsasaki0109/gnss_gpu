const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const { gunzipSync } = require("node:zlib");
const { createHash } = require("node:crypto");
const folder = path.resolve(__dirname, "../../docs/assets/plateau");
const manifest = JSON.parse(fs.readFileSync(path.join(folder, "manifest.json")));
const scene = JSON.parse(gunzipSync(fs.readFileSync(path.join(folder, manifest.scene.file))));
  // Preserve the original GPS-only numerical regression alongside multi-GNSS tests.
  scene.navigation = scene.navigation.filter(record => record.id.startsWith("G"));
const meshBytes = gunzipSync(fs.readFileSync(path.join(folder, manifest.mesh.file)));
const modules = Promise.all([import("../../docs/assets/simulation/mesh.mjs"), import("../../docs/assets/simulation/orbit.mjs"), import("../../docs/assets/simulation/city.mjs")]);
const close = (a, b, tolerance = 1e-9) => assert.ok(Math.abs(a - b) <= tolerance, `${a} != ${b} (tolerance ${tolerance})`);
const minus = (a, b) => a.map((v, i) => v - b[i]);
const dot = (a, b) => a.reduce((sum, v, i) => sum + v * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => a.map(v => v / Math.hypot(...a));
function meshOf(triangles, owners = triangles.map(() => 0)) {
  return { vertices: Float64Array.from(triangles.flat(2)), indices: Uint32Array.from({ length: triangles.length * 3 }, (_, i) => i), owners: Uint32Array.from(owners), types: new Uint8Array(triangles.length), triangleCount: triangles.length, vertexCount: triangles.length * 3 };
}
const quad = (a, b, c, d) => [[a, b, c], [a, c, d]];
let simulationPromise;
async function simulation() {
  if (!simulationPromise) simulationPromise = modules.then(([{ decodeMesh }, , { CitySimulation }]) => new CitySimulation(scene, decodeMesh(meshBytes.buffer.slice(meshBytes.byteOffset, meshBytes.byteOffset + meshBytes.byteLength))));
  return simulationPromise;
}

test("published assets retain every full-resolution epoch and verify compressed/raw SHA-256", async () => {
  for (const entry of [manifest.mesh, manifest.scene]) {
    const data = fs.readFileSync(path.join(folder, entry.file));
    assert.equal(data.length, entry.bytes);
    assert.equal(createHash("sha256").update(data).digest("hex"), entry.sha256);
    assert.equal(createHash("sha256").update(gunzipSync(data)).digest("hex"), entry.raw_sha256);
  }
  assert.equal(scene.track.length, 12410); assert.equal(scene.week, 2032);
  assert.deepEqual(scene.track[0], [273375.1, -3963426.7981, 3350882.1576, 3694865.5458]);
  assert.equal(scene.track.at(-1)[0], 274616);
  for (let i = 1; i < scene.track.length; i++) close(scene.track[i][0] - scene.track[i - 1][0], .1, 1e-9);
  assert.equal(scene.stats.triangles, 320932); assert.equal(scene.stats.holes, 59);
  assert.equal(scene.provenance.files.length, 28); assert.equal(scene.stats.topology_repaired_faces, 1);
  assert.ok(scene.provenance.geoid.min_m > 36 && scene.provenance.geoid.max_m < 37);
  const [{ decodeMesh }] = await modules;
  assert.throws(() => decodeMesh(new ArrayBuffer(64)), /mesh/);
});

test("GPS propagation and clock agree with 96 independent native Python CPU reference cases", async () => {
  const [, { orbit }] = await modules;
  const fixture = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures/gps-orbits.json")));
  assert.equal(fixture.cases.length, 96);
  let maximum = 0;
  for (const item of fixture.cases) {
    const result = orbit(item.record, item.time, item.week);
    const error = Math.hypot(...minus(result.position, item.position));
    maximum = Math.max(maximum, error); assert.ok(error < 1e-6, `${item.record.id}: ${error} m`);
    close(result.clock, item.clock, 1e-15);
    const rolled = orbit(item.record, item.time - 604800, item.week + 1);
    rolled.position.forEach((v, i) => close(v, result.position[i], 1e-6));
  }
  console.log(`GPS CPU reference: max ECEF difference ${maximum} m over 96 cases`);
});

test("ephemeris selection enforces health, fit, transmission causality and GPS week rollover", async () => {
  const [, { selectNavigation }] = await modules;
  const record = structuredClone(scene.navigation[0]);
  record.id = "G01"; record.parameters[18] = 2000; record.parameters[8] = 1000;
  record.parameters[21] = 0; record.parameters[24] = 900; record.parameters[25] = 4;
  const future = structuredClone(record); future.id = "G02"; future.parameters[24] = 1200;
  const unhealthy = structuredClone(record); unhealthy.id = "G03"; unhealthy.parameters[21] = 1;
  const stale = structuredClone(record); stale.id = "G04"; stale.parameters[8] = 1100 - 7201;
  assert.deepEqual(selectNavigation([record, future, unhealthy, stale], 1100, 2000).map(r => r.id), ["G01"]);
  assert.equal(selectNavigation([record], 8200, 2000).length, 1);
  assert.equal(selectNavigation([record], 8200.1, 2000).length, 0);
  record.parameters[8] = 604700; record.parameters[24] = 604600;
  assert.equal(selectNavigation([record], 50, 2001).length, 1);
});

test("transmit-time orbit and Sagnac rotation satisfy the reception-frame light-time equation", async () => {
  const [, { receivedSatellite, orbit, C, OMEGA_E, selectNavigation }] = await modules;
  const row = scene.track[8000], receiver = row.slice(1);
  for (const record of selectNavigation(scene.navigation, row[0], scene.week)) {
    const state = receivedSatellite(record, row[0], receiver, scene.week);
    close(Math.hypot(...minus(state.position, receiver)), C * state.transitTime, 3e-5);
    const raw = orbit(record, row[0] - state.transitTime, scene.week).position;
    const theta = OMEGA_E * state.transitTime;
    close(Math.atan2(state.position[1], state.position[0]), Math.atan2(raw[1], raw[0]) - theta, 1e-13);
    close(state.position[2], raw[2], 1e-8);
    assert.ok(Math.hypot(...minus(state.position, raw)) > 1);
  }
});

test("triangle BVH preserves holes, edge hits and tiny faces without inventing area", async () => {
  const [{ MeshBVH }] = await modules;
  const triangles = [
    ...quad([-4,-4,0], [4,-4,0], [4,-1,0], [-4,-1,0]),
    ...quad([-4,1,0], [4,1,0], [4,4,0], [-4,4,0]),
    ...quad([-4,-1,0], [-1,-1,0], [-1,1,0], [-4,1,0]),
    ...quad([1,-1,0], [4,-1,0], [4,1,0], [1,1,0]),
    [[7,0,0], [7,0,0], [7,0,0]],
    [[10,0,0], [10+1e-9,0,0], [10,1e-9,0]],
  ];
  const bvh = new MeshBVH(meshOf(triangles));
  assert.equal(bvh.firstHit([0,0,10], [0,0,-1]), null);
  close(bvh.firstHit([1,0,10], [0,0,-1]).distance, 10);
  close(bvh.firstHit([3,0,10], [0,0,-1]).distance, 10);
  assert.equal(bvh.firstHit([3,0,10], [1,0,0]), null);
  assert.equal(bvh.firstHit([3,0,-1], [0,0,-1]), null);
  assert.equal(bvh.firstHit([3,0,10], [0,0,-1], 5), null);
  close(bvh.firstHit([10+2e-10,2e-10,.001], [0,0,-1]).distance, .001);
});

test("specular triangles agree with an analytic image solution and reject an occluded leg", async () => {
  const [{ MeshBVH }] = await modules;
  const wall = quad([0,-10,0], [0,10,0], [0,10,10], [0,-10,10]);
  const rx = [4,-2,1.5], source = [6,4,6];
  const path = new MeshBVH(meshOf(wall)).reflections(rx, source)[0];
  assert.ok(path); path.point.forEach((v, i) => close(v, [0,.4,3.3][i], 1e-12));
  const incoming = unit(minus(path.point, source)), outgoing = unit(minus(rx, path.point));
  const reflected = [-incoming[0], incoming[1], incoming[2]];
  assert.ok(Math.hypot(...minus(reflected, outgoing)) < 1e-12); assert.ok(path.extra > 0);
  const obstacle = quad([2,-3,0], [2,2,0], [2,2,5], [2,-3,5]);
  const blocked = new MeshBVH(meshOf([...wall, ...obstacle], [0,0,1,1])).reflections(rx, source);
  assert.ok(!blocked.some(p => p.object === 0));
  const far = [20200000,4,6], longPath = new MeshBVH(meshOf(wall)).reflections(rx, far)[0];
  const expected = [0, ...[1,2].map(i => (rx[i]*far[0]+far[i]*rx[0])/(rx[0]+far[0]))];
  longPath.point.forEach((v, i) => close(v, expected[i], 1e-12));
  assert.equal(new MeshBVH(meshOf(wall)).reflections([4,0,5], [6,0,5]).length, 1);
});

test("full-city BVH nearest hits match an independent plane-and-edge brute-force oracle", async () => {
  const sim = await simulation(), mesh = sim.bvh.mesh, v = mesh.vertices, ix = mesh.indices;
  // Independent plane intersections and oriented edge half-planes, without BVH,
  // Moller-Trumbore, cached edges, or the production intersection routine.
  const faces = [];
  for (let t = 0; t < mesh.triangleCount; t++) {
    const points = [0,1,2].map(k => Array.from(v.subarray(ix[t*3+k]*3, ix[t*3+k]*3+3)));
    const n = cross(minus(points[1], points[0]), minus(points[2], points[0]));
    const length = Math.hypot(...n); if (length === 0) continue;
    faces.push({ points, n: n.map(x => x / length), id: t });
  }
  let state = 1942;
  const random = () => ((state = (Math.imul(state,1664525)+1013904223)>>>0) / 4294967296);
  for (let ray = 0; ray < 96; ray++) {
    const row = scene.track[Math.floor(random() * scene.track.length)];
    const [, orbit] = await modules;
    const origin = orbit.localPoint(row.slice(1), scene); origin[2] += 3;
    const direction = ray % 3 === 0 ? [0,0,1] : unit([random()-.5, random()-.5, random()*.8+.1]);
    let expected = Infinity;
    for (const face of faces) {
      const denominator = dot(face.n, direction); if (Math.abs(denominator) < 1e-12) continue;
      const distance = dot(face.n, minus(face.points[0], origin)) / denominator;
      if (distance <= 1e-5 || distance >= expected) continue;
      const point = origin.map((value, i) => value + direction[i] * distance);
      let inside = true;
      for (let i = 0; i < 3; i++) {
        const a = face.points[i], b = face.points[(i+1)%3];
        if (dot(cross(minus(b, a), minus(point, a)), face.n) < -1e-8) { inside = false; break; }
      }
      if (inside) expected = distance;
    }
    const actual = sim.bvh.firstHit(origin, direction);
    assert.equal(actual === null, expected === Infinity, `ray ${ray}`);
    if (actual) close(actual.distance, expected, 1e-6);
  }
});

test("Householder QR recovers a known solution and reports insufficient/rank-deficient geometry", async () => {
  const [, { leastSquares, geometricSPP }] = await modules;
  const rows = [[1,0,0,1],[0,1,0,1],[0,0,1,1],[-1,0,0,1],[0,-1,0,1],[0,0,-1,1]];
  const expected = [12,-5,.4,80], rhs = rows.map(row => dot(row, expected));
  leastSquares(rows, rhs).forEach((value, i) => close(value, expected[i], 1e-12));
  assert.equal(leastSquares(rows.slice(0,3), rhs.slice(0,3)), null);
  assert.equal(leastSquares(rows.map(() => [1,1,1,1]), rhs), null);
  assert.equal(geometricSPP([0,0,0], []), null);
});

test("real epochs are deterministic, finite, and keep ideal-bias results distinct from measured accuracy", async () => {
  const sim = await simulation();
  for (const epoch of [0,2000,4000,6000,8000,12409]) {
    const result = sim.compute({ epoch });
    assert.deepEqual(result, sim.compute({ epoch }));
    assert.equal(Object.values(result.counts).reduce((a,b) => a+b,0), result.satellites.length);
    for (const signal of result.satellites) {
      assert.ok(signal.elevation >= 10 && signal.id.startsWith("G"));
      assert.equal(signal.status === "LOS", signal.directHit === null);
      if (signal.status === "NLOS") assert.ok(signal.reflection);
      if (signal.status === "BLOCKED") assert.equal(signal.reflection, null);
      if (signal.reflection) assert.ok(Number.isFinite(signal.reflection.extra) && signal.reflection.extra >= 0);
    }
    if (result.spp.los) close(result.spp.los.horizontal, 0, 1e-6);
  }
  const urban = sim.compute({epoch:8000});
  assert.deepEqual(urban.counts, {LOS:4,NLOS:1,BLOCKED:3});
  assert.ok(urban.spp.all.horizontal > 60 && urban.spp.all.horizontal < 70);
  assert.equal(sim.compute({epoch:2000}).spp.all, null);
  assert.throws(() => sim.compute({epoch:12410}), RangeError);
  assert.throws(() => sim.compute({mask:NaN}), RangeError);
  assert.throws(() => sim.compute({antenna:21}), RangeError);
});
