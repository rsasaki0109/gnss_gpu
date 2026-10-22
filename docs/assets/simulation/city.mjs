import { C, GNSS_SYSTEMS, validateSystems, add, sub, dot, mul, norm, geodetic, enuRotation, localPoint, selectNavigation, receivedSatellite, geometricSPP } from "./orbit.mjs?v=glonass-1";
import { MeshBVH } from "./mesh.mjs?v=reflection-loops-1";

export class CitySimulation {
  constructor(scene, mesh, options) {
    if (scene.version !== 1 || scene.stats.triangles !== mesh.triangleCount || scene.stats.vertices !== mesh.vertexCount) throw new Error("Scene and mesh versions differ");
    this.scene = scene;
    this.bvh = new MeshBVH(mesh, options);
  }

  compute({ epoch = 0, antenna = 0, mask = 10, systems = GNSS_SYSTEMS } = {}) {
    validateSystems(systems);
    if (!Number.isInteger(epoch) || epoch < 0 || epoch >= this.scene.track.length || !Number.isFinite(antenna) || antenna < 0 || antenna > 20 || !Number.isFinite(mask) || mask < 5 || mask > 30) throw new RangeError("Invalid epoch or receiver setting");
    const row = this.scene.track[epoch], time = row[0];
    const base = row.slice(1), baseLLH = geodetic(base);
    const receiverECEF = add(base, mul(enuRotation(baseLLH.lat, baseLLH.lon)[2], antenna));
    const receiver = localPoint(receiverECEF, this.scene), llh = geodetic(receiverECEF);
    const orientation = enuRotation(llh.lat, llh.lon), candidates = [];
    const records = selectNavigation(this.scene.navigation.filter(record => systems.includes(record.id[0])), time, this.scene.week);
    for (const record of records) {
      const state = receivedSatellite(record, time, receiverECEF, this.scene.week);
      const delta = sub(state.position, receiverECEF), range = norm(delta), unit = mul(delta, 1 / range);
      const azimuth = (Math.atan2(dot(orientation[0], unit), dot(orientation[1], unit)) * 180 / Math.PI + 360) % 360;
      const elevation = Math.asin(Math.max(-1, Math.min(1, dot(orientation[2], unit)))) * 180 / Math.PI;
      if (elevation < mask) continue;
      const source = localPoint(state.position, this.scene), direction = mul(sub(source, receiver), 1 / range);
      const directHit = this.bvh.firstHit(receiver, direction, range);
      candidates.push({ id: record.id, azimuth, elevation, source, direction, range, directHit, clock: state.clock, transitTime: state.transitTime });
    }
    const batches = this.bvh.reflectionsBatch(receiver, candidates.map(satellite => satellite.source));
    const satellites = candidates.map((satellite, i) => {
      const { id, azimuth, elevation, source, direction, range, directHit, clock, transitTime } = satellite;
      const paths = batches[i], reflection = paths[0] ?? null;
      const status = !directHit ? "LOS" : reflection ? "NLOS" : "BLOCKED";
      if (directHit) directHit.vertices = this.bvh.triangle(directHit.triangle);
      if (reflection) reflection.vertices = this.bvh.triangle(reflection.triangle);
      return { id, azimuth, elevation, source, direction, range, status, directHit, reflection, reflectionCount: paths.length, clock, transitTime };
    });
    const counts = { LOS: 0, NLOS: 0, BLOCKED: 0 };
    satellites.forEach(satellite => counts[satellite.status]++);
    return { epoch, tow: row[0], week: this.scene.week, antenna, mask, systems, receiver, receiverECEF, latitude: llh.lat * 180 / Math.PI, longitude: llh.lon * 180 / Math.PI, height: llh.height,
      satellites, counts, healthy: records.length, spp: { all: geometricSPP(receiver, satellites), los: geometricSPP(receiver, satellites.filter(satellite => satellite.status === "LOS")) }, speedOfLight: C };
  }
}
