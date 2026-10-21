import { CitySimulation } from "./city.mjs?v=reflection-1";
import { decodeMesh } from "./mesh.mjs?v=reflection-1";

let simulation;
self.onmessage = ({ data }) => {
  try {
    if (data.type === "init") {
      simulation = new CitySimulation(data.scene, decodeMesh(data.buffer));
      self.postMessage({ type: "ready", nodes: simulation.bvh.links.length / 5 });
    } else {
      if (!simulation) throw new Error("City mesh has not loaded");
      const started = performance.now();
      const snapshot = simulation.compute(data.settings);
      self.postMessage({ type: "result", id: data.id, revision: data.revision, snapshot, milliseconds: performance.now() - started });
    }
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, revision: data.revision, error: error.message });
  }
};
