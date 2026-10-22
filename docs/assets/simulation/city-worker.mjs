import { CitySimulation } from "./city.mjs?v=glonass-1";
import { decodeMesh } from "./mesh.mjs?v=glonass-1";
import { RouteEvaluation } from "./route-analysis.mjs?v=route-summary-1";

let simulation, evaluation;
self.onmessage = ({ data }) => {
  try {
    if (data.type === "init") {
      simulation = new CitySimulation(data.scene, decodeMesh(data.buffer));
      evaluation = new RouteEvaluation(simulation,message=>self.postMessage(message));
      self.postMessage({ type: "ready", nodes: simulation.bvh.links.length / 5 });
    } else if (data.type === "analysis-start") {
      if(!evaluation)throw new Error("City mesh has not loaded");
      evaluation.start(data);
    } else if (data.type === "analysis-pause") {
      evaluation?.pause();
    } else if (data.type === "analysis-resume") {
      evaluation?.resume();
    } else if (data.type === "analysis-cancel") {
      evaluation?.cancel();
    } else {
      if (!simulation) throw new Error("City mesh has not loaded");
      const started = performance.now();
      const snapshot = simulation.compute(data.settings);
      self.postMessage({ type: "result", id: data.id, revision: data.revision, snapshot, milliseconds: performance.now() - started });
    }
  } catch (error) {
    self.postMessage({ type: data.type?.startsWith("analysis-")?"analysis-error":"error", id: data.id, revision: data.revision, error: error.message });
  }
};
