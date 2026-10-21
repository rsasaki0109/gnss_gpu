import { simulate } from "./model.mjs";

self.onmessage = ({ data }) => {
  try {
    self.postMessage({ id: data.id, revision: data.revision, snapshot: simulate(data.settings) });
  } catch (error) {
    self.postMessage({ id: data.id, revision: data.revision, error: error.message });
  }
};
