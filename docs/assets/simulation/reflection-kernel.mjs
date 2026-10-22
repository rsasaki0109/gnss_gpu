import { encoded } from "./reflection-candidates.bytes.mjs?v=reflection-wasm-1";

export const REFLECTION_SOURCE_LIMIT = 512;
let attempted = false, compiled = null;
function candidateModule() {
  if (!attempted) {
    attempted = true;
    try {
      compiled = new WebAssembly.Module(Uint8Array.from(atob(encoded), value => value.charCodeAt(0)));
    } catch { /* The caller retains the JavaScript implementation. */ }
  }
  return compiled;
}

export function createReflectionKernel(count, capacity = 4096) {
  if (!Number.isSafeInteger(count) || count < 1 || !Number.isInteger(capacity) || capacity < 0 || capacity > 65536) return null;
  const plane = count * 72, projection = plane + count * 32;
  const source = Math.ceil((projection + count) / 8) * 8, output = source + REFLECTION_SOURCE_LIMIT * 24;
  const pages = Math.ceil((output + capacity * 32) / 65536), module = candidateModule();
  if (!module || pages > 32768) return null;
  try {
    const memory = new WebAssembly.Memory({initial:pages,maximum:pages});
    const instance = new WebAssembly.Instance(module,{env:{memory}}), buffer = memory.buffer;
    const input = new Float64Array(buffer,source,REFLECTION_SOURCE_LIMIT * 3);
    return {
      // Store the BVH arrays directly in this memory.
      edges:new Float64Array(buffer,0,count * 9), planes:new Float64Array(buffer,plane,count * 4),
      projection:new Uint8Array(buffer,projection,count), words:new Uint32Array(buffer,output,capacity * 8),
      values:new Float64Array(buffer,output,capacity * 4), scans:0, fallbacks:0,
      scan(receiver,sources) {
        this.scans++;
        if (sources.length > REFLECTION_SOURCE_LIMIT) { this.fallbacks++; return null; }
        for (let i=0;i<sources.length;i++) {
          input[i*3]=sources[i][0];input[i*3+1]=sources[i][1];input[i*3+2]=sources[i][2];
        }
        const result = instance.exports.scan(0,plane,projection,count,source,sources.length,receiver[0],receiver[1],receiver[2],output,capacity);
        if (result < 0) { this.fallbacks++; return null; }
        return result;
      },
    };
  } catch { return null; }
}
