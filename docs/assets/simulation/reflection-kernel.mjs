import { encoded } from "./reflection-candidates.bytes.mjs?v=reflection-simd-1";
import { encoded as simdEncoded } from "./reflection-candidates-simd.bytes.mjs?v=reflection-simd-1";

export const REFLECTION_SOURCE_LIMIT = 512;
let attempted = false, compiled = null;
function candidateModule() {
  if (!attempted) {
    attempted = true;
    // Prefer strict SIMD, then the existing scalar Float64 module.
    for (const [bytes,simd] of [[simdEncoded,true],[encoded,false]]) {
      try {
        compiled = {module:new WebAssembly.Module(Uint8Array.from(atob(bytes), value => value.charCodeAt(0))),simd};
        break;
      } catch { /* Try the next backend; the caller also retains JavaScript. */ }
    }
  }
  return compiled;
}

export function createReflectionKernel(count, capacity = 4096) {
  if (!Number.isSafeInteger(count) || count < 1 || !Number.isInteger(capacity) || capacity < 0 || capacity > 65536) return null;
  const plane = count * 72, projection = plane + count * 32;
  const source = Math.ceil((projection + count) / 16) * 16, output = source + REFLECTION_SOURCE_LIMIT * 24;
  const pages = Math.ceil((output + capacity * 32) / 65536), backend = candidateModule();
  if (!backend || pages > 32768) return null;
  try {
    const memory = new WebAssembly.Memory({initial:pages,maximum:pages});
    const instance = new WebAssembly.Instance(backend.module,{env:{memory}}), buffer = memory.buffer;
    const input = new Float64Array(buffer,source,REFLECTION_SOURCE_LIMIT * 3);
    return {
      // Store the BVH arrays directly in this memory.
      edges:new Float64Array(buffer,0,count * 9), planes:new Float64Array(buffer,plane,count * 4),
      projection:new Uint8Array(buffer,projection,count), words:new Uint32Array(buffer,output,capacity * 8),
      values:new Float64Array(buffer,output,capacity * 4), simd:backend.simd, scans:0, fallbacks:0,
      scan(receiver,sources) {
        this.scans++;
        if (sources.length > REFLECTION_SOURCE_LIMIT) { this.fallbacks++; return null; }
        if (backend.simd) {
          // Packed pairs allow three vector loads. An unused odd lane cannot emit.
          for (let i=0;i<sources.length;i++) {
            const base=(i>>1)*6+(i&1);
            input[base]=sources[i][0];input[base+2]=sources[i][1];input[base+4]=sources[i][2];
          }
        } else {
          for (let i=0;i<sources.length;i++) {
            input[i*3]=sources[i][0];input[i*3+1]=sources[i][1];input[i*3+2]=sources[i][2];
          }
        }
        const result = instance.exports.scan(0,plane,projection,count,source,sources.length,receiver[0],receiver[1],receiver[2],output,capacity);
        if (result < 0) { this.fallbacks++; return null; }
        return result;
      },
    };
  } catch { return null; }
}
