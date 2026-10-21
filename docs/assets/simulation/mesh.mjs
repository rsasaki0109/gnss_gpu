// Full-resolution Float64 triangle geometry. Rendering precision is separate.
export function decodeMesh(buffer) {
  const view = new DataView(buffer);
  if (new TextDecoder().decode(new Uint8Array(buffer, 0, 8)) !== "PLTGEO1\0") throw new Error("Unknown mesh format");
  const nv = view.getUint32(8, true), nt = view.getUint32(12, true);
  if (!nv || !nt || buffer.byteLength !== 16 + nv * 24 + nt * 17) throw new Error("Incomplete city mesh");
  const vertices = new Float64Array(buffer, 16, nv * 3);
  const indices = new Uint32Array(buffer, 16 + nv * 24, nt * 3);
  const owners = new Uint32Array(buffer, 16 + nv * 24 + nt * 12, nt);
  const types = new Uint8Array(buffer, 16 + nv * 24 + nt * 16, nt);
  return { vertices, indices, owners, types, triangleCount: nt, vertexCount: nv };
}

const CONTACT = 1e-5; // 10 micrometres; below the source model's millimetre scale.

export class MeshBVH {
  constructor(mesh) {
    this.mesh = mesh;
    const count = mesh.triangleCount;
    this.edges = new Float64Array(count * 9);
    this.planes = new Float64Array(count * 4);
    this.areas = new Float64Array(count);
    this.projection = new Uint8Array(count);
    this.order = Uint32Array.from({ length: count }, (_, i) => i);
    const boxes = new Float64Array(count * 6), centers = new Float64Array(count * 3);
    for (let t = 0; t < count; t++) {
      const offset = t * 9;
      for (let axis = 0; axis < 3; axis++) {
        const a = mesh.vertices[mesh.indices[t * 3] * 3 + axis];
        const b = mesh.vertices[mesh.indices[t * 3 + 1] * 3 + axis];
        const c = mesh.vertices[mesh.indices[t * 3 + 2] * 3 + axis];
        this.edges[offset + axis] = a;
        this.edges[offset + 3 + axis] = b - a;
        this.edges[offset + 6 + axis] = c - a;
        boxes[t * 6 + axis] = Math.min(a, b, c) - 1e-7;
        boxes[t * 6 + 3 + axis] = Math.max(a, b, c) + 1e-7;
        centers[t * 3 + axis] = (a + b + c) / 3;
      }
      const e = this.edges, n = [e[offset + 4] * e[offset + 8] - e[offset + 5] * e[offset + 7], e[offset + 5] * e[offset + 6] - e[offset + 3] * e[offset + 8], e[offset + 3] * e[offset + 7] - e[offset + 4] * e[offset + 6]];
      const size = Math.hypot(...n);
      this.areas[t] = size;
      let plane = 0;
      for (let axis = 0; axis < 3; axis++) { const value = size === 0 ? 0 : n[axis] / size; this.planes[t * 4 + axis] = value; plane += value * e[offset + axis]; }
      this.planes[t * 4 + 3] = plane;
      this.projection[t] = Math.abs(n[0]) > Math.abs(n[1]) ? (Math.abs(n[0]) > Math.abs(n[2]) ? 0 : 2) : (Math.abs(n[1]) > Math.abs(n[2]) ? 1 : 2);
    }
    const nodes = [];
    const compare = (a, b, axis) => centers[a * 3 + axis] - centers[b * 3 + axis] || a - b;
    const partition = (start, end, middle, axis) => {
      let left = start, right = end - 1;
      while (left < right) {
        const pivot = this.order[(left + right) >>> 1];
        let i = left, j = right;
        while (i <= j) {
          while (compare(this.order[i], pivot, axis) < 0) i++;
          while (compare(this.order[j], pivot, axis) > 0) j--;
          if (i <= j) { [this.order[i], this.order[j]] = [this.order[j], this.order[i]]; i++; j--; }
        }
        if (middle <= j) right = j;
        else if (middle >= i) left = i;
        else return;
      }
    };
    const build = (start, end) => {
      const minimum = [Infinity, Infinity, Infinity], maximum = [-Infinity, -Infinity, -Infinity];
      for (let i = start; i < end; i++) for (let axis = 0; axis < 3; axis++) { minimum[axis] = Math.min(minimum[axis], boxes[this.order[i] * 6 + axis]); maximum[axis] = Math.max(maximum[axis], boxes[this.order[i] * 6 + 3 + axis]); }
      const node = { minimum, maximum, start, count: end - start, left: -1, right: -1, axis: 0 }, id = nodes.length;
      nodes.push(node);
      if (end - start > 8) {
        node.axis = maximum[0] - minimum[0] > maximum[1] - minimum[1] ? 0 : 1;
        if (maximum[2] - minimum[2] > maximum[node.axis] - minimum[node.axis]) node.axis = 2;
        const middle = (start + end) >>> 1;
        partition(start, end, middle, node.axis);
        node.left = build(start, middle); node.right = build(middle, end); node.count = 0;
      }
      return id;
    };
    build(0, count);
    this.bounds = new Float64Array(nodes.length * 6);
    this.links = new Int32Array(nodes.length * 5);
    nodes.forEach((node, i) => { this.bounds.set([...node.minimum, ...node.maximum], i * 6); this.links.set([node.left, node.right, node.start, node.count, node.axis], i * 5); });
  }

  triangle(t) {
    const o = t * 9, e = this.edges;
    return [Array.from(e.subarray(o, o + 3)), [e[o] + e[o + 3], e[o + 1] + e[o + 4], e[o + 2] + e[o + 5]], [e[o] + e[o + 6], e[o + 1] + e[o + 7], e[o + 2] + e[o + 8]]];
  }

  triangleDistance(t, origin, direction, minimum, maximum) {
    const e = this.edges, o = t * 9;
    const px = direction[1] * e[o + 8] - direction[2] * e[o + 7], py = direction[2] * e[o + 6] - direction[0] * e[o + 8], pz = direction[0] * e[o + 7] - direction[1] * e[o + 6];
    const det = e[o + 3] * px + e[o + 4] * py + e[o + 5] * pz;
    if (det === 0 || Math.abs(det) < this.areas[t] * 1e-12) return null;
    const x = origin[0] - e[o], y = origin[1] - e[o + 1], z = origin[2] - e[o + 2];
    const u = (x * px + y * py + z * pz) / det;
    if (u < -1e-10 || u > 1 + 1e-10) return null;
    const qx = y * e[o + 5] - z * e[o + 4], qy = z * e[o + 3] - x * e[o + 5], qz = x * e[o + 4] - y * e[o + 3];
    const v = (direction[0] * qx + direction[1] * qy + direction[2] * qz) / det;
    if (v < -1e-10 || u + v > 1 + 1e-10) return null;
    const distance = (e[o + 6] * qx + e[o + 7] * qy + e[o + 8] * qz) / det;
    return distance > minimum && distance < maximum ? distance : null;
  }

  firstHit(origin, direction, maximum = Infinity, minimum = CONTACT) {
    let distance = maximum, triangle = -1;
    const stack = [0], b = this.bounds, links = this.links;
    while (stack.length) {
      const node = stack.pop();
      let enter = minimum, exit = distance;
      for (let axis = 0; axis < 3; axis++) {
        if (direction[axis] === 0) { if (origin[axis] < b[node * 6 + axis] || origin[axis] > b[node * 6 + 3 + axis]) { exit = -Infinity; break; } }
        else {
          let near = (b[node * 6 + axis] - origin[axis]) / direction[axis], far = (b[node * 6 + 3 + axis] - origin[axis]) / direction[axis];
          if (near > far) [near, far] = [far, near];
          enter = Math.max(enter, near); exit = Math.min(exit, far);
          if (enter > exit) break;
        }
      }
      if (enter > exit) continue;
      const offset = node * 5;
      if (links[offset + 3]) {
        for (let i = links[offset + 2]; i < links[offset + 2] + links[offset + 3]; i++) {
          const t = this.order[i], hit = this.triangleDistance(t, origin, direction, minimum, distance + (triangle < 0 ? 0 : 1e-9));
          if (hit !== null && (hit < distance || (hit === distance && t < triangle))) { distance = hit; triangle = t; }
        }
      } else if (direction[links[offset + 4]] >= 0) stack.push(links[offset + 1], links[offset]);
      else stack.push(links[offset], links[offset + 1]);
    }
    return triangle < 0 ? null : { distance, triangle, object: this.mesh.owners[triangle], point: origin.map((v, i) => v + direction[i] * distance) };
  }

  clearSegment(a, b) {
    const delta = b.map((v, i) => v - a[i]), distance = Math.hypot(...delta);
    return distance <= 2 * CONTACT || this.firstHit(a, delta.map(v => v / distance), distance - CONTACT) === null;
  }

  containsPoint(t, point) {
    return this.containsXYZ(t, point[0], point[1], point[2]);
  }

  containsXYZ(t, px, py, pz) {
    const axis = this.projection[t], i = (axis + 1) % 3, j = (axis + 2) % 3, o = t * 9, e = this.edges;
    const x = (i === 0 ? px : i === 1 ? py : pz) - e[o + i], y = (j === 0 ? px : j === 1 ? py : pz) - e[o + j];
    const determinant = e[o + 3 + i] * e[o + 6 + j] - e[o + 3 + j] * e[o + 6 + i];
    const u = (x * e[o + 6 + j] - y * e[o + 6 + i]) / determinant;
    const v = (e[o + 3 + i] * y - e[o + 3 + j] * x) / determinant;
    return u >= -1e-10 && v >= -1e-10 && u + v <= 1 + 1e-10;
  }

  reflections(receiver, source) {
    const result = [], p = this.planes, direct = Math.hypot(...source.map((v, i) => v - receiver[i]));
    for (let t = 0; t < this.mesh.triangleCount; t++) {
      const o = t * 4, nx = p[o], ny = p[o + 1], nz = p[o + 2];
      const dr = nx * receiver[0] + ny * receiver[1] + nz * receiver[2] - p[o + 3];
      const ds = nx * source[0] + ny * source[1] + nz * source[2] - p[o + 3];
      if (Math.abs(dr) < CONTACT || dr * ds <= 0) continue;
      const factor = dr / (dr + ds);
      const mx = receiver[0] - 2 * dr * nx, my = receiver[1] - 2 * dr * ny, mz = receiver[2] - 2 * dr * nz;
      const qx = mx + (source[0] - mx) * factor, qy = my + (source[1] - my) * factor, qz = mz + (source[2] - mz) * factor;
      if (!this.containsXYZ(t, qx, qy, qz)) continue;
      const q = [qx, qy, qz];
      if (result.some(path => Math.hypot(...path.point.map((v, i) => v - q[i])) < 1e-7)) continue;
      if (!this.clearSegment(receiver, q) || !this.clearSegment(q, source)) continue;
      const distance = Math.hypot(...q.map((v, i) => v - receiver[i])) + Math.hypot(...source.map((v, i) => v - q[i]));
      const extra = distance - direct;
      if (extra < -1e-7) throw new Error("Reflection violated triangle inequality");
      result.push({ point: q, triangle: t, object: this.mesh.owners[t], distance, extra: Math.max(0, extra) });
    }
    result.sort((a, b) => a.distance - b.distance || a.triangle - b.triangle);
    return result;
  }
}
