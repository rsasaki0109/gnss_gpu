// Float32 is used only for drawing. The worker retains the Float64 source mesh.
import { localPoint, sub, mul, norm, add } from "./orbit.mjs";
import { OSMBasemap } from "./osm-layer.mjs";
import { viewAxes, surfaceOrientation, framePoints } from "./view-geometry.mjs";

const COLORS = { LOS: [.52, .89, .74], NLOS: [.94, .74, .47], BLOCKED: [.92, .56, .57] };

export class CityRenderer {
  constructor(canvas, overlay, scene, mesh, onMapChange = () => {}) {
    this.canvas = canvas; this.overlay = overlay; this.ctx = overlay.getContext("2d");
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: true });
    if (!gl) throw new Error("This view requires WebGL 2. Try a browser with hardware acceleration enabled.");
    this.gl = gl; this.scene = scene;
    this.camera = { yaw: -.63, pitch: .82, zoom: 1, target: [0, 0, 0], overview: false, span: 740 };
    const shader = (type, source) => {
      const result = gl.createShader(type);
      gl.shaderSource(result, source); gl.compileShader(result);
      if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result));
      return result;
    };
    const vertex = shader(gl.VERTEX_SHADER, `#version 300 es
      in vec3 a_position; in vec3 a_normal; in vec3 a_color;
      uniform mat3 u_view; uniform vec3 u_target; uniform vec3 u_scale;
      uniform float u_lit; out vec3 v_color;
      void main() {
        vec3 p = u_view * (a_position - u_target);
        gl_Position = vec4(p * u_scale + vec3(0., -.12, 0.), 1.);
        float shade = .53 + .20 * abs(a_normal.z) + .27 * abs(dot(a_normal, normalize(vec3(.4, -.6, .8))));
        v_color = a_color * mix(1., shade, u_lit);
      }`);
    const fragment = shader(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float; in vec3 v_color; out vec4 color;
      void main() { color = vec4(v_color, 1.); }`);
    this.program = gl.createProgram();
    gl.attachShader(this.program, vertex); gl.attachShader(this.program, fragment); gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    gl.deleteShader(vertex); gl.deleteShader(fragment);
    this.uniforms = Object.fromEntries(["view", "target", "scale", "lit"].map(key => [key, gl.getUniformLocation(this.program, `u_${key}`)]));
    this.locations = ["position", "normal", "color"].map(key => gl.getAttribLocation(this.program, `a_${key}`));
    const data = new Float32Array(mesh.triangleCount * 27), p = mesh.vertices, ix = mesh.indices;
    for (let t = 0; t < mesh.triangleCount; t++) {
      const a = ix[t * 3] * 3, b = ix[t * 3 + 1] * 3, c = ix[t * 3 + 2] * 3;
      const u = [p[b] - p[a], p[b + 1] - p[a + 1], p[b + 2] - p[a + 2]], v = [p[c] - p[a], p[c + 1] - p[a + 1], p[c + 2] - p[a + 2]];
      const normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
      const length = Math.hypot(...normal), tone = (mesh.owners[t] % 7) / 100;
      const color = scene.objects[mesh.owners[t]].kind === "bridge" ? [.40, .53, .56] : [.31 + tone, .43 + tone, .52 + tone];
      for (let k = 0; k < 3; k++) {
        const offset = t * 27 + k * 9, index = ix[t * 3 + k] * 3;
        data.set(p.subarray(index, index + 3), offset);
        data.set(normal.map(value => length ? value / length : 0), offset + 3); data.set(color, offset + 6);
      }
    }
    this.city = this.upload(data, gl.TRIANGLES);
    const lines = [], line = (a, b, color) => lines.push(...a, 0, 0, 0, ...color, ...b, 0, 0, 0, ...color);
    const [lo, hi] = scene.bounds, floor = lo[2] - 1;
    for (let x = Math.floor(lo[0] / 100) * 100; x <= hi[0]; x += 100) line([x, lo[1], floor], [x, hi[1], floor], [.12, .21, .27]);
    for (let y = Math.floor(lo[1] / 100) * 100; y <= hi[1]; y += 100) line([lo[0], y, floor], [hi[0], y, floor], [.12, .21, .27]);
    this.grid = this.upload(new Float32Array(lines), gl.LINES); lines.length = 0;
    const track = scene.track.map(row => localPoint(row.slice(1), scene));
    for (let i = 1; i < track.length; i++) line(track[i - 1], track[i], [.30, .49, .65]);
    this.track = this.upload(new Float32Array(lines), gl.LINES);
    this.dynamic = this.upload(new Float32Array(), gl.LINES, true);
    this.surface = this.upload(new Float32Array(), gl.TRIANGLES, true);
    this.basemap = new OSMBasemap(gl, scene, onMapChange); this.resize();
  }

  upload(data, mode, dynamic = false) {
    const gl = this.gl, vao = gl.createVertexArray(), buffer = gl.createBuffer();
    gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(gl.ARRAY_BUFFER, data, dynamic ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
    this.locations.forEach((location, i) => { gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 3, gl.FLOAT, false, 36, i * 12); });
    return { vao, buffer, count: data.length / 9, mode };
  }

  resize() {
    this.width = this.canvas.clientWidth; this.height = this.canvas.clientHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    for (const canvas of [this.canvas, this.overlay]) { canvas.width = Math.round(this.width * dpr); canvas.height = Math.round(this.height * dpr); }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0); this.gl.viewport(0, 0, this.canvas.width, this.canvas.height);
  }

  basis() {
    const { yaw, pitch } = this.camera;
    Object.assign(this, viewAxes(yaw, pitch));
    const [lo, hi] = this.scene.bounds;
    const span = this.camera.overview ? Math.max(hi[0] - lo[0], hi[1] - lo[1]) * 1.3 : this.camera.span;
    this.scale = Math.min(this.width, this.height * 1.3) / span * this.camera.zoom;
  }

  project(point) {
    const delta = sub(point, this.camera.target), dot = axis => delta.reduce((v, d, i) => v + d * axis[i], 0);
    return [this.width / 2 + dot(this.right) * this.scale, this.height * .56 - dot(this.up) * this.scale];
  }

  focusSurface(snapshot, signal, kind) {
    const hit = kind === "reflection" ? signal?.reflection : signal?.directHit;
    if (!hit) return false;
    if (!this.focus) this.previousCamera = { ...this.camera, target: this.camera.target.slice() };
    this.focus = { kind, satellite: signal.id, hit, points: [snapshot.receiver, hit.point, ...hit.vertices] };
    Object.assign(this.camera, surfaceOrientation(snapshot.receiver, hit), { overview: false, zoom: 1 });
    const color = COLORS[kind === "reflection" ? "NLOS" : "BLOCKED"].map(value => value * .8);
    const data = new Float32Array(hit.vertices.flatMap(point => [...point, 0, 0, 0, ...color]));
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.surface.buffer); this.gl.bufferData(this.gl.ARRAY_BUFFER, data, this.gl.DYNAMIC_DRAW);
    this.surface.count = 3; return true;
  }

  clearFocus() {
    if (!this.focus) return;
    Object.assign(this.camera, this.previousCamera); this.focus = null; this.previousCamera = null;
  }

  draw(snapshot, selected, reflections) {
    if (!this.width || !this.height) return;
    const gl = this.gl, ctx = this.ctx;
    if (this.focus) Object.assign(this.camera, framePoints(this.focus.points, this.camera.yaw, this.camera.pitch, this.width, this.height));
    else if (!this.camera.overview && snapshot) this.camera.target = snapshot.receiver.slice();
    if (this.camera.overview) this.camera.target = this.scene.bounds[0].map((value, i) => (value + this.scene.bounds[1][i]) / 2);
    this.basis(); gl.clearColor(.055, .10, .135, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const matrix = [];
    for (let i = 0; i < 3; i++) matrix.push(this.right[i], this.up[i], -this.depth[i]);
    const mapped = snapshot && this.basemap.draw(this.camera, this.width, this.height, this.scale, matrix);
    gl.useProgram(this.program); gl.enable(gl.DEPTH_TEST); gl.depthFunc(gl.LEQUAL);
    gl.uniformMatrix3fv(this.uniforms.view, false, matrix); gl.uniform3fv(this.uniforms.target, this.camera.target);
    gl.uniform3f(this.uniforms.scale, 2 * this.scale / this.width, 2 * this.scale / this.height, 1 / 25000);
    const draw = (item, lit) => { gl.uniform1f(this.uniforms.lit, lit); gl.bindVertexArray(item.vao); gl.drawArrays(item.mode, 0, item.count); };
    draw(this.city, 1); if (!mapped) draw(this.grid, 0); draw(this.track, 0); ctx.clearRect(0, 0, this.width, this.height);
    if (!snapshot) return;
    if (this.focus) {
      gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(-1, -1); draw(this.surface, 0); gl.disable(gl.POLYGON_OFFSET_FILL);
    }
    Object.assign(this.canvas.dataset, { view: this.focus?.kind ?? (this.camera.overview ? "overview" : "receiver"), focusSatellite: this.focus?.satellite ?? "", focusTriangle: this.focus?.hit.triangle ?? "", focusSide: this.focus ? this.camera.pitch < 0 ? "below" : "above" : "" });
    const data = [], labels = [];
    const line = (points, color) => { for (let i = 1; i < points.length; i++) for (const point of [points[i - 1], points[i]]) data.push(...point, 0, 0, 0, ...color); };
    const end = (point, direction) => add(point, mul(direction, (this.focus ? this.camera.span * .8 : 420) / this.camera.zoom));
    const ray = signal => {
      const color = COLORS[this.focus?.kind === "blockage" ? "BLOCKED" : signal.status], active = signal.id === selected, tint = active ? color : color.map(v => v * .58);
      const directEnd = signal.directHit?.point ?? end(snapshot.receiver, signal.direction);
      if (this.focus?.kind !== "reflection") {
        line([snapshot.receiver, directEnd], tint);
        if (active && signal.directHit) line([directEnd, end(directEnd, signal.direction)], [.34, .23, .26]);
      }
      let labelPoint = end(snapshot.receiver, signal.direction);
      if (reflections && signal.reflection && this.focus?.kind !== "blockage") {
        const q = signal.reflection.point, direction = sub(signal.source, q);
        const sourceEnd = end(q, mul(direction, 1 / norm(direction)));
        line([snapshot.receiver, q, sourceEnd], COLORS.NLOS.map(v => v * (active ? 1 : .6)));
        if (this.focus) labelPoint = sourceEnd;
      }
      const hits = this.focus ? [this.focus.hit] : [signal.directHit, reflections ? signal.reflection : null];
      if (active) for (const hit of hits) if (hit) line([...hit.vertices, hit.vertices[0]], COLORS[hit === signal.reflection ? "NLOS" : "BLOCKED"]);
      labels.push({ point: this.project(labelPoint), text: signal.id, color: tint });
    };
    if (!this.focus) for (const signal of snapshot.satellites) if (signal.id !== selected) ray(signal);
    const active = snapshot.satellites.find(s => s.id === selected); if (active) ray(active);
    this.canvas.dataset.displaySignals = labels.length;
    line([snapshot.receiver, add(snapshot.receiver, [0, 0, 9])], COLORS.LOS);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.dynamic.buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(data), gl.DYNAMIC_DRAW);
    this.dynamic.count = data.length / 9; draw(this.dynamic, 0);
    for (const label of labels) {
      const [x, y] = label.point;
      if (x < 25 || x > this.width - 25 || y < 75 || y > this.height - 65) continue;
      ctx.font = "11px ui-monospace, monospace"; ctx.fillStyle = "#0c141ce8"; ctx.fillRect(x - 18, y - 10, 36, 19);
      ctx.fillStyle = `rgb(${label.color.map(v => Math.round(v * 255)).join(",")})`;
      ctx.textAlign = "center"; ctx.fillText(label.text, x, y + 3);
    }
    const [x, y] = this.project(snapshot.receiver);
    ctx.beginPath(); ctx.arc(x, y, 10, 0, 2 * Math.PI); ctx.fillStyle = "#85e4bc22"; ctx.fill();
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.fillStyle = "#85e4bc"; ctx.fill();
    ctx.strokeStyle = "#0c141c"; ctx.lineWidth = 2; ctx.stroke();
    if (this.focus) {
      const [hx, hy] = this.project(this.focus.hit.point), text = this.focus.kind === "reflection" ? "Reflection" : "Obstruction";
      const color = this.focus.kind === "reflection" ? "#efbc79" : "#ec8f91";
      ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI * 2); ctx.fillStyle = color; ctx.fill(); ctx.stroke();
      ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left";
      const textWidth = ctx.measureText(text).width, lx = Math.max(8, Math.min(this.width - textWidth - 18, hx + 10));
      ctx.fillStyle = "#0c141ce8"; ctx.fillRect(lx - 5, hy - 12, textWidth + 10, 21);
      ctx.fillStyle = color; ctx.fillText(text, lx, hy + 2);
    }
    ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#b0c3cf"; ctx.textAlign = "left";
    const maximum = Math.min(120, this.width * .25) / this.scale;
    const power = 10 ** Math.floor(Math.log10(maximum));
    const metres = [5,2,1].find(value => value * power <= maximum) * power, size = metres * this.scale;
    ctx.fillStyle = "#0c141ce8"; ctx.fillRect(16, this.height - 76, Math.max(size + 18, 100), 36);
    ctx.fillStyle = "#dce8ef";
    ctx.fillRect(24, this.height - 50, size, 1); ctx.fillRect(24, this.height - 54, 1, 5); ctx.fillRect(24 + size, this.height - 54, 1, 5);
    ctx.fillText(`${metres} m · ENU`, 24, this.height - 61);
  }
}
