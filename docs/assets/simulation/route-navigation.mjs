// Navigation uses every original reference pose. Only this display is projected.
import { localPoint } from "./orbit.mjs";

export function routeLayout(points, width, height, padding = 22) {
  const bounds = [[Infinity, Infinity], [-Infinity, -Infinity]];
  for (const point of points) for (let axis = 0; axis < 2; axis++) {
    bounds[0][axis] = Math.min(bounds[0][axis], point[axis]);
    bounds[1][axis] = Math.max(bounds[1][axis], point[axis]);
  }
  const span = bounds[0].map((value, axis) => Math.max(1, bounds[1][axis] - value));
  const scale = Math.min(Math.max(1, width - 2 * padding) / span[0], Math.max(1, height - 2 * padding) / span[1]);
  const center = bounds[0].map((value, axis) => (value + bounds[1][axis]) / 2);
  return { scale, points: points.map(point => [width / 2 + (point[0] - center[0]) * scale, height / 2 - (point[1] - center[1]) * scale]) };
}

export function nearestRecordedEpoch(points, x, y, current, radius = 18) {
  let best = null, distance = radius * radius;
  for (let epoch = 0; epoch < points.length; epoch++) {
    const squared = (points[epoch][0] - x) ** 2 + (points[epoch][1] - y) ** 2;
    // Exact spatial ties prefer the current time, then the earlier source row.
    if (squared < distance || (squared === distance && (best === null || Math.abs(epoch - current) < Math.abs(best - current)))) {
      best = epoch; distance = squared;
    }
  }
  return best;
}

export function stepEpoch(epoch, delta, count) {
  return Math.max(0, Math.min(count - 1, Math.round(epoch + delta)));
}

export class RouteNavigator {
  constructor(canvas, scene, onSelect, onPreview) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.positions = scene.track.map(row => localPoint(row.slice(1), scene));
    this.onSelect = onSelect; this.onPreview = onPreview; this.enabled = false; this.epoch = 0; this.hover = null;
    canvas.dataset.poseCount = this.positions.length;
    canvas.setAttribute("aria-valuemin", "1"); canvas.setAttribute("aria-valuemax", String(this.positions.length));
    const pick = event => {
      const box = canvas.getBoundingClientRect();
      return nearestRecordedEpoch(this.layout.points, event.clientX - box.left, event.clientY - box.top, this.epoch, event.pointerType === "touch" ? 24 : 18);
    };
    canvas.addEventListener("pointerdown", event => {
      if (this.enabled && event.isPrimary && event.button === 0) this.press = { id: event.pointerId, x: event.clientX, y: event.clientY };
    });
    canvas.addEventListener("pointerup", event => {
      const press = this.press; this.press = null;
      if (!this.enabled || press?.id !== event.pointerId || Math.hypot(event.clientX - press.x, event.clientY - press.y) > 8) return;
      const epoch = pick(event);
      if (epoch !== null) { canvas.focus({ preventScroll: true }); this.onSelect(epoch); }
    });
    canvas.addEventListener("pointercancel", () => { this.press = null; this.preview(null); });
    canvas.addEventListener("pointermove", event => { if (this.enabled && event.pointerType !== "touch") this.preview(pick(event)); });
    canvas.addEventListener("pointerleave", () => { this.press = null; this.preview(null); });
    canvas.addEventListener("keydown", event => {
      if (!this.enabled) return;
      const direction = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[event.key];
      let epoch;
      if (direction) epoch = stepEpoch(this.epoch, direction * (event.shiftKey ? 10 : 1), this.positions.length);
      else if (event.key === "Home") epoch = 0;
      else if (event.key === "End") epoch = this.positions.length - 1;
      else return;
      event.preventDefault(); this.onSelect(epoch);
    });
    new ResizeObserver(() => this.resize()).observe(canvas); this.resize();
  }

  resize() {
    this.width = this.canvas.clientWidth; this.height = this.canvas.clientHeight;
    this.dpr = Math.min(devicePixelRatio || 1, 2);
    this.canvas.width = Math.round(this.width * this.dpr); this.canvas.height = Math.round(this.height * this.dpr);
    this.layout = routeLayout(this.positions, this.width, this.height);
    this.path = new Path2D();
    this.layout.points.forEach(([x, y], i) => i ? this.path.lineTo(x, y) : this.path.moveTo(x, y));
    this.schedule();
  }

  update(epoch, ready, enabled, valueText) {
    if (this.epoch !== epoch) this.preview(null);
    this.epoch = epoch; this.ready = ready; this.enabled = enabled;
    this.canvas.tabIndex = enabled ? 0 : -1;
    this.canvas.setAttribute("aria-disabled", String(!enabled));
    this.canvas.setAttribute("aria-valuenow", String(epoch + 1)); this.canvas.setAttribute("aria-valuetext", valueText);
    Object.assign(this.canvas.dataset, { epoch, state: !enabled ? "loading" : ready ? "ready" : "tracing" });
    this.schedule();
  }

  preview(epoch) {
    if (epoch === this.hover) return;
    this.hover = epoch; this.onPreview(epoch); this.schedule();
  }

  schedule() {
    if (this.scheduled) return;
    this.scheduled = requestAnimationFrame(() => { this.scheduled = 0; this.draw(); });
  }

  draw() {
    if (!this.width || !this.height) return;
    const ctx = this.ctx, points = this.layout.points;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); ctx.clearRect(0, 0, this.width, this.height);
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    ctx.strokeStyle = "#79a0bd"; ctx.lineWidth = 1.6; ctx.stroke(this.path);
    ctx.beginPath();
    for (let i = Math.max(0, this.epoch - 50); i <= Math.min(points.length - 1, this.epoch + 50); i++) {
      const [x, y] = points[i]; if (i === Math.max(0, this.epoch - 50)) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = this.ready ? "#85e4bc" : "#efbc79"; ctx.lineWidth = 3; ctx.stroke();
    const dot = (epoch, radius, fill, stroke) => {
      ctx.beginPath(); ctx.arc(...points[epoch], radius, 0, Math.PI * 2);
      ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1.5; ctx.stroke();
    };
    dot(0, 3, "#12202b", "#afbfca"); dot(points.length - 1, 3, "#afbfca", "#afbfca");
    if (this.hover !== null) dot(this.hover, 5, "#12202b", "#e8eff3");
    dot(this.epoch, 5, this.ready ? "#85e4bc" : "#efbc79", "#0c141c");
    ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#a0afb9";
    ctx.fillText("N ↑", 12, 17);
    const target = 46 / this.layout.scale, power = 10 ** Math.floor(Math.log10(target));
    const metres = [1, 2, 5, 10].map(v => v * power).find(v => v >= target);
    const length = metres * this.layout.scale, y = this.height - 14;
    ctx.strokeStyle = "#a0afb9"; ctx.lineWidth = 1; ctx.beginPath();
    ctx.moveTo(12, y - 3); ctx.lineTo(12, y); ctx.lineTo(12 + length, y); ctx.lineTo(12 + length, y - 3); ctx.stroke();
    ctx.fillText(`${metres} m`, 12, y - 7);
  }
}
