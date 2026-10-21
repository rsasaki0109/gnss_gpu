// Camera geometry only. Source coordinates and signal solutions are never changed.
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

export function viewAxes(yaw, pitch) {
  const c = Math.cos(yaw), s = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  return { right: [c, -s, 0], up: [-s * sp, -c * sp, cp], depth: [s * cp, c * cp, sp] };
}

export function surfaceOrientation(receiver, hit) {
  const [a, b, c] = hit.vertices, u = b.map((v, i) => v - a[i]), v = c.map((value, i) => value - a[i]);
  let normal = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  const towardReceiver = receiver.map((value, i) => value - hit.point[i]);
  if (dot(normal, towardReceiver) < 0) normal = normal.map(value => -value);
  const horizontal = Math.hypot(normal[0], normal[1]);
  const yaw = horizontal > Math.hypot(...normal) * 1e-10 ? Math.atan2(normal[0], normal[1]) : Math.atan2(towardReceiver[0], towardReceiver[1]);
  const angle = Math.atan2(normal[2], horizontal);
  const pitch = Math.abs(angle) < .4 ? .4 : Math.max(-1.3, Math.min(1.3, angle));
  return { yaw, pitch };
}

export function clampViewPitch(pitch, surface = false, direction = 0) {
  if (!surface) return Math.max(.15, Math.min(1.48, pitch));
  const limited = Math.max(-1.48, Math.min(1.48, pitch));
  // The map's ground-plane projection is singular at a horizontal view.
  return Math.abs(limited) >= .15 ? limited : (direction || limited || 1) < 0 ? -.15 : .15;
}

export function framePoints(points, yaw, pitch, width, height) {
  const axes = Object.values(viewAxes(yaw, pitch)), anchor = points[0];
  const low = [Infinity, Infinity, Infinity], high = [-Infinity, -Infinity, -Infinity];
  for (const point of points) {
    const delta = point.map((value, i) => value - anchor[i]);
    axes.forEach((axis, i) => { const value = dot(delta, axis); low[i] = Math.min(low[i], value); high[i] = Math.max(high[i], value); });
  }
  const center = low.map((value, i) => (value + high[i]) / 2);
  const target = anchor.map((value, i) => value + axes.reduce((sum, axis, j) => sum + axis[i] * center[j], 0));
  // Leave room for the city caption, camera controls, scale and map credit.
  const halfWidth = Math.max(20, width / 2 - 50);
  const halfHeight = Math.max(20, Math.min(height * .56 - 150, height * .44 - 100));
  const unit = Math.min(width, height * 1.3);
  const scale = Math.min(halfWidth / Math.max(.5, (high[0] - low[0]) / 2), halfHeight / Math.max(.5, (high[1] - low[1]) / 2), unit / 48);
  return { target, span: unit / scale, scale };
}
