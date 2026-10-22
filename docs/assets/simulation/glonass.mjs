// GLONASS FDMA broadcast state integration in binary64 (RTKLIB deq/glorbit).
// The 2018 records use PZ-90.11; nominal WGS84 alignment, no frame transform.
const WEEK = 604800, OMEGA = 7.292115e-5;
const vector = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
function valid(record) {
  return vector(record.position) && vector(record.velocity) && vector(record.acceleration) &&
    vector(record.clock) && Number.isInteger(record.week) && Number.isFinite(record.toc) &&
    Number.isInteger(record.utcOffset) && record.utcOffset >= 0 && record.utcOffset <= 64 &&
    Math.hypot(...record.position) > 1e6 && Math.hypot(...record.position) < 1e8;
}
export function glonassElapsed(record, time, week) {
  return (week - record.week) * WEEK + time - record.utcOffset - record.toc % WEEK;
}
export function usableGlonass(record, time, week) {
  // Broadcast interpolation interval is +/-15 minutes; future reference epochs
  // are allowed only when their message was already transmitted.
  return valid(record) && record.health === 0 && Number.isFinite(record.transmission) &&
    Number.isInteger(week) && Number.isFinite(time) && Math.abs(glonassElapsed(record,time,week)) <= 900 &&
    record.transmission - week * WEEK + record.utcOffset <= time + 1e-6;
}
function derivative(x, acceleration) {
  const r2 = x[0]*x[0] + x[1]*x[1] + x[2]*x[2], r3 = r2*Math.sqrt(r2);
  const j2 = 1.5*1.0826257e-3*3.9860044e14*6378136**2/r2/r3;
  const gravity = -3.9860044e14/r3 - j2*(1 - 5*x[2]*x[2]/r2);
  return [x[3],x[4],x[5],
    (gravity+OMEGA*OMEGA)*x[0]+2*OMEGA*x[4]+acceleration[0],
    (gravity+OMEGA*OMEGA)*x[1]-2*OMEGA*x[3]+acceleration[1],
    (gravity-2*j2)*x[2]+acceleration[2]];
}
export function glonassOrbit(record, time, week) {
  if (!valid(record) || !Number.isFinite(time) || !Number.isInteger(week)) throw new RangeError("Invalid GLONASS orbit");
  const elapsed = glonassElapsed(record,time,week);
  if (Math.abs(elapsed) > 1800) throw new RangeError("GLONASS orbit is outside its propagation interval");
  const x = [...record.position,...record.velocity];
  let remaining = elapsed;
  while (Math.abs(remaining) > 1e-9) {
    const step = Math.sign(remaining)*Math.min(60,Math.abs(remaining));
    const k1 = derivative(x,record.acceleration);
    const k2 = derivative(x.map((v,i)=>v+k1[i]*(step/2)),record.acceleration);
    const k3 = derivative(x.map((v,i)=>v+k2[i]*(step/2)),record.acceleration);
    const k4 = derivative(x.map((v,i)=>v+k3[i]*step),record.acceleration);
    for (let i=0;i<6;i++) x[i] += (k1[i]+2*k2[i]+2*k3[i]+k4[i])*(step/6);
    remaining -= step;
  }
  // RINEX stores -TauN, so no second sign reversal or relativistic term.
  return {position:x.slice(0,3),clock:record.clock[0]+record.clock[1]*elapsed};
}
