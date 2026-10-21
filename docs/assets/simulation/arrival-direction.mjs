// Display-only direction toward the existing shortest reflection point.
// A source bearing looks from the receiver toward the reflector, opposite to
// the propagation velocity of the final ray leg. No new paths are traced.
import { geodetic, enuRotation } from "./orbit.mjs";

const vector = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
const dot = (a,b) => a.reduce((sum,value,i) => sum+value*b[i],0);

export function reflectedArrival(sceneRotation, snapshot, signal) {
  if (!signal?.reflection || signal.status === "BLOCKED" || !vector(signal.reflection.point) || !vector(snapshot?.receiver) || !vector(snapshot.receiverECEF) || sceneRotation?.length !== 3 || !sceneRotation.every(vector) || !Number.isFinite(signal.azimuth) || !Number.isFinite(signal.elevation)) return null;
  const displacement = signal.reflection.point.map((value,i) => value-snapshot.receiver[i]);
  // Transpose the fixed scene's ECEF-to-ENU rotation, then use the current
  // receiver's geodetic ENU frame. The scene origin is not the receiver.
  const ecef = [0,1,2].map(i => sceneRotation.reduce((sum,axis,j) => sum+axis[i]*displacement[j],0));
  const location = geodetic(snapshot.receiverECEF);
  const enu = enuRotation(location.lat,location.lon).map(axis => dot(axis,ecef));
  const length = Math.hypot(...enu), horizontal = Math.hypot(enu[0],enu[1]);
  if (!(length > 0) || !Number.isFinite(length)) return null;
  const azimuth = horizontal === 0 ? null : (Math.atan2(enu[0],enu[1])*180/Math.PI+360)%360;
  const elevation = Math.atan2(enu[2],horizontal)*180/Math.PI;
  const az = signal.azimuth*Math.PI/180, el = signal.elevation*Math.PI/180;
  const satellite = [Math.cos(el)*Math.sin(az),Math.cos(el)*Math.cos(az),Math.sin(el)], arrival = enu.map(value=>value/length);
  const cross = [satellite[1]*arrival[2]-satellite[2]*arrival[1],satellite[2]*arrival[0]-satellite[0]*arrival[2],satellite[0]*arrival[1]-satellite[1]*arrival[0]];
  const separation = Math.atan2(Math.hypot(...cross),dot(satellite,arrival))*180/Math.PI;
  // Keep negative elevation and directions below the satellite mask intact.
  return {azimuth,elevation,separation};
}
