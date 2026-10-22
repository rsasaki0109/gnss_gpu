// Display-only RF assumptions. The positioning solver never consumes these values.
// Fresnel convention matches python/gnss_gpu/fresnel.py and ITU-R P.2040.
import { reflectionGeometry } from "./reflection-geometry.mjs";
import { reflectedArrival } from "./arrival-direction.mjs";
import { selectNavigation } from "./orbit.mjs?v=glonass-1";

export const MATERIALS = Object.freeze({concrete:[5.31,0.0326],glass:[6.27,0.0043],brick:[3.75,0.038],wood:[1.99,0.0047]});
export const DEFAULT_POWER = Object.freeze({material:"fixed",polarization:"rhcp",antenna:"isotropic"});
export function validatePower(options=DEFAULT_POWER) {
  const value = {...DEFAULT_POWER,...options};
  if (!(value.material==="fixed" || Object.hasOwn(MATERIALS,value.material)) || !["rhcp","rhcp_cross","unpolarized"].includes(value.polarization) || !["isotropic","zenith"].includes(value.antenna)) throw new RangeError("Invalid RF assumptions");
  return value;
}
const add=(a,b)=>[a[0]+b[0],a[1]+b[1]], sub=(a,b)=>[a[0]-b[0],a[1]-b[1]];
const scale=(a,k)=>[a[0]*k,a[1]*k];
const divide=(a,b)=>{const d=b[0]*b[0]+b[1]*b[1];return [(a[0]*b[0]+a[1]*b[1])/d,(a[1]*b[0]-a[0]*b[1])/d];};
const squared=a=>a[0]*a[0]+a[1]*a[1];

export function fresnelPower(angle, material, frequency=1575.42e6, polarization="rhcp") {
  if (!Number.isFinite(angle) || angle<0 || angle>Math.PI/2 || !Object.hasOwn(MATERIALS,material) || !Number.isFinite(frequency) || frequency<=0 || !["rhcp","rhcp_cross","unpolarized"].includes(polarization)) throw new RangeError("Invalid Fresnel input");
  const [eps,sigma]=MATERIALS[material], imaginary=-sigma/(2*Math.PI*frequency*8.8541878128e-12);
  const c=Math.cos(angle), a=eps-Math.sin(angle)**2, length=Math.hypot(a,imaginary);
  const q=[Math.sqrt((length+a)/2),-Math.sqrt(Math.max(0,(length-a)/2))];
  const te=divide(sub([c,0],q),add([c,0],q)), ec=scale([eps,imaginary],c);
  const tm=divide(sub(ec,q),add(ec,q));
  const power=polarization==="unpolarized" ? (squared(te)+squared(tm))/2 : squared(scale(polarization==="rhcp"?add(tm,te):sub(tm,te),0.5));
  return Math.max(0,Math.min(1,power));
}

export function signalMetrics(snapshot, scene, options=DEFAULT_POWER) {
  const model=validatePower(options);
  const channels = new Map(selectNavigation(scene.navigation.filter(r=>r.id[0]==="R"),snapshot.tow,snapshot.week).map(r=>[r.id,r.frequencyChannel]));
  return snapshot.satellites.map(signal=>{
    const extra=signal.status==="LOS" ? 0 : signal.status==="NLOS" ? signal.reflection.extra : null;
    let power=null, powerState=signal.status==="BLOCKED"?"no-path":"unavailable", frequency=signal.id[0]==="C" ? 1561.098e6 : signal.id[0]==="R" ? (Number.isInteger(channels.get(signal.id)) ? 1602e6+channels.get(signal.id)*0.5625e6 : null) : 1575.42e6;
    if (signal.status!=="BLOCKED") {
      let factor=1;
      if (signal.status==="NLOS") {
        if (model.material==="fixed") factor=0.25;
        else { const geometry=reflectionGeometry(snapshot,signal); factor=geometry && frequency ? fresnelPower(geometry.incidence*Math.PI/180,model.material,frequency,model.polarization) : null; }
      }
      if (model.antenna==="zenith") {
        const elevation=signal.status==="LOS" ? signal.elevation : reflectedArrival(scene.rotation,snapshot,signal)?.elevation;
        factor = factor===null || !Number.isFinite(elevation) ? null : factor*Math.max(0,Math.sin(elevation*Math.PI/180))**2;
      }
      if (factor!==null && factor>0) { power=20*Math.log10(20000000/(signal.range+extra))+10*Math.log10(factor);powerState="finite"; }
      // Exact zero power is separate from a missing geometric path; CSV retains it.
      if (factor===0) powerState="zero";
    }
    return {id:signal.id,status:signal.status,elevation:signal.elevation,extra,power,powerState,frequency};
  });
}
