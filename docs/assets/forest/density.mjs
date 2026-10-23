// Synthetic, uncalibrated density surrogates. Same mean mass in every ellipsoid.
export const DENSITY_PROFILES=Object.freeze({
  uniform:Object.freeze({label:"Uniform",formula:"rho = rho0",calibrated:false}),
  "radial-height-v1":Object.freeze({label:"Radial + height",formula:"rho = rho0 (1 - 0.65 |u|^2) (1 + 0.45 u_z) / 0.61",coordinates:"u = (point - crown center) / crown radii",normalization:"volume mean equals rho0; <|u|^2> = 3/5 and odd moments vanish",calibrated:false})
});
export function checkDensityProfile(profile){if(!Object.hasOwn(DENSITY_PROFILES,profile))throw new RangeError("Unknown crown density profile");}
export function crownDensity(crown,point,density=1,profile="uniform"){
  checkDensityProfile(profile);
  const u=point.map((v,i)=>(v-crown.center[i])/crown.radii[i]),r2=u.reduce((s,v)=>s+v*v,0);
  if(r2>1)return 0;
  return profile==="uniform"?density:density*(1-.65*r2)*(1+.45*u[2])/.61;
}
// Closed-form integral of a cubic along an already clipped, unit-speed chord.
// Midpoint coordinates avoid subtracting two large antiderivative values.
export function crownIntegral(origin,direction,crown,span,density,profile="uniform"){
  checkDensityProfile(profile);
  const length=span[1]-span[0];
  if(profile==="uniform")return length*density;
  const mid=span[0]+length/2,half=length/2;
  const q=origin.map((v,i)=>(v+mid*direction[i]-crown.center[i])/crown.radii[i]),d=direction.map((v,i)=>half*v/crown.radii[i]);
  const q2=q.reduce((s,v)=>s+v*v,0),d2=d.reduce((s,v)=>s+v*v,0),qd=q.reduce((s,v,i)=>s+v*d[i],0);
  return length*density*((1-.65*q2)*(1+.45*q[2])-(.65*d2*(1+.45*q[2])+1.3*qd*.45*d[2])/3)/.61;
}
