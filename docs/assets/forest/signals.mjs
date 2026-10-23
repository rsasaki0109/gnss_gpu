import {selectNavigation,receivedSatellite,sub,mul,norm,dot,geodetic,enuRotation} from "../simulation/orbit.mjs?v=glonass-1";
// Shared by the live view and route worker; identical received-state convention.
export function forestSignals(scene,receiver,seconds,systems="GECJR"){
  const ecef=scene.origin.map((v,i)=>v+scene.rotation.reduce((sum,axis,j)=>sum+axis[i]*receiver[j],0)),llh=geodetic(ecef),orientation=enuRotation(llh.lat,llh.lon),tow=(scene.startTow??scene.track[0][0])+seconds,signals=[];
  for(const record of selectNavigation(scene.navigation.filter(r=>systems.includes(r.id[0])),tow,scene.week)){
    const state=receivedSatellite(record,tow,ecef,scene.week),delta=sub(state.position,ecef),direction=mul(delta,1/norm(delta));
    const elevation=Math.asin(Math.max(-1,Math.min(1,dot(orientation[2],direction))))*180/Math.PI;
    if(elevation<10)continue;
    signals.push({id:record.id,azimuth:(Math.atan2(dot(orientation[0],direction),dot(orientation[1],direction))*180/Math.PI+360)%360,elevation,sourceECEF:state.position,sourceENU:scene.rotation.map(axis=>dot(axis,sub(state.position,scene.origin))),directionENU:scene.rotation.map(axis=>dot(axis,direction))});
  }
  return signals;
}
