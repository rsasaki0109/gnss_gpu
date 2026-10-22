// Statistics describe completed sampled poses, never missing-as-zero accuracy.
const quantile=(sorted,p)=>sorted.length?sorted[Math.max(0,Math.ceil(p*sorted.length)-1)]:null;
function distribution(values) {
  const sorted=[...values].sort((a,b)=>a-b);
  const n=sorted.length;
  const median=n?(n%2?sorted[(n-1)/2]:(sorted[n/2-1]+sorted[n/2])/2):null;
  return {median,p95:quantile(sorted,.95),maximum:sorted.at(-1)??null};
}
export function summarizeRoute(rows) {
  const values=[[],[]],deltas=[];
  for(const row of rows){
    if(row.conditions?.length!==2)throw new RangeError("Two route conditions required");
    const pair=row.conditions.map(c=>c.bias);
    pair.forEach((bias,i)=>{if(bias!==null){if(!Number.isFinite(bias)||bias<0)throw new RangeError("Invalid geometric bias");values[i].push(bias);}});
    if(pair.every(v=>v!==null))deltas.push(pair[0]-pair[1]);
  }
  return {sampledPoses:rows.length,conditions:values.map(v=>({available:v.length,unavailable:rows.length-v.length,availableFraction:rows.length?v.length/rows.length:null,...distribution(v)})),paired:{count:deltas.length,aLower:deltas.filter(d=>d<0).length,equal:deltas.filter(d=>d===0).length,bLower:deltas.filter(d=>d>0).length,...distribution(deltas)},method:"Median averages the two central values for even counts; P95 uses nearest rank over available sampled poses, not time weighting. Paired quantiles use per-pose A-minus-B only where both fits exist; they are not differences of marginal quantiles. Geometric bias is not measured positioning accuracy."};
}

// RFC 7946: longitude, latitude, optional WGS84 ellipsoidal height in metres.
// Points are reference receivers, not estimated/fitted positions or interpolated lines.
export function routeGeoJSON(rows,metadata={}) {
  const features=[];
  for(const row of rows)for(const [index,condition]of row.conditions.entries()){
    const geo=condition.geodetic, valid=Number.isFinite(geo?.longitude)&&Math.abs(geo.longitude)<=180&&Number.isFinite(geo?.latitude)&&Math.abs(geo.latitude)<=90;
    const coordinates=valid?[geo.longitude,geo.latitude]:null;
    if(coordinates&&Number.isFinite(geo.height))coordinates.push(geo.height);
    features.push({type:"Feature",id:`${row.epoch}-${index===0?"A":"B"}`,geometry:coordinates?{type:"Point",coordinates}:null,properties:{epoch:row.epoch,gps_week:row.week,gps_tow_s:row.tow,condition:index===0?"A":"B",systems:condition.systems,antenna_offset_m:condition.antenna,mask_deg:condition.mask,horizontal_geometric_bias_m:condition.bias,fit_available:condition.bias!==null,counts:condition.counts,coordinate_role:"reference receiver",coordinate_status:valid?"available":"unavailable"}});
  }
  return {type:"FeatureCollection",metadata:{...metadata,coordinateFrame:"Nominal WGS84 alignment; no epoch-dependent frame or antenna lever-arm correction",heightDatum:"WGS84 ellipsoid, metres; virtual antenna offset applied",geometryRole:"Reference receiver at each evaluated original pose; no fitted positions or interpolation"},summary:summarizeRoute(rows),features};
}
