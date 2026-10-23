// Published RET extinction parameters, not a full ITU-R P.833 implementation.
// tau = sigma_t * distance is a POWER optical depth: dB = 10/ln(10) * tau.
const DB_PER_NEPER=10/Math.LN10;
export const EXTINCTION_REFERENCES=Object.freeze({
  "silver-maple":Object.freeze({species:"Acer saccharinum",anchorGHz:1.3,inLeafPerM:.241,outOfLeafPerM:.139}),
  "korean-pine":Object.freeze({species:"Pinus koraiensis",anchorGHz:1.5,inLeafPerM:.215,outOfLeafPerM:null}),
  "himalayan-cedar":Object.freeze({species:"Cedrus deodara",anchorGHz:1.5,inLeafPerM:.271,outOfLeafPerM:null})
});
export const MATERIAL_PRESETS=Object.freeze({"maple-pine":"korean-pine","maple-cedar":"himalayan-cedar"});
export const EXTINCTION_SOURCE=Object.freeze({
  title:"ITU-R P.833-10 (2021), Table 8; optical density definition in section 3.2.1.4",
  url:"https://www.itu.int/rec/R-REC-P.833-10-202109-I/en",
  table:"fitted combined absorption and scatter coefficient sigma_t, m^-1",
  frequencyPolicy:"Fixed published anchors: maple 1.3 GHz, conifers 1.5 GHz. No interpolation, carrier assignment or GNSS-band frequency scaling.",
  scope:"Ballistic power extinction only, exp(-sigma_t * density integral); not the full RET/slant-tree model, diffuse gain, diffraction, coherent RF, C/N0 or receiver tracking.",
  transfer:"Reference analogues for synthetic growth forms. Relative density/profile and bare-crown fine-branch volume are synthetic mappings, not site/LAI calibration. RHCP GNSS transfer and reflected-leg use are unvalidated; no positioning-accuracy claim.",
  seasonal:"Leaf-off maple retains the published out-of-leaf extinction; evergreen conifers retain their in-leaf reference. Missing conifer out-of-leaf data are not invented.",
  calibratedToScene:false
});
export function checkAttenuation(model="linear-v1",preset="maple-pine",foliage=1){
  if(!["linear-v1","itu-extinction-v1"].includes(model)||!Object.hasOwn(MATERIAL_PRESETS,preset))throw new RangeError("Unknown attenuation model or material preset");
  if(model==="itu-extinction-v1"&&foliage!==1)throw new RangeError("Reference extinction requires unit foliage; use linear mode for multiplier studies");
}
export function extinctionMaterial(tree,preset="maple-pine"){
  checkAttenuation("itu-extinction-v1",preset);
  if(!["broadleaf","conifer"].includes(tree.species)||![0,1].includes(tree.leafRetention))throw new RangeError("Reference extinction needs a supported tree type and binary leaf retention");
  const reference=tree.species==="broadleaf"?"silver-maple":MATERIAL_PRESETS[preset],data=EXTINCTION_REFERENCES[reference],leafState=tree.leafRetention?"in-leaf":"out-of-leaf",sigmaPerM=tree.leafRetention?data.inLeafPerM:data.outOfLeafPerM;
  if(sigmaPerM===null)throw new RangeError("No out-of-leaf extinction reference for this evergreen");
  return {reference,leafState,anchorGHz:data.anchorGHz,sigmaPerM,dbPerM:sigmaPerM*DB_PER_NEPER};
}
export function extinctionLoss(crowns,profile="uniform"){
  return crowns.reduce((sum,c)=>sum+(profile==="uniform"?(c.exit-c.enter)*c.density:c.densityIntegral)*c.material.dbPerM,0);
}
export function attenuationDescription(model="linear-v1",preset="maple-pine"){
  checkAttenuation(model,preset);
  return model==="linear-v1"?{model,calibratedToScene:false,scope:"Illustrative specific loss times leaf-density integral; no residual crown loss after leaf-off."}:{model,preset,references:EXTINCTION_REFERENCES,source:EXTINCTION_SOURCE};
}
