import {generateObservations,solvePosition} from "./positioning.mjs?v=position-1";
self.onmessage=({data})=>{
  try{
    const {forest,receiver,signals,seed,sigma,mapBias,context}=data;
    const generated=generateObservations(forest,receiver,signals,{seed,sigma});
    // Solver searches horizontal coordinates only; prior z is unused.
    const prior=[receiver[0]+7.37,receiver[1]-5.13,0];
    const solution=solvePosition(generated.observations,prior,forest,{mapBias,onProgress:progress=>self.postMessage({progress})});
    const results=solution.results.map(r=>({...r,horizontalErrorM:r.receiverENU?Math.hypot(r.receiverENU[0]-receiver[0],r.receiverENU[1]-receiver[1]):null,clockErrorM:r.clock===undefined?null:r.clock-generated.truth.clock}));
    self.postMessage({result:{schema:"synthetic-forest-positioning-v1",simulationOnly:true,context,scene:forest,...generated,prior,results,search:solution.search,
      assumptions:{measurement:"artificial single-arrival range plus common clock and Gaussian noise; not coherent multipath/code tracking",sources:"GPS only; direct-flight-time broadcast endpoints held fixed for every candidate",height:"terrain plus known 1.6 m receiver height; 2D horizontal search",map:"same synthetic geometry family for generation and inference; mapBias shifts solver terrain, wood, crowns and height aiding",initialization:"synthetic prior offset [+7.37,-5.13] m from truth; bounded local search, no global guarantee",pathWeights:"equal normalized feasible paths; no RF or truth association weights",limitations:"matched-map inverse crime; no real-data accuracy claim; candidate grid may miss other modes"}}});
  }catch(error){self.postMessage({error:error.message});}
};
