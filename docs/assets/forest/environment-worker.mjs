import {runComparison} from './environment-comparison.mjs?v=locations-1';
self.onmessage=({data})=>{try{self.postMessage({result:runComparison(data.scene,data.settings,progress=>self.postMessage({progress}))});}catch(error){self.postMessage({error:error.message});}};
