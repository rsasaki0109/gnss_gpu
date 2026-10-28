import {sensitivityStudy} from './sensitivity.mjs?v=1';
self.onmessage=({data})=>{try{self.postMessage({result:sensitivityStudy(data.scene,data.settings,progress=>self.postMessage({progress}))});}catch(error){self.postMessage({error:error.message});}};
