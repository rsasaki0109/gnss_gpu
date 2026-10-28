// Run from repository root; optional first argument is JSON {location, seed}.
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {sensitivityStudy} from '../docs/assets/forest/sensitivity.mjs';
const asset=JSON.parse(readFileSync('docs/assets/plateau/manifest.json')).scene;
const bytes=readFileSync(`docs/assets/plateau/${asset.file}`),sha256=createHash('sha256').update(bytes).digest('hex');
if(bytes.length!==asset.bytes||sha256!==asset.sha256)throw Error('Orbit asset checksum mismatch');
const result=sensitivityStudy(JSON.parse(gunzipSync(bytes)),process.argv[2]?JSON.parse(process.argv[2]):{});
console.log(JSON.stringify({...result,source:{file:asset.file,sha256}},null,2));
