// Run from the repository root; prints the same complete artifact as the UI.
import {readFileSync} from 'node:fs';
import {gunzipSync} from 'node:zlib';
import {createHash} from 'node:crypto';
import {runComparison} from '../docs/assets/forest/environment-comparison.mjs';
const asset=JSON.parse(readFileSync('docs/assets/plateau/manifest.json')).scene;
const bytes=readFileSync(`docs/assets/plateau/${asset.file}`),sha256=createHash('sha256').update(bytes).digest('hex');
if(bytes.length!==asset.bytes||sha256!==asset.sha256)throw Error('Orbit asset checksum mismatch');
const settings=process.argv[2]?JSON.parse(process.argv[2]):{};
const result=runComparison(JSON.parse(gunzipSync(bytes)),settings);
console.log(JSON.stringify({...result,source:{file:asset.file,sha256}},null,2));
