const test = require("node:test");
const assert = require("node:assert/strict");
const { gunzipSync } = require("node:zlib");
const fs = require("node:fs");
const path = require("node:path");
const geometry = import("../../docs/assets/simulation/map-geometry.mjs");
const cacheModule = import("../../docs/assets/simulation/tile-cache.mjs");
const scene = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname,"../../docs/assets/plateau/odaiba-scene.json.gz"))));
const close = (a, b, epsilon = 1e-9) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);
const tile = key => ({ key });
const flush = () => new Promise(resolve => setImmediate(resolve));

test("Web Mercator matches the equator, world limits and independent log-tangent coordinates", async () => {
  const { lonLatToTile, tileToLonLat } = await geometry;
  assert.deepEqual(lonLatToTile(0,0,0), [.5,.5]);
  close(tileToLonLat(0,0,0)[1], 85.0511287798066);
  for (const [lon,lat] of [[139.775924,35.618026],[-73.9,40.7],[170,-45],[0,0]]) {
    for (const zoom of [0,14,17,19]) {
      const xy = lonLatToTile(lon,lat,zoom), n = 2 ** zoom;
      close(xy[1], n * (1 - Math.log(Math.tan(Math.PI/4+lat*Math.PI/360))/Math.PI)/2, 1e-8);
      const restored = tileToLonLat(...xy,zoom);
      close(restored[0],lon); close(restored[1],lat);
    }
  }
  close(tileToLonLat(...lonLatToTile(0,90,0),0)[1],85.0511287798066);
});

test("map coordinates align with the recorded WGS84 ECEF origin and an analytic equatorial ENU frame", async () => {
  const { localToLonLat, lonLatToLocal } = await geometry;
  const ll = localToLonLat([0,0,0],scene);
  close(ll[0],139.78712595,1e-8); close(ll[1],35.62931853,1e-8);
  const p = lonLatToLocal(139.78712595,35.62931853,44.6995,scene);
  assert.ok(Math.hypot(...p) < .0002);
  const equator = {origin:[6378137,0,0],rotation:[[0,1,0],[0,0,1],[1,0,0]]};
  assert.deepEqual(lonLatToLocal(0,0,12,equator), [0,0,12]);
  assert.deepEqual(localToLonLat([0,0,12],equator), [0,0]);
  close(lonLatToLocal(90,0,0,equator)[0],6378137);
  close(lonLatToLocal(0,90,0,equator)[1],6356752.314245179,1e-6);
});

test("inverse camera footprint projects to all four canvas corners", async () => {
  const { groundFootprint } = await geometry;
  for (const yaw of [-1.2,0,.8]) for (const pitch of [.15,.82,1.48]) {
    const camera = {yaw,pitch,target:[-400,500,12]}, width=940,height=565,scale=1.23;
    const corners = groundFootprint(camera,width,height,scale,-8);
    for (let i=0;i<4;i++) {
      const [x,y,z] = corners[i].map((v,j)=>v-camera.target[j]);
      const px = width/2+(Math.cos(yaw)*x-Math.sin(yaw)*y)*scale;
      const py = height*.56-(-Math.sin(yaw)*Math.sin(pitch)*x-Math.cos(yaw)*Math.sin(pitch)*y+Math.cos(pitch)*z)*scale;
      close(px,[0,width,width,0][i],1e-8); close(py,[0,0,height,height][i],1e-8);
    }
  }
});

test("visible map tile planning covers the viewport and stays bounded at oblique and overview scales", async () => {
  const { groundFootprint,visibleTiles,localToLonLat,lonLatToTile } = await geometry;
  for (const scale of [.08,.5,1,5]) for (const pitch of [.15,.82,1.48]) {
    const footprint = groundFootprint({yaw:-.63,pitch,target:[-1000,-1250,0]},940,565,scale,-8);
    const tiles = visibleTiles(footprint,scene,scale);
    assert.ok(tiles.length>=1 && tiles.length<=32);
    assert.equal(new Set(tiles.map(t=>t.key)).size,tiles.length);
    for (const point of footprint) {
      const xy = lonLatToTile(...localToLonLat(point,scene),tiles[0].z);
      assert.ok(tiles.some(t=>xy[0]>=t.x-1e-8&&xy[0]<=t.x+1+1e-8&&xy[1]>=t.y-1e-8&&xy[1]<=t.y+1+1e-8));
    }
  }
  assert.throws(()=>visibleTiles([],scene,0),RangeError);
});

test("tessellated map buffers preserve shared tile edges and north-to-south texture orientation", async () => {
  const { tileMesh,lonLatToTile } = await geometry;
  const [tx,ty] = lonLatToTile(139.78,35.62,17);
  const a = {x:Math.floor(tx),y:Math.floor(ty),z:17};
  const left = tileMesh(a,scene,36.3), right = tileMesh({...a,x:a.x+1},scene,36.3);
  assert.equal(left.length,8*8*6*5); assert.ok([...left].every(Number.isFinite));
  const edge = (data,u) => {
    const vertices = new Map();
    for(let i=0;i<data.length;i+=5)if(data[i+3]===u)vertices.set(data[i+4],Array.from(data.slice(i,i+3)));
    return vertices;
  };
  assert.deepEqual(edge(left,1),edge(right,0));
  assert.ok(edge(left,0).get(0)[1] > edge(left,0).get(1)[1]);
});

test("tile requests obey concurrency and reuse the current view without duplicate loads", async () => {
  const {VisibleTileCache}=await cacheModule;
  const calls=[],pending=new Map();
  const cache=new VisibleTileCache(t=>{calls.push(t.key);return new Promise(resolve=>pending.set(t.key,resolve));},()=>{},{concurrency:2,capacity:4});
  cache.select(['a','b','c'].map(tile)); await flush();
  assert.deepEqual(calls,['a','b']); assert.equal(cache.active,2);
  cache.select(['a','b','c'].map(tile)); await flush(); assert.equal(calls.length,2);
  pending.get('a')({dispose(){}}); await flush(); assert.deepEqual(calls,['a','b','c']);
  pending.get('b')({dispose(){}}); pending.get('c')({dispose(){}}); await flush();
  assert.deepEqual(cache.stats(),{state:'ready',ready:3,failed:0,total:3});
  cache.select(['a','b','c'].map(tile)); await flush(); assert.equal(calls.length,3);
});

test("stale camera queues and disabled maps do not request unseen tiles", async () => {
  const {VisibleTileCache}=await cacheModule;
  const calls=[],pending=new Map();
  const cache=new VisibleTileCache(t=>{calls.push(t.key);return new Promise(resolve=>pending.set(t.key,resolve));},()=>{},{concurrency:2,capacity:4});
  cache.select(['a','b','c'].map(tile)); cache.select(['d','e'].map(tile)); await flush();
  assert.deepEqual(calls,['d','e']);
  cache.setEnabled(false); cache.select([tile('f')]);
  pending.get('d')({dispose(){}}); pending.get('e')({dispose(){}}); await flush();
  assert.deepEqual(calls,['d','e']); assert.equal(cache.stats().state,'off');
  cache.setEnabled(true); await flush(); assert.deepEqual(calls,['d','e','f']);
  pending.get('f')({dispose(){}}); await flush(); assert.equal(cache.stats().state,'ready');
});

test("LRU eviction releases GPU resources and keeps memory bounded", async () => {
  const {VisibleTileCache}=await cacheModule;
  const calls=[],disposed=[];
  const cache=new VisibleTileCache(async t=>{calls.push(t.key);return {dispose(){disposed.push(t.key);}};},()=>{},{concurrency:1,capacity:2});
  for(const key of ['a','b','c']){cache.select([tile(key)]);await flush();assert.ok(cache.entries.size<=2);}
  assert.deepEqual(disposed,['a']);
  cache.select([tile('b')]);await flush();assert.deepEqual(calls,['a','b','c']);
  cache.select([tile('d')]);await flush();assert.deepEqual(disposed,['a','c']);
  assert.equal(cache.entries.size,2);
});

test("tile outages remain local to the map and retry only after an explicit toggle", async () => {
  const {VisibleTileCache}=await cacheModule;
  let calls=0;
  const cache=new VisibleTileCache(async()=>{if(++calls===1)throw Error('offline');return {dispose(){}};},()=>{});
  cache.select([tile('a')]);await flush();assert.equal(cache.stats().state,'unavailable');
  cache.select([tile('a')]);await flush();assert.equal(calls,1);
  cache.setEnabled(false);cache.setEnabled(true);await flush();
  assert.equal(calls,2);assert.equal(cache.stats().state,'ready');
});
