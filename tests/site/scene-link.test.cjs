const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const {gunzipSync} = require("node:zlib");
const links = import("../../docs/assets/simulation/scene-link.mjs");
const manifest = JSON.parse(fs.readFileSync(path.join(__dirname,"../../docs/assets/plateau/manifest.json")));
const scene = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname,"../../docs/assets/plateau/odaiba-scene.json.gz"))));
const assets = {mesh:manifest.mesh.sha256,data:manifest.scene.sha256,epochs:scene.track.length,satellites:[...new Set(scene.navigation.map(record=>record.id))]};

test("scene links round-trip exact original epochs, receiver inputs and full binary64 camera values", async () => {
  const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await links;
  for(const epoch of [0,8000,12409])for(const view of ["receiver","overview","reflection","blockage"]) {
    const state={...DEFAULT_SCENE,epoch,antenna:7.5,mask:5,satellite:"G12",view,yaw:-.6300000000000001,pitch:view==="blockage"?-1.3:.8200000000000001,zoom:1.2500000000000002,basemap:false};
    const url=new URL(writeSceneLink("https://example.org/gnss_gpu/simulation.html",state,assets));
    const restored=readSceneLink(url.hash,assets);
    assert.deepEqual(restored,state);
    assert.deepEqual(scene.track[restored.epoch],scene.track[epoch]);
    assert.equal(url.search,"");
  }
  const zero={...DEFAULT_SCENE,yaw:-0};
  assert.ok(Object.is(readSceneLink(new URL(writeSceneLink("http://localhost/simulation.html",zero,assets)).hash,assets).yaw,-0));
});

test("links retain the Pages project path and strip unrelated query values and URL credentials", async () => {
  const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await links;
  const url=new URL(writeSceneLink("https://user:password@example.org/gnss_gpu/simulation.html?unrelated=value#stage",DEFAULT_SCENE,assets));
  assert.equal(url.pathname,"/gnss_gpu/simulation.html"); assert.equal(url.username,""); assert.equal(url.password,""); assert.equal(url.search,"");
  assert.equal(new URLSearchParams(url.hash.slice(1)).get("mesh"),assets.mesh);
  assert.equal(new URLSearchParams(url.hash.slice(1)).get("data"),assets.data);
  assert.deepEqual(readSceneLink(url.hash,assets),DEFAULT_SCENE);
  assert.throws(()=>writeSceneLink("javascript:alert(1)",DEFAULT_SCENE,assets),/HTTP/);
  for(const hash of ["","#stage","#scene","#other=1"])assert.equal(readSceneLink(hash,assets),null);
});

test("invalid, non-finite and out-of-range values are rejected instead of clamped or partly applied", async () => {
  const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await links;
  const base=new URL(writeSceneLink("https://example.org/simulation.html",{...DEFAULT_SCENE,satellite:"G12"},assets));
  for(const [key,value] of [["epoch","-1"],["epoch","12410"],["epoch","1.5"],["height","20.5"],["height","0.25"],["height","0x10"],["mask","31"],["mask","NaN"],["yaw","Infinity"],["yaw"," 1"],["zoom","0.2"],["pitch","0"],["pitch","-0.82"],["view","__proto__"],["sat","G99"],["map","true"],["reflections","yes"]]) {
    const fields=new URLSearchParams(base.hash.slice(1)); fields.set(key,value);
    assert.throws(()=>readSceneLink(`#${fields}`,assets),undefined,`${key}=${value}`);
  }
  const original=structuredClone(DEFAULT_SCENE);
  assert.deepEqual(DEFAULT_SCENE,original);
});

test("version, dataset fingerprints, duplicate fields and incomplete links fail explicitly", async () => {
  const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await links;
  const base=new URL(writeSceneLink("https://example.org/simulation.html",DEFAULT_SCENE,assets));
  for(const [key,value,pattern] of [["scene","2",/version/],["mesh","0".repeat(64),/different dataset/],["data","0".repeat(64),/different dataset/]]) {
    const fields=new URLSearchParams(base.hash.slice(1));fields.set(key,value);
    assert.throws(()=>readSceneLink(`#${fields}`,assets),pattern);
  }
  assert.throws(()=>readSceneLink(`${base.hash}&epoch=1`,assets),/duplicate/);
  assert.throws(()=>readSceneLink(`${base.hash}&extra=1`,assets),/unknown/);
  const missing=new URLSearchParams(base.hash.slice(1));missing.delete("mask");
  assert.throws(()=>readSceneLink(`#${missing}`,assets),/missing/);
});

test("view modes obey the same pitch and reflection-display contract as the interactive controls", async () => {
  const {DEFAULT_SCENE,readSceneLink,writeSceneLink}=await links;
  const state={...DEFAULT_SCENE,satellite:"G12",view:"blockage",pitch:-1.48,zoom:6,reflections:false};
  assert.deepEqual(readSceneLink(new URL(writeSceneLink("https://example.org/simulation.html",state,assets)).hash,assets),state);
  const bad={...state,view:"reflection"};
  assert.throws(()=>readSceneLink(new URL(writeSceneLink("https://example.org/simulation.html",bad,assets)).hash,assets),/inconsistent/);
  assert.throws(()=>readSceneLink(new URL(writeSceneLink("https://example.org/simulation.html",{...state,satellite:""},assets)).hash,assets),/view/);
});
