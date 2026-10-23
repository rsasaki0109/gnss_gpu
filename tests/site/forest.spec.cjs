const {test,expect}=require("@playwright/test");
const fs=require("node:fs");
// Deterministic offscreen WebGL: the host's default headless GPU context can be lost.
// No software-renderer flags are added to the website or normal browser launches.
test.use({launchOptions:{args:["--use-angle=swiftshader","--enable-unsafe-swiftshader"]}});

test("controlled positioning captures a fixed epoch, runs blindly and exports every method",async({page},testInfo)=>{
  test.setTimeout(180000);
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready");
  await page.locator("#position-experiment summary").click();
  await page.locator("#position-run").click();await page.locator("#position-cancel").click();
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","cancelled");
  await expect(page.locator("#position-export")).toBeDisabled();
  await page.locator("#forest-play").click();await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","true");
  await page.locator("#position-run").click();await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","false");
  const captured=Number(await page.locator("#forest-lab").getAttribute("data-seconds"));
  // Subsequent scene edits must not mutate the worker's captured inputs/results.
  await page.locator("#forest-count").selectOption("32");
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","ready",{timeout:120000});
  await expect(page.locator("#position-results tr")).toHaveCount(3);
  const pending=page.waitForEvent("download");await page.locator("#position-export").click();const download=await pending;
  const data=JSON.parse(fs.readFileSync(await download.path(),"utf8"));
  expect(data.schema).toBe("synthetic-forest-positioning-v1");expect(data.scene.count).toBe(128);expect(data.context.elapsedSeconds).toBe(captured);
  expect(data.context.source.sha256).toMatch(/^[a-f0-9]{64}$/);expect(data.search.globalSpatialOptimum).toBe(false);expect(data.search.clockGridAdditiveNllBound).toBe(1/32);
  expect(data.observations.length).toBeGreaterThanOrEqual(3);expect(data.observations.every(o=>o.id.startsWith("G")&&!('index' in o))).toBe(true);
  expect(data.results.map(r=>r.method)).toEqual(["straight","robust","paths"]);
  expect(data.results.every(r=>r.status==="solution"&&Number.isFinite(r.horizontalErrorM))).toBe(true);
  expect(data.assumptions.limitations).toContain("inverse crime");
  console.log(JSON.stringify({project:testInfo.project.name,results:data.results.map(r=>({method:r.method,error:r.horizontalErrorM,boundary:r.boundary})),observations:data.observations.length}));
  await page.screenshot({path:testInfo.outputPath("positioning.png"),fullPage:true});
  await page.locator("#forest-systems").selectOption("E");await page.locator("#position-run").click();
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","error");await expect(page.locator("#position-export")).toBeDisabled();
  expect(errors).toEqual([]);
});

test("forest ray geometry, live sky, assumptions and exported snapshot remain consistent",async({page},testInfo)=>{
  const errors=[],requests=[];page.on("pageerror",e=>errors.push(e.message));page.on("request",r=>requests.push(r.url()));
  await page.goto("forest.html");const root=page.locator("#forest-lab");
  await expect(root).toHaveAttribute("data-state","ready",{timeout:45000});await expect(root).toHaveAttribute("data-count","128");
  const drawing=await page.locator("#forest-scene").evaluate(canvas=>{
    document.getElementById("forest-reset-view").click();const gl=canvas.getContext("webgl2"),size=120,pixels=new Uint8Array(size*size*4);
    gl.readPixels(Math.floor(canvas.width/2-size/2),Math.floor(canvas.height/2-size/2),size,size,gl.RGBA,gl.UNSIGNED_BYTE,pixels);
    const colors=new Set();for(let i=0;i<pixels.length;i+=16)colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
    return {lost:gl.isContextLost(),error:gl.getError(),colors:colors.size};
  });expect(drawing.lost).toBe(false);expect(drawing.error).toBe(0);expect(drawing.colors).toBeGreaterThan(6);
  await expect(page.locator('#forest-signals button[data-channel^="R"]')).not.toHaveCount(0);
  const g12=page.locator('#forest-sky [data-satellite="G12"] circle'),before=await g12.getAttribute("cx");
  await page.locator("#forest-progress").press("End");await expect(root).toHaveAttribute("data-seconds","1200");
  await expect(g12).not.toHaveAttribute("cx",before);
  await page.locator("#foliage").press("Home");
  await expect(page.locator("#forest-signals tr td:last-child")).toHaveText(Array(Number(await root.getAttribute("data-satellites"))).fill("0.0 dB"));
  await page.locator("#forest-count").selectOption("96");await expect(root).toHaveAttribute("data-count","96");
  const downloadPromise=page.waitForEvent("download");await page.locator("#forest-export").click();const download=await downloadPromise;
  const data=JSON.parse(fs.readFileSync(await download.path(),"utf8"));
  expect(data.schema).toBe("synthetic-forest-snapshot-v2");expect(data.simulationOnly).toBe(true);expect(data.options.foliage).toBe(0);expect(data.scene.trees).toHaveLength(96);
  expect(data.assumptions.reflections).toContain("single bounce");expect(data.source.sha256).toMatch(/^[a-f0-9]{64}$/);expect(data.signals.every(s=>s.trace.canopyLoss===0)).toBe(true);
  expect(data.signals.filter(s=>s.trace.wood.length||s.trace.ground!==null).every(s=>s.trace.directRelativeDb===null)).toBe(true);
  await page.locator("#forest-progress").press("Home");await page.locator("#forest-play").click();
  await expect.poll(async()=>Number(await root.getAttribute("data-seconds"))).toBeGreaterThan(1);
  await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","true");await page.locator("#forest-play").click();
  await page.locator("#forest-reset-view").click();await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("forest-bare-3d.png")});
  await page.locator("#foliage").press("End");await page.locator("#forest-count").selectOption("64");
  await page.locator("#forest-progress").focus();await page.locator("#forest-progress").press("ArrowRight");
  await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("forest-3d.png")});
  await page.screenshot({path:testInfo.outputPath("forest-page.png"),fullPage:true});
  const layout=await page.evaluate(()=>({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));expect(layout.scroll).toBeLessThanOrEqual(layout.client+2);
  expect(requests.some(url=>url.includes("odaiba-mesh"))).toBe(false);expect(requests.some(url=>url.includes("tile.openstreetmap.org"))).toBe(false);expect(errors).toEqual([]);
  await expect(root).toHaveAttribute("data-state","ready");
});

test("selected ground path renders, changes assumed power and exports coherent full segments",async({page},testInfo)=>{
  await page.goto("forest.html");const root=page.locator("#forest-lab"),readout=page.locator("#forest-ground");
  await expect(root).toHaveAttribute("data-state","ready",{timeout:45000});
  const ids=await page.locator("[data-channel]").evaluateAll(nodes=>nodes.map(n=>n.dataset.channel));
  for(const id of ids){await page.locator(`[data-channel="${id}"]`).click();if(await readout.getAttribute("data-state")==="ready")break;}
  await expect(readout).toHaveAttribute("data-state","ready");await expect(page.locator("#forest-scene")).toHaveAttribute("data-reflection","true");
  const extra=Number(await readout.getAttribute("data-extra")),delay=Number(await readout.getAttribute("data-delay-ns")),power=Number(await readout.getAttribute("data-power-db"));
  expect(extra).toBeGreaterThan(0);expect(Math.abs(delay-extra/299792458*1e9)).toBeLessThan(1e-10);
  await page.locator("#bounce-loss").press("End");expect(Number(await readout.getAttribute("data-extra"))).toBe(extra);expect(Math.abs(Number(await readout.getAttribute("data-power-db"))-power+24)).toBeLessThan(1e-10);
  const time=await root.getAttribute("data-seconds");await page.locator("#forest-trail-view").click();await page.locator("#forest-focus-reflection").click();await expect(page.locator("#forest-scene")).toHaveAttribute("data-view","overview");expect(await root.getAttribute("data-seconds")).toBe(time);
  await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("ground-reflection.png")});
  const pending=page.waitForEvent("download");await page.locator("#forest-export").click();const data=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8")),packet=data.selectedReflection;
  expect(data.view.follow).toBe(false);expect(data.view.pitch).toBeGreaterThan(.5);expect(packet.model).toBe("finite-terrain-reflection-v1");expect(packet.options.completeSegments).toBe(true);expect(packet.options.bounceLoss).toBe(30);expect(packet.tow).toBe(data.tow);
  expect(packet.satelliteStateConvention).toContain("no reflected-flight-time orbit iteration");
  const signal=data.signals.find(s=>s.id===packet.satellite);
  for(const p of packet.paths){
    expect(p.reason).toBe(null);expect(p.relativePowerDb).not.toBe(null);expect(p.sourceRange).toBeGreaterThan(1e6);
    expect(Math.abs(Math.hypot(...p.point.map((v,i)=>v-data.receiverENU[i]))-p.receiverRange)).toBeLessThan(1e-8);
    expect(Math.abs(Math.hypot(...p.point.map((v,i)=>v-signal.sourceENU[i]))-p.sourceRange)).toBeLessThan(1e-7);
  }
});

test("live export uses one completed receiver and orbit epoch; pause and end retain it",async({page})=>{
  await page.goto("forest.html");const root=page.locator("#forest-lab");await expect(root).toHaveAttribute("data-state","ready",{timeout:45000});
  await page.locator("#forest-progress").press("Home");await page.locator("#forest-play").click();await expect.poll(async()=>Number(await root.getAttribute("data-seconds"))).toBeGreaterThan(1);
  const pending=page.waitForEvent("download");await page.locator("#forest-export").click();const data=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));
  const {forestReceiver}=await import("../../docs/assets/forest/model.mjs");expect(data.receiverENU).toEqual(forestReceiver(data.elapsedSeconds));expect(data.tow).toBe(data.startTow+data.elapsedSeconds);expect(data.selectedReflection.tow).toBe(data.tow);
  await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","true");await page.locator("#forest-play").click();
  const stopped=await root.getAttribute("data-seconds");await page.locator("#bounce-loss").press("Home");expect(await root.getAttribute("data-seconds")).toBe(stopped);
  await page.locator("#forest-progress").press("End");await expect(root).toHaveAttribute("data-seconds","1200");await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","false");
});

test("losing the graphics context stops playback and cannot leave a ready result",async({page})=>{
  await page.goto("forest.html");const root=page.locator("#forest-lab");await expect(root).toHaveAttribute("data-state","ready",{timeout:45000});
  await page.locator("#forest-play").click();
  await page.locator("#forest-scene").evaluate(canvas=>canvas.getContext("webgl2").getExtension("WEBGL_lose_context").loseContext());
  await expect(root).toHaveAttribute("data-state","error");await expect(page.locator("#forest-status")).toContainText("graphics context lost");
  await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","false");await expect(page.locator("#forest-export")).toBeDisabled();
});

test("corrupted orbit source fails visibly without enabling a plausible forest result",async({page})=>{
  await page.route("**/odaiba-scene.json.gz*",route=>route.fulfill({status:200,body:Buffer.from("bad orbit bytes"),contentType:"application/gzip"}));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","error");
  await expect(page.locator("#forest-status")).toContainText("checksum mismatch");await expect(page.locator("#forest-play")).toBeDisabled();await expect(page.locator("#forest-export")).toBeDisabled();
});

test("trail perspective follows the completed receiver and keeps appearance out of RF assumptions",async({page},testInfo)=>{
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:45000});
  const canvas=page.locator("#forest-scene");await expect(canvas).toHaveAttribute("data-view","overview");
  expect(Number(await canvas.getAttribute("data-vertices"))).toBeLessThan(800000);
  await page.locator("#forest-trail-view").click();await expect(canvas).toHaveAttribute("data-view","trail");
  await canvas.screenshot({path:testInfo.outputPath("forest-trail.png")});
  for(const end of [false,true]){
    if(end)await page.locator("#forest-progress").press("End");
    const pending=page.waitForEvent("download");await page.locator("#forest-export").click();const data=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));
    expect(data.view.follow).toBe(true);expect(data.view.target).toEqual(data.receiverENU);expect(data.assumptions.appearance).toContain("display only");expect(data.scene.count).toBe(128);
  }
  const drawing=await canvas.evaluate(c=>{document.getElementById("forest-trail-view").click();const gl=c.getContext("webgl2"),pixels=new Uint8Array(64*64*4);gl.readPixels(Math.floor(c.width/2)-32,Math.floor(c.height*.2),64,64,gl.RGBA,gl.UNSIGNED_BYTE,pixels);const colors=new Set();for(let i=0;i<pixels.length;i+=16)colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);return {lost:gl.isContextLost(),error:gl.getError(),colors:colors.size};});
  expect(drawing.lost).toBe(false);expect(drawing.error).toBe(0);expect(drawing.colors).toBeGreaterThan(10);
  await page.locator("#forest-reset-view").click();await expect(canvas).toHaveAttribute("data-view","overview");expect(errors).toEqual([]);
});

test("density control changes both-leg attenuation while preserving geometry and snapshot time",async({page})=>{
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:45000});
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  await page.locator("#density-profile").selectOption("uniform");const uniform=await snapshot();
  await page.locator("#density-profile").selectOption("radial-height-v1");const structured=await snapshot();
  expect(uniform.elapsedSeconds).toBe(structured.elapsedSeconds);expect(structured.options.densityProfile).toBe("radial-height-v1");
  let different=0;
  for(const s of structured.signals){const a=uniform.signals.find(a=>a.id===s.id);expect(s.sourceENU).toEqual(a.sourceENU);expect(s.trace.wood).toEqual(a.trace.wood);expect(s.trace.ground).toBe(a.trace.ground);expect(s.trace.canopyPath).toBe(a.trace.canopyPath);if(Math.abs(s.trace.canopyLoss-a.trace.canopyLoss)>1e-8)different++;}
  expect(different).toBeGreaterThan(0);expect(structured.selectedReflection.options.densityProfile).toBe("radial-height-v1");expect(uniform.selectedReflection.options.densityProfile).toBe("uniform");
  expect(structured.selectedReflection.paths.map(p=>p.point)).toEqual(uniform.selectedReflection.paths.map(p=>p.point));
});

test("route worker compares both profiles, retains fixed-sigma baseline and exports all epochs",async({page},testInfo)=>{
  test.setTimeout(240000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:45000});
  await page.locator("#forest-count").selectOption("32");await page.locator("#position-experiment summary").click();await page.locator("#position-route-step").selectOption("60");
  await page.locator("#position-route-run").click();await page.locator("#position-cancel").click();await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","cancelled");await expect(page.locator("#position-export")).toBeDisabled();
  await page.locator("#position-route-step").selectOption("300");  await page.locator("#forest-play").click();await page.locator("#position-route-run").click();await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","false");
  await page.locator("#forest-count").selectOption("64");
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","ready",{timeout:180000});
  await expect(page.locator("#route-results")).toBeVisible();await expect(page.locator("#position-results tr")).toHaveCount(15);
  const pending=page.waitForEvent("download");await page.locator("#position-export").click();const data=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));
  expect(data.schema).toBe("synthetic-forest-route-v1");expect(data.scene.count).toBe(32);expect(data.settings.stepSeconds).toBe(300);expect(data.cases.map(c=>c.seconds)).toEqual([0,300,600,900,1200]);expect(data.assumptions.positioning).toContain("NOT used");
  expect(data.cases.some(c=>c.positioning.results.some(r=>r.status==="solution"))).toBe(true);
  for(const c of data.cases){expect(c.tow).toBe(data.context.startTow+c.seconds);expect(c.observations.every(o=>o.sigma===data.settings.sigma)).toBe(true);expect(c.channels.every(s=>Number.isInteger(s.feasiblePaths)&&s.feasiblePaths>=0)).toBe(true);}
  const ids=await page.locator("#route-satellite option").evaluateAll(nodes=>nodes.map(n=>n.value));expect(ids.some(id=>id.startsWith("G"))).toBe(true);expect(ids.some(id=>id.startsWith("R"))).toBe(true);
  const crossing=ids.find(id=>data.cases.some(c=>c.channels.some(s=>s.id===id&&!s.directBlocked&&s.canopyPathM>0)));if(crossing)await expect(page.locator("#route-satellite")).toHaveValue(crossing);
  const selected=ids.find(id=>id.startsWith("R"));await page.locator("#route-satellite").selectOption(selected);await expect(page.locator("#route-loss")).toHaveAttribute("aria-label",`${selected} · direct canopy attenuation`);
  await expect(page.locator('#route-loss path[data-series="Uniform"]')).toHaveCount(1);await expect(page.locator('#route-position path[data-series="paths"]')).toHaveCount(1);
  const {traceForest}=await import("../../docs/assets/forest/model.mjs");
  for(const c of data.cases){const s=c.channels.find(s=>s.id===selected);if(!s)continue;const maximum=Math.hypot(...s.sourceENU.map((v,i)=>v-c.receiverENU[i]));for(const profile of ["uniform","radial-height-v1"]){const trace=traceForest(data.scene,c.receiverENU,s.directionENU,{maximum,specificLoss:data.settings.specificLoss,foliage:data.settings.foliage,densityProfile:profile});expect(trace.canopyLoss).toBeCloseTo(s.profiles[profile].canopyLossDb,9);}}
  if(crossing)await page.locator("#route-satellite").selectOption(crossing);
  await page.locator("#route-results").screenshot({path:testInfo.outputPath("route-profiles.png")});expect(errors).toEqual([]);
  const layout=await page.evaluate(()=>({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));expect(layout.scroll).toBeLessThanOrEqual(layout.client+2);
});
