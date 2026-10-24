const {test,expect}=require("@playwright/test");
const fs=require("node:fs");
// Deterministic offscreen WebGL: the host's default headless GPU context can be lost.
// No software-renderer flags are added to the website or normal browser launches.
test.use({launchOptions:{args:["--use-angle=swiftshader","--enable-unsafe-swiftshader"]}});

test("tree loss attribution reconciles with traces and focuses RF envelopes without changing measurements",async({page},testInfo)=>{
  test.setTimeout(150000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:60000});
  const causes=page.locator("#forest-tree-causes"),canvas=page.locator("#forest-scene"),rays=page.locator("#forest-rays");
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  const first=await snapshot(),id=await causes.locator("[data-tree-select]").first().getAttribute("data-tree-select"),sum=rows=>rows.reduce((s,r)=>s+r.crownLossDb,0);
  expect(first.treeAttribution.totalCanopyLossDb).toBeCloseTo(sum(first.treeAttribution.rows),9);
  expect(first.treeAttribution.satellite).toBe(first.signalHistory.samples.at(-1).id);
  expect(sum(first.treeAttribution.rows)).toBeCloseTo(first.signals.find(s=>s.id===first.signalHistory.samples.at(-1).id).trace.canopyLoss,9);
  await causes.locator(`[data-tree-select="${id}"]`).click();await expect(canvas).toHaveAttribute("data-view","overview");await expect(rays).toHaveAttribute("data-highlighted-tree",id);
  expect(await page.locator(`[data-tree-outline="${id}"] path`).count()).toBeGreaterThan(0);
  const focused=await snapshot();expect(focused.signals).toEqual(first.signals);expect(focused.signalHistory.samples).toEqual(first.signalHistory.samples);
  await page.locator(".stage").screenshot({path:testInfo.outputPath("tree-cause-focus.png")});
  await page.locator("#forest-path-mode").selectOption("off");await expect(page.locator("[data-tree-outline]")).toHaveCount(0);await page.locator("#forest-path-mode").selectOption("selected");
  await page.locator("#forest-tree-scope").selectOption("ground");const ground=await snapshot(),path=ground.selectedReflection.paths[0];
  expect(ground.treeAttribution.available).toBe(Boolean(path));if(path)expect(sum(ground.treeAttribution.rows)).toBeCloseTo(path.canopyLoss,9);
  await page.locator("#forest-tree-scope").selectOption("direct");await page.locator("#attenuation-model").selectOption("itu-extinction-v1");await page.locator("#forest-season").selectOption("leaf-off");
  const winter=await snapshot();expect(sum(winter.treeAttribution.rows)).toBeCloseTo(winter.signals.find(s=>s.id===winter.signalHistory.samples.at(-1).id).trace.canopyLoss,9);
  await page.locator("#forest-play").click();await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","true");await causes.locator("[data-tree-select]").first().click();await expect(page.locator("#forest-play")).toHaveAttribute("aria-pressed","false");
  await causes.screenshot({path:testInfo.outputPath("tree-cause-breakdown.png")});expect(errors).toEqual([]);expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test("live signal charts retain completed epochs and reset on seeks, RF edits and satellite selection",async({page},testInfo)=>{
  test.setTimeout(150000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");const root=page.locator("#forest-lab"),history=page.locator("#forest-live-history");await expect(root).toHaveAttribute("data-state","ready",{timeout:60000});
  await page.locator("#forest-count").selectOption("32");await expect(history).toHaveAttribute("data-samples","1");
  await page.locator("#forest-play").click();await expect.poll(async()=>Number(await history.getAttribute("data-samples")),{timeout:45000}).toBeGreaterThan(2);await page.locator("#forest-play").click();
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  const first=await snapshot(),samples=first.signalHistory.samples,last=samples.at(-1);
  expect(last.seconds).toBe(first.elapsedSeconds);const channel=first.signals.find(s=>s.id===last.id);expect(channel).toBeDefined();
  expect(last.directLossDb).toBe(channel.trace.directRelativeDb===null?null:channel.trace.canopyLoss);
  expect(last.reflectionExtraM).toBe(first.selectedReflection.paths[0]?.extra??null);
  expect(samples.every((s,i)=>i===0||s.seconds>samples[i-1].seconds)).toBe(true);
  await expect(page.locator('[data-history-plot="directLossDb"] [data-live-series]')).toHaveCount(1);
  await page.locator("#forest-light").selectOption("overcast");const relit=await snapshot();expect(relit.signalHistory.samples).toEqual(samples);expect(relit.signals).toEqual(first.signals);
  await page.locator("#forest-progress").press("Home");await expect(history).toHaveAttribute("data-samples","1");expect((await snapshot()).signalHistory.reason).toContain("Timeline");
  await page.locator("#forest-play").click();await expect.poll(async()=>Number(await history.getAttribute("data-samples")),{timeout:45000}).toBeGreaterThan(1);await page.locator("#forest-play").click();
  await page.locator("#density-profile").selectOption("uniform");await expect(history).toHaveAttribute("data-samples","1");
  const target=page.locator('#forest-signals button[aria-pressed="false"]').first(),id=await target.getAttribute("data-channel");await target.click();await expect(history).toHaveAttribute("data-selected",id);await expect(history).toHaveAttribute("data-samples","1");
  await page.locator(".viewer").screenshot({path:testInfo.outputPath("live-signal-changes.png")});expect(errors).toEqual([]);expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
});

test("GNSS paths stay visible in trail and overview, selectable without changing measurements",async({page},testInfo)=>{
  test.setTimeout(120000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");const root=page.locator("#forest-lab");await expect(root).toHaveAttribute("data-state","ready",{timeout:60000});
  const rays=page.locator("#forest-rays"),paths=page.locator('#forest-rays path[data-path-kind="direct"]');
  expect(await paths.count()).toBeGreaterThan(3);
  for(const path of await paths.all())expect(await path.evaluate(p=>p.getTotalLength())).toBeGreaterThan(.5);
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  const before=await snapshot();
  const target=await page.locator('#forest-rays [data-signal-select][aria-pressed="false"]').first().getAttribute("data-signal-select");
  await page.locator(`#forest-rays [data-signal-select="${target}"]`).click();await expect(root).toHaveAttribute("data-selected",target);
  await page.locator("#forest-path-mode").selectOption("selected");await expect(paths).toHaveCount(1);
  await page.locator("#forest-track-signal").click();await expect(paths).toHaveCount(1);
  const eye=await page.evaluate(async()=>{const {forestCameraFrame}=await import('./assets/forest/signal-overlay.mjs?v=signal-paths-1');return forestCameraFrame({yaw:0,pitch:-1.4,zoom:1,target:[0,0,1.6],follow:true},800,600).eye;});expect(eye[2]).toBeCloseTo(2.6);
  await page.locator("#forest-path-mode").selectOption("off");await expect(page.locator('#forest-rays path[data-signal]')).toHaveCount(0);
  const hidden=await snapshot();expect(hidden.signals).toEqual(before.signals);expect(hidden.receiverENU).toEqual(before.receiverENU);expect(hidden.view.signalMode).toBe("off");
  await page.locator("#forest-path-mode").selectOption("all");await page.locator("#forest-reset-view").click();expect(await paths.count()).toBeGreaterThan(3);
  await page.locator(".stage").screenshot({path:testInfo.outputPath("gnss-overview.png")});await page.locator("#forest-trail-view").click();await page.locator(".stage").screenshot({path:testInfo.outputPath("gnss-trail.png")});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);expect(errors).toEqual([]);
});

test("evergreen needles remain visible at trail distance in leaf-off season",async({page},testInfo)=>{
  test.setTimeout(90000);
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:60000});
  await page.locator("#forest-count").selectOption("32");await page.locator("#forest-stand").selectOption("conifer");await page.locator("#forest-season").selectOption("leaf-off");
  const canvas=page.locator("#forest-scene");
  const drawing=await canvas.evaluate(c=>{
    document.getElementById("forest-trail-view").click();const gl=c.getContext("webgl2"),w=c.width,h=Math.floor(c.height*.55),pixels=new Uint8Array(w*h*4);
    gl.readPixels(0,c.height-h,w,h,gl.RGBA,gl.UNSIGNED_BYTE,pixels);let green=0;
    for(let i=0;i<pixels.length;i+=4)if(pixels[i+1]>pixels[i]*1.15&&pixels[i+1]>pixels[i+2]*1.1)green++;
    return {fraction:green/(w*h),error:gl.getError()};
  });
  expect(drawing.error).toBe(0);expect(drawing.fraction).toBeGreaterThan(.004);
  await canvas.screenshot({path:testInfo.outputPath("evergreen-needles.png")});
});

test("woodland lighting changes appearance while preserving the captured RF scene",async({page},testInfo)=>{
  test.setTimeout(120000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:60000});
  const canvas=page.locator("#forest-scene");await expect(canvas).toHaveAttribute("data-appearance","woodland-1");await expect(canvas).toHaveAttribute("data-view","trail");
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  const sunny=await snapshot();await canvas.screenshot({path:testInfo.outputPath("woodland-sunlit.png")});
  await page.locator("#forest-light").selectOption("overcast");await expect(canvas).toHaveAttribute("data-lighting","overcast");
  const cloudy=await snapshot();assertSameScene(sunny,cloudy);
  expect(cloudy.view.lighting).toBe("overcast");await canvas.screenshot({path:testInfo.outputPath("woodland-overcast.png")});
  await page.locator("#forest-reset-view").click();await expect(canvas).toHaveAttribute("data-view","overview");await page.locator("#forest-trail-view").click();
  const drawing=await canvas.evaluate(c=>{const gl=c.getContext("webgl2");return {error:gl.getError(),lost:gl.isContextLost()};});
  expect(drawing).toEqual({error:0,lost:false});expect(errors).toEqual([]);
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  function assertSameScene(a,b){expect(b.scene).toEqual(a.scene);expect(b.signals).toEqual(a.signals);expect(b.receiverENU).toEqual(a.receiverENU);expect(b.elapsedSeconds).toBe(a.elapsedSeconds);}
});

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
  await page.locator(".stage").screenshot({path:testInfo.outputPath("ground-reflection.png")});
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
  const canvas=page.locator("#forest-scene");await expect(canvas).toHaveAttribute("data-view","trail");
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

test("tree type and season rebuild shared geometry, RF, exports and captured worker inputs",async({page},testInfo)=>{
  test.setTimeout(180000);
  const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");const root=page.locator("#forest-lab");await expect(root).toHaveAttribute("data-state","ready");
  await page.locator("#forest-count").selectOption("32");
  const snapshot=async()=>{const pending=page.waitForEvent("download");await page.locator("#forest-export").click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  const summer=await snapshot(),summerVertices=Number(await page.locator("#forest-scene").getAttribute("data-vertices"));
  expect(summer.scene.stand).toBe("broadleaf");expect(summer.scene.season).toBe("leaf-on");expect(summer.signals.some(s=>s.trace.canopyLoss>0)).toBe(true);
  await page.locator("#forest-season").selectOption("leaf-off");await expect(root).toHaveAttribute("data-season","leaf-off");
  const winter=await snapshot();expect(winter.scene.trees.every(t=>t.leafRetention===0)).toBe(true);expect(winter.signals.every(s=>s.trace.canopyLoss===0&&s.trace.crowns.length===0)).toBe(true);
  expect(Number(await page.locator("#forest-scene").getAttribute("data-vertices"))).toBeLessThan(summerVertices);
  for(const s of winter.signals)expect(s.trace.wood).toEqual(summer.signals.find(a=>a.id===s.id).trace.wood);
  await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("broadleaf-leaf-off.png")});
  await page.locator("#forest-stand").selectOption("conifer");await expect(root).toHaveAttribute("data-stand","conifer");
  const evergreenWinter=await snapshot();expect(evergreenWinter.scene.trees.every(t=>t.canopies.length===5&&t.leafRetention===1)).toBe(true);
  expect(evergreenWinter.scene.trees.map(t=>[t.x,t.y])).toEqual(summer.scene.trees.map(t=>[t.x,t.y]));
  await page.locator("#forest-season").selectOption("leaf-on");const evergreenSummer=await snapshot();expect(evergreenSummer.signals).toEqual(evergreenWinter.signals);
  expect(evergreenWinter.signals.some(s=>s.trace.canopyLoss>0)).toBe(true);
  await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("conifer-leaf-on.png")});
  await page.locator("#forest-stand").selectOption("mixed");await page.locator("#forest-season").selectOption("leaf-off");const mixed=await snapshot();
  expect(mixed.scene.trees.filter(t=>t.leafRetention===0)).toHaveLength(16);expect(mixed.scene.trees.filter(t=>t.leafRetention===1)).toHaveLength(16);
  const {traceForest}=await import("../../docs/assets/forest/model.mjs");
  for(const data of [summer,winter,evergreenWinter,mixed])for(const s of data.signals){
    const actual=traceForest(data.scene,data.receiverENU,s.trace.direction,data.options);
    expect(actual.canopyLoss).toBeCloseTo(s.trace.canopyLoss,9);expect(actual.wood).toEqual(s.trace.wood);
    expect(data.assumptions.phenology).toContain("not calibrated");
  }
  await page.locator("#forest-trail-view").click();await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("mixed-leaf-off-trail.png")});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.locator("#position-experiment summary").click();await page.locator("#position-run").click();await page.locator("#forest-stand").selectOption("broadleaf");
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","ready",{timeout:120000});
  const pending=page.waitForEvent("download");await page.locator("#position-export").click();const result=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));
  expect(result.scene).toEqual(mixed.scene);expect(result.results).toHaveLength(3);expect(result.results.every(r=>r.status==="solution")).toBe(true);expect(errors).toEqual([]);
});

test("published extinction reference retains bare-crown loss and survives a captured route study",async({page},testInfo)=>{
  test.setTimeout(240000);const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("forest.html");await expect(page.locator("#forest-lab")).toHaveAttribute("data-state","ready",{timeout:45000});await page.locator("#forest-count").selectOption("32");
  const download=async button=>{const pending=page.waitForEvent("download");await page.locator(button).click();return JSON.parse(fs.readFileSync(await (await pending).path(),"utf8"));};
  await page.locator("#attenuation-model").selectOption("itu-extinction-v1");
  for(const id of ["#specific-loss","#foliage"])await expect(page.locator(id)).toBeDisabled();await expect(page.locator("#material-preset")).toBeEnabled();
  const summer=await download("#forest-export");await page.locator("#forest-season").selectOption("leaf-off");const winter=await download("#forest-export");
  expect(winter.assumptions.vegetationModel.source.calibratedToScene).toBe(false);expect(winter.assumptions.vegetationModel.source.frequencyPolicy).toContain("No interpolation");
  expect(winter.signals.some(s=>s.trace.canopyLoss>0)).toBe(true);
  for(const s of winter.signals){const a=summer.signals.find(a=>a.id===s.id);expect(s.trace.wood).toEqual(a.trace.wood);expect(s.trace.ground).toBe(a.trace.ground);expect(s.trace.canopyPath).toBe(a.trace.canopyPath);expect(s.trace.canopyLoss).toBeCloseTo(a.trace.canopyLoss*.139/.241,9);expect(s.trace.crowns.every(c=>c.material.leafState==="out-of-leaf"&&c.material.anchorGHz===1.3)).toBe(true);}
  expect(winter.selectedReflection.options.attenuationModel).toBe("itu-extinction-v1");
  await page.screenshot({path:testInfo.outputPath("reference-leaf-off.png"),fullPage:true});
  await page.locator("#forest-stand").selectOption("conifer");const pine=await download("#forest-export");await page.locator("#material-preset").selectOption("maple-cedar");const cedar=await download("#forest-export");
  expect(pine.signals.some(s=>s.trace.canopyLoss>0)).toBe(true);
  for(const s of cedar.signals){const a=pine.signals.find(a=>a.id===s.id);expect(s.trace.canopyLoss).toBeCloseTo(a.trace.canopyLoss*.271/.215,9);expect(s.trace.wood).toEqual(a.trace.wood);expect(s.trace.crowns.every(c=>c.material.reference==="himalayan-cedar"&&c.material.anchorGHz===1.5)).toBe(true);}
  await page.locator("#forest-stand").selectOption("mixed");await page.locator("#position-experiment summary").click();await page.locator("#position-route-run").click();
  await page.locator("#attenuation-model").selectOption("linear-v1");await expect(page.locator("#foliage")).toBeEnabled();await expect(page.locator("#material-preset")).toBeDisabled();
  await expect(page.locator("#position-experiment")).toHaveAttribute("data-state","ready",{timeout:180000});const route=await download("#position-export");
  await expect(page.locator("#position-status")).toContainText("captured mixed / leaf-off");await expect(page.locator("#position-status")).toContainText("extinction reference (maple / Himalayan cedar)");expect(route.settings.attenuationModel).toBe("itu-extinction-v1");expect(route.settings.materialPreset).toBe("maple-cedar");expect(route.scene.season).toBe("leaf-off");expect(route.cases).toHaveLength(5);expect(route.assumptions.vegetationModel.source.scope).toContain("Ballistic");
  const {traceForest}=await import("../../docs/assets/forest/model.mjs");let hits=0;
  for(const c of route.cases)for(const s of c.channels)for(const densityProfile of ["uniform","radial-height-v1"]){
    const maximum=Math.hypot(...s.sourceENU.map((v,i)=>v-c.receiverENU[i])),tr=traceForest(route.scene,c.receiverENU,s.directionENU,{...route.settings,maximum,densityProfile});
    expect(s.profiles[densityProfile].canopyLossDb).toBeCloseTo(tr.canopyLoss,9);if(tr.canopyLoss>0)hits++;
  }
  expect(hits).toBeGreaterThan(0);expect(route.cases.every(c=>c.observations.every(o=>o.sigma===route.settings.sigma))).toBe(true);
  await page.locator("#route-results").screenshot({path:testInfo.outputPath("reference-route-plots.png")});  await page.screenshot({path:testInfo.outputPath("reference-route.png"),fullPage:true});expect(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);expect(errors).toEqual([]);
});
