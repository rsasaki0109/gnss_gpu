const {test,expect}=require("@playwright/test");
const fs=require("node:fs");
// Deterministic offscreen WebGL: the host's default headless GPU context can be lost.
// No software-renderer flags are added to the website or normal browser launches.
test.use({launchOptions:{args:["--use-angle=swiftshader","--enable-unsafe-swiftshader"]}});

test("forest ray geometry, live sky, assumptions and exported snapshot remain consistent",async({page},testInfo)=>{
  const errors=[],requests=[];page.on("pageerror",e=>errors.push(e.message));page.on("request",r=>requests.push(r.url()));
  await page.goto("forest.html");const root=page.locator("#forest-lab");
  await expect(root).toHaveAttribute("data-state","ready",{timeout:45000});await expect(root).toHaveAttribute("data-count","64");
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
  const time=await root.getAttribute("data-seconds");await page.locator("#forest-focus-reflection").click();expect(await root.getAttribute("data-seconds")).toBe(time);
  await page.locator("#forest-scene").screenshot({path:testInfo.outputPath("ground-reflection.png")});
  const pending=page.waitForEvent("download");await page.locator("#forest-export").click();const data=JSON.parse(fs.readFileSync(await (await pending).path(),"utf8")),packet=data.selectedReflection;
  expect(packet.model).toBe("finite-terrain-reflection-v1");expect(packet.options.completeSegments).toBe(true);expect(packet.options.bounceLoss).toBe(30);expect(packet.tow).toBe(data.tow);
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
