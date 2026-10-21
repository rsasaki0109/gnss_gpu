const { test, expect } = require("@playwright/test");
const { deflateSync } = require("node:zlib");
const { gunzipSync } = require("node:zlib");
const fs = require("node:fs");
const path = require("node:path");
const SCENE = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname, "../../docs/assets/plateau/odaiba-scene.json.gz"))));
const POSITIONS = SCENE.track.map(row => SCENE.rotation.map(axis => axis.reduce((sum, value, i) => sum + value * (row[i + 1] - SCENE.origin[i]), 0)));

// A generated one-pixel test tile. Headless tests never contact OSM servers.
function tileFixture() {
  const chunk = (type, data) => {
    const content = Buffer.concat([Buffer.from(type), data]);
    let crc = 0xffffffff;
    for (const byte of content) {
      crc ^= byte;
      for (let i=0;i<8;i++) crc = (crc>>>1)^((crc&1)?0xedb88320:0);
    }
    const size = Buffer.alloc(4), checksum = Buffer.alloc(4);
    size.writeUInt32BE(data.length); checksum.writeUInt32BE((crc^0xffffffff)>>>0);
    return Buffer.concat([size,content,checksum]);
  };
  const header = Buffer.alloc(13); header.writeUInt32BE(1,0); header.writeUInt32BE(1,4); header[8]=8; header[9]=6;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk("IHDR",header),chunk("IDAT",deflateSync(Buffer.from([0,230,238,230,255]))),chunk("IEND",Buffer.alloc(0))]);
}
const TILE = tileFixture();
test.beforeEach(async ({page}) => {
  await page.route("https://tile.openstreetmap.org/**", route => route.fulfill({
    status:200, contentType:"image/png", body:TILE,
    headers:{"Access-Control-Allow-Origin":"*","Cache-Control":"public, max-age=604800"},
  }));
});

test("PLATEAU full-mesh simulation responds to exact epochs and receiver controls", async ({ page }) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => { if (message.type() === "error") errors.push(message.text()); });
  await page.goto("simulation.html");
  const canvas = page.locator("#scene");
  await expect(canvas).toHaveAttribute("data-ready", "true", {timeout:45000});
  await expect(canvas).toHaveAttribute("data-triangles", "320932");
  await expect(canvas).toHaveAttribute("data-epoch", "8000");
  await expect(page.getByRole("heading", { name: "Real city. Real satellite geometry." })).toBeVisible();
  await expect(page.locator(".satellite")).toHaveCount(8);
  await expect(page.locator(".metrics strong")).toHaveText(["4","1","3"]);
  await expect(page.locator("#spp-all")).toHaveText("65.74 m");
  await page.locator('[data-satellite="G12"]').click();
  await expect(page.locator("#selected-id")).toHaveText("G12");
  await expect(page.locator("#excess-value")).toHaveText("+23.038 m");
  await expect(page.locator("#feature-id")).toContainText("LoD");

  await page.locator("#preset").selectOption("0");
  await expect(canvas).toHaveAttribute("data-epoch", "0");
  await expect(page.locator(".satellite")).toHaveCount(7);
  await expect(page.locator(".metrics strong")).toHaveText(["5","0","2"]);
  await page.locator("#antenna").press("End");
  await expect(canvas).toHaveAttribute("data-antenna", "20");
  await page.locator("#mask").press("End");
  await expect(canvas).toHaveAttribute("data-mask", "30");
  await page.locator("#progress").press("End");
  await expect(canvas).toHaveAttribute("data-epoch", "12409");
  await page.locator("#reset").click();
  await expect(canvas).toHaveAttribute("data-epoch", "8000");
  await expect(canvas).toHaveAttribute("data-mask", "10");
  await expect(canvas).toHaveAttribute("data-antenna", "0");

  await page.locator("#play").click();
  await expect(page.locator("#play")).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => Number(await canvas.getAttribute("data-epoch"))).toBeGreaterThan(8000);
  await page.locator("#play").click();
  await expect(page.locator("#play")).toHaveAttribute("aria-pressed", "false");
  await page.locator("#reset").click();
  await expect(canvas).toHaveAttribute("data-epoch", "8000");
  const counts = await page.locator(".metrics strong").allTextContents();
  await page.locator("#reflections").uncheck();
  await expect(page.locator(".metrics strong")).toHaveText(counts);
  await page.locator("#overview").click();
  await expect(page.locator("#overview")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#reset-view").click();
  await expect(page.locator("#overview")).toHaveAttribute("aria-pressed", "false");
  await page.getByText("Data sources, method & limits", {exact:true}).click();
  await expect(page.locator("#geometry-readout")).toContainText("59 interior rings");
  await expect(page.locator("#repair-readout")).toContainText("1 source face repaired");
  const layout = await page.evaluate(() => ({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
  expect(layout.scroll).toBeLessThanOrEqual(layout.client + 2); expect(errors).toEqual([]);
});

test("results page links to PLATEAU simulation within the Pages project prefix", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("link", { name: "Open the 3D simulation" }).click();
  await expect(page).toHaveURL(/\/gnss_gpu\/simulation\.html$/);
  await expect(page.locator("#scene")).toHaveAttribute("data-ready", "true", {timeout:45000});
});

test("a mismatched asset fails visibly without substituting synthetic geometry", async ({ page }) => {
  await page.route("**/plateau/manifest.json", async route => {
    const response = await route.fetch(), manifest = await response.json();
    manifest.mesh.sha256 = "0".repeat(64);
    await route.fulfill({response, json:manifest});
  });
  await page.goto("simulation.html");
  await expect(page.locator("#scene")).toHaveAttribute("data-error", "true", {timeout:45000});
  await expect(page.locator("#toast")).toContainText("Data integrity check failed");
  await expect(page.locator("#play")).toBeDisabled();
  await expect(page.locator("#scene")).not.toHaveAttribute("data-ready", "true");
});

test("OSM is georeferenced, visibly credited and independent of the signal calculation", async ({page}) => {
  const tileRequests=[];
  page.on("request", request => { if(new URL(request.url()).hostname === "tile.openstreetmap.org") tileRequests.push(request.url()); });
  await page.goto("simulation.html");
  const canvas=page.locator("#scene");
  await expect(canvas).toHaveAttribute("data-map-state","ready",{timeout:45000});
  await expect(canvas).toHaveAttribute("data-epoch","8000");
  await expect(page.locator("#map-credit")).toBeVisible();
  await expect(page.locator("#map-credit a").first()).toHaveAttribute("href","https://www.openstreetmap.org/copyright");
  await expect(page.locator("#osm-location")).toHaveAttribute("href",/mlat=35\.618026&mlon=139\.775924/);
  expect(tileRequests.length).toBeGreaterThan(0); expect(tileRequests.length).toBeLessThanOrEqual(32);
  for(const url of tileRequests){const parsed=new URL(url);expect(parsed.pathname).toMatch(/^\/\d+\/\d+\/\d+\.png$/);expect(parsed.search).toBe("");}
  const before=await page.locator(".metrics strong").allTextContents(), receiver=await canvas.getAttribute("data-receiver");
  const count=tileRequests.length;
  await page.locator("#basemap").uncheck();
  await expect(canvas).toHaveAttribute("data-map-state","off");
  await expect(page.locator("#map-credit")).toBeHidden();
  await expect(page.locator(".metrics strong")).toHaveText(before);
  await expect(canvas).toHaveAttribute("data-receiver",receiver);
  await page.locator("#basemap").check();
  await expect(canvas).toHaveAttribute("data-map-state","ready");
  await expect(page.locator(".metrics strong")).toHaveText(before);
  expect(tileRequests.length).toBe(count);
  const box=await page.locator("#map-credit").boundingBox(),stage=await page.locator("#stage").boundingBox();
  expect(box.x).toBeGreaterThanOrEqual(stage.x);expect(box.x+box.width).toBeLessThanOrEqual(stage.x+stage.width);
  expect(box.y+box.height).toBeLessThanOrEqual(stage.y+stage.height);
});

test("a tile service outage leaves PLATEAU tracing operational", async ({page}) => {
  const errors=[];page.on("pageerror",error=>errors.push(error.message));
  await page.route("https://tile.openstreetmap.org/**",route=>route.abort("failed"));
  await page.goto("simulation.html");
  const canvas=page.locator("#scene");
  await expect(canvas).toHaveAttribute("data-ready","true",{timeout:45000});
  await expect(canvas).toHaveAttribute("data-map-state","unavailable");
  await expect(page.locator("#map-status")).toHaveText("Map unavailable · simulation active");
  await expect(page.locator(".metrics strong")).toHaveText(["4","1","3"]);
  await expect(page.locator("#play")).toBeEnabled();
  await page.locator("#preset").selectOption("0");
  await expect(canvas).toHaveAttribute("data-epoch","0");
  await expect(page.locator(".metrics strong")).toHaveText(["5","0","2"]);
  expect(errors).toEqual([]);
});

test("route keyboard and step buttons evaluate exact source poses, pause playback and respect endpoints", async ({page}) => {
  const errors=[]; page.on("pageerror", error => errors.push(error.message));
  await page.goto("simulation.html");
  const canvas=page.locator("#scene"), route=page.getByRole("slider", {name:"Recorded route position"});
  await expect(route).toHaveAttribute("data-state", "ready", {timeout:45000});
  await expect(route).toHaveAttribute("data-pose-count", "12410");
  await expect(route).toHaveAttribute("aria-valuenow", "8001");
  await route.press("ArrowRight");
  await expect(canvas).toHaveAttribute("data-epoch", "8001");
  const receiver=(await canvas.getAttribute("data-receiver")).split(",").map(Number);
  expect(Math.hypot(...receiver.map((v,i)=>v-POSITIONS[8001][i]))).toBeLessThan(1e-7);
  await expect(page.locator("#preset")).toHaveValue("");
  await expect(route).toHaveAttribute("aria-valuetext", /Pose 8,002 of 12,410/);
  await route.press("Shift+ArrowRight");
  await expect(canvas).toHaveAttribute("data-epoch", "8011");
  await page.getByRole("button", {name:"Back 1 second", exact:true}).click();
  await page.getByRole("button", {name:"Previous recorded pose, 0.1 seconds", exact:true}).click();
  await expect(canvas).toHaveAttribute("data-epoch", "8000");
  await expect(page.locator(".metrics strong")).toHaveText(["4","1","3"]);
  await expect(page.locator("#spp-all")).toHaveText("65.74 m");
  await route.press("Home");
  await expect(canvas).toHaveAttribute("data-epoch", "0");
  await expect(page.getByRole("button", {name:"Back 1 second", exact:true})).toBeDisabled();
  await expect(page.getByRole("button", {name:"Previous recorded pose, 0.1 seconds", exact:true})).toBeDisabled();
  await route.press("End");
  await expect(canvas).toHaveAttribute("data-epoch", "12409");
  await expect(page.getByRole("button", {name:"Forward 1 second", exact:true})).toBeDisabled();
  await expect(page.getByRole("button", {name:"Next recorded pose, 0.1 seconds", exact:true})).toBeDisabled();
  await route.press("Home");
  await page.locator("#play").click();
  await expect.poll(async()=>Number(await canvas.getAttribute("data-epoch"))).toBeGreaterThan(0);
  await route.press("Home");
  await expect(page.locator("#play")).toHaveAttribute("aria-pressed", "false");
  await expect(canvas).toHaveAttribute("data-epoch", "0");
  await expect(route).toHaveAttribute("data-state", "ready");
  expect(errors).toEqual([]);
});

test("clicking or tapping the route seeks an original nearby pose while empty space and drags leave it intact", async ({page,isMobile}) => {
  await page.goto("simulation.html");
  const canvas=page.locator("#scene"), route=page.locator("#route");
  await expect(route).toHaveAttribute("data-state", "ready", {timeout:45000});
  await route.scrollIntoViewIfNeeded();
  const box=await route.boundingBox();
  const dimensions=await route.evaluate(el=>({width:el.clientWidth,height:el.clientHeight}));
  // Independently project the requested original position into the north-up overview.
  const low=[0,1].map(i=>Math.min(...POSITIONS.map(p=>p[i]))),high=[0,1].map(i=>Math.max(...POSITIONS.map(p=>p[i])));
  const scale=Math.min((dimensions.width-44)/(high[0]-low[0]),(dimensions.height-44)/(high[1]-low[1]));
  const target=POSITIONS[6000];
  const x=dimensions.width/2+(target[0]-(low[0]+high[0])/2)*scale;
  const y=dimensions.height/2-(target[1]-(low[1]+high[1])/2)*scale;
  if(isMobile) await page.touchscreen.tap(box.x+x,box.y+y);
  else await route.click({position:{x,y}});
  await expect.poll(async()=>Number(await canvas.getAttribute("data-epoch"))).not.toBe(8000);
  await expect(route).toHaveAttribute("data-state", "ready");
  const epoch=Number(await canvas.getAttribute("data-epoch"));
  // Screen rounding can select an adjacent source row; the receiver remains exactly that row.
  const receiver=(await canvas.getAttribute("data-receiver")).split(",").map(Number);
  expect(Math.hypot(...receiver.map((v,i)=>v-POSITIONS[epoch][i]))).toBeLessThan(1e-7);
  expect(Math.hypot(POSITIONS[epoch][0]-target[0],POSITIONS[epoch][1]-target[1])*scale).toBeLessThan(1.5);
  await expect(page.locator("#progress")).toHaveValue(String(epoch));
  await expect(route).toHaveAttribute("aria-valuenow", String(epoch+1));
  await route.click({position:{x:1,y:1}});
  await expect(canvas).toHaveAttribute("data-epoch", String(epoch));
  const fresh=await route.boundingBox();
  await page.mouse.move(fresh.x+x,fresh.y+y); await page.mouse.down();
  await page.mouse.move(fresh.x+x+30,fresh.y+y+20,{steps:4}); await page.mouse.up();
  await expect(canvas).toHaveAttribute("data-epoch", String(epoch));
  const layout=await page.evaluate(()=>({client:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth}));
  expect(layout.scroll).toBeLessThanOrEqual(layout.client+2);
});
