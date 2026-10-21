const { test, expect } = require("@playwright/test");
const { deflateSync } = require("node:zlib");

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
