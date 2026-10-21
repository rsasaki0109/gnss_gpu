const { test, expect } = require("@playwright/test");

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
