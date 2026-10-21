const { test, expect } = require("@playwright/test");

test("3D simulation runs under the Pages project prefix and responds to controls", async ({ page }) => {
  const errors = [];
  page.on("pageerror", error => errors.push(error.message));
  page.on("console", message => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto("simulation.html");
  const canvas = page.locator("#scene");
  await expect(canvas).toHaveAttribute("data-ready", "true");
  await expect(page.getByRole("heading", { name: "Trace signals through a city." })).toBeVisible();
  await expect(page.locator(".satellite")).toHaveCount(10);
  const total = await page.locator(".metrics strong").allTextContents();
  expect(total.map(Number).reduce((a, b) => a + b, 0)).toBe(10);

  await page.locator("#preset").selectOption("plaza");
  await expect(canvas).toHaveAttribute("data-preset", "plaza");
  await page.locator("#height").press("End");
  await expect(canvas).toHaveAttribute("data-height", "2");
  await page.locator("#rotation").press("ArrowRight");
  await expect(canvas).toHaveAttribute("data-rotation", "1");
  await page.locator('[data-satellite="E24"]').click();
  await expect(page.locator("#selected-id")).toHaveText("E24");

  await page.getByRole("button", { name: "Play route" }).click();
  await expect(page.locator("#play")).toHaveAttribute("aria-pressed", "true");
  await expect.poll(async () => Number(await page.locator("#progress").inputValue())).toBeGreaterThan(35);
  await page.getByRole("button", { name: "Pause", exact: true }).click();
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  await expect(canvas).toHaveAttribute("data-preset", "canyon");
  await expect(canvas).toHaveAttribute("data-rotation", "0");
  await expect(canvas).toHaveAttribute("data-height", "1");

  // The camera starts centered on the crossing of two streets.
  const previousReceiver = await canvas.getAttribute("data-receiver");
  const size = await canvas.boundingBox();
  await canvas.click({ position: { x: size.width / 2, y: size.height * 0.63 } });
  await expect(page.locator("#progress-out")).toHaveText("Placed manually");
  await expect(canvas).not.toHaveAttribute("data-receiver", previousReceiver);
  const counts = await page.locator(".metrics strong").allTextContents();
  await page.locator("#reflections").uncheck();
  await expect(page.locator(".metrics strong")).toHaveText(counts);
  await page.getByRole("button", { name: "Reset", exact: true }).click();

  const layout = await page.evaluate(() => ({
    client: document.documentElement.clientWidth,
    scroll: document.documentElement.scrollWidth,
  }));
  expect(layout.scroll).toBeLessThanOrEqual(layout.client + 2);
  expect(errors).toEqual([]);
});

test("results page links to the simulation within the project prefix", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("link", { name: "Open the 3D simulation" }).click();
  await expect(page).toHaveURL(/\/gnss_gpu\/simulation\.html$/);
  await expect(page.locator("#scene")).toHaveAttribute("data-ready", "true");
});
