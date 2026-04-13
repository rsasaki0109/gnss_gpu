const { test, expect } = require("@playwright/test");

const IGNORED_CONSOLE_ERRORS = [
  "fonts.googleapis.com",
  "fonts.gstatic.com",
  "/favicon.ico",
];

test("artifact snapshot renders cleanly", async ({ page }) => {
  const issues = [];
  page.on("pageerror", (err) => issues.push(`pageerror: ${err.message}`));
  page.on("console", (msg) => {
    if (msg.type() !== "error") {
      return;
    }
    const text = msg.text();
    if (IGNORED_CONSOLE_ERRORS.some((pattern) => text.includes(pattern))) {
      return;
    }
    issues.push(`console: ${text}`);
  });

  await page.goto("/", { waitUntil: "domcontentloaded" });

  await expect(page.locator("#title")).toHaveText("gnss_gpu Artifact Snapshot");
  await expect(page.locator("#subtitle")).toContainText("PF+RobustClear-10K");
  const showcaseCards = page.locator("#showcase-media article");
  await expect(showcaseCards).toHaveCount(3);
  await expect(showcaseCards.filter({ hasText: "UrbanNav LOS/NLOS Map Sweep" })).toHaveCount(1);
  await expect(page.locator("#hero-cards article")).toHaveCount(7);
  await expect(page.locator("#method-freeze article")).toHaveCount(4);
  await expect(page.locator("#quick-links article")).toHaveCount(7);
  await expect(page.locator("#figures article")).toHaveCount(4);
  await expect(page.locator("#analysis-charts article")).toHaveCount(5);
  await expect(page.locator("#tables section")).toHaveCount(8);
  await expect(page.locator(".error-box")).toHaveCount(0);
  await expect(page.locator("#validation-card .metric-value")).toContainText("440 passed");

  const snapshotReady = await page.evaluate(() => Boolean(window.__GNSS_GPU_SNAPSHOT__));
  expect(snapshotReady).toBeTruthy();

  const brokenImages = await page.locator("img").evaluateAll((imgs) =>
    imgs.filter((img) => !img.complete || img.naturalWidth === 0).length,
  );
  expect(brokenImages).toBe(0);

  const layout = await page.evaluate(() => ({
    clientWidth: document.documentElement.clientWidth,
    scrollWidth: document.documentElement.scrollWidth,
  }));
  expect(layout.scrollWidth).toBeLessThanOrEqual(layout.clientWidth + 2);

  const headings = await page.locator("h2").allInnerTexts();
  expect(headings).toEqual(
    expect.arrayContaining([
      "At A Glance",
      "Visual Summary",
      "Method Freeze",
      "Featured Figures",
      "Extra Charts",
      "Results Tables",
      "Validation",
      "Limitations",
    ]),
  );

  expect(issues).toEqual([]);
});
