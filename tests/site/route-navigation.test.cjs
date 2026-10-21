const test = require("node:test");
const assert = require("node:assert/strict");
const { gunzipSync } = require("node:zlib");
const fs = require("node:fs");
const path = require("node:path");
const navigation = import("../../docs/assets/simulation/route-navigation.mjs");
const scene = JSON.parse(gunzipSync(fs.readFileSync(path.join(__dirname, "../../docs/assets/plateau/odaiba-scene.json.gz"))));

test("north-up route projection preserves aspect, orientation and every original pose", async () => {
  const { routeLayout } = await navigation;
  const points = [[-100, -50, 9], [0, 0, 0], [100, 50, -3]], original = structuredClone(points);
  const layout = routeLayout(points, 440, 240, 20);
  assert.equal(layout.scale, 2);
  assert.deepEqual(layout.points, [[20, 220], [220, 120], [420, 20]]);
  assert.deepEqual(points, original);
  const narrow = routeLayout(points, 140, 240, 20);
  assert.equal(narrow.scale, .5);
  assert.deepEqual(narrow.points, [[20, 145], [70, 120], [120, 95]]);
  assert.deepEqual(routeLayout([[12, 34, 56]], 100, 100).points, [[50, 50]]);
});

test("picking returns an original row within the hit radius and leaves empty map space unselected", async () => {
  const { nearestRecordedEpoch } = await navigation;
  const points = [[0, 0], [10, 0], [20, 0], [30, 0]];
  assert.equal(nearestRecordedEpoch(points, 17, 2, 0, 4), 2);
  assert.equal(nearestRecordedEpoch(points, 15, 0, 3, 5), 2);
  assert.equal(nearestRecordedEpoch(points, 15, 0, 0, 5), 1);
  assert.equal(nearestRecordedEpoch(points, 15, 20, 0, 18), null);
  assert.equal(nearestRecordedEpoch([], 0, 0, 0), null);
});

test("overlapping outward and return poses prefer current time, then the earlier source row", async () => {
  const { nearestRecordedEpoch } = await navigation;
  const points = [[0, 0], [10, 0], [20, 0], [10, 0], [0, 0]];
  assert.equal(nearestRecordedEpoch(points, 0, 0, 0), 0);
  assert.equal(nearestRecordedEpoch(points, 0, 0, 4), 4);
  assert.equal(nearestRecordedEpoch(points, 10, 0, 4), 3);
  assert.equal(nearestRecordedEpoch(points, 10, 0, 2), 1);
  // A genuinely closer screen position wins even when it is far away in time.
  assert.equal(nearestRecordedEpoch([[0, 0], [.00001, 0]], .00001, 0, 0), 1);
});

test("step navigation clamps source endpoints without inventing an intermediate epoch", async () => {
  const { stepEpoch } = await navigation;
  assert.equal(stepEpoch(8000, 1, 12410), 8001);
  assert.equal(stepEpoch(8000, -10, 12410), 7990);
  assert.equal(stepEpoch(0, -1, 12410), 0);
  assert.equal(stepEpoch(12405, 10, 12410), 12409);
});

test("all 12,410 real route rows survive resizing and return exact source-row selections", async () => {
  const { routeLayout, nearestRecordedEpoch } = await navigation;
  // Independent ECEF-to-local multiplication, without the production helper.
  const positions = scene.track.map(row => scene.rotation.map(axis => axis.reduce((sum, value, i) => sum + value * (row[i + 1] - scene.origin[i]), 0)));
  const original = structuredClone(scene.track);
  for (const [width, height] of [[180, 190], [365, 185], [430, 190]]) {
    const layout = routeLayout(positions, width, height);
    assert.equal(layout.points.length, 12410);
    for (let epoch = 0; epoch < positions.length; epoch += 97) {
      const [x, y] = layout.points[epoch];
      assert.ok(x >= 22 - 1e-9 && x <= width - 22 + 1e-9);
      assert.ok(y >= 22 - 1e-9 && y <= height - 22 + 1e-9);
      assert.equal(nearestRecordedEpoch(layout.points, x, y, epoch), epoch);
    }
  }
  assert.deepEqual(scene.track, original);
});
