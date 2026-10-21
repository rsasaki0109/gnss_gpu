// Only the current viewport is queued. Resources are kept in a bounded LRU cache;
// HTTP caching is left to the browser, with no prefetch or cache-bypass requests.
export class VisibleTileCache {
  constructor(load, onChange, { concurrency = 4, capacity = 64 } = {}) {
    if (concurrency < 1 || capacity < concurrency) throw new RangeError("Invalid tile cache limits");
    this.load = load; this.onChange = onChange;
    this.concurrency = concurrency; this.capacity = capacity;
    this.entries = new Map(); this.wanted = new Map();
    this.active = 0; this.clock = 0; this.enabled = true;
  }

  select(tiles) {
    if (tiles.length > this.capacity) throw new RangeError("Viewport exceeds tile cache capacity");
    this.wanted = new Map(tiles.map(tile => [tile.key, tile]));
    for (const key of this.wanted.keys()) if (this.entries.has(key)) this.entries.get(key).used = ++this.clock;
    this.evict(); this.pump(); this.notify();
  }

  setEnabled(enabled) {
    if (enabled && !this.enabled) for (const [key, entry] of this.entries) if (entry.state === "error") this.entries.delete(key);
    this.enabled = enabled; this.pump(); this.notify();
  }

  pump() {
    if (!this.enabled) return;
    for (const [key, tile] of this.wanted) {
      if (this.active >= this.concurrency) break;
      if (this.entries.has(key)) continue;
      this.evict(this.capacity - 1);
      if (this.entries.size >= this.capacity) break;
      const entry = { tile, state: "loading", used: ++this.clock, value: null };
      this.entries.set(key, entry); this.active++;
      Promise.resolve().then(() => {
        if (!this.enabled || !this.wanted.has(key)) return null;
        return this.load(tile);
      }).then(value => {
        if (value === null) { this.entries.delete(key); return; }
        entry.value = value; entry.state = "ready";
      }, () => { entry.state = "error"; }).finally(() => {
        this.active--; this.evict(); this.pump(); this.notify();
      });
    }
  }

  evict(limit = this.capacity) {
    const candidates = [...this.entries].filter(([key, entry]) => !this.wanted.has(key) && entry.state !== "loading").sort((a, b) => a[1].used - b[1].used);
    for (const [key, entry] of candidates) {
      if (this.entries.size <= limit) break;
      entry.value?.dispose(); this.entries.delete(key);
    }
  }

  ready() {
    if (!this.enabled) return [];
    return [...this.wanted.keys()].map(key => this.entries.get(key)).filter(entry => entry?.state === "ready").map(entry => entry.value);
  }

  stats() {
    let ready = 0, failed = 0;
    for (const key of this.wanted.keys()) {
      const state = this.entries.get(key)?.state;
      ready += state === "ready"; failed += state === "error";
    }
    const total = this.wanted.size;
    const state = !this.enabled ? "off" : ready + failed < total || !total ? "loading" : failed ? ready ? "partial" : "unavailable" : "ready";
    return { state, ready, failed, total };
  }

  notify() { this.onChange(this.stats()); }
}
