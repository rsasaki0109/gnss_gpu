// Read the existing ideal-range SPP result. No new solve, path or satellite state.
const vector = value => value?.length === 3 && Array.from(value).every(Number.isFinite);
const distance = (a,b) => Math.hypot(...a.map((value,i)=>value-b[i]));

export function rangeFit(snapshot) {
  if (!vector(snapshot?.receiver) || !Array.isArray(snapshot.satellites)) return null;
  const rows = [];
  for (const signal of snapshot.satellites) {
    if (signal.status === "BLOCKED") {
      rows.push({id:signal.id,status:signal.status,used:false,excess:null,rangeChange:null,residual:null}); continue;
    }
    if (!["LOS","NLOS"].includes(signal.status) || !vector(signal.source)) return null;
    const excess = signal.status === "LOS" ? 0 : signal.reflection?.extra;
    if ((signal.status === "LOS" && signal.directHit) || (signal.status === "NLOS" && !signal.directHit) || !Number.isFinite(excess) || excess < 0) return null;
    // Match geometricSPP's scene-coordinate norms, not the independently
    // calculated ECEF signal.range, which can differ in the last bits.
    const direct = distance(signal.source,snapshot.receiver);
    if (!(direct > 0) || !Number.isFinite(direct)) return null;
    rows.push({id:signal.id,status:signal.status,used:true,excess,direct,rangeChange:null,residual:null});
  }
  const used = rows.filter(row=>row.used), fitted = snapshot.spp?.all;
  let solution = null;
  if (fitted) {
    if (!vector(fitted.position) || !vector(fitted.offset) || !Number.isFinite(fitted.clock) || !Number.isFinite(fitted.horizontal) || fitted.horizontal < 0 || fitted.satellites !== used.length || used.length < 4) return null;
    for (let i=0;i<rows.length;i++) {
      const row = rows[i]; if (!row.used) continue;
      const modeled = distance(snapshot.satellites[i].source,fitted.position);
      row.rangeChange = modeled-row.direct;
      // Preserve the solver's subtraction order, including the clock sign.
      row.residual = row.direct+row.excess-modeled-fitted.clock;
      if (!Number.isFinite(row.rangeChange) || !Number.isFinite(row.residual)) return null;
    }
    solution = {horizontal:fitted.horizontal,up:fitted.offset[2],clock:fitted.clock,rms:Math.hypot(...used.map(row=>row.residual))/Math.sqrt(used.length)};
  }
  return {rows,used:used.length,solution};
}

const metres = (value,signed=false) => value === null ? "—" : `${signed && value > 0 ? "+" : ""}${value.toFixed(3)} m`;
function readout(node,value,text=metres(value,true)) {
  node.textContent = text;
  if (value === null) { node.removeAttribute("data-value"); node.removeAttribute("title"); }
  else { node.dataset.value = String(value); node.title = `${value} m`; }
}

export class RangeFit {
  constructor(root,onSelect) {
    this.root = root; this.onSelect = onSelect;
    const find = id => root.querySelector(`#${id}`);
    this.label = find("range-label"); this.status = find("range-status"); this.results = find("range-results");
    this.table = find("range-signals"); this.selectedLabel = find("range-selected");
    this.values = Object.fromEntries(["horizontal","up","clock","rms","excess","change","selected-clock","residual"].map(name=>[name,find(`range-${name}`)]));
  }

  update(snapshot,selected,{ready,error}) {
    if (this.snapshot === snapshot && this.selected === selected && this.ready === ready && this.error === error) return;
    const keepRows = this.snapshot === snapshot && this.ready && ready && !this.error && !error;
    if (this.snapshot !== snapshot) this.fit = ready && !error ? rangeFit(snapshot) : null;
    this.snapshot = snapshot; this.selected = selected; this.ready = ready; this.error = error;
    this.results.hidden = true; if (!keepRows) this.table.replaceChildren(); this.selectedLabel.textContent = "";
    for (const key of Object.keys(this.root.dataset)) delete this.root.dataset[key];
    for (const node of Object.values(this.values)) readout(node,null);
    const message = (state,text) => { this.root.dataset.state = state; this.status.textContent = text; this.label.textContent = state === "pending" ? "Updating…" : "Paths → position bias²"; };
    if (error) { message("unavailable","Range fit unavailable · simulation stopped."); return; }
    if (!snapshot) { message("loading","Preparing modeled range diagnostics…"); return; }
    if (!ready) { message("pending","Waiting for the selected pose and receiver settings to finish tracing…"); return; }
    if (!this.fit) this.fit = rangeFit(snapshot);
    const fit = this.fit;
    if (!fit) { message("unavailable","The completed trace has no usable range-fit details."); return; }
    message(fit.solution ? "ready" : "no-solution",fit.solution ? `${fit.used} usable paths · existing ideal-range solution · residuals in metres.` : `${fit.used} usable paths · no position solution. Excess ranges remain available; fit terms do not.`);
    Object.assign(this.root.dataset,{epoch:snapshot.epoch,antenna:snapshot.antenna,mask:snapshot.mask});
    const solution = fit.solution;
    readout(this.values.horizontal,solution?.horizontal ?? null,solution ? `${solution.horizontal.toFixed(2)} m` : "—");
    for (const name of ["up","clock","rms"]) readout(this.values[name],solution?.[name] ?? null,metres(solution?.[name] ?? null,name!=="rms"));
    const chosen = fit.rows.find(row=>row.id===selected);
    this.selectedLabel.textContent = chosen ? `${chosen.id} · ${chosen.status === "BLOCKED" ? "no usable path; excluded from fit" : chosen.status === "LOS" ? "direct arrives first; zero modeled excess" : "shortest reflection; earliest modeled path"}${!solution && chosen.used ? " · fit unavailable" : ""}.` : "Select a satellite to inspect its range terms.";
    if (chosen) this.root.dataset.fitSatellite = chosen.id;
    readout(this.values.excess,chosen?.excess ?? null);
    readout(this.values.change,chosen?.rangeChange ?? null);
    readout(this.values["selected-clock"],chosen?.used && solution ? solution.clock : null);
    readout(this.values.residual,chosen?.residual ?? null);
    if (!keepRows) for (const row of fit.rows) {
      const tr = document.createElement("tr"); tr.dataset.fitRow = row.id; tr.dataset.path = row.status; tr.dataset.selected = String(row.id===selected);
      const heading = document.createElement("th"); heading.scope = "row";
      const button = document.createElement("button"); button.type = "button"; button.dataset.fitSatellite = row.id;
      button.textContent = row.id; button.setAttribute("aria-label",`Inspect ${row.id} range fit, ${row.status}`); button.setAttribute("aria-pressed",String(row.id===selected));
      button.addEventListener("click",()=>this.onSelect(row.id)); heading.append(button); tr.append(heading);
      const path = document.createElement("td"); path.textContent = row.status; tr.append(path);
      for (const name of ["excess","residual"]) { const cell = document.createElement("td"); cell.dataset.term = name; readout(cell,row[name]); tr.append(cell); }
      this.table.append(tr);
    }
    for (const tr of this.table.children) {
      const active = tr.dataset.fitRow === selected;
      tr.dataset.selected = String(active); tr.querySelector("button").setAttribute("aria-pressed",String(active));
    }
    this.results.hidden = false;
  }
}
