const $ = (s) => document.querySelector(s);
let currentES = null;
const QS = new URLSearchParams(location.search);
let profile = /^[A-Za-z0-9_-]{1,40}$/.test(QS.get("profile") || "") ? QS.get("profile") : "priya";
const charts = {};
// Chart.js is loaded from a CDN without SRI (known, accepted for this demo).
const TABS = [["dashboard", "Dashboard"], ["weather", "Money Weather"], ["afford", "Can I afford…?"], ["duel", "Decision duel"],
              ["recs", "Recommendations"], ["goals", "Goals"], ["chat", "Chat"]];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const { money, pct, fmtDate, weekdayMon, worstLabel, deltaPts, fmtStamp, pick } = window.FPU;  // pure helpers: static/util.js
let STATE = null;  // last rendered /api/state (read by the guided demo)
const DARK = !!(window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches);
const C = DARK ? { acc: "#7c93ff", bad: "#ff8a80", mute: "#7f8fae", text: "#c4cee3", grid: "#27385a" }
               : { acc: "#2b4eff", bad: "#c8372d", mute: "#9aa7bd", text: "#44516b", grid: "#e5ebf3" };
if (window.Chart) { Chart.defaults.color = C.text; Chart.defaults.borderColor = C.grid; }
const emit = (name, detail) => document.dispatchEvent(new CustomEvent(name, { detail }));

async function api(path, body, form) {
  const opt = form ? { method: "POST", body: form }
    : body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || r.statusText);
  return data;
}
function chart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
}
function show(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("on", t.id === name));
  document.querySelectorAll("#nav button").forEach((b) => b.classList.toggle("on", b.dataset.t === name));
}
function addEvent(e) {
  const d = el("div", "ev " + e.kind);
  d.append(el("small", "", e.agent + " · " + e.kind), el("div", "", e.summary));
  const box = $("#trace");
  box.append(d);
  box.scrollTop = box.scrollHeight;
}
function follow(runId, then) {
  $("#trace").replaceChildren();
  if (currentES) currentES.close();
  const es = new EventSource(`/trace/${encodeURIComponent(runId)}/stream`);
  currentES = es;
  es.addEventListener("trace", (m) => {
    const e = JSON.parse(m.data);
    addEvent(e);
    if (e.kind === "done") { es.close(); if (then) then(); }
  });
  es.onerror = () => {
    es.close();
    addEvent({ agent: "client", kind: "error", summary: "trace stream disconnected" });
    if (then) then();
  };
}
function fail(e) {
  $("#err").textContent = e.message; setTimeout(() => ($("#err").textContent = ""), 6000);
  emit("fp:error", { message: e.message });
}

function renderState(st) {
  STATE = st;
  if (!st.has_data || !st.forecast) {
    $("#kpis").replaceChildren(); $("#findings").replaceChildren(); $("#recList").replaceChildren();
    $("#goList").replaceChildren(); $("#narrative").textContent = "";
    ["cSpend", "cFore"].forEach((id) => { if (charts[id]) { charts[id].destroy(); delete charts[id]; } });
    $("#overview").textContent = st.has_data ? "Analysis is still running or failed; see the trace."
      : "Load a persona or upload a CSV to begin.";
    return;
  }
  const f = st.forecast;
  const k = $("#kpis"); k.replaceChildren();
  [[money(st.balance), "Current balance"], [money(f.summary.end_p50), `Median in ${f.horizon} days`],
   [pct(f.summary.p_overdraft), "Overdraft risk"], [f.summary.risk_day || "none", "First risk day"]].forEach(([v, l]) => {
    const d = el("div", "kpi"); d.append(el("b", "", v), el("span", "", l)); k.append(d);
  });
  $("#overview").textContent = st.overview;
  $("#narrative").textContent = f.narrative;
  renderHero(st.profile || profile);
  const palette = ["#2b4eff", "#e0803a", "#1f8a5b", "#c8372d", "#8a63d2", "#d4a017", "#2aa7b8", "#d6568c", "#8793a8"];
  chart("cSpend", { type: "bar", data: { labels: st.monthly.months,
    datasets: Object.entries(st.monthly.series).filter(([c]) => c !== "income").map(([c, v], i) =>
      ({ label: c, data: v, backgroundColor: palette[i % palette.length] })) },
    options: { scales: { x: { stacked: true }, y: { stacked: true } }, plugins: { title: { display: true, text: "Monthly spend by category" } } } });
  chart("cFore", { type: "line", data: { labels: f.bands.dates, datasets: [
    { label: "p90", data: f.bands.p90, borderColor: C.mute, pointRadius: 0, fill: false },
    { label: "p50", data: f.bands.p50, borderColor: C.acc, pointRadius: 0, fill: false },
    { label: "p10", data: f.bands.p10, borderColor: C.bad, pointRadius: 0, fill: "-2", backgroundColor: C.acc + "22" }] },
    options: { plugins: { title: { display: true, text: "Balance forecast (p10 / p50 / p90)" } } } });
  const fl = $("#findings"); fl.replaceChildren();
  st.findings.forEach((x) => { const c = el("div", "card"); c.append(el("span", "tag", kind(x.type)), el("span", "", x.summary)); fl.append(c); });
  const rl = $("#recList"); rl.replaceChildren();
  st.recs.forEach((r) => {
    const c = el("div", "card");
    c.append(el("span", "tag " + (r.verified ? "ok" : "bad"), r.verified ? "Numbers checked" : "Not checked"),
             el("span", "tag", kind(r.trigger_type)), el("div", "", r.text));
    const byId = Object.fromEntries(st.findings.map((x) => [x.id, x.summary]));
    const basis = r.finding_ids.map((i) => byId[i]).filter(Boolean);
    const d = el("details"); d.append(el("summary", "", "Why?"));
    if (basis.length) { d.append(el("div", "", "Based on:")); basis.forEach((t) => d.append(el("div", "ev-row", t))); }
    d.append(el("div", "vnote", (r.verification === "recomputed" ? "Re-calculated from your transactions. " : "Matches the analysis figures, not re-calculated. ") + (/^all claims recomputed/i.test(r.verifier_notes || "") ? "" : (r.verifier_notes || ""))),
      el("div", "", "Figures used:"), evidenceRows(r.evidence));
    c.append(d); rl.append(c);
  });
  const gl = $("#goList"); gl.replaceChildren();
  (f.goals || []).forEach((g) => {
    const c = el("div", "card");
    if (g.error) { c.append(el("div", "", `${g.name}: ${g.error}`)); gl.append(c); return; }
    c.append(el("b", "", g.name), el("div", "", `Target ${money(g.target)} by ${g.deadline}: ${pct(g.p_met)} chance` +
      (g.eta ? `, median ETA ${g.eta}` : ", not reached within a year")));
    gl.append(c);
  });
}
let heroSeq = 0;
async function renderHero(p) {
  const seq = ++heroSeq, head = $("#heroHead");
  try {
    const w = await api(`/api/weather/${encodeURIComponent(p)}`);
    if (seq !== heroSeq) return;
    const cal = w.calendar, worst = FPU.worstLabel(cal), first = cal.find((c) => c.label === worst);
    head.dataset.w = worst;
    head.textContent = { clear: `Clear skies for the next ${w.horizon} days`, cloudy: "Mostly clear, with some cloud ahead",
      rain: `Rain likely from ${FPU.fmtDate(first.date)}`, storm: `Storm ahead around ${FPU.fmtDate(first.date)}` }[worst];
    $("#heroSub").textContent = w.headline_text;
    const strip = $("#heroStrip"); strip.replaceChildren();
    strip.style.gridTemplateColumns = `repeat(${cal.length},1fr)`;
    cal.forEach((c) => { const i = el("i", c.label + (worst !== "clear" && c === first ? " first" : "")); i.title = `${FPU.fmtDate(c.date)}: ${FPU.pct(c.p_below_buffer)} chance below buffer`; strip.append(i); });
    $("#stripFrom").textContent = FPU.fmtDate(cal[0].date); $("#stripTo").textContent = FPU.fmtDate(cal[cal.length - 1].date);
    $("#heroGo").onclick = () => { show("weather"); weatherEnter(); };
  } catch (e) { head.textContent = "Forecast"; $("#heroSub").textContent = ""; $("#heroStrip").replaceChildren(); }
}
async function loadDigests() {
  const box = $("#digests"); box.replaceChildren();
  (await api(`/api/digests/${encodeURIComponent(profile)}`)).slice(0, 5).forEach((d) => {
    const c = el("div", "card");
    c.append(el("small", "", `${fmtStamp(d.created_at)} · ${d.channel}${d.sent ? " (emailed)" : ""}`),
             el("pre", "", d.content));
    box.append(c);
  });
  emit("fp:digest");
}
async function refresh() {
  renderState(await api(`/api/state/${encodeURIComponent(profile)}`));
  weatherInvalidate();
  loadDigests().catch(fail);
  emit("fp:rendered");
}

// ---------------------------------------------------------------- Money Weather
const W = { scenario: "none", lifestyle: 1.0, cuts: {}, seq: 0, timer: null, loaded: false, data: null, hl: null,
            shown: null, activeDate: null };
const SCENARIOS = [["none", "Normal 60 days"], ["salary_skip", "Job-loss drill"],
                   ["purchase", "Big purchase"], ["emergency", "Medical emergency"]];
const LABELS = { clear: ["☀", "Clear skies"], cloudy: ["⛅", "Cloudy"], rain: ["🌧", "Rain possible"], storm: ["⛈", "Storm ahead"] };

// A scenario flow in words; a skipped salary is missing income, never an outflow.
const isSalary = (f) => /salary/i.test(f.label);
function flowText(f) {
  if (isSalary(f)) return `missing salary ${money(Math.abs(f.amount))}`;
  return `${f.label} ${f.amount < 0 ? "−" : "+"}${money(Math.abs(f.amount))}`;
}
function marker(cls, glyph, label) {
  const m = el("span", "mk " + cls, glyph);
  m.setAttribute("role", "img"); m.setAttribute("aria-label", label); m.title = label;
  return m;
}
function dayTitle(c, buffer) {
  return `${fmtDate(c.date)}: ${pct(c.p_below_buffer)} chance of dropping below your safety buffer (${money(buffer)}); ` +
    `balance p10 ${money(c.p10)}, median ${money(c.p50)}, p90 ${money(c.p90)}`;
}
function weatherQueryString() {
  const q = new URLSearchParams({ lifestyle: String(W.lifestyle), horizon: "60", scenario: W.scenario });
  const amt = $("#wAmount").value.trim();
  if (amt && W.scenario !== "none" && W.scenario !== "salary_skip") q.set("amount", amt);
  if (Object.keys(W.cuts).length) q.set("cuts", JSON.stringify(W.cuts));
  return q.toString();
}
function weatherSetErr(msg) {
  $("#wErr").textContent = msg || "";
  if (msg) $("#wAmount").setAttribute("aria-invalid", "true"); else $("#wAmount").removeAttribute("aria-invalid");
}
// Pressed/disabled/slider state always mirrors W so the controls never contradict what is on screen.
function weatherSyncControls() {
  document.querySelectorAll("#wScen button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.s === W.scenario)));
  $("#wAmount").disabled = !(W.scenario === "purchase" || W.scenario === "emergency");
  $("#wLife").value = String(Math.round(W.lifestyle * 100));
  $("#wLifeLbl").textContent = `Non-bill spending ${Math.round(W.lifestyle * 100)}%`;
}
async function weatherQuery() {
  clearTimeout(W.timer);
  const mine = ++W.seq;  // request counter: only the newest response may render
  $("#wBusy").textContent = "Updating…";
  $("#weather").setAttribute("aria-busy", "true");
  try {
    const d = await api(`/api/weather/${encodeURIComponent(profile)}?${weatherQueryString()}`);
    if (mine !== W.seq) return;
    W.data = d; W.loaded = true;
    W.shown = { scenario: W.scenario, lifestyle: W.lifestyle, cuts: { ...W.cuts } };
    weatherSetErr("");
    follow(d.run_id);
    weatherRender(d);
    emit("fp:weather");
  } catch (e) {
    if (mine !== W.seq) return;
    if (W.data && W.shown) {
      // keep the previous render; roll the controls back to what is actually displayed
      W.scenario = W.shown.scenario; W.lifestyle = W.shown.lifestyle; W.cuts = { ...W.shown.cuts };
      weatherSyncControls();
      renderUmbrellas(W.data);
      weatherSetErr(e.message);
    } else {
      W.loaded = false;
      weatherEmpty(/no data/i.test(e.message) ? "Load a persona first." : "Could not load the weather: " + e.message);
      weatherSetErr(e.message);
    }
    emit("fp:error", { message: e.message });
  } finally {
    if (mine === W.seq) { $("#wBusy").textContent = ""; $("#weather").removeAttribute("aria-busy"); }
  }
}
function weatherEmpty(msg) {
  $("#wMsg").textContent = msg;
  ["#wHeadline", "#wFlows", "#wCal", "#wStorms", "#wAutopsy", "#wUmbrellas", "#wApplied", "#wRisk", "#wDay"].forEach((s) => $(s).replaceChildren());
  $("#wChip").replaceChildren(); $("#wChip").className = "";
  $("#wBody").hidden = true;
}
// A different persona was loaded: nothing from the old one may survive, including in-flight responses.
function weatherInvalidate() {
  clearTimeout(W.timer);
  W.seq++;  // stale responses from the old persona are dropped
  W.loaded = false; W.data = null; W.shown = null; W.cuts = {}; W.hl = null; W.activeDate = null;
  W.scenario = "none"; W.lifestyle = 1.0;
  $("#wAmount").value = "";
  weatherSetErr("");
  weatherSyncControls();
  $("#wBusy").textContent = ""; $("#weather").removeAttribute("aria-busy");
  weatherEmpty("Load a persona first.");
  if ($("#weather").classList.contains("on")) weatherQuery().catch(fail);
}
function weatherEnter() { if (!W.loaded) weatherQuery().catch(fail); }

function weatherRender(d) {
  $("#wMsg").textContent = "";
  $("#wBody").hidden = false;
  renderWeatherHeadline(d);
  renderCalendar(d);
  renderStorms(d);
  // keep the storm highlight only if the new answer still has a storm with the same start
  highlightStorm(W.hl ? (d.storms.find((s) => s.start === W.hl.start) || null) : null, false);
  renderAutopsy(d);
  renderUmbrellas(d);
  if (d.scenario_amount != null && (d.scenario === "purchase" || d.scenario === "emergency") && !$("#wAmount").value) {
    $("#wAmount").value = Math.round(d.scenario_amount);
  }
  weatherSyncControls();
}
function renderWeatherHeadline(d) {
  const worst = worstLabel(d.calendar);
  const [glyph, text] = LABELS[worst];
  const chip = $("#wChip"); chip.replaceChildren(el("span", "", glyph + " "), el("span", "", text));
  chip.className = "chip " + worst;
  $("#wHeadline").replaceChildren(el("p", "", d.headline_text));
  $("#wRisk").textContent = "Chance of dipping below your safety buffer: " + pct(d.headline.p_any_breach);
  const fl = $("#wFlows"); fl.replaceChildren();
  (d.scenario_flows || []).forEach((f) => fl.append(el("span", "tag", `${flowText(f)} on ${fmtDate(f.date)}`)));
  if (d.scenario_note) fl.append(el("div", "wnote", d.scenario_note));
}
function renderCalendar(d) {
  const g = $("#wCal"); g.replaceChildren();
  const hdr = el("div", "crow"); hdr.setAttribute("role", "row");
  ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"].forEach((n) => {
    const h = el("div", "dow", n); h.setAttribute("role", "columnheader"); hdr.append(h);
  });
  g.append(hdr);
  if (!d.calendar.length) return;
  const keep = d.calendar.some((c) => c.date === W.activeDate) ? W.activeDate : d.calendar[0].date;
  W.activeDate = keep;
  let row = null, col = weekdayMon(d.calendar[0].date);
  const newRow = () => { row = el("div", "crow"); row.setAttribute("role", "row"); g.append(row); };
  newRow();
  for (let i = 0; i < col; i++) { const e = el("div", "day empty"); e.setAttribute("role", "gridcell"); e.setAttribute("aria-hidden", "true"); row.append(e); }
  const flows = {};
  (d.scenario_flows || []).forEach((f) => { (flows[f.date] = flows[f.date] || []).push(f); });
  const pays = new Set(d.paydays || []);
  d.calendar.forEach((c, i) => {
    if (col === 7) { col = 0; newRow(); }
    const cell = el("div", "day " + c.label);
    cell.setAttribute("role", "gridcell");
    cell.tabIndex = c.date === keep ? 0 : -1;  // roving tabindex: the grid is one tab stop
    cell.dataset.date = c.date;
    const mstart = i === 0 || c.date.slice(8) === "01";  // month marker: "Oct 1" instead of "1"
    if (mstart) cell.classList.add("mstart");
    const top = el("div", "dtop"), notes = [];
    top.append(el("small", "", mstart ? fmtDate(c.date) : String(+c.date.slice(8))));
    (flows[c.date] || []).forEach((f) => {
      const t = flowText(f); notes.push(t);
      top.append(isSalary(f) ? marker("miss", "₹✕", t) : marker("out", "₹−", t));
    });
    if (pays.has(c.date)) { notes.push("payday"); top.append(marker("pay", "₹+", "payday")); }
    const t = dayTitle(c, d.buffer) + (notes.length ? "; " + notes.join("; ") : "");
    cell.title = t; cell.setAttribute("aria-label", t);
    cell.append(top, el("span", "g", LABELS[c.label][0]), el("b", "", pct(c.p_below_buffer)));
    row.append(cell); col++;
  });
  $("#wDay").replaceChildren();
}
function calCells() { return Array.from(document.querySelectorAll("#wCal .day[data-date]")); }
function calFocus(cell) {
  calCells().forEach((c) => { c.tabIndex = c === cell ? 0 : -1; });
  W.activeDate = cell.dataset.date;
  cell.focus();
}
function calDetail(cell) {
  const c = W.data && W.data.calendar.find((x) => x.date === cell.dataset.date);
  if (!c) return;
  const extra = (W.data.scenario_flows || []).filter((f) => f.date === c.date).map(flowText)
    .concat((W.data.paydays || []).includes(c.date) ? ["payday"] : []);
  $("#wDay").textContent = `${fmtDate(c.date)} · ${LABELS[c.label][1]} · ${pct(c.p_below_buffer)} risk · balance p10 ${money(c.p10)} to p90 ${money(c.p90)} (median ${money(c.p50)})` +
    (extra.length ? " · " + extra.join(" · ") : "");
}
function calKey(e) {
  const cell = e.target.closest && e.target.closest(".day[data-date]");
  if (!cell) return;
  const cells = calCells(), i = cells.indexOf(cell);
  const move = { ArrowLeft: -1, ArrowRight: 1, ArrowUp: -7, ArrowDown: 7 }[e.key];
  let j = i;
  if (move !== undefined) j = Math.min(cells.length - 1, Math.max(0, i + move));
  else if (e.key === "Home") j = 0;
  else if (e.key === "End") j = cells.length - 1;
  else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); calDetail(cell); return; }
  else return;
  e.preventDefault();
  calFocus(cells[j]);
}
function highlightStorm(st, scroll) {
  W.hl = st;
  document.querySelectorAll("#wCal .day[data-date]").forEach((c) => {
    c.classList.toggle("hl", !!st && c.dataset.date >= st.start && c.dataset.date <= st.end);
  });
  document.querySelectorAll("#wStorms .storm").forEach((c) => {
    const on = !!st && c.dataset.start === st.start;
    c.classList.toggle("sel", on); c.setAttribute("aria-pressed", String(on));
  });
  if (st && scroll) $("#wAutopsyBox").scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
}
function renderStorms(d) {
  const box = $("#wStorms"); box.replaceChildren();
  if (!d.storms.length) { box.append(el("div", "card", "No storms: no run of days at 25% risk or more.")); return; }
  d.storms.forEach((st) => {
    const c = el("div", "card storm " + st.label);
    c.tabIndex = 0; c.setAttribute("role", "button"); c.setAttribute("aria-pressed", "false"); c.dataset.start = st.start;
    c.append(el("b", "", `${LABELS[st.label][0]} ${fmtDate(st.start)} – ${fmtDate(st.end)}`),
             el("div", "", `${st.days} days, peak ${pct(st.peak_p)} on ${fmtDate(st.peak_date)} (${LABELS[st.label][1]})`));
    const pick = () => highlightStorm(W.hl && W.hl.start === st.start ? null : st, true);
    c.onclick = pick;
    c.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } };
    box.append(c);
  });
}
function renderAutopsy(d) {
  const a = d.autopsy, box = $("#wAutopsy"); box.replaceChildren();
  box.append(el("p", "", `${a.n_failing} of ${a.n_paths} simulated futures dip below the buffer (${pct(a.p_any_breach)}).`),
             el("p", "", d.autopsy_text));
  const fb = a.first_breach || {};
  if (fb.already_below) box.append(el("div", "wnote", "You start below the safety buffer."));
  else if (fb.date) box.append(el("div", "", `Most common first dip: ${fmtDate(fb.date)} (${pct(fb.share || 0)} of failing futures)`));
  if (a.insufficient_failing && a.n_failing > 0) box.append(el("div", "wnote", "Too few failing futures to rank causes reliably."));
  if ((a.causes || []).length) {
    box.append(el("h4", "", "What drives the failing futures"));
    a.causes.forEach((c) => {
      const row = el("div", "bar"), track = el("div", "track"), fill = el("div", "fill");
      fill.style.width = Math.min(100, Math.max(0, c.share_of_gap * 100)) + "%";
      track.append(fill);
      row.append(el("span", "bl", c.category.replace(/_/g, " ")), track, el("span", "bv", pct(c.share_of_gap)));
      row.title = `${money(c.excess_per_path)} more spend per failing path (${money(c.failing_mean)} vs ${money(c.other_mean)})`;
      box.append(row);
    });
  }
  if ((a.bills || []).length) {
    box.append(el("h4", "", "Outflows before the first dip"));
    a.bills.forEach((b) => {
      const planned = /^planned:/i.test(b.merchant);  // "planned purchase" (no colon) already says it
      const r = el("div", "bill");
      r.append(el("span", "", fmtDate(b.date)), el("span", "", b.merchant.replace(/^planned:\s*/i, "")),
               el("span", "", money(Math.abs(b.amount))));
      if (planned) r.insertBefore(el("span", "tag", "planned"), r.children[1]);
      box.append(r);
    });
  }
}
function renderUmbrellas(d) {
  const box = $("#wUmbrellas"); box.replaceChildren();
  box.append(el("p", "", d.umbrella_text));
  (d.umbrellas || []).forEach((u) => {
    const c = el("div", "card");
    const total = u.cumulative_cut_pct != null ? u.cumulative_cut_pct : u.cut_pct;
    c.append(el("b", "", `Cut ${u.category.replace(/_/g, " ")} by ${Math.round(total)}%`),
             el("div", "", `Risk ${pct(u.p_any_breach_before)} → ${pct(u.p_any_breach_after)}` +
               (deltaPts(u.p_any_breach_before, u.p_any_breach_after) ? ` (${deltaPts(u.p_any_breach_before, u.p_any_breach_after)})` : "") +
               `, costs about ${money(u.monthly_sacrifice)}/month`));
    [["before", u.p_any_breach_before], ["after", u.p_any_breach_after]].forEach(([k, v]) => {
      const row = el("div", "bar"), track = el("div", "track"), fill = el("div", "fill " + k);
      fill.style.width = Math.min(100, v * 100) + "%"; track.append(fill);
      row.append(el("span", "bl", k), track, el("span", "bv", pct(v))); c.append(row);
    });
    const btn = el("button", "act", "Apply to the map"); btn.type = "button";
    btn.onclick = () => { W.cuts[u.category] = total / 100; weatherQuery().catch(fail); };
    c.append(btn); box.append(c);
  });
  const ap = $("#wApplied"); ap.replaceChildren();
  const cats = Object.keys(W.cuts);
  if (cats.length) {
    ap.append(el("small", "", "Applied cuts: "));
    cats.forEach((k) => {
      const t = el("button", "tag rm", `${k.replace(/_/g, " ")} −${Math.round(W.cuts[k] * 100)}% ✕`);
      t.type = "button"; t.title = "Remove this cut"; t.setAttribute("aria-label", "Remove cut " + k);
      t.onclick = () => { delete W.cuts[k]; weatherQuery().catch(fail); };
      ap.append(t);
    });
    const clr = el("button", "act", "Clear cuts"); clr.type = "button";
    clr.onclick = () => { W.cuts = {}; weatherQuery().catch(fail); };
    ap.append(clr);
  }
}
function weatherInit() {
  SCENARIOS.forEach(([id, label]) => {
    const b = el("button", "chipbtn", label); b.type = "button"; b.dataset.s = id; b.setAttribute("aria-pressed", String(id === W.scenario));
    b.onclick = () => { W.scenario = id; $("#wAmount").value = ""; weatherSetErr(""); weatherSyncControls(); weatherQuery().catch(fail); };
    $("#wScen").append(b);
  });
  $("#wAmount").disabled = true;
  $("#wCal").addEventListener("keydown", calKey);
  $("#wCal").addEventListener("click", (e) => {
    const cell = e.target.closest(".day[data-date]");
    if (cell) { calFocus(cell); calDetail(cell); }
  });
  $("#wAmount").onchange = () => weatherQuery().catch(fail);
  const sl = $("#wLife");
  sl.oninput = () => {
    W.lifestyle = +sl.value / 100;
    $("#wLifeLbl").textContent = `Non-bill spending ${sl.value}%`;
    clearTimeout(W.timer);
    W.timer = setTimeout(() => weatherQuery().catch(fail), 300);
  };
  $("#wReset").onclick = () => {
    W.scenario = "none"; W.lifestyle = 1.0; W.cuts = {}; W.hl = null;
    $("#wAmount").value = ""; weatherSetErr("");
    weatherSyncControls();
    weatherQuery().catch(fail);
  };
  $("#wRefresh").onclick = () => weatherQuery().catch(fail);
}

function begin(p) {
  profile = p.profile;
  if (Array.from($("#persona").options).some((o) => o.value === p.profile)) $("#persona").value = p.profile;
  follow(p.run_id, () => { history.replaceState(null, "", `?profile=${encodeURIComponent(p.profile)}`); refresh(); });
}

const pl = (n, one, many) => `${n} ${n === 1 ? one : (many || one + "s")}`;
const KIND = { duplicate_charge: "Possible double charge", large_charge: "Unusually large charge", category_spike: "Spending spike",
  unused_subscription: "Unused subscription", subscription: "Subscription", behavior_trigger: "Payday habit",
  storm_warning: "Storm warning", forecast_risk: "Cash-flow risk", goal_behind: "Goal at risk" };
const kind = (t) => KIND[t] || String(t).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
const label = (k) => String(k).replace(/_/g, " ");
const HIDE = new Set(["id", "type", "summary", "txn_id", "cid", "ids", "finding_id"]);
const FIELD = { value: ["Amount", "₹"], baseline: ["Your usual", "₹"], excess: ["Extra spend", "₹"], pct_change: ["Change", "%"],
  amount: ["Amount", "₹"], monthly_cost: ["Monthly cost", "₹"], annual_cost: ["Yearly cost", "₹"], buffer: ["Safety buffer", "₹"],
  p_overdraft: ["Chance of overdraft", "p"], p_any_breach: ["Chance of dipping below buffer", "p"], p_met: ["Chance of meeting goal", "p"],
  share_pct: ["Share of spending", "%"], ratio: ["Times normal rate", "x"], horizon: ["Days looked ahead", ""], window_days: ["Days after payday", ""],
  merchant: ["Merchant", ""], category: ["Category", ""], month: ["Month", ""], date: ["Date", ""], risk_day: ["First risk day", ""], detail: ["Reason", ""] };
function evidenceRows(ev) {
  const box = el("div", "ev-list");
  Object.entries(ev || {}).forEach(([k, v]) => {
    if (HIDE.has(k) || v == null || (typeof v === "object" && !Object.keys(v).length)) return;
    const [name, unit] = FIELD[k] || [label(k).replace(/^./, (c) => c.toUpperCase()), ""];
    let text;
    if (typeof v === "number") {
      text = unit === "₹" ? inr(v) : unit === "p" ? `${Math.round(v * 100)}%` : unit === "%" ? `${Math.round(v)}%`
        : unit === "x" ? `${v.toFixed(1)}×` : v.toLocaleString("en-IN", { maximumFractionDigits: 2 });
    } else if (typeof v === "object") return;  // nested engine detail is not user-facing
    else text = String(v).replace(/_/g, " ");
    const row = el("div", "ev-row"); row.append(el("span", "", name), el("b", "", text)); box.append(row);
  });
  return box;
}
const inr = (n) => "₹" + Math.round(n).toLocaleString("en-IN");
function importInit() {
  const dlg = $("#importDlg"), file = $("#impFile");
  let token = null, opening = null;
  const step = (n) => { $("#impStep1").hidden = n !== 1; $("#impStep2").hidden = n !== 2; $("#impBusy").hidden = n !== 0; };
  const reset = () => { token = null; file.value = ""; $("#impErr").textContent = ""; step(1); };
  $("#importBtn").onclick = () => { reset(); dlg.showModal(); };
  $("#impClose").onclick = () => dlg.close();
  $("#impBack").onclick = reset;
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files.length) { file.files = e.dataTransfer.files; file.onchange(); } });
  file.onchange = () => {
    if (!file.files.length) return;
    const f = file.files[0], big = /\.(pdf|png|jpe?g|tiff?|bmp|webp)$/i.test(f.name);
    $("#impBusyMsg").textContent = big ? "Reading and running OCR where needed… (first run can take ~10 s)" : "Reading statement…";
    $("#impErr").textContent = ""; step(0);
    const fd = new FormData(); fd.append("file", f);
    api("/api/import/preview", null, fd).then((r) => { token = r.token; opening = r.opening_balance; showPreview(r); step(2); })
      .catch((e) => { $("#impErr").textContent = e.message; step(1); file.value = ""; });
  };
  function showPreview(r) {
    const k = $("#impKpis"); k.replaceChildren();
    [[r.rows, "transactions"], [`${r.from} → ${r.to}`, `${r.days} days`], [inr(r.money_in), "money in"], [inr(r.money_out), "money out"]]
      .forEach(([v, l]) => { const d = el("div", "kpi"); d.append(el("b", "", String(v)), el("span", "", l)); k.append(d); });
    const n = $("#impNotes"); n.replaceChildren();
    const piiN = Object.values(r.pii || {}).reduce((a, b) => a + b, 0);
    n.append(el("div", "nt good", piiN ? `🔒 ${pl(piiN, "personal identifier")} masked: ` + Object.entries(r.pii).map(([a, b]) => `${b} ${a.replace(/_/g, " ")}`).join(", ") : "🔒 No personal identifiers found in descriptions."));
    n.append(el("div", "nt " + (r.opening_balance == null ? "warn" : "good"),
      r.opening_balance == null ? "No balance column found: enter your opening balance below, otherwise 0 is assumed and forecasts will be off."
        : `Opening balance ${inr(r.opening_balance)} was read from the statement's balance column.`));
    if (r.days < 60) n.append(el("div", "nt warn", `Only ${r.days} days of history: forecasts need roughly 90 days to be reliable.`));
    (r.notes || []).filter((x) => !/^masked |^opening balance|headers \(name/.test(x)).forEach((x) =>
      n.append(el("div", "nt " + (/low:|guess|ambiguous|skipped|unreadable/.test(x) ? "warn" : ""), x)));
    const tb = $("#impTable tbody"); tb.replaceChildren();
    r.sample.forEach((x) => { const tr = el("tr"); tr.append(el("td", "", x.date), el("td", "", x.description), el("td", "r " + (x.amount < 0 ? "neg" : "pos"), x.amount.toLocaleString("en-IN", { minimumFractionDigits: 2 }))); tb.append(tr); });
    if (r.rows > r.sample.length) { const tr = el("tr"); const td = el("td", "", `… and ${r.rows - r.sample.length} more`); td.colSpan = 3; tr.append(td); tb.append(tr); }
    $("#upForm").elements.opening_balance.value = "";
    $("#upForm").elements.opening_balance.placeholder = r.opening_balance == null ? "opening balance ₹" : `opening balance ₹ (${Math.round(r.opening_balance)} from file)`;
  }
  $("#impConfirm").onclick = () => {
    const f = $("#upForm");
    if (!f.elements.profile.checkValidity()) { f.elements.profile.reportValidity(); return; }
    api("/api/import/confirm", { token, profile: f.elements.profile.value, opening_balance: f.elements.opening_balance.value })
      .then((p) => { dlg.close(); begin(p); }).catch((e) => ($("#impErr").textContent = e.message));
  };
}

async function init() {
  weatherInit();
  TABS.forEach(([id, label]) => { const b = el("button", "", label); b.dataset.t = id; b.onclick = () => { show(id); if (id === "weather") weatherEnter(); }; $("#nav").append(b); });
  show("dashboard");
  const personas = await api("/api/personas");
  personas.forEach((p) => $("#persona").append(new Option(p, p)));
  if (personas.includes(profile)) $("#persona").value = profile;  // show the profile actually on screen
  $("#loadBtn").onclick = () => api(`/api/load/${encodeURIComponent($("#persona").value)}`, {}).then(begin).catch(fail);
  importInit();
  $("#digestBtn").onclick = () => api(`/api/digest/${encodeURIComponent(profile)}`, {})
    .then(loadDigests).catch(fail);
  $("#afForm").onsubmit = (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    const body = { amount: +f.amount, financed_months: +f.financed_months || 0 };
    if (f.date) body.date = f.date;
    api(`/api/afford/${encodeURIComponent(profile)}`, body).then((r) => {
      follow(r.run_id);
      const a = r.answer, o = $("#afOut"); o.replaceChildren();
      o.append(el("div", "card", a.narrative), el("div", "card", "Verdict: " + a.verdict));
      if (a.notice && !a.narrative.includes(a.notice)) o.append(el("div", "card", "Notice: " + a.notice));
      a.best_dates.forEach((d) => o.append(el("div", "card", `${d.date}: overdraft risk ${pct(d.p_overdraft)}, median ${money(d.end_p50)}`)));
      emit("fp:afford");
    }).catch(fail);
  };
  $("#duForm").onsubmit = (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    const body = { a: { name: f.an, amount: +f.aa }, b: { name: f.bn, amount: +f.ba, date: f.bd } };
    if (f.ad) body.a.date = f.ad;
    api(`/api/duel/${encodeURIComponent(profile)}`, body).then((r) => {
      follow(r.run_id);
      const a = r.answer;
      // the deterministic computed result always shows; the narration only when it adds something
      $("#duOut").replaceChildren(el("div", "card reason", "Result: " + a.reason));
      if (a.narrative && a.narrative !== a.reason) $("#duOut").append(el("div", "card", a.narrative));
      if (a.notice) $("#duOut").append(el("div", "card", "Notice: " + a.notice));
      [["cDuelA", a.a], ["cDuelB", a.b]].forEach(([id, side]) => chart(id, { type: "line",
        data: { labels: side.bands.dates, datasets: [
          { label: "p90", data: side.bands.p90, borderColor: C.mute, pointRadius: 0 },
          { label: "p50", data: side.bands.p50, borderColor: C.acc, pointRadius: 0 },
          { label: "p10", data: side.bands.p10, borderColor: C.bad, pointRadius: 0, fill: "-2", backgroundColor: C.acc + "22" }] },
        options: { plugins: { title: { display: true,
          text: `${side.name}: overdraft ${pct(side.summary.p_overdraft)}, median ${money(side.summary.end_p50)}` } } } }));
      emit("fp:duel");
    }).catch(fail);
  };
  $("#goForm").onsubmit = (e) => {
    e.preventDefault();
    const f = Object.fromEntries(new FormData(e.target));
    api(`/api/goals/${encodeURIComponent(profile)}`, { name: f.name, target: +f.target, deadline: f.deadline })
      .then(() => api(`/api/rerun/${encodeURIComponent(profile)}`, {})).then(begin).catch(fail);
  };
  $("#chForm").onsubmit = (e) => {
    e.preventDefault();
    const q = new FormData(e.target).get("q");
    api(`/api/chat/${encodeURIComponent(profile)}`, { question: q }).then((r) => {
      follow(r.run_id); $("#chOut").prepend(el("div", "card", "Q: " + q), el("div", "card", r.answer.text));
    }).catch(fail);
  };
  try { await refresh(); } catch (e) { fail(e); return; }
  await deepLink();
}
// ?profile=priya&tab=weather&scenario=purchase[&apply=1]: open a tab (and a weather scenario) on load.
// Values are checked against fixed lists; nothing plays by itself beyond this one query (and one umbrella).
async function deepLink() {
  const tab = pick(QS.get("tab"), TABS.map((t) => t[0]));
  if (!tab) return;
  show(tab);
  if (tab !== "weather") return;
  const sc = pick(QS.get("scenario"), SCENARIOS.map((x) => x[0]));
  if (sc) { W.scenario = sc; weatherSyncControls(); }
  await weatherQuery();
  const u = QS.get("apply") === "1" && W.data && (W.data.umbrellas || [])[0];
  if (u) {
    W.cuts[u.category] = (u.cumulative_cut_pct != null ? u.cumulative_cut_pct : u.cut_pct) / 100;
    await weatherQuery();
  }
}
init();
