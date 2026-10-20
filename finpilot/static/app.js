const $ = (s) => document.querySelector(s);
let currentES = null;
let profile = new URLSearchParams(location.search).get("profile") || "priya";
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
const { money, pct, fmtDate, weekdayMon, worstLabel, deltaPts } = window.FPU;  // pure helpers: static/util.js
let STATE = null;  // last rendered /api/state (read by the guided demo)
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
  const palette = ["#5eead4", "#60a5fa", "#a78bfa", "#f472b6", "#fbbf24", "#fb923c", "#34d399", "#f87171", "#94a3b8"];
  chart("cSpend", { type: "bar", data: { labels: st.monthly.months,
    datasets: Object.entries(st.monthly.series).filter(([c]) => c !== "income").map(([c, v], i) =>
      ({ label: c, data: v, backgroundColor: palette[i % palette.length] })) },
    options: { scales: { x: { stacked: true }, y: { stacked: true } }, plugins: { title: { display: true, text: "Monthly spend by category" } } } });
  chart("cFore", { type: "line", data: { labels: f.bands.dates, datasets: [
    { label: "p90", data: f.bands.p90, borderColor: "#334155", pointRadius: 0, fill: false },
    { label: "p50", data: f.bands.p50, borderColor: "#5eead4", pointRadius: 0, fill: false },
    { label: "p10", data: f.bands.p10, borderColor: "#f87171", pointRadius: 0, fill: "-2", backgroundColor: "#5eead41a" }] },
    options: { plugins: { title: { display: true, text: "Balance forecast (p10 / p50 / p90)" } } } });
  const fl = $("#findings"); fl.replaceChildren();
  st.findings.forEach((x) => { const c = el("div", "card"); c.append(el("span", "tag", x.type), el("span", "", x.summary)); fl.append(c); });
  const rl = $("#recList"); rl.replaceChildren();
  st.recs.forEach((r) => {
    const c = el("div", "card");
    c.append(el("span", "tag " + (r.verified ? "ok" : "bad"), r.verified ? "verified" : "unverified"),
             el("span", "tag", r.trigger_type), el("div", "", r.text));
    const d = el("details"); d.append(el("summary", "", "Why?"),
      el("div", "", "Triggered by: " + r.finding_ids.join(", ")),
      el("div", "vnote", "Verifier (" + (r.verification || "n/a") + "): " + r.verifier_notes),
      el("pre", "", JSON.stringify(r.evidence, null, 1)));
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
async function loadDigests() {
  const box = $("#digests"); box.replaceChildren();
  (await api(`/api/digests/${encodeURIComponent(profile)}`)).slice(0, 5).forEach((d) => {
    const c = el("div", "card");
    c.append(el("small", "", `${d.created_at} · ${d.channel}${d.sent ? " (emailed)" : ""}`),
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
  $("#wLifeLbl").textContent = `Lifestyle spending ${Math.round(W.lifestyle * 100)}%`;
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
  (d.scenario_flows || []).forEach((f) => {
    const verb = f.amount < 0 ? "−" : "+";
    fl.append(el("span", "tag", `planned: ${f.label} ${verb}${money(Math.abs(f.amount))} on ${fmtDate(f.date)}`));
  });
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
  d.calendar.forEach((c) => {
    if (col === 7) { col = 0; newRow(); }
    const cell = el("div", "day " + c.label);
    cell.setAttribute("role", "gridcell");
    cell.tabIndex = c.date === keep ? 0 : -1;  // roving tabindex: the grid is one tab stop
    cell.dataset.date = c.date;
    const t = dayTitle(c, d.buffer);
    cell.title = t; cell.setAttribute("aria-label", t);
    cell.append(el("small", "", String(+c.date.slice(8))), el("span", "g", LABELS[c.label][0]), el("b", "", pct(c.p_below_buffer)));
    row.append(cell); col++;
  });
  $("#wDay").replaceChildren();
  highlightStorm(W.hl, false);
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
  $("#wDay").textContent = `${fmtDate(c.date)} · ${LABELS[c.label][1]} · ${pct(c.p_below_buffer)} risk · balance p10 ${money(c.p10)} to p90 ${money(c.p90)} (median ${money(c.p50)})`;
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
      const planned = /^planned:/i.test(b.merchant);
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
    $("#wLifeLbl").textContent = `Lifestyle spending ${sl.value}%`;
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

function begin(p) { profile = p.profile; follow(p.run_id, () => { history.replaceState(null, "", `?profile=${encodeURIComponent(p.profile)}`); refresh(); }); }

async function init() {
  weatherInit();
  TABS.forEach(([id, label]) => { const b = el("button", "", label); b.dataset.t = id; b.onclick = () => { show(id); if (id === "weather") weatherEnter(); }; $("#nav").append(b); });
  show("dashboard");
  (await api("/api/personas")).forEach((p) => $("#persona").append(new Option(p, p)));
  $("#loadBtn").onclick = () => api(`/api/load/${$("#persona").value}`, {}).then(begin).catch(fail);
  $("#upForm").onsubmit = (e) => {
    e.preventDefault();
    const fd = new FormData(e.target);  // includes the profile-name field (validated server side)
    api("/api/upload", null, fd).then(begin).catch(fail);
  };
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
      if (a.notice) o.append(el("div", "card", "Notice: " + a.notice));
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
      const a = r.answer; $("#duOut").replaceChildren(el("div", "card", a.narrative));
      if (a.notice) $("#duOut").append(el("div", "card", "Notice: " + a.notice));
      [["cDuelA", a.a], ["cDuelB", a.b]].forEach(([id, side]) => chart(id, { type: "line",
        data: { labels: side.bands.dates, datasets: [
          { label: "p90", data: side.bands.p90, borderColor: "#334155", pointRadius: 0 },
          { label: "p50", data: side.bands.p50, borderColor: "#5eead4", pointRadius: 0 },
          { label: "p10", data: side.bands.p10, borderColor: "#f87171", pointRadius: 0, fill: "-2", backgroundColor: "#5eead41a" }] },
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
  refresh().catch(fail);
}
init();
