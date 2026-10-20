const $ = (s) => document.querySelector(s);
let currentES = null;
let profile = new URLSearchParams(location.search).get("profile") || "priya";
const charts = {};
// Chart.js is loaded from a CDN without SRI (known, accepted for this demo).
const TABS = [["dashboard", "Dashboard"], ["afford", "Can I afford…?"], ["duel", "Decision duel"],
              ["recs", "Recommendations"], ["goals", "Goals"], ["chat", "Chat"]];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}
const money = (x) => "₹" + Math.round(x).toLocaleString("en-IN");
const pct = (x) => Math.round(x * 100) + "%";

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
function fail(e) { $("#err").textContent = e.message; setTimeout(() => ($("#err").textContent = ""), 6000); }

function renderState(st) {
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
      el("div", "", "Verifier (" + (r.verification || "n/a") + "): " + r.verifier_notes),
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
}
async function refresh() {
  renderState(await api(`/api/state/${encodeURIComponent(profile)}`));
  loadDigests().catch(fail);
}

function begin(p) { profile = p.profile; follow(p.run_id, () => { history.replaceState(null, "", `?profile=${encodeURIComponent(p.profile)}`); refresh(); }); }

async function init() {
  TABS.forEach(([id, label]) => { const b = el("button", "", label); b.dataset.t = id; b.onclick = () => show(id); $("#nav").append(b); });
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
