// Guided demo: a self-playing tour that drives the REAL controls (clicks, input events, form submits).
// Part 1 is a pure step runner (injected timers, node-testable); part 2 wires it to the page.
(function (root) {
  const ABORT = { aborted: true };

  // Step runner. steps: [{id, dwell, run(ctx)}]. ctx: {sleep, check, skipping, reduced}.
  // Semantics: pause freezes every running sleep (remaining time is kept); skip ends the current step's waits
  // immediately; stop rejects them and nothing further runs; an error thrown by a step stops the tour and is reported.
  function createRunner(o) {
    const T = o.timers, steps = o.steps, cb = o.callbacks || {};
    let state = "idle", paused = false, skip = false, idx = -1;
    const sleepers = new Set();

    function arm(s) { s.t0 = T.now(); s.id = T.setTimeout(() => { sleepers.delete(s); s.resolve(); }, s.left); }
    function sleep(ms) {
      if (state !== "running") return Promise.reject(ABORT);
      if (skip || ms <= 0) return Promise.resolve();
      return new Promise((resolve, reject) => {
        const s = { left: ms, resolve, reject, id: null, t0: 0 };
        sleepers.add(s);
        if (!paused) arm(s);
      });
    }
    function pause() {
      if (state !== "running" || paused) return;
      paused = true;
      sleepers.forEach((s) => { T.clearTimeout(s.id); s.left = Math.max(0, s.left - (T.now() - s.t0)); });
      if (cb.onPause) cb.onPause(true);
    }
    function resume() {
      if (state !== "running" || !paused) return;
      paused = false;
      sleepers.forEach(arm);
      if (cb.onPause) cb.onPause(false);
    }
    function skipStep() {
      if (state !== "running") return;
      skip = true;
      if (paused) { paused = false; if (cb.onPause) cb.onPause(false); }
      const all = Array.from(sleepers); sleepers.clear();
      all.forEach((s) => { T.clearTimeout(s.id); s.resolve(); });
    }
    function end(status, err) {
      if (state !== "running") return;
      state = status;
      const all = Array.from(sleepers); sleepers.clear();
      all.forEach((s) => { T.clearTimeout(s.id); s.reject(ABORT); });
      if (cb.onEnd) cb.onEnd(status, err);
    }
    const ctx = { sleep, check() { if (state !== "running") throw ABORT; }, get skipping() { return skip; },
                  reduced: !!o.reduced };
    async function start() {
      state = "running"; paused = false;
      for (let i = 0; i < steps.length; i++) {
        if (state !== "running") return;
        idx = i; skip = false;
        if (cb.onStep) cb.onStep(i, steps.length, steps[i]);
        try {
          await steps[i].run(ctx);
          ctx.check();
          await sleep(steps[i].dwell * (o.reduced ? (o.reducedFactor || 1.5) : 1));
        } catch (e) {
          if (e === ABORT) return;
          end("error", e);
          return;
        }
      }
      end("done");
    }
    return { start, pause, resume, skip: skipStep, stop: () => end("stopped"),
             get state() { return state; }, get paused() { return paused; }, get index() { return idx; } };
  }

  if (typeof module !== "undefined" && module.exports) { module.exports = { createRunner, ABORT }; return; }

  // ------------------------------------------------------------------ page wiring (browser only)
  const $ = (s) => document.querySelector(s);
  const U = root.FPU;
  const reduced = () => !!(root.matchMedia && root.matchMedia("(prefers-reduced-motion: reduce)").matches);
  const TIMERS = { setTimeout: (f, ms) => root.setTimeout(f, ms), clearTimeout: (i) => root.clearTimeout(i), now: () => Date.now() };
  const pending = new Set();
  let runner = null, hlEl = null, t0 = 0;

  const caption = (txt, isErr) => { const c = $("#tourCap"); c.textContent = txt; c.classList.toggle("err", !!isErr); };
  function highlight(elm) {
    if (hlEl) hlEl.classList.remove("tour-hl");
    hlEl = elm || null;
    if (hlEl) {
      hlEl.classList.add("tour-hl");
      hlEl.scrollIntoView({ behavior: reduced() ? "auto" : "smooth", block: "center" });
    }
  }
  function need(sel, what) {
    const e = typeof sel === "string" ? $(sel) : sel;
    if (!e) throw new Error(`Guided demo could not find ${what || sel} on the page.`);
    return e;
  }
  // Resolve on the next `name` event; reject on an app error event, a timeout, or when the tour is stopped.
  function waitEvt(name, what, ms) {
    return new Promise((resolve, reject) => {
      const limit = ms || 20000;
      const done = (fn, v) => {
        clearTimeout(tm);
        document.removeEventListener(name, onOk); document.removeEventListener("fp:error", onErr);
        pending.delete(cancel); fn(v);
      };
      const onOk = () => done(resolve);
      const onErr = (e) => done(reject, new Error(`${what} failed: ${e.detail && e.detail.message}`));
      const cancel = () => done(reject, ABORT);
      const tm = setTimeout(() => done(reject, new Error(`Timed out after ${limit / 1000} s waiting for ${what}.`)), limit);
      pending.add(cancel);
      document.addEventListener(name, onOk); document.addEventListener("fp:error", onErr);
    });
  }
  // Run `trigger` (which clicks / dispatches on a real control) and wait for the app to report completion.
  async function act(trigger, evt, what, ctx, ms) {
    const p = waitEvt(evt, what, ms);
    p.catch(() => {});
    try { trigger(); } catch (e) { Array.from(pending).forEach((f) => f()); throw e; }
    await p; ctx.check();
  }
  const setVal = (input, v) => {
    input.value = String(v);
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  };
  const showTab = (id) => need(`#nav button[data-t="${id}"]`, "the " + id + " tab").click();
  const riskText = () => need("#wRisk", "the risk line").textContent.replace(/^.*?:\s*/, "");

  async function enterWeather(ctx) {
    if (W.loaded) { showTab("weather"); return; }
    await act(() => showTab("weather"), "fp:weather", "the Money Weather map", ctx);
  }

  const cap1 = (w) => w.charAt(0).toUpperCase() + w.slice(1);
  const monthsOf = (st) => (st && st.monthly && st.monthly.months && st.monthly.months.length) || 0;
  const monthsText = (n) => (n ? `${n} months of` : "its");

  // Duel presets: amounts of 95%, 90%, 85% of the balance; option B just after the next payday (or ~8 weeks
  // out when no payday is known). Each preset is run through the real duel endpoint and the first whose options
  // differ (overdraft risk by >= 2 points or median lowest balance by >= 10% of the balance) is used.
  async function pickDuel(S) {
    const bal = S.st.balance, d0 = S.st.forecast.bands.dates[0], last = U.addDays(d0, 89);
    const pays = ((W.data && W.data.paydays) || []).filter((d) => d > d0 && d <= last);
    const bDate = pays.length ? U.addDays(pays[0], 1) : U.addDays(d0, 56);
    const bName = `Wait until ${U.fmtDate(bDate)}`;
    const presets = [0.95, 0.9, 0.85].map((f) => ({ frac: f, amt: U.roundTo(bal * f, 1000) }));
    for (const p of presets) {
      const r = await api(`/api/duel/${encodeURIComponent(profile)}`,
        { a: { name: "Buy now", amount: p.amt }, b: { name: bName, amount: p.amt, date: bDate } });
      const a = r.answer.a.summary, b = r.answer.b.summary;
      const dp = Math.abs(a.p_overdraft - b.p_overdraft), dl = Math.abs(a.floor_p50 - b.floor_p50);
      if (dp >= 0.02 || dl >= 0.1 * Math.abs(bal)) return { ...p, bDate, bName, payday: pays[0] || null, differs: true };
    }
    return { ...presets[0], bDate, bName, payday: pays[0] || null, differs: false };
  }

  function buildSteps(S) {
    const bal = () => { if (!S.st || !S.st.has_data) throw new Error("No data loaded, cannot continue."); return S.st.balance; };
    return [
      { id: "load", dwell: 3000, async run(ctx) {
        const who = need("#persona", "the persona list").value || "priya";
        S.name = cap1(who);
        const known = STATE && STATE.profile === who ? monthsOf(STATE) : 0;
        caption(`Loading ${S.name}, a synthetic persona with ${monthsText(known)} bank transactions. The agents analyse it live in the trace.`);
        highlight(need("#loadBtn"));
        await act(() => need("#loadBtn").click(), "fp:rendered", "the persona to load", ctx, 20000);
        // an earlier render (e.g. the page's own first load) may land first: wait for this persona's state
        while (!STATE || STATE.profile !== who) { await waitEvt("fp:rendered", "the persona to load", 20000); ctx.check(); }
        S.st = STATE;
        if (!S.st || !S.st.has_data || !S.st.forecast) throw new Error("The pipeline did not finish: no forecast available.");
        highlight(need("#trace"));
        caption(`Pipeline done on ${monthsText(monthsOf(S.st))} transactions. The categoriser, analyst, forecaster, advisor and verifier have all reported in the trace.`);
      } },
      { id: "dashboard", dwell: 2500, async run(ctx) {
        showTab("dashboard");
        const k = need("#kpis", "the KPIs").children;
        highlight(need("#kpis"));
        caption(`Balance ${k[0].firstChild.textContent}. Overdraft risk over the next 60 days: ${k[2].firstChild.textContent}.`);
        await ctx.sleep(4000);
        highlight(need("#cFore", "the forecast chart"));
        caption("The fan chart is the 60-day outlook: p10, median and p90 of a Monte Carlo over recent habits and recurring bills.");
        await ctx.sleep(4500);
      } },
      { id: "weather", dwell: 2000, async run(ctx) {
        await enterWeather(ctx);
        highlight(need("#wChip"));
        caption(`Money Weather, Normal 60 days: ${$("#wChip").textContent}. ${riskText()} chance of dipping below the safety buffer.`);
        await ctx.sleep(4000);
        const chip = need('#wScen button[data-s="purchase"]', "the Big purchase chip");
        highlight(chip);
        await act(() => chip.click(), "fp:weather", "the Money Weather map", ctx);
        highlight(need("#wChip"));
        caption(`Now a big purchase: ${$("#wChip").textContent}. ${riskText()} chance of a dip. ${need("#wHeadline").textContent}`);
        S.duel = pickDuel(S);  // runs in the background; the duel step waits for it
        S.duel.catch(() => {});
        await ctx.sleep(4500);
        const card = $("#wStorms .storm");
        if (card) {
          highlight(card); card.click();
          caption("The autopsy re-runs the failing futures to show which habits and bills cause them.");
        } else {
          highlight(need("#wStorms", "the storm list"));
          caption(`No storm window for ${S.name} even with this purchase. The autopsy still explains any failing futures.`);
        }
        await ctx.sleep(1500);
        highlight(need("#wAutopsy"));
        await ctx.sleep(3500);
        const umb = $("#wUmbrellas button.act");
        let withUmb = false;
        if (umb) {
          highlight(umb.closest(".card"));
          const before = riskText();
          caption("An umbrella is a spending cut ranked by measured risk reduction. Applying the first one to the map.");
          await ctx.sleep(1500);
          await act(() => umb.click(), "fp:weather", "the Money Weather map", ctx);
          highlight(need("#wRisk"));
          caption(`With the umbrella: ${riskText()} (was ${before}).`);
          withUmb = true;
        } else {
          highlight(need("#wUmbrellas", "the umbrella list"));
          caption(`No umbrella offered: ${need("#wUmbrellas").textContent.slice(0, 160)}`);
        }
        await ctx.sleep(3500);
        // lifestyle slider 100 -> 70 -> 100, each step through a real input event on the real slider
        const sl = need("#wLife", "the lifestyle slider");
        highlight(sl);
        const base = riskText();
        const seq = ctx.reduced ? [70, 100] : [85, 70, 85, 100];
        for (const v of seq) {
          await act(() => setVal(sl, v), "fp:weather", "the Money Weather map", ctx);
          caption(`Non-bill spending (groceries, dining, shopping...) at ${v}%: ${riskText()} chance of a dip ` +
            `(${withUmb ? "with the umbrella; " : ""}${base} at 100%).`);
          await ctx.sleep(v === 70 || v === 100 ? 2600 : 1200);
        }
      } },
      { id: "recs", dwell: 2500, async run(ctx) {
        showTab("recs");
        const d = need("#recList details", "a recommendation's Why? details");
        d.open = true;
        highlight(d.closest(".card"));
        const note = need("#recList details .vnote", "the verifier note").textContent;
        caption(`Why? opens the evidence. ${note.length > 220 ? note.slice(0, 217) + "..." : note}`);
        await ctx.sleep(3000);
      } },
      { id: "afford", dwell: 2500, async run(ctx) {
        showTab("afford");
        const f = need("#afForm");
        const amt = U.roundTo(bal() * 0.6, 1000);
        f.elements.amount.value = String(amt);
        highlight(f);
        caption(`Can I afford ${U.money(amt)} (about 60% of the balance)? Simulating it now.`);
        await ctx.sleep(1500);
        await act(() => f.requestSubmit(), "fp:afford", "the affordability answer", ctx);
        highlight(need("#afOut"));
        const v = Array.from($("#afOut").children).map((c) => c.textContent).find((t) => /^Verdict:/.test(t)) || "Answer ready.";
        caption(`${v}. The date options below come from re-running the simulation.`);
        await ctx.sleep(1500);
      } },
      { id: "duel", dwell: 2500, async run(ctx) {
        showTab("duel");
        const f = need("#duForm");
        bal();
        const pickP = S.duel || pickDuel(S);
        const P = await pickP; ctx.check();
        S.duelPick = P;
        f.elements.an.value = "Buy now"; f.elements.aa.value = String(P.amt); f.elements.ad.value = "";
        f.elements.bn.value = P.bName; f.elements.ba.value = String(P.amt); f.elements.bd.value = P.bDate;
        highlight(f);
        const when = P.payday ? `just after the next payday (${U.fmtDate(P.payday)})` : "about eight weeks later";
        caption(`Decision duel: spend ${U.money(P.amt)} (${Math.round(P.frac * 100)}% of the balance) now, or on ${U.fmtDate(P.bDate)}, ${when}. ` +
          (P.differs ? "Both futures run on identical random draws."
            : `For ${S.name} the options barely differ at 95%, 90% or 85% of the balance, so this shows the 95% case as it is.`));
        await ctx.sleep(2500);
        await act(() => f.requestSubmit(), "fp:duel", "the duel result", ctx);
        highlight(need("#duOut"));
        caption(need("#duOut .card", "the duel result").textContent);
        await ctx.sleep(2500);
      } },
      { id: "digest", dwell: 2500, async run(ctx) {
        showTab("dashboard");
        const b = need("#digestBtn", "the digest button");
        highlight(b);
        caption("Finally the digest: a summary of findings, stored here and emailed only if SMTP is configured.");
        await ctx.sleep(1500);
        await act(() => b.click(), "fp:digest", "the digest", ctx);
        highlight(need("#digests .card", "the new digest"));
      } },
      { id: "end", dwell: 0, async run() {
        highlight(null);
        caption("That is FinPilot: agents you can watch, forecasts you can stress-test, and advice whose numbers are checked against recomputed values or the engine's own evidence. Press Replay to watch again.");
      } },
    ];
  }

  function setBusy(on) {
    document.body.classList.toggle("touring", on);
    ["main", "header"].forEach((s) => { if (on) $(s).setAttribute("inert", ""); else $(s).removeAttribute("inert"); });
    $("#tourBtn").disabled = on;
  }
  function cleanup() {
    Array.from(pending).forEach((f) => f());
    highlight(null);
    setBusy(false);
  }
  function startTour() {
    if (runner && runner.state === "running") return;
    const S = { st: null };
    t0 = Date.now();
    $("#tourBar").hidden = false;
    $("#tourReplay").hidden = true; $("#tourPause").hidden = false; $("#tourSkip").hidden = false;
    $("#tourPause").textContent = "Pause"; $("#tourStop").textContent = "Stop";
    setBusy(true);
    caption("Starting the guided demo...");
    runner = createRunner({
      timers: TIMERS, reduced: reduced(), reducedFactor: 1.6, steps: buildSteps(S),
      callbacks: {
        onStep: (i, n) => { $("#tourStep").textContent = `Step ${i + 1} of ${n}`; },
        onPause: (p) => { $("#tourPause").textContent = p ? "Resume" : "Pause"; },
        onEnd: (status, err) => {
          cleanup();
          if (status === "stopped") { $("#tourBar").hidden = true; return; }
          if (status === "error") caption("Guided demo stopped: " + (err && err.message ? err.message : String(err)), true);
          else $("#tourStep").textContent = `Finished in ${Math.round((Date.now() - t0) / 1000)} s`;
          $("#tourReplay").hidden = false; $("#tourPause").hidden = true; $("#tourSkip").hidden = true;
          $("#tourStop").textContent = "Close";
        },
      },
    });
    runner.start();
  }
  function stopTour() {
    if (runner && runner.state === "running") runner.stop();
    else $("#tourBar").hidden = true;
  }
  $("#tourBtn").addEventListener("click", startTour);
  $("#tourReplay").addEventListener("click", startTour);
  $("#tourStop").addEventListener("click", stopTour);
  $("#tourSkip").addEventListener("click", () => { if (runner) runner.skip(); });
  $("#tourPause").addEventListener("click", () => { if (runner) { if (runner.paused) runner.resume(); else runner.pause(); } });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#tourBar").hidden) stopTour(); });
})(typeof window !== "undefined" ? window : globalThis);
