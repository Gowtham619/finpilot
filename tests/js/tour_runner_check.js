// Fake-clock checks for the guided-demo step runner. Prints a JSON object of results.
const { createRunner, ABORT } = require("../../finpilot/static/tour.js");

function clock() {
  let now = 0, seq = 0; const q = new Map();
  return {
    setTimeout(f, ms) { const id = ++seq; q.set(id, { at: now + ms, f }); return id; },
    clearTimeout(id) { q.delete(id); },
    now: () => now,
    async advance(ms) {
      const end = now + ms;
      for (;;) {
        let best = null;
        for (const [id, t] of q) if (t.at <= end && (!best || t.at < best.t.at)) best = { id, t };
        if (!best) break;
        q.delete(best.id); now = best.t.at; best.t.f();
        for (let i = 0; i < 20; i++) await Promise.resolve();
      }
      now = end;
      for (let i = 0; i < 20; i++) await Promise.resolve();
    },
  };
}
const mk = (steps, extra = {}) => {
  const c = clock(), log = [], ends = [];
  const r = createRunner({ timers: c, steps, callbacks: { onStep: (i) => log.push("step" + i),
    onEnd: (s, e) => ends.push([s, e && e.message]) }, ...extra });
  return { c, r, log, ends };
};
const res = {};
(async () => {
  // 1. runs steps in order, honouring dwell
  { const log = []; const t = mk([{ dwell: 1000, run: async () => { log.push("a"); } }, { dwell: 2000, run: async () => { log.push("b"); } }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(0); const s0 = t.r.index; await t.c.advance(999); const s1 = t.r.index; await t.c.advance(1);
    const s2 = t.r.index; await t.c.advance(1999); const mid = t.ends.length; await t.c.advance(1);
    res.order = { s0, s1, s2, mid, ends: t.ends, log }; }
  // 2. pause freezes the dwell, resume continues with the remaining time
  { const t = mk([{ dwell: 1000, run: async () => {} }, { dwell: 10, run: async () => {} }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(400); t.r.pause(); await t.c.advance(5000); const stillFirst = t.r.index;
    t.r.resume(); await t.c.advance(599); const before = t.r.index; await t.c.advance(1); res.pause = { stillFirst, before, after: t.r.index }; }
  // 3. skip ends the dwell and step sleeps immediately
  { const t = mk([{ dwell: 5000, run: async (ctx) => { await ctx.sleep(3000); } }, { dwell: 1, run: async () => {} }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(100); t.r.skip(); await t.c.advance(0); res.skip = { index: t.r.index }; }
  // 4. skip while paused resumes and moves on
  { const t = mk([{ dwell: 5000, run: async () => {} }, { dwell: 1, run: async () => {} }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(100); t.r.pause(); t.r.skip(); await t.c.advance(0); res.skipPaused = { index: t.r.index, paused: t.r.paused }; }
  // 5. stop cancels pending sleeps and never runs later steps
  { let ran2 = false; const t = mk([{ dwell: 5000, run: async () => {} }, { dwell: 1, run: async () => { ran2 = true; } }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(100); t.r.stop(); await t.c.advance(10000); res.stop = { ran2, ends: t.ends, state: t.r.state }; }
  // 6. a throwing step stops the tour with the error; later steps do not run
  { let ran2 = false; const t = mk([{ dwell: 1, run: async () => { throw new Error("boom"); } }, { dwell: 1, run: async () => { ran2 = true; } }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(100); res.error = { ran2, ends: t.ends, state: t.r.state }; }
  // 7. a step that sleeps while stopped is aborted, not reported as an error
  { const t = mk([{ dwell: 1, run: async (ctx) => { await ctx.sleep(10000); } }]);
    t.r.start(); await t.c.advance(0); await t.c.advance(10); t.r.stop(); await t.c.advance(10); res.abort = { ends: t.ends }; }
  // 8. reduced motion lengthens the dwell
  { const t = mk([{ dwell: 1000, run: async () => {} }], { reduced: true, reducedFactor: 2 });
    t.r.start(); await t.c.advance(0); await t.c.advance(1999); const m = t.ends.length; await t.c.advance(1); res.reduced = { m, ends: t.ends }; }
  console.log(JSON.stringify(res));
})();
