// Pure helpers shared by the UI (window.FPU) and the node unit tests (module.exports).
(function (root) {
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const SEVERITY = ["clear", "cloudy", "rain", "storm"];

  // "2026-03-07" -> "Mar 7". Pure string work: no Date, so no timezone shift.
  function fmtDate(iso) { const [, m, d] = iso.split("-"); return `${MONTHS[+m - 1]} ${+d}`; }
  // Monday-first weekday index (Mon=0 .. Sun=6) of an ISO date, computed in UTC.
  function weekdayMon(iso) {
    const [y, m, d] = iso.split("-").map(Number);
    return (new Date(Date.UTC(y, m - 1, d)).getUTCDay() + 6) % 7;
  }
  function worstLabel(cal) {
    return cal.reduce((w, c) => (SEVERITY.indexOf(c.label) > SEVERITY.indexOf(w) ? c.label : w), "clear");
  }
  // Probability (0..1) as a percent string; a tiny but non-zero value shows "<1%", never "0%".
  function pct(x) {
    const v = x * 100;
    if (v > 0 && Math.round(v) === 0) return "<1%";
    return Math.round(v) + "%";
  }
  // Risk reduction in percentage points; null when the change is too small to show.
  function deltaPts(before, after) {
    const d = (before - after) * 100;
    if (Math.abs(d) < 0.5) return null;
    return (d > 0 ? "−" : "+") + Math.round(Math.abs(d)) + " pts";
  }
  // Indian digit grouping (12,34,567), done by hand so it is identical in every runtime.
  function money(x) {
    const n = Math.round(Math.abs(x));
    let s = String(n);
    if (s.length > 3) {
      const head = s.slice(0, -3).replace(/\B(?=(\d{2})+(?!\d))/g, ",");
      s = head + "," + s.slice(-3);
    }
    return (x < 0 && n !== 0 ? "−₹" : "₹") + s;
  }
  function addDays(iso, n) {
    const [y, m, d] = iso.split("-").map(Number);
    const t = new Date(Date.UTC(y, m - 1, d + n));
    return t.toISOString().slice(0, 10);
  }
  const roundTo = (x, step) => Math.max(step, Math.round(x / step) * step);

  const api = { MONTHS, SEVERITY, fmtDate, weekdayMon, worstLabel, pct, deltaPts, money, addDays, roundTo };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.FPU = api;
})(typeof window !== "undefined" ? window : globalThis);
