import threading
import time
from collections import OrderedDict

MAX_RUNS = 500


class Trace:
    """In-memory run traces. At most `max_runs` runs are retained; when over the cap, the oldest
    finished runs are evicted first (a run still in progress is never evicted)."""

    def __init__(self, max_runs=MAX_RUNS):
        self._ev = OrderedDict()
        self._done = set()
        self._lock = threading.Lock()
        self.max_runs = max_runs

    def _evict(self):
        if len(self._ev) <= self.max_runs:
            return
        for rid in [r for r in self._ev if r in self._done]:
            if len(self._ev) <= self.max_runs:
                break
            del self._ev[rid]
            self._done.discard(rid)

    def emit(self, run_id, agent, kind, summary, data=None):
        with self._lock:
            evs = self._ev.get(run_id)
            if evs is None:
                evs = self._ev[run_id] = []
                self._evict()
            ev = {"seq": len(evs), "ts": time.time(), "agent": agent,
                  "kind": kind, "summary": summary, "data": data or {}}
            evs.append(ev)
        return ev

    def since(self, run_id, n=0):
        with self._lock:
            return list(self._ev.get(run_id, [])[n:])

    def finish(self, run_id):
        """Idempotent. The 'done' event is emitted before is_done() flips; consumers should treat
        either the 'done' event or is_done() as terminal."""
        with self._lock:
            if run_id in self._done:
                return
        self.emit(run_id, "coordinator", "done", "run complete")
        with self._lock:
            self._done.add(run_id)

    def is_done(self, run_id):
        with self._lock:
            return run_id in self._done

    def __len__(self):
        with self._lock:
            return len(self._ev)


trace = Trace()
