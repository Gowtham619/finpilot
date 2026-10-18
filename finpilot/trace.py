import threading
import time
from collections import defaultdict


class Trace:
    def __init__(self):
        self._ev = defaultdict(list)
        self._done = set()
        self._lock = threading.Lock()

    def emit(self, run_id, agent, kind, summary, data=None):
        with self._lock:
            ev = {"seq": len(self._ev[run_id]), "ts": time.time(), "agent": agent,
                  "kind": kind, "summary": summary, "data": data or {}}
            self._ev[run_id].append(ev)
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
        return run_id in self._done


trace = Trace()
