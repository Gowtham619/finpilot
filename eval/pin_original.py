"""Hash of the first-generator result sections, used by tests/test_generate_messy.py.

The pin guards that adding the OOD block did not move the first-generator results. When forecast,
backtest, calibration, detection or categorization engine behaviour is changed ON PURPOSE, recompute
it with `python -m eval.pin_original`, paste it into ORIGINAL_SECTIONS_SHA256 in the test, and record
the reason in the commit message. An unexplained mismatch is a regression, not a stale pin.
"""
import hashlib
import json
import tempfile

SECTIONS = ("categorization", "detection", "forecast", "forecast_meta", "calibration", "backtest")


def original_sections_hash(res):
    keep = {k: res[k] for k in SECTIONS}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()


def main():
    import contextlib
    import io
    import pathlib

    from eval import run_eval
    saved = run_eval.OUT
    with tempfile.TemporaryDirectory() as d:
        run_eval.OUT = pathlib.Path(d)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                res = run_eval.main(["--no-llm"])
        finally:
            run_eval.OUT = saved
    print(original_sections_hash(res))


if __name__ == "__main__":
    main()
