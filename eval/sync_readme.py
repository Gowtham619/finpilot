"""Keep README evaluation tables generated from eval/results.md.

Regions `<!-- eval-results:begin NAME -->` ... `<!-- eval-results:end NAME -->` in README.md are rewritten
with the tables of the matching results.md section. `python -m eval.sync_readme` rewrites (idempotent);
`--check` exits 1 if the README is out of sync. Prose outside the markers is never touched.
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SECTIONS = {  # region name -> results.md heading prefix
    "categorization": "## Categorization",
    "detection": "## Detection",
    "forecast": "## Forecast",
    "calibration": "## Calibration",
    "backtest": "## Advice backtest",
    "ood": "## Out-of-distribution check",
}


def tables(md, prefix):
    sec = None
    for part in re.split(r"(?m)^(?=## )", md):
        if part.startswith(prefix):
            sec = part
            break
    if sec is None:
        raise SystemExit(f"results.md has no section {prefix!r}")
    lines = sec.splitlines()
    idx = [i for i, ln in enumerate(lines) if ln.startswith("|")]
    if not idx:
        raise SystemExit(f"section {prefix!r} has no table")
    return "\n".join(lines[idx[0]:idx[-1] + 1])


def render(readme, md):
    for name, prefix in SECTIONS.items():
        pat = re.compile(rf"(<!-- eval-results:begin {name} -->\n).*?(\n<!-- eval-results:end {name} -->)", re.S)
        if not pat.search(readme):
            raise SystemExit(f"README.md is missing region {name!r}")
        body = tables(md, prefix)
        readme = pat.sub(lambda m: m.group(1) + body + m.group(2), readme)
    return readme


def main(argv=None, root=ROOT):
    argv = sys.argv[1:] if argv is None else argv
    rp, mp = root / "README.md", root / "eval" / "results.md"
    cur = rp.read_text()
    new = render(cur, mp.read_text())
    if "--check" in argv:
        if new != cur:
            print("README.md evaluation tables are out of sync with eval/results.md; run `make eval-sync`.")
            return 1
        return 0
    rp.write_text(new)
    return 0


if __name__ == "__main__":
    sys.exit(main())
