"""Render every narration line to a 24 kHz mono wav with macOS `say`; prints durations."""
import json, subprocess, sys, wave
from pathlib import Path
from narration import LINES, RATE, VOICE

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
durs = {}
for key, text in LINES:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    aiff, wav = out / f"{safe}.aiff", out / f"{safe}.wav"
    subprocess.run(["say", "-v", VOICE, "-r", str(RATE), "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@24000", "-c", "1", str(aiff), str(wav)], check=True)
    aiff.unlink()
    with wave.open(str(wav)) as w:
        durs[key] = w.getnframes() / w.getframerate()
    print(f"{durs[key]:5.1f}s  {key}")
(out / "durations.json").write_text(json.dumps(durs, indent=1))
print("total %.0fs" % sum(durs.values()))
