"""Turn the recording + voice clips into the final narrated mp4.

usage: build.py REC.pkl VO_DIR OUT.mp4
Narration is placed at its scene marker; if a clip is longer than the scene, the video holds its last frame
(a freeze) so the picture never talks over the next scene.
"""
import json, pickle, re, subprocess, sys, tempfile, wave
from pathlib import Path
import cv2, numpy as np
from narration import LINES

REC, VO, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
W, H, FPS, SR, GAP = 1280, 800, 12, 24000, 0.35
rec = pickle.load(open(REC, "rb"))
frames, markers = rec["frames"], rec["markers"]
t0 = frames[0][0]
dur = json.loads((VO / "durations.json").read_text())

# anchors: (source_time, key)
anchors, used = [], set()
tour_start = next(t for t, k in markers if k == "tour_intro")
for key, _ in LINES:
    if key.startswith("cap:"):
        rx = re.compile(key[4:])
        hit = next((t for t, k in markers if k.startswith("cap:") and t >= tour_start and rx.search(k[4:]) and t not in used), None)
    else:
        hit = next((t for t, k in markers if k == key), None)
    if hit is None:
        print("no anchor for", key); continue
    used.add(hit); anchors.append((hit - t0, key))
anchors.sort()

freezes, d = [], 0.0   # (source_time, seconds)
placed = []            # (output_start, key)
for i, (T, key) in enumerate(anchors):
    start = T + d
    placed.append((start, key))
    end = start + dur[key] + GAP
    nxt = anchors[i + 1][0] + d if i + 1 < len(anchors) else None
    if nxt is not None and nxt < end:
        extra = end - nxt
        freezes.append((anchors[i + 1][0] - 0.05, extra)); d += extra
total_src = frames[-1][0] - t0
last_end = placed[-1][0] + dur[placed[-1][1]] + 0.8
total_out = max(total_src + d, last_end)
print(f"source {total_src:.0f}s  freezes {sum(e for _, e in freezes):.0f}s  output {total_out:.0f}s")

# audio
audio = np.zeros(int((total_out + 1) * SR), dtype=np.float32)
for start, key in placed:
    safe = "".join(c if c.isalnum() else "_" for c in key)
    with wave.open(str(VO / f"{safe}.wav")) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768
    i = int(start * SR); audio[i:i + len(pcm)] += pcm
tmp = Path(tempfile.mkdtemp())
with wave.open(str(tmp / "vo.wav"), "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", "-b", "48000", str(tmp / "vo.wav"), str(tmp / "vo.m4a")], check=True)

# video: output time -> source time through the freezes
def src_time(o):
    cum = 0.0
    for fs, extra in freezes:
        out_start = fs + cum
        if o < out_start: break
        if o < out_start + extra: return fs
        cum += extra
    return o - cum
vw = None
for fourcc in ("avc1", "mp4v"):
    vw = cv2.VideoWriter(str(tmp / "v.mp4"), cv2.VideoWriter_fourcc(*fourcc), FPS, (W, H))
    if vw.isOpened(): break
i, cache = 0, (None, None)
for k in range(int(total_out * FPS)):
    s = min(src_time(k / FPS), total_src)
    while i + 1 < len(frames) and frames[i + 1][0] - t0 <= s: i += 1
    if cache[0] != i:
        img = cv2.imdecode(np.frombuffer(frames[i][1], np.uint8), cv2.IMREAD_COLOR)
        if img.shape[:2] != (H, W): img = cv2.resize(img, (W, H))
        cache = (i, img)
    vw.write(cache[1])
vw.release()
subprocess.run(["swift", str(Path(__file__).with_name("mux.swift")), str(tmp / "v.mp4"), str(tmp / "vo.m4a"), OUT], check=True)
print("wrote", OUT)
