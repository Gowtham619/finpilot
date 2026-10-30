"""Drive the real UI in headless Chrome through every feature and record it (CDP screencast).

usage: record.py OUT.pkl CHROME_PROFILE_DIR VO_DIR   (needs the app on :8000 and the clips from tts.py)
Saves frames [(epoch_seconds, jpeg_bytes)] and markers [(epoch_seconds, key)] for build.py.
"""
import base64, json, pickle, subprocess, sys, time, urllib.request
from datetime import date, timedelta
from pathlib import Path
from websockets.sync.client import connect

OUT, PROFILE_DIR, VO = sys.argv[1], sys.argv[2], Path(sys.argv[3])
DUR = json.loads((VO / "durations.json").read_text())
ROOT = Path(__file__).resolve().parents[2]
W, H, PORT = 1280, 800, 9333
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={PORT}", f"--user-data-dir={PROFILE_DIR}",
                         "--no-first-run", "--disable-gpu", f"--window-size={W},{H}", "about:blank"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
frames, markers = [], []
try:
    for _ in range(60):
        try:
            page = next(t for t in json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json")) if t["type"] == "page"); break
        except Exception:
            time.sleep(0.2)
    ws = connect(page["webSocketDebuggerUrl"], max_size=None)
    nid = 0

    def handle(m):
        if m.get("method") == "Page.screencastFrame":
            p = m["params"]
            frames.append((p["metadata"]["timestamp"], base64.b64decode(p["data"])))
            ws.send(json.dumps({"id": 0, "method": "Page.screencastFrameAck", "params": {"sessionId": p["sessionId"]}}))

    def call(method, **params):
        global nid
        nid += 1
        ws.send(json.dumps({"id": nid, "method": method, "params": params}))
        while True:
            m = json.loads(ws.recv())
            if m.get("id") == nid:
                if "error" in m: raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})
            handle(m)

    def js(expr):
        r = call("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r: raise RuntimeError(r["exceptionDetails"]["exception"].get("description", str(r["exceptionDetails"])))
        return r["result"].get("value")

    def pump(sec):
        end = time.time() + sec
        while time.time() < end:
            try: handle(json.loads(ws.recv(timeout=min(0.3, max(0.01, end - time.time())))))
            except TimeoutError: pass

    def wait(expr, timeout=40, what=""):
        end = time.time() + timeout
        while time.time() < end:
            if js(f"!!({expr})"): return
            pump(0.3)
        raise RuntimeError(f"timeout waiting for {what or expr}")

    def mark(key):
        markers.append((js("Date.now()/1000"), key)); print(f"  [{key}]", flush=True)

    def caption(text):
        js("""(t)=>0""" if False else f"""(()=>{{const b=document.querySelector('#tourBar');b.hidden=false;document.body.classList.add('touring');
          document.querySelector('.tctl').style.display='none';document.querySelector('#tourCap').textContent={json.dumps(text)};}})()""")

    def hide_caption():
        js("document.querySelector('#tourBar').hidden=true;document.body.classList.remove('touring')")

    def dwell_for(key, extra=0.6):
        pump(DUR[key] + extra)

    def nav(tab):
        js(f"document.querySelector('#nav button[data-t={tab}]').click()")

    call("Page.enable"); call("Runtime.enable"); call("DOM.enable")
    call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
    call("Page.navigate", url="http://127.0.0.1:8000/?profile=priya")
    pump(5)
    call("Page.startScreencast", format="jpeg", quality=85, maxWidth=W, maxHeight=H, everyNthFrame=1)
    pump(0.5)

    # ---- intro on the loaded dashboard
    mark("intro"); caption("FinPilot: a financial copilot that shows you the weather, not just the balance."); dwell_for("intro")

    # ---- import
    mark("import_open"); caption("Import a statement: CSV, text PDF, or a scan/photo. Read locally, personal data masked.")
    js("document.querySelector('#importBtn').click()"); pump(1.2)

    def pick(path):
        doc = call("DOM.getDocument")["root"]["nodeId"]
        node = call("DOM.querySelector", nodeId=doc, selector="#impFile")["nodeId"]
        call("DOM.setFileInputFiles", files=[str(path)], nodeId=node)
    dwell_for("import_open", 0.0)
    mark("import_scan"); caption("Scanned page: local OCR, with a warning when digits may be misread.")
    pick(ROOT / "testdata/statements/scanned_page.png")
    wait("!document.querySelector('#impStep2').hidden", 60, "scan preview"); dwell_for("import_scan", 1.0)
    js("document.querySelector('#impBack').click()"); pump(0.8)
    mark("import_text"); caption("Text PDF: preview first, nothing is saved until you confirm. Names, UPI IDs, phone and account numbers are masked.")
    pick(ROOT / "testdata/statements/text_statement.pdf")
    wait("!document.querySelector('#impStep2').hidden", 40, "pdf preview"); pump(3)
    js("document.querySelector('.tblwrap').scrollTop=0; document.querySelector('#importDlg').scrollTop=9999"); dwell_for("import_text", 0.2)
    js("document.querySelector('#upForm').elements.profile.value='mybank'")
    mark("import_confirm"); caption("Confirm, and the agents analyse it.")
    js("document.querySelector('#impConfirm').click()"); dwell_for("import_confirm", 0.3)
    wait("!document.querySelector('#importDlg').open && /run complete/.test(document.querySelector('#trace').textContent)", 60, "pipeline")
    pump(1.0); mark("import_done"); caption("The new profile has its own forecast and findings.")
    js("window.scrollTo(0,0)"); dwell_for("import_done", 1.5)
    hide_caption()

    # ---- guided demo (it loads Priya itself)
    mark("tour_intro")
    js("document.querySelector('#persona').value='priya'")
    js("document.querySelector('#tourBtn').click()")
    last, start = None, time.time()
    while time.time() - start < 150:
        pump(0.25)
        st = json.loads(js("JSON.stringify({cap:document.querySelector('#tourCap').textContent,done:!document.querySelector('#tourReplay').hidden})"))
        if st["cap"] != last:
            last = st["cap"]; markers.append((js("Date.now()/1000"), "cap:" + last)); print("  cap:", last[:70], flush=True)
        if st["done"]:
            pump(2.5); break
    else:
        raise RuntimeError("tour did not finish")
    if last.startswith("Guided demo stopped"): raise RuntimeError(last)
    js("document.querySelector('#tourBar').hidden=true;document.body.classList.remove('touring')"); js("document.querySelector('main').removeAttribute('inert');document.querySelector('header').removeAttribute('inert')")

    # ---- extra Money Weather scenarios
    js("window.scrollTo(0,0)"); nav("weather"); pump(1.5)
    js("document.querySelector('#wReset').click()"); pump(1.5)

    def scenario(label):
        js(f"[...document.querySelectorAll('#wScen button')].find(b=>b.textContent.includes({json.dumps(label)})).click()")
        pump(1.0); wait("!document.querySelector('#wBusy').textContent", 30, "weather"); pump(0.8)
    mark("weather_jobloss"); caption("Job-loss drill: your next paychecks are removed."); scenario("Job-loss")
    js("window.scrollTo(0,150)"); pump(DUR["weather_jobloss"] - 2.0)
    mark("weather_medical"); caption("Medical emergency: a sudden expense lands in every future."); scenario("Medical")
    pump(DUR["weather_medical"] - 1.0)

    # ---- recommendations
    mark("recs"); caption("Recommendations in plain words, with a Why? panel showing the figures used.")
    nav("recs"); pump(1.0)
    js("(()=>{const d=document.querySelectorAll('#recList details')[2]||document.querySelector('#recList details');d.open=true;d.scrollIntoView({block:'center'})})()")
    pump(DUR["recs"])

    # ---- goals
    mark("goals"); caption("Goals: a target balance by a deadline, with the chance of reaching it.")
    nav("goals"); pump(1.0)
    dl = (date(2026, 10, 19) + timedelta(days=270)).isoformat()
    js(f"""(()=>{{const f=document.querySelector('#goForm');f.elements.name.value='Emergency fund';f.elements.target.value='150000';f.elements.deadline.value='{dl}';f.requestSubmit();}})()""")
    wait("document.querySelector('#goList .card')", 20, "goal card"); pump(DUR["goals"] - 1.0)

    # ---- chat
    mark("chat"); caption("Ask about your own spending in plain language.")
    nav("chat"); pump(1.0)
    js("""(()=>{const f=document.querySelector('#chForm');f.elements.q.value='How much did I spend on food delivery in July?';f.requestSubmit();})()""")
    wait("document.querySelector('#chOut').textContent.trim().length>10", 30, "chat answer"); pump(DUR["chat"] - 1.0)

    # ---- outro
    mark("outro"); caption("FinPilot: engines compute, models narrate, a verifier checks. All data is synthetic.")
    nav("dashboard"); js("window.scrollTo(0,0)"); pump(DUR["outro"] + 1.0)
    call("Page.stopScreencast")
    print("frames", len(frames), "markers", len(markers))
    pickle.dump({"frames": frames, "markers": markers}, open(OUT, "wb"))
finally:
    proc.terminate()
