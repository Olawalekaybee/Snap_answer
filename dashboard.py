#!/usr/bin/env python3
"""
SnapAnswer Dashboard - live D435i view in the browser, adjustable capture area,
Online (Claude API) / Offline (local Ollama) switch, Snap / Save buttons,
answers shown large on the side and spoken through the Spark's speaker.

Run:   python dashboard.py            then open  http://localhost:8000
"""

import argparse
import json
import logging
import os
import re
import shutil
import threading
import time
import traceback
from collections import deque
from datetime import datetime

import cv2
import requests
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from online import ClaudeBrain, load_key
from snapanswer import Brain, open_camera, Voice, compose_speech, norm_label, sharpness

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.join(HERE, "settings.json")
DEFAULT_ROI = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}


def crop(frame, roi):
    h, w = frame.shape[:2]
    x0, y0 = int(roi["x"] * w), int(roi["y"] * h)
    x1, y1 = int((roi["x"] + roi["w"]) * w), int((roi["y"] + roi["h"]) * h)
    return frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]


def merge_pages(qs, new):
    """Append questions read from the next page. Joins a question that runs across the page
    break (stem on one page, options on the next) and drops exact repeats (overlapping photos)."""
    norm = lambda t: re.sub(r"\W+", " ", t or "").strip().lower()[:80]
    seen = {norm(q["stem"]) for q in qs}
    for q in new:
        q = dict(q)
        q.setdefault("options", [])
        if qs and (not q.get("number") or not norm(q["stem"])) and not qs[-1].get("options") and q["options"]:
            qs[-1]["options"] = q["options"]            # options continued from previous page
            continue
        if norm(q["stem"]) and norm(q["stem"]) in seen:
            continue                                   # same question photographed twice
        seen.add(norm(q["stem"]))
        qs.append(q)
    return qs


class Engine:
    def __init__(self, a):
        self.a = a
        self.cam = None                      # opened (and re-opened) by the camera thread
        self.cam_status = "Connecting to camera…"
        self.offline = Brain(a.ollama, a.vision_model, a.reasoner or a.vision_model, a.think)
        key = load_key()
        self.online = ClaudeBrain(a.online_model, key) if key else None
        print(f"[online] {'ready: ' + a.online_model if self.online else 'no API key -> online mode unavailable'}")
        self.voice = Voice(a.tts, a.voice, a.speed, a.piper_model)

        s = self._load_settings()
        self.roi = s.get("roi", DEFAULT_ROI)
        self.mode = s.get("mode", a.mode)
        if self.mode == "online" and not self.online:
            self.mode = "offline"

        self.lock = threading.Lock()
        self.frame = self.jpeg = self.dist = None
        self.frame_id = 0
        self.recent = deque(maxlen=15)       # last frames, to pick the sharpest one on SNAP
        self.pages = []                      # multi-page batch: [{"name": ..., "jpg": bytes}]
        self.auto = False
        self.want_auto = a.auto
        self.busy, self.stage = False, "Ready"
        self.armed, self.stable_since, self.prev_gray = True, None, None
        self.results, self.order, self.current, self.version = {}, [], None, 0
        self.job = 0
        os.makedirs(a.save_dir, exist_ok=True)
        os.makedirs(a.saved_dir, exist_ok=True)
        os.makedirs(os.path.join(a.save_dir, "pages"), exist_ok=True)
        threading.Thread(target=self._camera_loop, daemon=True).start()

    # ---------------- settings
    def _load_settings(self):
        try:
            with open(SETTINGS) as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_settings(self):
        with open(SETTINGS, "w") as f:
            json.dump({"roi": self.roi, "mode": self.mode}, f, indent=2)

    def set_roi(self, r):
        x, y = min(max(float(r["x"]), 0), 0.95), min(max(float(r["y"]), 0), 0.95)
        w, h = min(max(float(r["w"]), 0.05), 1 - x), min(max(float(r["h"]), 0.05), 1 - y)
        self.roi = {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}
        self._save_settings()
        self.bump()

    def set_mode(self, mode):
        if mode == "online" and not self.online:
            return False, "No API key. Put it in .anthropic_key and restart."
        self.mode = mode
        self._save_settings()
        self.bump()
        return True, f"{mode.capitalize()} mode"

    def bump(self):
        self.version += 1

    # ---------------- camera
    def has_depth(self):
        return self.cam is not None and self.cam.has_depth

    def _open_camera(self):
        """Keep trying until the camera opens, so a busy/unplugged camera never kills the app."""
        attempt = 0
        while self.cam is None:
            attempt += 1
            try:
                cam = open_camera(self.a.camera, self.a.width, self.a.height, self.a.fps, self.a.uvc,
                                  self.a.uvc_width, self.a.uvc_height, self.a.uvc_fps)
                self.cam, self.cam_status = cam, ""
                self.auto = self.want_auto
                print("[camera] connected")
            except BaseException as e:  # Camera() calls sys.exit on failure
                if isinstance(e, KeyboardInterrupt):
                    raise
                msg = str(e) or type(e).__name__
                if "temporarily unavailable" in msg or "busy" in msg.lower():
                    msg = "camera is busy (another program is using it)"
                self.cam_status = f"Camera not available: {msg}. Retrying… (try unplug/replug)"
                if attempt == 1 or attempt % 10 == 0:
                    print(f"[camera] {self.cam_status}")
                time.sleep(3)
            self.bump()

    def _camera_loop(self):
        last_ok = time.time()
        while True:
            if self.cam is None:
                self._open_camera()
            try:
                frame, dist = self.cam.read()
            except Exception as e:
                print(f"[camera] read error: {e}")
                frame, dist = None, None
            if frame is None:
                if time.time() - last_ok > 10:  # no frames for 10 s -> unplugged/hung: reopen it
                    print("[camera] lost - reconnecting")
                    try:
                        self.cam.close()
                    except Exception:
                        pass
                    self.cam, self.auto = None, False
                    last_ok = time.time()
                    with self.lock:
                        self.jpeg = None
                    self.cam_status = "Camera disconnected. Reconnecting…"
                    self.bump()
                time.sleep(0.05)
                continue
            last_ok = time.time()
            h, w = frame.shape[:2]
            prev = cv2.resize(frame, (960, int(h * 960 / w))) if w > 960 else frame
            ok, jpg = cv2.imencode(".jpg", prev, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with self.lock:
                self.frame, self.dist, self.jpeg = frame, dist, jpg.tobytes()
                self.frame_id += 1
                self.recent.append((time.time(), frame))
            if self.auto and not self.busy and self.voice.idle() and self._auto_ready(frame, dist):
                self.snap(source="auto")

    def _auto_ready(self, frame, dist):
        gray = cv2.cvtColor(cv2.resize(crop(frame, self.roi), (640, 360)), cv2.COLOR_BGR2GRAY)
        diff = (cv2.absdiff(gray, self.prev_gray).mean()
                if self.prev_gray is not None and self.prev_gray.shape == gray.shape else 0.0)
        moving = diff > 4
        self.prev_gray = gray
        if self.has_depth():
            if dist is None or not (self.a.near <= dist <= self.a.far):
                self.armed, self.stable_since = True, None
                return False
        else:
            # No depth (webcam): re-arm when the picture changes a lot (page removed / turned),
            # then fire once it is sharp and held still.
            ref = getattr(self, "snap_gray", None)
            if ref is None or (ref.shape == gray.shape and cv2.absdiff(gray, ref).mean() > 18):
                self.armed = True
            if not self.armed:
                return False
        if not self.armed or moving or sharpness(gray) < self.a.sharpness:
            self.stable_since = None
            return False
        self.stable_since = self.stable_since or time.time()
        if time.time() - self.stable_since >= self.a.hold:
            self.armed, self.stable_since = False, None
            self.snap_gray = gray
            return True
        return False

    # ---------------- actions
    def _best_crop_jpg(self):
        """Sharpest capture-area crop from the last ~0.8 s of frames (beats hand shake / blur)."""
        with self.lock:
            now = time.time()
            cands = [f for t, f in self.recent if now - t <= 0.8] or ([self.frame] if self.frame is not None else [])
        if not cands:
            return None
        best, best_score = None, -1.0
        for f in cands:
            c = crop(f, self.roi)
            g = cv2.cvtColor(cv2.resize(c, (640, max(1, int(640 * c.shape[0] / max(1, c.shape[1]))))),
                             cv2.COLOR_BGR2GRAY)
            score = sharpness(g)
            if score > best_score:
                best, best_score = c, score
        return cv2.imencode(".jpg", best, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()

    def _start(self, jpgs, source, ask=""):
        """Begin answering a list of page images. Caller must already hold busy=True."""
        self.voice.stop()
        self.job += 1
        threading.Thread(target=self._solve, args=(jpgs, source, self.mode, ask, self.job), daemon=True).start()

    def cancel(self):
        if not self.busy:
            return False, "Nothing is running"
        self.job += 1            # the running job's result is ignored when it finishes
        self.busy, self.stage = False, "Ready"
        self.voice.stop()
        self.bump()
        return True, "Cancelled"

    def snap(self, source="button", ask=""):
        with self.lock:
            if self.busy:
                return False, "Still working on the last photo"
            if self.frame is None:
                return False, "No camera picture yet"
            self.busy = True
        jpg = self._best_crop_jpg()
        if jpg is None:
            self.busy = False
            return False, "No camera picture yet"
        self._start([jpg], source, ask)
        return True, "Snapped"

    # ---------------- multi-page batch
    def add_page(self):
        if len(self.pages) >= self.a.max_pages:
            return False, f"Maximum {self.a.max_pages} pages - press Answer all"
        jpg = self._best_crop_jpg()
        if jpg is None:
            return False, "No camera picture yet"
        name = f"page_{datetime.now():%Y%m%d_%H%M%S_%f}.jpg"
        with open(os.path.join(self.a.save_dir, "pages", name), "wb") as f:
            f.write(jpg)
        self.pages.append({"name": name, "jpg": jpg})
        self.bump()
        return True, f"Page {len(self.pages)} added"

    def remove_page(self, i):
        if 0 <= i < len(self.pages):
            self.pages.pop(i)
            self.bump()
            return True, "Page removed"
        return False, "No such page"

    def clear_pages(self):
        self.pages = []
        self.bump()
        return True, "Pages cleared"

    def answer_pages(self, ask=""):
        if not self.pages:
            return False, "Add at least one page first"
        with self.lock:
            if self.busy:
                return False, "Still working on the last photo"
            self.busy = True
        jpgs = [p["jpg"] for p in self.pages]
        self.pages = []
        self._start(jpgs, "pages", ask)
        return True, f"Answering {len(jpgs)} page(s)"

    def _solve(self, jpgs, source, mode, ask="", job=0):
        rid = datetime.now().strftime("%Y%m%d_%H%M%S")
        names = [f"{rid}.jpg"] if len(jpgs) == 1 else [f"{rid}_p{i + 1}.jpg" for i in range(len(jpgs))]
        res = {"id": rid, "time": datetime.now().strftime("%H:%M:%S"), "source": source, "mode": mode,
               "image": f"/captures/{names[0]}", "images": [f"/captures/{n}" for n in names], "files": names,
               "pages": len(jpgs), "ask": ask, "questions": [], "error": None, "note": None, "done": False}
        t_start = time.time()
        try:
            for n, jpg in zip(names, jpgs):
                with open(os.path.join(self.a.save_dir, n), "wb") as f:
                    f.write(jpg)
            with self.lock:
                self.results[rid] = res
                self.order = (self.order + [rid])[-30:]
                self.current = rid
            if mode == "online":
                try:
                    self._solve_online(jpgs, res, ask)
                except Exception as e:
                    if isinstance(e, self.online.anthropic.AuthenticationError):
                        raise RuntimeError("API key was rejected. Check .anthropic_key.")
                    print(f"[online] failed ({type(e).__name__}: {e}) -> falling back to offline")
                    detail = str(getattr(e, "message", "") or e).replace("\n", " ")[:160]
                    res["note"] = f"Online failed ({type(e).__name__}: {detail}) - answered offline instead."
                    res["mode"] = "offline"
                    res["questions"] = []
                    self._solve_offline(jpgs, res, ask)
            else:
                self._solve_offline(jpgs, res, ask)
        except requests.exceptions.ConnectionError:
            res["error"] = "Can't reach the offline AI model. Is Ollama running?  (sudo systemctl start ollama)"
            self.voice.say("I can't reach the AI model.")
        except Exception as e:
            traceback.print_exc()
            res["error"] = f"Something went wrong: {e}"
        finally:
            res["seconds"] = round(time.time() - t_start, 1)
            res["done"] = True
            print(f"[{res['mode']}] done in {res['seconds']}s")
            with open(os.path.join(self.a.save_dir, f"{rid}.json"), "w") as f:
                json.dump(res, f, indent=2)
            if job == self.job:          # not cancelled / superseded
                self.busy, self.stage = False, "Ready"
            self.bump()

    def _unreadable(self, res):
        res["error"] = "I can't make out anything clearly. Adjust the capture box, hold it flat, in good light."
        self.voice.say("I can't make that out clearly. Please adjust the picture.")

    def _solve_online(self, jpgs, res, ask=""):
        self.stage = "Asking Claude (online)…" if len(jpgs) == 1 else f"Asking Claude about {len(jpgs)} pages…"
        self.bump()
        data = self.online.solve(jpgs, ask)
        qs = (data.get("questions") or [])[: self.a.max_questions * len(jpgs)]
        if not data.get("readable", True) or not qs:
            return self._unreadable(res)
        for q in qs:
            q.setdefault("options", [])
            q["kind"] = "mcq" if q["options"] else ("info" if q.get("kind") == "info" else "theory")
            if q["kind"] == "mcq":
                letter = norm_label(q.get("letter", ""))
                match = next((o for o in q["options"] if norm_label(o["label"]) == letter), None)
                q.update(letter=letter, option_text=match["text"] if match else "", reason=q.get("reason", "").strip())
                out = {"letter": letter, "reason": q["reason"]}
            else:
                q["answer"] = q.get("answer", "").strip()
                out = {"answer": q["answer"]}
            q["speech"] = compose_speech(q, out, multi=len(qs) > 1)
            q["pending"] = False
        res["questions"] = qs
        self.bump()
        for q in qs:
            self.voice.say(q["speech"])

    def _solve_offline(self, jpgs, res, ask=""):
        if (ask or "").strip():
            return self._describe_offline(jpgs, res, ask)
        qs = []
        for i, jpg in enumerate(jpgs, 1):
            self.stage = ("Reading the question (offline)…" if len(jpgs) == 1
                          else f"Reading page {i} of {len(jpgs)} (offline)…")
            self.bump()
            try:
                data = self.offline.transcribe(jpg)
            except (requests.exceptions.ConnectionError, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"[offline] page {i} unreadable: {e}")
                continue
            if data.get("readable", True):
                qs = merge_pages(qs, data.get("questions") or [])
        qs = qs[: self.a.max_questions * len(jpgs)]
        if not qs:  # no questions on the page: explain whatever it shows instead
            return self._describe_offline(jpgs, res, "")
        res["questions"] = [dict(q, kind="mcq" if q.get("options") else "theory", pending=True) for q in qs]
        self.bump()
        for i, q in enumerate(res["questions"]):
            self.stage = f"Solving {i + 1} of {len(qs)} (offline)…"
            self.bump()
            try:
                out = self.offline.answer(q)
            except (requests.exceptions.ConnectionError, KeyboardInterrupt):
                raise
            except Exception as e:  # one bad question must not lose the others
                print(f"[offline] Q{q.get('number')} failed: {e}")
                q.update(pending=False, answer="", failed=True)
                self.bump()
                continue
            if q["kind"] == "mcq":
                letter = norm_label(out.get("letter", ""))
                match = next((o for o in q["options"] if norm_label(o["label"]) == letter), None)
                q.update(letter=letter, option_text=match["text"] if match else "",
                         reason=out.get("reason", "").strip())
            else:
                q["answer"] = out.get("answer", "").strip()
            q["speech"] = compose_speech(q, out, multi=len(qs) > 1)
            q["pending"] = False
            self.bump()
            self.voice.say(q["speech"])

    def _describe_offline(self, jpgs, res, ask):
        items = []
        for i, jpg in enumerate(jpgs, 1):
            self.stage = f"Looking at the picture (offline)…" if len(jpgs) == 1 else f"Looking at page {i} of {len(jpgs)}…"
            self.bump()
            try:
                out = self.offline.describe(jpg, ask)
            except (requests.exceptions.ConnectionError, KeyboardInterrupt):
                raise
            except Exception as e:
                print(f"[offline] describe page {i} failed: {e}")
                continue
            if (out.get("answer") or "").strip():
                q = {"number": str(i) if len(jpgs) > 1 else "", "stem": out.get("title", "") or (ask or ""),
                     "options": [], "kind": "info", "answer": out["answer"].strip(), "pending": False}
                q["speech"] = compose_speech(q, {"answer": q["answer"]}, multi=False)
                items.append(q)
                res["questions"] = items
                self.bump()
                self.voice.say(q["speech"])
        if not items:
            self._unreadable(res)

    def save(self, rid=None):
        res = self.results.get(rid or self.current)
        if res is None:  # nothing answered yet -> save what is inside the capture box now
            with self.lock:
                frame = None if self.frame is None else crop(self.frame, self.roi).copy()
            if frame is None:
                return False, "No camera picture yet"
            name = f"photo_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            cv2.imwrite(os.path.join(self.a.saved_dir, name), frame)
            return True, f"Photo saved: {self.a.saved_dir}/{name}"
        rid = res["id"]
        files = res.get("files") or [f"{rid}.jpg"]
        for n in files:
            shutil.copy(os.path.join(self.a.save_dir, n), os.path.join(self.a.saved_dir, n))
        lines = [f"SnapAnswer - {rid} ({res.get('mode', '')})", ""]
        for q in res["questions"]:
            lines.append((f"Q{q.get('number') or ''}: " if q["kind"] != "info" else "") + q["stem"])
            for o in q.get("options") or []:
                lines.append(f"   {o['label']}. {o['text']}")
            if q["kind"] == "mcq":
                lines.append(f"ANSWER: {q.get('letter', '')}. {q.get('option_text', '')}")
                lines.append(f"REASON: {q.get('reason', '')}")
            else:
                lines.append(f"ANSWER: {q.get('answer', '')}")
            lines.append("")
        if res.get("error"):
            lines.append(f"Error: {res['error']}")
        with open(os.path.join(self.a.saved_dir, f"{rid}.txt"), "w") as f:
            f.write("\n".join(lines))
        return True, f"Saved: {self.a.saved_dir}/{rid} ({len(files)} photo(s) + .txt)"

    def repeat(self, rid=None):
        res = self.results.get(rid or self.current)
        speeches = [q["speech"] for q in (res or {}).get("questions", []) if q.get("speech")]
        if not speeches:
            return False, "Nothing to repeat yet"
        self.voice.stop()
        for s in speeches:
            self.voice.say(s)
        return True, "Repeating"

    def state(self):
        speaking = not self.voice.idle()
        history = []
        for rid in reversed(self.order):
            r = self.results[rid]
            bits = [q.get("letter") if q["kind"] == "mcq" else q["kind"] for q in r["questions"] if not q.get("pending")]
            history.append({"id": rid, "time": r["time"], "mode": r.get("mode"),
                            "summary": ", ".join(b for b in bits if b) or ("error" if r.get("error") else "…")})
        return {"status": self.stage if self.busy else ("Speaking…" if speaking else "Ready"),
                "busy": self.busy, "speaking": speaking, "auto": self.auto,
                "has_depth": self.has_depth(), "auto_available": self.cam is not None, "cam_status": self.cam_status, "dist": self.dist, "near": self.a.near, "far": self.a.far,
                "camera": getattr(self.cam, "info", "") if self.cam else "camera offline", "version": self.version,
                "mode": self.mode, "online_available": self.online is not None,
                "online_model": self.a.online_model, "offline_model": self.a.reasoner or self.a.vision_model,
                "roi": self.roi, "current": self.results.get(self.current), "history": history,
                "pages": [f"/captures/pages/{p['name']}" for p in self.pages], "max_pages": self.a.max_pages}


def build_app(eng):
    app = Flask(__name__, static_folder=None)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    @app.get("/")
    def index():
        r = send_from_directory(os.path.join(HERE, "web"), "index.html")
        r.headers["Cache-Control"] = "no-store"   # always load the newest page after an update
        return r

    @app.get("/video")
    def video():
        def gen():
            last = -1
            while True:
                with eng.lock:
                    fid, jpg = eng.frame_id, eng.jpeg
                if jpg is None or fid == last:
                    time.sleep(0.02)
                    continue
                last = fid
                yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                       + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/captures/<path:name>")
    def captures(name):
        return send_from_directory(os.path.abspath(eng.a.save_dir), name)

    @app.get("/state")
    def state():
        return jsonify(eng.state())

    @app.get("/result/<rid>")
    def result(rid):
        r = eng.results.get(rid)
        return jsonify(r) if r else abort(404)

    def reply(ok_msg):
        ok, msg = ok_msg
        return jsonify({"ok": ok, "message": msg})

    def ask_text():
        body = request.get_json(silent=True) or {}
        return str(body.get("ask", ""))[:500]

    @app.post("/snap")
    def snap():
        return reply(eng.snap(ask=ask_text()))

    @app.post("/cancel")
    def cancel():
        return reply(eng.cancel())

    @app.post("/page/add")
    def page_add():
        return reply(eng.add_page())

    @app.post("/page/remove")
    def page_remove():
        return reply(eng.remove_page(int(request.args.get("i", -1))))

    @app.post("/page/clear")
    def page_clear():
        return reply(eng.clear_pages())

    @app.post("/page/answer")
    def page_answer():
        return reply(eng.answer_pages(ask_text()))

    @app.post("/save")
    def save():
        return reply(eng.save(request.args.get("id")))

    @app.post("/repeat")
    def repeat():
        return reply(eng.repeat(request.args.get("id")))

    @app.post("/stop")
    def stop():
        eng.voice.stop()
        return reply((True, "Stopped"))

    @app.post("/auto")
    def auto():
        if eng.cam is None:
            return reply((False, "Camera not connected"))
        eng.auto = not eng.auto
        eng.armed = True
        return reply((True, f"Auto {'on' if eng.auto else 'off'}"))

    @app.post("/mode")
    def mode():
        m = request.args.get("m", "")
        if m not in ("online", "offline"):
            return reply((False, "Unknown mode"))
        return reply(eng.set_mode(m))

    @app.post("/roi")
    def roi():
        try:
            eng.set_roi(request.get_json(force=True))
            return reply((True, "Capture area saved"))
        except Exception as e:
            return reply((False, f"Bad capture area: {e}"))

    return app


def main():
    p = argparse.ArgumentParser(description="SnapAnswer browser dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--mode", choices=["online", "offline"], default="offline", help="Starting mode (remembered after)")
    p.add_argument("--online-model", default="claude-sonnet-5-5", help="e.g. claude-sonnet-5-5, claude-haiku-5-5")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--vision-model", default="qwen3-vl:8b")
    p.add_argument("--reasoner", default=None)
    p.add_argument("--think", default=None)
    p.add_argument("--tts", choices=["kokoro", "piper", "none"], default="kokoro")
    p.add_argument("--voice", default="bm_george")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--piper-model", default="en_GB-alan-medium.onnx")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--uvc", type=int, default=None, help="force webcam /dev/videoN")
    p.add_argument("--camera", choices=["auto", "uvc", "realsense"], default="auto",
                   help="auto = EMEET/webcam first, then RealSense")
    p.add_argument("--uvc-width", type=int, default=3840)
    p.add_argument("--uvc-height", type=int, default=2160)
    p.add_argument("--uvc-fps", type=int, default=30)
    p.add_argument("--auto", action="store_true")
    p.add_argument("--near", type=float, default=0.25)
    p.add_argument("--far", type=float, default=0.60)
    p.add_argument("--hold", type=float, default=1.2)
    p.add_argument("--sharpness", type=float, default=80.0)
    p.add_argument("--max-questions", type=int, default=8, help="per page")
    p.add_argument("--max-pages", type=int, default=10)
    p.add_argument("--save-dir", default="captures")
    p.add_argument("--saved-dir", default="saved")
    a = p.parse_args()

    eng = Engine(a)
    print(f"\n  Dashboard ready:  http://localhost:{a.port}   (mode: {eng.mode})\n")
    eng.voice.say("Dashboard ready.")
    build_app(eng).run(host=a.host, port=a.port, threaded=True)


if __name__ == "__main__":
    main()