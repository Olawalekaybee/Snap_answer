#!/usr/bin/env python3
"""
SnapAnswer - point an Intel RealSense D435i at a question and hear the answer.

Runs fully local on an NVIDIA DGX Spark:
  D435i colour frame -> vision model (Ollama) transcribes question + options
                     -> reasoning model answers (MCQ letter + reason, or brief theory)
                     -> Kokoro TTS speaks it in a natural human voice

Preview-window keys:   SPACE capture   A auto-capture (depth)   R repeat   Q/ESC quit
Headless (--no-preview) terminal keys:   Enter capture   r repeat   q quit
"""

import argparse
import base64
import glob
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime

import cv2
import numpy as np
import requests

# =============================================================== prompts

TRANSCRIBE_PROMPT = """Transcribe every question visible in this photo of a question paper or screen.
- Copy each question stem exactly, including numbers, units and symbols. Write maths as plain text (x^2, sqrt(3), 3/4, pi).
- If a question has choices labelled A, B, C, D (or a-e, i-iv, 1-4), put each choice in "options" with its label exactly as printed. Otherwise "options" is an empty list.
- If a diagram, table or graph is needed to answer, describe the facts it shows in "diagram_notes"; otherwise use "".
- Ignore page headers, general instructions, marks, fingers and background.
- Do NOT answer anything.
- If no question is legible, set "readable" to false and "questions" to []."""

ANSWER_SYSTEM = """You are an expert teacher answering exam questions out loud. Your words go straight to a text-to-speech engine.
Rules:
- Be correct above all. Work the problem out fully and check it before you answer.
- Never restate the question. Never say the same idea twice in different words. No filler ("great question", "basically", "in conclusion", "it is important to note").
- Speak like a clear human teacher: plain sentences only, no markdown, no lists, no symbols. Read maths naturally ("x squared", "root three", "metres per second").
- Always give numerical results with units.
- Reply with JSON only."""

MCQ_TEMPLATE = """Multiple-choice question:
{stem}
{notes}
Options:
{options}

Return JSON with:
"letter": the label of the single best option, exactly as printed.
"reason": one or two short sentences giving the specific fact, rule or calculation that makes it right. Mention why the most tempting wrong option fails only if that adds real value. Do not repeat the option text and never say "because it is correct"."""

THEORY_TEMPLATE = """Theory question:
{stem}
{notes}

Return JSON with:
"answer": the answer in at most three short sentences, starting directly with the answer itself. If it asks for several points, name them in one sentence. If it is a calculation, give the final result with units plus the single key step."""

TRANSCRIBE_SCHEMA = {
    "type": "object",
    "properties": {
        "readable": {"type": "boolean"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "number": {"type": "string"},
                    "stem": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"label": {"type": "string"}, "text": {"type": "string"}},
                            "required": ["label", "text"],
                        },
                    },
                    "diagram_notes": {"type": "string"},
                },
                "required": ["number", "stem", "options", "diagram_notes"],
            },
        },
    },
    "required": ["readable", "questions"],
}
MCQ_SCHEMA = {"type": "object", "properties": {"letter": {"type": "string"}, "reason": {"type": "string"}},
              "required": ["letter", "reason"]}
THEORY_SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}

DESCRIBE_PROMPT = """Look at this photo. It may be a passage, diagram, chart, formula, code, sign, label or object.
{task}
Your words will be read aloud: plain spoken sentences, no lists or symbols, at most three short sentences, no filler.
Return JSON with "title" (a few words naming what is shown) and "answer"."""

DESCRIBE_SCHEMA = {"type": "object", "properties": {"title": {"type": "string"}, "answer": {"type": "string"}},
                   "required": ["title", "answer"]}

# =============================================================== camera


class Camera:
    """D435i through pyrealsense2 (colour + depth); optional plain-UVC fallback (colour only)."""

    def __init__(self, width, height, fps, uvc_device=None):
        self.rs_pipe = None
        if uvc_device is not None:
            self._open_uvc(uvc_device, width, height)
            return
        try:
            import pyrealsense2 as rs
        except ImportError:
            sys.exit("[camera] pyrealsense2 not installed. Run setup_spark.sh, or use --uvc N "
                     "(find N with: v4l2-ctl --list-devices).")
        self.rs = rs
        dev = self._find_device()
        usb = (dev.get_info(rs.camera_info.usb_type_descriptor)
               if dev.supports(rs.camera_info.usb_type_descriptor) else "?")
        fw = dev.get_info(rs.camera_info.firmware_version)
        self.serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[camera] {dev.get_info(rs.camera_info.name)}  USB {usb}  firmware {fw}")
        self.info = f"{dev.get_info(rs.camera_info.name)} · USB {usb}"
        if usb.startswith("2"):
            print("[camera] WARNING: connected at USB 2 - limited to 720p. "
                  "Use a blue USB 3 port and a USB 3 (data) cable for sharp text.")

        # Ask the camera which modes it actually offers on this connection, instead of guessing.
        # (Guessing wrong on USB 2 can make the D435i drop off the bus.)
        color_modes, depth_modes = set(), set()
        for s in dev.query_sensors():
            for prof in s.get_stream_profiles():
                if not prof.is_video_stream_profile():
                    continue
                v = prof.as_video_stream_profile()
                key = (v.width(), v.height(), v.fps())
                if prof.stream_type() == rs.stream.color and prof.format() == rs.format.bgr8:
                    color_modes.add(key)
                elif prof.stream_type() == rs.stream.depth and prof.format() == rs.format.z16:
                    depth_modes.add(key)

        combos = []
        for w, h in [(width, height), (1920, 1080), (1280, 720), (960, 540), (640, 480)]:
            for f in [fps, 15, 30, 6]:
                if (w, h, f) not in color_modes:
                    continue
                for dw, dh in [(848, 480), (640, 480), (480, 270)]:
                    if (dw, dh, f) in depth_modes:
                        combos.append((w, h, f, (dw, dh)))
                        break
        # colour-only fallbacks (no auto-capture)
        for w, h in [(1920, 1080), (1280, 720), (640, 480)]:
            for f in [15, 30, 6]:
                if (w, h, f) in color_modes:
                    combos.append((w, h, f, None))
                    break
        combos = list(dict.fromkeys(combos))
        if not combos:
            sys.exit("[camera] The camera reports no usable colour modes. Unplug/replug it and try again.")

        for w, h, f, d in combos:
            pipe = rs.pipeline()
            try:
                cfg = rs.config()
                cfg.enable_device(self.serial)
                cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, f)
                if d:
                    cfg.enable_stream(rs.stream.depth, d[0], d[1], rs.format.z16, f)
                pipe.start(cfg)
                for _ in range(max(10, 2 * f)):  # let auto-exposure settle
                    pipe.wait_for_frames(timeout_ms=5000)
                self.rs_pipe, self.depth_on = pipe, bool(d)
                self.info += f" · {w}x{h}@{f}" + (" + depth" if d else "")
                print(f"[camera] running: colour {w}x{h}@{f}"
                      + (f" + depth {d[0]}x{d[1]}" if d else " (no depth -> auto-capture off)"))
                return
            except Exception as e:
                print(f"[camera] {w}x{h}@{f}{' + depth' if d else ''} failed: {e}")
                try:
                    pipe.stop()
                except Exception:
                    pass
                self._find_device(wait=True)  # camera may have reset; wait for it to come back
        sys.exit("[camera] Could not start any stream. Unplug/replug the camera and run `rs-enumerate-devices`.")

    def _find_device(self, wait=False):
        rs = self.rs
        for i in range(20):  # up to ~10 s, so a camera that is still re-connecting is picked up
            devs = rs.context().query_devices()
            if len(devs):
                if wait or i:
                    time.sleep(1.0)
                return devs[0]
            if i == 0:
                print("[camera] waiting for the camera to appear...")
            time.sleep(0.5)
        sys.exit("[camera] No RealSense camera found. Unplug and replug it (USB 3 port), "
                 "check `rs-enumerate-devices`, then run again.")

    def _open_uvc(self, idx, width, height):
        self.cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            sys.exit(f"[camera] Cannot open /dev/video{idx}")
        self.info = f"UVC /dev/video{idx}"
        print(f"[camera] UVC /dev/video{idx} (colour only, no auto-capture)")

    @property
    def has_depth(self):
        return self.rs_pipe is not None and getattr(self, "depth_on", False)

    def read(self):
        """Return (bgr_frame, distance_to_centre_in_metres or None)."""
        if self.rs_pipe is None:
            ok, img = self.cap.read()
            return (img if ok else None), None
        try:
            frames = self.rs_pipe.wait_for_frames(timeout_ms=5000)
        except RuntimeError as e:
            print(f"[camera] frame timeout: {e}")
            return None, None
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color:
            return None, None
        img = np.asanyarray(color.get_data()).copy()
        dist = None
        if depth:
            cx, cy = depth.get_width() // 2, depth.get_height() // 2
            vals = [depth.get_distance(x, y) for x in range(cx - 40, cx + 41, 10) for y in range(cy - 40, cy + 41, 10)]
            vals = [v for v in vals if v > 0]
            dist = float(np.median(vals)) if vals else None
        return img, dist

    def close(self):
        if self.rs_pipe is not None:
            self.rs_pipe.stop()
        else:
            self.cap.release()


def list_v4l2():
    """[(index, name)] of /dev/videoN devices, from sysfs."""
    out = []
    for p in glob.glob("/sys/class/video4linux/video*"):
        try:
            idx = int(re.sub(r"\D", "", os.path.basename(p)))
            with open(os.path.join(p, "name")) as f:
                out.append((idx, f.read().strip()))
        except Exception:
            pass
    return sorted(out)


class UVCCamera:
    """Ordinary USB webcam (e.g. EMEET SmartCam C60E 4K) via V4L2: MJPEG, high resolution, autofocus."""
    has_depth = False
    rs_pipe = None

    def __init__(self, device=None, width=3840, height=2160, fps=30, prefer="emeet"):
        cands = [device] if device is not None else self.candidates(prefer)
        if not cands:
            raise RuntimeError("No webcam found - check the USB cable (`v4l2-ctl --list-devices`)")
        names = dict(list_v4l2())
        last = "webcam would not start"
        for idx in cands:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            # MJPEG is what makes 4K possible over USB; raw YUYV is limited to low resolutions.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # always the newest frame, no lag
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            img = None
            for _ in range(30):                     # first frames can be empty while it starts
                ok, img = cap.read()
                if ok and img is not None:
                    break
                img = None
            if img is None:
                cap.release()
                last = f"/dev/video{idx} gives no picture"
                continue
            h, w = img.shape[:2]
            self.cap, self.device = cap, idx
            self.info = f"{names.get(idx, 'Webcam')} · {w}x{h}"
            print(f"[camera] {self.info} on /dev/video{idx} (autofocus on)")
            if w < width:
                print(f"[camera] note: asked for {width}x{height}, camera gave {w}x{h}")
            return
        raise RuntimeError(last)

    @staticmethod
    def candidates(prefer="emeet"):
        devs = list_v4l2()
        pref = [i for i, n in devs if prefer and prefer.lower() in n.lower()]
        rest = [i for i, n in devs if "realsense" not in n.lower() and i not in pref]
        return pref + rest

    def read(self):
        ok, img = self.cap.read()
        return (img if ok else None), None

    def close(self):
        self.cap.release()


def open_camera(kind="auto", width=1920, height=1080, fps=15, uvc=None,
                uvc_width=3840, uvc_height=2160, uvc_fps=30):
    """auto = EMEET (or other webcam named in prefer) first, then RealSense, then any webcam."""
    if kind == "realsense":
        return Camera(width, height, fps)
    if kind == "uvc" or uvc is not None:
        return UVCCamera(uvc, uvc_width, uvc_height, uvc_fps)
    if any("emeet" in n.lower() for _, n in list_v4l2()):
        try:
            return UVCCamera(None, uvc_width, uvc_height, uvc_fps)
        except Exception as e:
            print(f"[camera] EMEET found but failed to start: {e}")
    try:
        import pyrealsense2 as rs
        if len(rs.context().query_devices()):
            return Camera(width, height, fps)
    except ImportError:
        pass
    return UVCCamera(None, uvc_width, uvc_height, uvc_fps)


def small_gray(frame):
    h, w = frame.shape[:2]
    crop = frame[h // 6: 5 * h // 6, w // 6: 5 * w // 6]
    return cv2.cvtColor(cv2.resize(crop, (640, 360)), cv2.COLOR_BGR2GRAY)


def sharpness(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# =============================================================== LLM (Ollama)


class Brain:
    def __init__(self, host, vision_model, reasoner_model, think=None):
        self.host, self.vision, self.reasoner, self.think = host.rstrip("/"), vision_model, reasoner_model, think

    def _chat(self, model, messages, schema, temperature, think=None):
        """Ask Ollama for JSON. Tries 3 ways, because some models (e.g. Qwen3 'thinking'
        builds) return an empty answer when forced into a schema or told not to think."""
        attempts = [
            {"format": schema, "think": think or False},   # strict schema, no thinking
            {"format": "json", "think": think or False},   # plain JSON mode
            {"format": None, "think": think},              # model's default behaviour
        ]
        if "gpt-oss" in model:  # gpt-oss can't switch thinking off
            attempts = [{"format": schema, "think": think or "low"}, {"format": None, "think": think or "low"}]
        last = ""
        for i, opt in enumerate(attempts, 1):
            body = {"model": model, "messages": messages, "stream": False, "keep_alive": "30m",
                    "options": {"temperature": temperature, "num_ctx": 16384}}
            if opt["format"] is not None:
                body["format"] = opt["format"]
            if opt["think"] is not None:
                body["think"] = opt["think"]
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=600)
            if r.status_code == 400 and "think" in body:  # model doesn't support the think flag
                body.pop("think")
                r = requests.post(f"{self.host}/api/chat", json=body, timeout=600)
            r.raise_for_status()
            m = r.json().get("message", {})
            for text in (m.get("content") or "", m.get("thinking") or ""):
                if text.strip():
                    try:
                        return parse_json(text)
                    except (ValueError, json.JSONDecodeError):
                        last = text
            print(f"[offline] attempt {i} with {model}: no usable JSON "
                  f"(content={len(m.get('content') or '')} chars, thinking={len(m.get('thinking') or '')} chars)")
        raise ValueError(f"Model did not return JSON. Last reply: {last[:150]!r}")

    @staticmethod
    def _img(jpg_bytes, max_side=1600):
        """Local vision models get slow on 4K images; ~1600 px keeps text sharp and speed good."""
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        if img is not None and max(img.shape[:2]) > max_side:
            sc = max_side / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            jpg_bytes = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        return base64.b64encode(jpg_bytes).decode()

    def transcribe(self, jpg_bytes):
        msg = {"role": "user", "content": TRANSCRIBE_PROMPT,
               "images": [self._img(jpg_bytes)]}
        return self._chat(self.vision, [msg], TRANSCRIBE_SCHEMA, temperature=0.0)

    def describe(self, jpg_bytes, ask=None):
        task = (f'Do this for the photo: "{ask.strip()}".' if (ask or "").strip()
                else "Say clearly what it is and explain what it means or its key points.")
        msg = {"role": "user", "content": DESCRIBE_PROMPT.format(task=task),
               "images": [self._img(jpg_bytes)]}
        return self._chat(self.vision, [msg], DESCRIBE_SCHEMA, temperature=0.2)

    def answer(self, q):
        notes = f"Diagram/table facts: {q['diagram_notes']}" if q.get("diagram_notes") else ""
        if q.get("options"):
            opts = "\n".join(f"{o['label']}. {o['text']}" for o in q["options"])
            prompt, schema = MCQ_TEMPLATE.format(stem=q["stem"], notes=notes, options=opts), MCQ_SCHEMA
        else:
            prompt, schema = THEORY_TEMPLATE.format(stem=q["stem"], notes=notes), THEORY_SCHEMA
        msgs = [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}]
        return self._chat(self.reasoner, msgs, schema, temperature=0.2, think=self.think)


def parse_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise ValueError(f"Model did not return JSON: {text[:200]}")


def norm_label(s):
    return re.sub(r"[^A-Za-z0-9]", "", str(s)).upper()


def compose_speech(q, result, multi):
    prefix = f"Question {q.get('number') or ''}. " if multi else ""
    if q.get("options"):
        letter = norm_label(result.get("letter", ""))
        match = next((o for o in q["options"] if norm_label(o["label"]) == letter), None)
        opt_text = f" {match['text'].rstrip('.')}." if match else ""
        return f"{prefix}The answer is {letter}.{opt_text} {result.get('reason', '').strip()}"
    return f"{prefix}{result.get('answer', '').strip()}"


def clean_for_speech(text):
    text = re.sub(r"[*_#`>|]", "", text)
    text = text.replace("->", " to ").replace("=", " equals ").replace("%", " percent")
    return re.sub(r"\s+", " ", text).strip()

# =============================================================== voice


class Voice:
    """Queued speech so the next question can be solved while the current one is spoken."""

    def __init__(self, engine, voice, speed, piper_model):
        self.engine, self.voice, self.speed, self.piper_model = engine, voice, speed, piper_model
        self.q = queue.Queue()
        if engine == "kokoro":
            from kokoro import KPipeline
            import sounddevice as sd
            self.sd = sd
            # lang code is the voice's first letter: a = US English, b = British English
            self.kpipe = KPipeline(lang_code=voice[0], repo_id="hexgrad/Kokoro-82M")
        threading.Thread(target=self._worker, daemon=True).start()

    def say(self, text):
        print(f"\n>>> {text}\n")
        self.q.put(clean_for_speech(text))

    def idle(self):
        return self.q.unfinished_tasks == 0

    def stop(self):
        """Cut off current speech and drop anything queued."""
        self._interrupt = True
        while True:
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break
        if self.engine == "kokoro":
            self.sd.stop()

    _interrupt = False

    def _worker(self):
        while True:
            text = self.q.get()
            self._interrupt = False
            t0 = time.time()
            try:
                if self.engine == "kokoro":
                    for _, _, audio in self.kpipe(text, voice=self.voice, speed=self.speed):
                        if self._interrupt:
                            break
                        audio = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
                        self.sd.play(audio, 24000)
                        self.sd.wait()
                    print(f"[voice] finished speaking ({time.time() - t0:.1f}s)"
                          + (" - interrupted" if self._interrupt else ""))
                elif self.engine == "piper":
                    p1 = subprocess.Popen(["piper", "--model", self.piper_model, "--output-raw"],
                                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    p2 = subprocess.Popen(["aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-"],
                                          stdin=p1.stdout)
                    p1.stdin.write(text.encode())
                    p1.stdin.close()
                    p2.wait()
            except Exception as e:
                print(f"[voice] {e}")
            finally:
                self.q.task_done()

# =============================================================== app


class App:
    def __init__(self, a):
        self.a = a
        self.cam = Camera(a.width, a.height, a.fps, a.uvc)
        self.brain = Brain(a.ollama, a.vision_model, a.reasoner or a.vision_model, a.think)
        self.voice = Voice(a.tts, a.voice, a.speed, a.piper_model)
        self.auto = a.auto and self.cam.has_depth
        self.busy, self.armed, self.stable_since, self.prev_gray = False, True, None, None
        self.last_speech, self.status = "", "Ready"
        self.cmd = queue.Queue()
        os.makedirs(a.save_dir, exist_ok=True)

    # ---------- solving (runs in a background thread)
    def solve(self, frame):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = {"time": stamp}
        try:
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            jpg = jpg.tobytes()
            with open(os.path.join(self.a.save_dir, f"{stamp}.jpg"), "wb") as f:
                f.write(jpg)

            self.status, t0 = "Reading the question...", time.time()
            data = self.brain.transcribe(jpg)
            log["transcription"] = data
            qs = data.get("questions") or []
            print(f"[vision] {len(qs)} question(s) in {time.time() - t0:.1f}s")
            if not data.get("readable", True) or not qs:
                self.voice.say("I can't read a question clearly. Hold the page flat, a bit closer, in good light.")
                return

            qs = qs[: self.a.max_questions]
            spoken = []
            for i, q in enumerate(qs, 1):
                self.status = f"Solving {i}/{len(qs)}..."
                t1 = time.time()
                res = self.brain.answer(q)
                text = compose_speech(q, res, multi=len(qs) > 1)
                print(f"[answer] {time.time() - t1:.1f}s")
                self.voice.say(text)  # speak now; next question solves meanwhile
                spoken.append(text)
                log.setdefault("answers", []).append(res)
            self.last_speech = " ".join(spoken)
        except requests.exceptions.ConnectionError:
            self.voice.say("I can't reach the AI model. Please check that Ollama is running.")
        except Exception:
            traceback.print_exc()
            self.voice.say("Something went wrong. Please try again.")
        finally:
            with open(os.path.join(self.a.save_dir, f"{stamp}.json"), "w") as f:
                json.dump(log, f, indent=2)
            self.busy, self.status = False, "Ready"

    def trigger(self, frame, manual=False):
        if self.busy:
            if manual:
                print(f"[capture] ignored - still working on the last photo ({self.status})")
            return
        if not self.voice.idle():
            if not manual:
                return            # auto mode waits for speech to finish
            self.voice.stop()     # manual press cuts the voice off and captures
        print("[capture] photo taken, reading the question...")
        self.busy = True
        threading.Thread(target=self.solve, args=(frame.copy(),), daemon=True).start()

    # ---------- depth-based auto capture: page held 25-60 cm away, sharp and still
    def auto_ready(self, frame, dist):
        gray = small_gray(frame)
        moving = self.prev_gray is not None and cv2.absdiff(gray, self.prev_gray).mean() > 4
        self.prev_gray = gray
        if dist is None or not (self.a.near <= dist <= self.a.far):
            self.armed, self.stable_since = True, None  # page removed -> re-arm
            return False
        if not self.armed or moving or sharpness(gray) < self.a.sharpness:
            self.stable_since = None
            return False
        self.stable_since = self.stable_since or time.time()
        if time.time() - self.stable_since >= self.a.hold:
            self.armed, self.stable_since = False, None  # one shot per page
            return True
        return False

    # ---------- UI
    def draw(self, frame, dist):
        view = cv2.resize(frame, (960, int(960 * frame.shape[0] / frame.shape[1])))
        h, w = view.shape[:2]
        cv2.rectangle(view, (w // 6, h // 6), (5 * w // 6, 5 * h // 6), (0, 200, 255), 2)
        lines = [f"{self.status}   |   auto: {'ON' if self.auto else 'off'}"
                 + (f"   |   {dist * 100:.0f} cm" if dist else ""),
                 "SPACE capture   A auto   R repeat   Q quit"]
        for i, t in enumerate(lines):
            cv2.putText(view, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
            cv2.putText(view, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("SnapAnswer", view)

    def stdin_loop(self):
        print("Headless mode: Enter = capture, r = repeat, q = quit")
        for line in sys.stdin:
            self.cmd.put(line.strip().lower() or "capture")

    def run(self):
        if self.a.no_preview:
            threading.Thread(target=self.stdin_loop, daemon=True).start()
        self.voice.say("Ready. Show me a question.")
        try:
            while True:
                frame, dist = self.cam.read()
                if frame is None:
                    continue
                if self.auto and not self.busy and self.auto_ready(frame, dist):
                    self.trigger(frame)

                key = ""
                if self.a.no_preview:
                    try:
                        key = self.cmd.get_nowait()
                    except queue.Empty:
                        pass
                else:
                    self.draw(frame, dist)
                    k = cv2.waitKey(1) & 0xFF
                    key = {32: "capture", ord("a"): "a", ord("r"): "r", ord("q"): "q", 27: "q"}.get(k, "")

                if key == "q":
                    break
                if key == "capture":
                    self.trigger(frame, manual=True)
                elif key == "r" and self.last_speech:
                    self.voice.say(self.last_speech)
                elif key == "a":
                    self.auto = (not self.auto) and self.cam.has_depth
        finally:
            self.cam.close()
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="RealSense D435i question answerer with a human voice")
    p.add_argument("--ollama", default="http://localhost:11434")
    p.add_argument("--vision-model", default="qwen3-vl:8b", help="Ollama vision model that reads the page")
    p.add_argument("--reasoner", default=None, help="Ollama model that answers (default: same as vision model)")
    p.add_argument("--think", default=None, help="Reasoning effort for thinking models, e.g. low/medium/high for gpt-oss")
    p.add_argument("--tts", choices=["kokoro", "piper", "none"], default="kokoro")
    p.add_argument("--voice", default="bm_george", help="Kokoro voice: bm_george, bf_emma, am_michael, af_heart ...")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--piper-model", default="en_GB-alan-medium.onnx")
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--uvc", type=int, default=None, help="Use /dev/videoN via OpenCV instead of pyrealsense2")
    p.add_argument("--auto", action="store_true", help="Auto-capture when a page is held steady in range")
    p.add_argument("--near", type=float, default=0.25, help="Auto-capture min distance (m)")
    p.add_argument("--far", type=float, default=0.60, help="Auto-capture max distance (m)")
    p.add_argument("--hold", type=float, default=1.2, help="Seconds the page must be steady")
    p.add_argument("--sharpness", type=float, default=80.0, help="Min focus score for auto-capture")
    p.add_argument("--max-questions", type=int, default=5)
    p.add_argument("--save-dir", default="captures")
    p.add_argument("--no-preview", action="store_true", help="Headless (SSH) mode")
    App(p.parse_args()).run()


if __name__ == "__main__":
    main()