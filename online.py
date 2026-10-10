"""
Online mode for SnapAnswer: one Claude API call reads the photo AND answers every question.

API key (from console.anthropic.com - billed separately from a Claude Pro/Max subscription):
  either   export ANTHROPIC_API_KEY=sk-ant-...
  or       put the key alone in a file named  .anthropic_key  next to this file
"""

import base64
import json
import re
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_FILE = os.path.join(HERE, ".anthropic_key")

SYSTEM = """You are an expert teacher. You are shown a photo (often exam questions, but it can be anything: a passage, diagram, chart, formula, code, sign, label or object). Your words are read aloud by a text-to-speech engine.
Rules:
- Correctness above all. Work each problem out fully and check it before answering.
- Copy each question stem exactly as printed (maths as plain text, e.g. x^2, sqrt(3)).
- If a question has labelled choices (A-D, a-e, i-iv, 1-4) it is "mcq": give the single best label exactly as printed in "letter", and in "reason" one or two short sentences stating the specific fact, rule or calculation that makes it right. Mention why the most tempting wrong option fails only if it adds real value. Never say "because it is correct".
- Otherwise it is "theory": put the answer in "answer" in at most three short sentences, starting directly with the answer. Several points -> name them in one sentence. Calculation -> final result with units plus the single key step.
- Never restate the question, never say the same idea twice, no filler. Plain spoken sentences: no markdown, lists or symbols; read maths naturally ("x squared", "root three").
- If the photo has no explicit question, still help: return one item with kind "info", "stem" = a short title of what is shown (e.g. "Circuit diagram of a full-wave rectifier"), and "answer" = a brief, clear explanation of what it is, what it means or its key points (at most three sentences).
- If the user gives a request, follow it for the photo (it overrides the default behaviour) and put the reply in one "info" item, still brief and spoken-style.
- Ignore headers, instructions, marks, fingers and background. Set readable to false only if nothing in the photo can be made out at all."""

TOOL = {
    "name": "submit_answers",
    "description": "Submit the transcribed questions and their answers.",
    "input_schema": {
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
                        "kind": {"type": "string", "enum": ["mcq", "theory", "info"]},
                        "letter": {"type": "string", "description": "mcq only"},
                        "reason": {"type": "string", "description": "mcq only"},
                        "answer": {"type": "string", "description": "theory and info"},
                    },
                    "required": ["number", "stem", "options", "kind"],
                },
            },
        },
        "required": ["readable", "questions"],
    },
}


def load_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key and os.path.exists(KEY_FILE):
        with open(KEY_FILE) as f:
            key = f.read().strip()
    return key or None


class ClaudeBrain:
    def __init__(self, model, api_key):
        import anthropic
        self.anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key, timeout=180.0, max_retries=1)

    @staticmethod
    def _prep(jpg_bytes, max_side):
        # Claude works best with images up to ~1568 px on the long side; smaller = faster upload.
        img = cv2.imdecode(np.frombuffer(jpg_bytes, np.uint8), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        if max(h, w) > max_side:
            s = max_side / max(h, w)
            img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()).decode()

    def solve(self, jpgs, ask=None):
        """jpgs: one JPEG (bytes) or a list of JPEGs = consecutive pages of the same paper."""
        if isinstance(jpgs, (bytes, bytearray)):
            jpgs = [jpgs]
        n = len(jpgs)
        content = []
        for i, jpg in enumerate(jpgs, 1):
            if n > 1:
                content.append({"type": "text", "text": f"Page {i} of {n}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                                        "data": self._prep(jpg, 1568 if n <= 4 else 1200)}})
        if n == 1:
            instr = ("Read and answer every question in this photo (or, if there is no question, explain what it "
                     "shows), then call submit_answers with the result.")
        else:
            instr = (f"These {n} photos are consecutive pages of the same question paper, in order. "
                   "A question may continue from one page onto the next (for example the stem on one page "
                   "and its options on the next) - treat it as one question. If two photos overlap, answer "
                   "each question only once. Answer every question in page order, then call submit_answers.")
        if (ask or "").strip():
            instr = f'User request: "{ask.strip()}". Apply it to the photo(s), then call submit_answers.'
        content.append({"type": "text", "text": instr})
        # Some newer models don't allow *forcing* a tool, so we offer it ("auto") and tell
        # the model to use it. If it answers in plain text instead, we read the JSON from that.
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=min(4000 + 2000 * (n - 1), 16000),
            system=SYSTEM + "\nAlways reply by calling the submit_answers tool exactly once.",
            tools=[TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": content}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "submit_answers":
                return block.input
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Claude returned no structured answer: {text[:150]!r}")