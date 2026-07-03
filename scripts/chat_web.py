#!/usr/bin/env python
"""Minimal local web chat: ThoughtVectors thinker or SmolLM2-135M, CPU.

Usage:
    .venv/bin/python scripts/chat_web.py [--ckpt checkpoints/FINAL_12H/best.pt] [--port 7860]

Serves a single-page chat UI backed by a model picked per-request:
  thoughtvec  ChatSession on the thinker checkpoint (loaded at startup)
  smollm      HuggingFaceTB/SmolLM2-135M-Instruct via transformers (lazy-loaded
              on first use; downloads ~270MB the first time)
One global session per model; "new chat" clears the active model's history.
CPU inference — GPU chat is HIP-broken on gfx1031 (RESEARCH_LOG 2026-06-25).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thoughtvec.chat import ChatSession  # noqa: E402

SMOLLM_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


class SmolLMSession:
    """ChatSession-compatible wrapper around SmolLM2-135M-Instruct (CPU)."""

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(SMOLLM_ID)
        self.model = AutoModelForCausalLM.from_pretrained(SMOLLM_ID)
        self.model.eval()
        self.history: list[dict[str, str]] = []

    def reset(self) -> None:
        self.history = []

    def reply(self, message: str, temperature: float = 0.0) -> str:
        self.history.append({"role": "user", "content": message})
        enc = self.tok.apply_chat_template(
            self.history, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        n_in = enc["input_ids"].shape[1]
        do_sample = temperature > 0
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=200,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=0.9 if do_sample else None,
                pad_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        self.history.append({"role": "assistant", "content": text})
        return text

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>ThoughtVectors chat</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; margin: 0; }
  body { font: 15px/1.5 system-ui, sans-serif; background: #14151a; color: #e6e6e9;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 16px; background: #1c1d24; display: flex; gap: 12px;
           align-items: center; border-bottom: 1px solid #2a2b33; flex-wrap: wrap; }
  header h1 { font-size: 15px; font-weight: 600; margin-right: auto; }
  header label { font-size: 13px; color: #9a9aa3; display: flex; gap: 6px; align-items: center; }
  #temp { width: 110px; }
  select { background: #14151a; color: #e6e6e9; border: 1px solid #2a2b33;
           border-radius: 6px; padding: 5px 8px; font-size: 13px; }
  button { background: #2f3450; color: #e6e6e9; border: 0; border-radius: 6px;
           padding: 6px 12px; cursor: pointer; font-size: 13px; }
  button:hover { background: #3a4064; }
  #log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 8px; }
  .msg { max-width: 70%; padding: 8px 12px; border-radius: 12px; white-space: pre-wrap; }
  .user { align-self: flex-end; background: #2f3450; }
  .bot  { align-self: flex-start; background: #23242c; }
  .bot.pending { color: #9a9aa3; font-style: italic; }
  form { display: flex; gap: 8px; padding: 12px 16px; background: #1c1d24;
         border-top: 1px solid #2a2b33; }
  input[type=text] { flex: 1; background: #14151a; color: #e6e6e9; border: 1px solid #2a2b33;
          border-radius: 8px; padding: 10px 12px; font-size: 15px; }
  input[type=text]:focus { outline: 1px solid #4a5284; }
</style></head><body>
<header>
  <h1>local chat</h1>
  <label>model
    <select id="model">
      <option value="thoughtvec">ThoughtVectors · __CKPT__</option>
      <option value="smollm">SmolLM2-135M-Instruct</option>
    </select></label>
  <label>temp <input id="temp" type="range" min="0" max="1.2" step="0.1" value="0">
    <span id="tval">0.0</span></label>
  <button id="reset">new chat</button>
</header>
<div id="log"></div>
<form id="f"><input id="box" type="text" autocomplete="off"
  placeholder="say something… (CPU inference, a few seconds per reply)" autofocus>
  <button>send</button></form>
<script>
const log = document.getElementById("log"), box = document.getElementById("box");
const temp = document.getElementById("temp"), tval = document.getElementById("tval");
const model = document.getElementById("model");
temp.oninput = () => tval.textContent = (+temp.value).toFixed(1);
model.onchange = async () => {  // fresh conversation on model switch (UI + backend agree)
  await fetch("/api/reset", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model.value }) });
  log.innerHTML = ""; box.focus();
};
function add(cls, text) {
  const d = document.createElement("div");
  d.className = "msg " + cls; d.textContent = text;
  log.appendChild(d); log.scrollTop = log.scrollHeight; return d;
}
document.getElementById("f").onsubmit = async e => {
  e.preventDefault();
  const text = box.value.trim(); if (!text) return;
  box.value = ""; box.disabled = true;
  add("user", text);
  const p = add("bot pending", "thinking…");
  try {
    const r = await fetch("/api/chat", { method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, temperature: +temp.value, model: model.value }) });
    const j = await r.json();
    p.textContent = j.reply ?? ("error: " + (j.error || r.status));
  } catch (err) { p.textContent = "error: " + err; }
  p.classList.remove("pending");
  box.disabled = false; box.focus();
};
document.getElementById("reset").onclick = async () => {
  await fetch("/api/reset", { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: model.value }) });
  log.innerHTML = ""; box.focus();
};
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    sessions: dict[str, object]  # "thoughtvec" always present; "smollm" lazy
    lock: threading.Lock
    page: bytes

    def _get_session(self, model: str):
        if model == "smollm" and "smollm" not in self.sessions:
            print("loading SmolLM2-135M-Instruct (first use)…", flush=True)
            self.sessions["smollm"] = SmolLMSession()
        return self.sessions.get(model)

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, self.page, "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.path == "/api/chat":
            msg = str(payload.get("message", "")).strip()
            temperature = float(payload.get("temperature", 0.0))
            model = str(payload.get("model", "thoughtvec"))
            if not msg:
                self._send(400, b'{"error":"empty message"}', "application/json")
                return
            with self.lock:  # sessions are stateful; serialize requests
                try:
                    session = self._get_session(model)
                    if session is None:
                        self._send(400, b'{"error":"unknown model"}', "application/json")
                        return
                    reply = session.reply(msg, temperature=temperature)
                except Exception as e:  # surface inference errors to the UI
                    self._send(500, json.dumps({"error": str(e)}).encode(),
                               "application/json")
                    return
            self._send(200, json.dumps({"reply": reply}).encode(), "application/json")
        elif self.path == "/api/reset":
            model = str(payload.get("model", "thoughtvec"))
            with self.lock:
                session = self.sessions.get(model)
                if session is not None:
                    session.reset()
            self._send(200, b"{}", "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def log_message(self, fmt: str, *args) -> None:  # quiet
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/FINAL_12H/best.pt")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0",
                    help="bind address (0.0.0.0 = whole local network)")
    args = ap.parse_args()

    print(f"loading {args.ckpt} on cpu …", flush=True)
    Handler.sessions = {"thoughtvec": ChatSession(args.ckpt, device="cpu")}
    Handler.lock = threading.Lock()
    Handler.page = PAGE.replace("__CKPT__", Path(args.ckpt).parent.name).encode()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"chat UI ready at http://{args.host}:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
