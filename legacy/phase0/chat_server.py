#!/usr/bin/env python3
"""Simple chat server for the thought-vector thinker model.

Uses the three-phase trained checkpoint with structural embeddings
(turn, speaker, decode-target) for multi-turn conversation context.
"""
from __future__ import annotations

import json
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "thought-vectors-main"))

from thought_vectors import SPTokenizer, ThoughtEncoder, ThoughtDecoder, LossPredictor, ThinkerModel
from thought_vectors.inference import decode_greedy

# ── Load model ──
device = torch.device("cpu")
print(f"Loading on {device}...")

CKPT_PATH = ROOT / "thought-vectors-main" / "artifacts" / "thinker_three_phase.pt"
print(f"Loading checkpoint: {CKPT_PATH}")
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)

enc = ThoughtEncoder(16384, 256, 4, 4, 0.1, 8192, 256).to(device)
dec = ThoughtDecoder(16384, 256, 4, 4, 0.1, 8192).to(device)
thinker = nn.TransformerEncoder(
    nn.TransformerEncoderLayer(256, 4, dropout=0.1, batch_first=True), 6
).to(device)
pred = LossPredictor(256, 256).to(device)

enc.load_state_dict(ckpt["encoder_state"])
dec.load_state_dict(ckpt["decoder_state"])
thinker.load_state_dict(ckpt["thinker_state"])
pred.load_state_dict(ckpt["predictor_state"])

model = ThinkerModel(enc, dec, thinker, pred, max_turns=4)
model.to(device)
model.eval()

# Load structural embeddings
emb = ckpt.get("thinker_embeddings")
if emb is not None:
    model.turn_embedding.load_state_dict(emb["turn_embedding"])
    model.speaker_embedding.load_state_dict(emb["speaker_embedding"])
    model.decode_embedding.data.copy_(emb["decode_embedding"].to(device))
    print("Loaded thinker structural embeddings")

tok = SPTokenizer()
tok.load("/tmp/sp_c4_16k.model")
pad_id = tok.pad_token_id
bos_id = tok.bos_token_id
eos_id = tok.eos_token_id

print("Model loaded. Starting server...")


# ── Helpers ──

def build_context_and_metadata(
    history: list[tuple[torch.Tensor, int, int]],
    current_thoughts: torch.Tensor,
    best_k: int,
    current_speaker: int,
    current_turn: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the multi-turn context tensor and structural metadata.

    Each history entry: (thought_vectors [1, k, D], speaker_id, turn_id)
    current_thoughts: [1, 256, D] from encoder
    best_k: number of vectors to take from current
    current_speaker: 0=user, 1=assistant
    current_turn: turn number for the current input

    Returns:
        context:     [1, total_k, D] — all vectors concatenated
        turn_ids:    [1, total_k] long
        speaker_ids: [1, total_k] long
        decode_mask: [1, total_k] bool — True for current input segment
    """
    all_vectors = [v for v, _, _ in history] + [current_thoughts[:, :best_k, :]]
    all_speakers = [s for _, s, _ in history] + [current_speaker]
    all_turns = [t for _, _, t in history] + [current_turn]

    pieces = []
    turn_ids_pieces = []
    speaker_ids_pieces = []
    decode_mask_pieces = []

    for i, (vecs, spk, turn) in enumerate(zip(all_vectors, all_speakers, all_turns)):
        k_i = vecs.size(1)
        pieces.append(vecs)
        turn_ids_pieces.append(torch.full((1, k_i), turn, dtype=torch.long, device=device))
        speaker_ids_pieces.append(torch.full((1, k_i), spk, dtype=torch.long, device=device))
        # Decode mask: true only for the current (last) turn
        decode_mask_pieces.append(torch.zeros(1, k_i, dtype=torch.bool, device=device) if i < len(all_vectors) - 1
                                  else torch.ones(1, k_i, dtype=torch.bool, device=device))

    context = torch.cat(pieces, dim=1)
    turn_ids = torch.cat(turn_ids_pieces, dim=1)
    speaker_ids = torch.cat(speaker_ids_pieces, dim=1)
    decode_mask = torch.cat(decode_mask_pieces, dim=1)

    return context, turn_ids, speaker_ids, decode_mask


# ── HTTP Server ──

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Thought Vector Chat</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; background: #1a1a2e; color: #e0e0e0; height: 100vh; display: flex; flex-direction: column; }
  #header { padding: 16px 20px; background: #16213e; border-bottom: 1px solid #0f3460; }
  #header h1 { font-size: 18px; color: #e94560; }
  #header .sub { font-size: 12px; color: #888; }
  #chat { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 12px; }
  .msg { max-width: 80%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; font-size: 14px; }
  .user { align-self: flex-end; background: #0f3460; color: #e0e0e0; }
  .assistant { align-self: flex-start; background: #16213e; border: 1px solid #0f3460; color: #ccc; }
  .msg .label { font-size: 11px; color: #888; margin-bottom: 4px; }
  #input-area { display: flex; gap: 8px; padding: 16px 20px; background: #16213e; border-top: 1px solid #0f3460; }
  #input { flex: 1; padding: 12px 16px; border-radius: 8px; border: 1px solid #0f3460; background: #1a1a2e; color: #e0e0e0; font-size: 14px; outline: none; }
  #input:focus { border-color: #e94560; }
  #send { padding: 12px 24px; border-radius: 8px; border: none; background: #e94560; color: white; font-size: 14px; cursor: pointer; }
  #send:hover { background: #c73650; }
  #send:disabled { opacity: 0.5; cursor: default; }
  .loading { opacity: 0.6; }
  .controls { display: flex; gap: 8px; align-items: center; padding: 8px 20px; background: #16213e; border-top: 1px solid #0f3460; }
  .controls label { font-size: 12px; color: #888; }
  .controls select { background: #1a1a2e; color: #e0e0e0; border: 1px solid #0f3460; padding: 4px 8px; border-radius: 4px; font-size: 12px; }
</style>
</head>
<body>
<div id="header">
  <h1>Thought Vector Chat</h1>
  <div class="sub">encoder → thinker (w/ turn/speaker/decode embeddings) → decoder</div>
</div>
<div class="controls">
  <span id="compression-info" style="font-size:12px;color:#888;"></span>
  <span style="flex:1"></span>
  <label style="font-size:11px;color:#888;">temp
    <input id="slider-temp" type="range" min="0" max="150" value="70"
           style="width:60px;vertical-align:middle;accent-color:#e94560;">
    <span id="val-temp" style="font-size:11px;color:#e94560;width:24px;display:inline-block;text-align:right;">0.70</span>
  </label>
  <label style="font-size:11px;color:#888;">top-k
    <input id="slider-topk" type="range" min="0" max="200" value="50"
           style="width:60px;vertical-align:middle;accent-color:#e94560;">
    <span id="val-topk" style="font-size:11px;color:#e94560;width:24px;display:inline-block;text-align:right;">50</span>
  </label>
  <label style="font-size:11px;color:#888;">top-p
    <input id="slider-topp" type="range" min="0" max="100" value="0"
           style="width:60px;vertical-align:middle;accent-color:#e94560;">
    <span id="val-topp" style="font-size:11px;color:#e94560;width:24px;display:inline-block;text-align:right;">—</span>
  </label>
  <button onclick="clearChat()" style="background:none;border:1px solid #0f3460;color:#888;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:12px">Clear</button>
</div>
<div id="chat"></div>
<div id="input-area">
  <input id="input" type="text" placeholder="Type a message..." autofocus>
  <button id="send" onclick="send()">Send</button>
</div>
<script>
const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const compressionInfo = document.getElementById('compression-info');

const sliderTemp = document.getElementById('slider-temp');
const sliderTopk = document.getElementById('slider-topk');
const sliderTopp = document.getElementById('slider-topp');
const valTemp = document.getElementById('val-temp');
const valTopk = document.getElementById('val-topk');
const valTopp = document.getElementById('val-topp');

sliderTemp.addEventListener('input', () => {
  valTemp.textContent = (sliderTemp.value / 100).toFixed(2);
});
sliderTopk.addEventListener('input', () => {
  valTopk.textContent = sliderTopk.value || '—';
});
sliderTopp.addEventListener('input', () => {
  const v = parseInt(sliderTopp.value);
  valTopp.textContent = v ? (v / 100).toFixed(2) : '—';
});

input.addEventListener('keydown', e => { if (e.key === 'Enter') send(); });

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  div.innerHTML = '<div class="label">' + (role === 'user' ? 'You' : 'Assistant') + '</div>' + escapeHtml(text);
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

async function send() {
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  addMsg('user', msg);
  sendBtn.disabled = true;
  sendBtn.textContent = '...';
  const loadingDiv = document.createElement('div');
  loadingDiv.className = 'msg assistant loading';
  loadingDiv.innerHTML = '<div class="label">Assistant</div><em>thinking...</em>';
  chat.appendChild(loadingDiv);
  chat.scrollTop = chat.scrollHeight;
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        message: msg,
        temperature: sliderTemp.value / 100 || 0,
        top_k: parseInt(sliderTopk.value) || 0,
        top_p: parseInt(sliderTopp.value) / 100 || 0,
      })
    });
    const data = await res.json();
    loadingDiv.remove();
    addMsg('assistant', data.response);
    if (data.compression) compressionInfo.textContent = data.compression;
  } catch(e) {
    loadingDiv.remove();
    addMsg('assistant', '[Error: ' + e.message + ']');
  }
  sendBtn.disabled = false;
  sendBtn.textContent = 'Send';
}

async function clearChat() {
  chat.innerHTML = '';
  await fetch('/clear', {method: 'POST'});
  compressionInfo.textContent = '';
  addMsg('assistant', 'Chat cleared. Start a new conversation!');
}

addMsg('assistant', 'Hello! I am the thought-vector thinker model. Ask me anything!');
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/chat":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            user_msg = body.get("message", "")
            temperature = float(body.get("temperature", 0.7))
            top_k = int(body.get("top_k", 0)) or None
            top_p = float(body.get("top_p", 0)) or None

            with torch.no_grad():
                # Encode current user message
                ids = torch.tensor(
                    [tok.encode(user_msg, add_special_tokens=True)], device=device
                )
                thoughts = model.encoder(ids, ids.eq(pad_id))
                n_t = ids.size(1) - 2  # token count minus BOS/EOS

                # Use predictor to find optimal k (relative threshold)
                pred_losses = model.predictor(thoughts)[0]
                full_loss = pred_losses[-1].item()
                threshold = full_loss * 1.3
                best_k = 256
                for k_ in range(1, 257):
                    if pred_losses[k_ - 1].item() <= threshold:
                        best_k = max(k_, 4)
                        break
                # Clamp to max_k used during training — the thinker was
                # never trained on sequences longer than max_k per turn.
                best_k = min(best_k, 64)

                # Build conversation context with structural metadata
                history = getattr(self.server, "conversation_history", [])
                current_turn = len(history) // 2  # each user+asst pair = 1 turn

                context, turn_ids, speaker_ids, decode_mask = build_context_and_metadata(
                    history, thoughts, best_k,
                    current_speaker=0,  # 0 = user
                    current_turn=current_turn,
                    device=device,
                )

                # Run thinker with structural embeddings
                t = model.thinker_forward(context, turn_ids, speaker_ids, decode_mask)

                # Decode from the current turn's segment only
                current_segment = t[:, decode_mask[0], :]  # [1, best_k, D]

                # Sample-based decoding with light temperature to break
                # the argmax monotony that causes samey responses.
                gen = decode_greedy(
                    model, current_segment,
                    bos_token_id=bos_id, eos_token_id=eos_id,
                    max_length=100,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                )
                response = tok.decode(gen[0].tolist(), skip_special_tokens=True)

                # Cache this turn for future context
                # Store user input vectors
                self.server.conversation_history = history + [
                    (thoughts[:, :best_k, :], 0, current_turn),  # (vecs, speaker=user, turn)
                ]
                # Encode and store assistant response vectors
                resp_ids = torch.tensor(
                    [tok.encode(response, add_special_tokens=True)], device=device
                )
                resp_thoughts = model.encoder(resp_ids, resp_ids.eq(pad_id))
                rk = min(best_k, resp_thoughts.size(1))
                self.server.conversation_history = self.server.conversation_history + [
                    (resp_thoughts[:, :rk, :], 1, current_turn),  # (vecs, speaker=asst, turn)
                ]

                # Trim to last ~5 turns to avoid memory growth
                max_turns = 5
                if len(self.server.conversation_history) > max_turns * 2:
                    self.server.conversation_history = self.server.conversation_history[-(max_turns * 2):]

            # Compression ratio display
            if best_k < n_t:
                ratio_label = f"{n_t // best_k}:1 compression"
            elif best_k > n_t:
                ratio_label = f"1:{best_k // max(1, n_t)} expansion"
            else:
                ratio_label = "1:1"
            turns = len(self.server.conversation_history) // 2
            compression_str = f"k={best_k} ({ratio_label}) | {turns} turn{'s' if turns != 1 else ''}"

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({
                "response": response,
                "compression": compression_str,
            }).encode("utf-8"))

        elif parsed.path == "/clear":
            self.server.conversation_history = []
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress request logging


if __name__ == "__main__":
    port = 8080
    server = HTTPServer(("0.0.0.0", port), Handler)
    server.conversation_history = []  # list of (vectors, speaker_id, turn_id)
    print(f"Server at http://localhost:{port}")
    print("Predictor selects optimal k automatically.")
    server.serve_forever()
