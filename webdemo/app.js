// Browser bootstrap: download models with progress, build sessions, wire UI.
// The chat loop itself is chat.js, parity-tested against the torch model.
import * as ort from "./lib/ort.wasm.min.mjs";
import { SentencePieceProcessor } from "./lib/sentencepiece.js";
import { OnnxChat, PAD_ID, BOS_ID, EOS_ID } from "./chat.js";

const FILES = {
  "models/encoder.fp16.onnx": 36599048,
  "models/thinker.onnx": 591180,
  "models/thinker.onnx.data": 57344000,
  "models/decoder.fp16.onnx": 40770427,
  "models/spm16k_bpe.model": 503814,
};

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const statusEl = $("status");
const barEl = $("bar");
const formEl = $("form");
const inputEl = $("input");
const sendEl = $("send");
const resetEl = $("reset");

function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

async function fetchProgress(path, onBytes) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const reader = res.body.getReader();
  const chunks = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    size += value.length;
    onBytes(value.length);
  }
  const buf = new Uint8Array(size);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return buf;
}

async function boot() {
  const total = Object.values(FILES).reduce((a, b) => a + b, 0);
  let got = 0;
  const onBytes = (n) => {
    got += n;
    const pct = Math.min(100, (100 * got) / total);
    barEl.style.width = `${pct}%`;
    statusEl.textContent =
      `downloading model — ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
  };
  const names = Object.keys(FILES);
  const bufs = Object.fromEntries(
    (await Promise.all(names.map((p) => fetchProgress(p, onBytes))))
      .map((b, i) => [names[i], b]));

  statusEl.textContent = "compiling model…";
  await new Promise((r) => setTimeout(r, 30)); // let the status paint

  // resolved against the document, not the ort bundle (which lives in lib/)
  ort.env.wasm.wasmPaths = new URL("lib/", document.baseURI).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated
    ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

  const spp = new SentencePieceProcessor();
  let b64 = "";
  const raw = bufs["models/spm16k_bpe.model"];
  for (let i = 0; i < raw.length; i += 0x8000) {
    b64 += String.fromCharCode(...raw.subarray(i, i + 0x8000));
  }
  await spp.loadFromB64StringModel(btoa(b64));
  const tok = {
    encode: (t) => Array.from(spp.encodeIds(t)),
    decode: (ids) => spp.decodeIds(new Int32Array(
      ids.filter((i) => i !== PAD_ID && i !== BOS_ID && i !== EOS_ID))),
  };

  const sessions = {
    encoder: await ort.InferenceSession.create(bufs["models/encoder.fp16.onnx"]),
    thinker: await ort.InferenceSession.create(bufs["models/thinker.onnx"], {
      externalData: [{ path: "thinker.onnx.data",
                       data: bufs["models/thinker.onnx.data"] }],
    }),
    decoder: await ort.InferenceSession.create(bufs["models/decoder.fp16.onnx"]),
  };

  barEl.parentElement.style.display = "none";
  statusEl.textContent = "ready — everything runs in your browser";
  inputEl.disabled = false;
  sendEl.disabled = false;
  inputEl.focus();
  return new OnnxChat(ort, sessions, tok);
}

const chatReady = boot().catch((e) => {
  statusEl.textContent = `failed to load: ${e.message}`;
  throw e;
});

let busy = false;
formEl.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = inputEl.value.trim();
  if (!text || busy) return;
  const chat = await chatReady;
  busy = true;
  inputEl.value = "";
  sendEl.disabled = true;
  addBubble("user", text);
  const bubble = addBubble("bot", "…");
  try {
    await chat.reply(text, (partial) => {
      bubble.textContent = partial || "…";
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
  } catch (e) {
    bubble.textContent = `error: ${e.message}`;
  }
  busy = false;
  sendEl.disabled = false;
  inputEl.focus();
});

resetEl.addEventListener("click", async () => {
  if (busy) return;
  (await chatReady).reset();
  messagesEl.textContent = "";
  addBubble("note", "conversation cleared");
});
