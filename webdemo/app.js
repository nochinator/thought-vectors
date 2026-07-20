// Browser bootstrap: download models with progress, build sessions, wire UI.
// The chat loops themselves are chat.js, parity-tested against torch.
import * as ort from "./lib/ort.wasm.min.mjs";
import { SentencePieceProcessor } from "./lib/sentencepiece.js";
import { OnnxChat, OnnxLmChat, PAD_ID, BOS_ID, EOS_ID } from "./chat.js";

const TV_FILES = {
  "models/encoder.fp16.onnx": 36599048,
  "models/thinker.onnx": 591535,
  "models/thinker.onnx.data": 57344000,
  "models/decoder.fp16.onnx": 40770427,
};
const LM_FILES = { "models/lm.fp16.onnx": 100772166 };
const TOK_FILE = { "models/spm16k_bpe.model": 503814 };
const LM_MAX_LEN = 384;
const LM_MAX_NEW = 64;

const THINKER = "thinker";
const LM = "lm";
const LABEL = { [THINKER]: "thinker", [LM]: "token-LM baseline" };

const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const statusEl = $("status");
const barEl = $("bar");
const formEl = $("form");
const inputEl = $("input");
const sendEl = $("send");
const resetEl = $("reset");
const modelRadios = document.querySelectorAll('input[name="model"]');

function addBubble(role, text, tag) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (tag) {
    const t = document.createElement("span");
    t.className = "tag";
    t.textContent = tag;
    div.appendChild(t);
  }
  const body = document.createElement("span");
  body.textContent = text;
  div.appendChild(body);
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return body;
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

async function downloadWithBar(files, label) {
  const total = Object.values(files).reduce((a, b) => a + b, 0);
  let got = 0;
  barEl.parentElement.style.display = "";
  const onBytes = (n) => {
    got += n;
    const pct = Math.min(100, (100 * got) / total);
    barEl.style.width = `${pct}%`;
    statusEl.textContent =
      `downloading ${label} — ${(got / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`;
  };
  const names = Object.keys(files);
  const bufs = Object.fromEntries(
    (await Promise.all(names.map((p) => fetchProgress(p, onBytes))))
      .map((b, i) => [names[i], b]));
  barEl.parentElement.style.display = "none";
  return bufs;
}

function makeTok(sppModelBytes) {
  const spp = new SentencePieceProcessor();
  let b64 = "";
  for (let i = 0; i < sppModelBytes.length; i += 0x8000) {
    b64 += String.fromCharCode(...sppModelBytes.subarray(i, i + 0x8000));
  }
  const ready = spp.loadFromB64StringModel(btoa(b64));
  return {
    ready,
    encode: (t) => Array.from(spp.encodeIds(t)),
    decode: (ids) => spp.decodeIds(new Int32Array(
      ids.filter((i) => i !== PAD_ID && i !== BOS_ID && i !== EOS_ID))),
  };
}

ort.env.wasm.wasmPaths = new URL("lib/", document.baseURI).href;
ort.env.wasm.numThreads = self.crossOriginIsolated
  ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const chats = {}; // { thinker: OnnxChat, lm: OnnxLmChat }, built lazily
let tok = null;

async function ensureTokenizer() {
  if (tok) return tok;
  const bufs = await downloadWithBar(TOK_FILE, "tokenizer");
  tok = makeTok(bufs["models/spm16k_bpe.model"]);
  await tok.ready;
  return tok;
}

async function ensureModel(name) {
  if (chats[name]) return chats[name];
  await ensureTokenizer();
  statusEl.textContent = `compiling ${LABEL[name]}…`;
  if (name === THINKER) {
    const bufs = await downloadWithBar(TV_FILES, "thinker model (~135 MB)");
    const sessions = {
      encoder: await ort.InferenceSession.create(bufs["models/encoder.fp16.onnx"]),
      thinker: await ort.InferenceSession.create(bufs["models/thinker.onnx"], {
        externalData: [{ path: "thinker.onnx.data",
                         data: bufs["models/thinker.onnx.data"] }],
      }),
      decoder: await ort.InferenceSession.create(bufs["models/decoder.fp16.onnx"]),
    };
    chats[THINKER] = new OnnxChat(ort, sessions, tok);
  } else {
    const bufs = await downloadWithBar(LM_FILES, "token-LM baseline (~100 MB)");
    const session = { lm: await ort.InferenceSession.create(bufs["models/lm.fp16.onnx"]) };
    chats[LM] = new OnnxLmChat(ort, session, tok, LM_MAX_LEN, LM_MAX_NEW);
  }
  return chats[name];
}

// The canonical conversation, shared across backends so switching models
// mid-conversation compares them on the same transcript (each backend's
// internal history is just reset to this before every reply).
let transcript = [];

function selectedModel() {
  for (const r of modelRadios) if (r.checked) return r.value;
  return THINKER;
}

let busy = false;

async function boot() {
  await ensureModel(THINKER);
  statusEl.textContent = "ready — everything runs in your browser";
  inputEl.disabled = false;
  sendEl.disabled = false;
  inputEl.focus();
}
const bootDone = boot().catch((e) => {
  statusEl.textContent = `failed to load: ${e.message}`;
  throw e;
});

formEl.addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const text = inputEl.value.trim();
  if (!text || busy) return;
  await bootDone;
  busy = true;
  inputEl.value = "";
  sendEl.disabled = true;
  addBubble("user", text);
  const modelName = selectedModel();
  let chat;
  try {
    chat = await ensureModel(modelName);
  } catch (e) {
    statusEl.textContent = `failed to load ${LABEL[modelName]}: ${e.message}`;
    busy = false;
    sendEl.disabled = false;
    return;
  }
  statusEl.textContent = "ready — everything runs in your browser";
  const bubble = addBubble("bot", "…", LABEL[modelName]);
  try {
    chat.history = transcript.slice();
    const reply = await chat.reply(text, (partial) => {
      bubble.textContent = partial || "…";
      messagesEl.scrollTop = messagesEl.scrollHeight;
    });
    transcript.push(text, reply);
  } catch (e) {
    bubble.textContent = `error: ${e.message}`;
  }
  busy = false;
  sendEl.disabled = false;
  inputEl.focus();
});

resetEl.addEventListener("click", () => {
  if (busy) return;
  transcript = [];
  messagesEl.textContent = "";
  addBubble("note", "conversation cleared");
});
