// End-to-end parity: the shipped JS chat loop (chat.js) on onnxruntime-web's
// WASM backend vs the 8 torch reference replies in reference_replies.json.
// Run from a directory with `npm install onnxruntime-web @sctg/sentencepiece-js`:
//   node webdemo/test_parity.node.mjs
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

import * as ort from "onnxruntime-web";
import { SentencePieceProcessor } from "@sctg/sentencepiece-js";

import { OnnxChat, OnnxLmChat, PAD_ID, BOS_ID, EOS_ID } from "./chat.js";

const here = dirname(fileURLToPath(import.meta.url));
const models = (n) => join(here, "models", n);

ort.env.wasm.numThreads = 1; // match a non-cross-origin-isolated browser

const spp = new SentencePieceProcessor();
await spp.loadFromB64StringModel(
  readFileSync(models("spm16k_bpe.model")).toString("base64"));
const tok = {
  encode: (t) => Array.from(spp.encodeIds(t)),
  decode: (ids) => spp.decodeIds(new Int32Array(
    ids.filter((i) => i !== PAD_ID && i !== BOS_ID && i !== EOS_ID))),
};

const buf = (n) => new Uint8Array(readFileSync(models(n)));
const sessions = {
  encoder: await ort.InferenceSession.create(buf("encoder.fp16.onnx")),
  thinker: await ort.InferenceSession.create(buf("thinker.onnx"), {
    externalData: [{ path: "thinker.onnx.data", data: buf("thinker.onnx.data") }],
  }),
  decoder: await ort.InferenceSession.create(buf("decoder.fp16.onnx")),
};
const lmSession = { lm: await ort.InferenceSession.create(buf("lm.fp16.onnx")) };

const ref = JSON.parse(readFileSync(join(here, "reference_replies.json"), "utf8"));
const chat = new OnnxChat(ort, sessions, tok);
const got = [];
const t0 = Date.now();
for (const t of ref.conversation) got.push(await chat.reply(t));
for (const t of ref.oneshot) {
  chat.reset();
  got.push(await chat.reply(t));
}
const dt = (Date.now() - t0) / 1000;

let ok = 0;
for (let i = 0; i < ref.replies.length; i++) {
  const match = got[i] === ref.replies[i];
  ok += match;
  if (!match) {
    console.log(`MISMATCH #${i}`);
    console.log(`  want: ${JSON.stringify(ref.replies[i])}`);
    console.log(`  got : ${JSON.stringify(got[i])}`);
  }
}
console.log(`tv: ${ok}/${ref.replies.length} replies match torch exactly `
  + `(${dt.toFixed(1)}s total, ${(dt / ref.replies.length).toFixed(1)}s/reply)`);

const lmChat = new OnnxLmChat(ort, lmSession, tok, ref.lm_max_len, ref.lm_max_new);
const lmGot = [];
const t1 = Date.now();
for (const t of ref.conversation) lmGot.push(await lmChat.reply(t));
for (const t of ref.oneshot) {
  lmChat.reset();
  lmGot.push(await lmChat.reply(t));
}
const lmDt = (Date.now() - t1) / 1000;

let lmOk = 0;
for (let i = 0; i < ref.lm_replies.length; i++) {
  const match = lmGot[i] === ref.lm_replies[i];
  lmOk += match;
  if (!match) {
    console.log(`LM MISMATCH #${i}`);
    console.log(`  want: ${JSON.stringify(ref.lm_replies[i])}`);
    console.log(`  got : ${JSON.stringify(lmGot[i])}`);
  }
}
console.log(`lm: ${lmOk}/${ref.lm_replies.length} replies match torch exactly `
  + `(${lmDt.toFixed(1)}s total, ${(lmDt / ref.lm_replies.length).toFixed(1)}s/reply)`);

process.exit(ok === ref.replies.length && lmOk === ref.lm_replies.length ? 0 : 1);
