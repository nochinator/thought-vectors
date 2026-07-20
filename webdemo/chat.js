// The chat loop, ported line-for-line from OnnxChat in
// scripts/quantize_web.py (the parity-proven python reference). No imports:
// the ONNX runtime, sessions, and tokenizer are injected so the same module
// runs in the browser and under Node for the parity test.

export const PAD_ID = 0;
export const BOS_ID = 1;
export const EOS_ID = 2;

const D = 384;
const K = 8; // thoughts per turn

export class OnnxChat {
  /**
   * @param ort onnxruntime-web module (for Tensor construction)
   * @param sessions {encoder, thinker, decoder} InferenceSessions
   * @param tok {encode(text)->number[], decode(ids)->string}
   */
  constructor(ort, sessions, tok) {
    this.ort = ort;
    this.s = sessions;
    this.tok = tok;
    this.history = [];
  }

  reset() {
    this.history = [];
  }

  ids64(ids) {
    return new this.ort.Tensor(
      "int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
  }

  /** Greedy reply; onToken(text-so-far) fires as tokens decode. */
  async reply(text, onToken) {
    this.history.push(text.trim());
    const turns = this.history.slice(-6);
    const n = turns.length;
    const firstRole = (this.history.length - n) % 2;
    const th = new Float32Array(n * K * D);
    const roles = [], dist = [];
    for (let j = 0; j < n; j++) {
      const ids = [BOS_ID, ...this.tok.encode(turns[j]).slice(0, 254), EOS_ID];
      const enc = await this.s.encoder.run({ ids: this.ids64(ids) });
      th.set(enc.thoughts.data, j * K * D);
      roles.push((firstRole + j) % 2);
      dist.push(Math.min(n - j, 6));
    }
    const out = await this.s.thinker.run({
      ctx_th: new this.ort.Tensor("float32", th, [1, n, K, D]),
      ctx_roles: this.ids64(roles),
      dist: this.ids64(dist),
    });
    const score = out.score.data;
    let best = 0;
    for (let h = 1; h < score.length; h++) if (score[h] < score[best]) best = h;
    const thoughts = new this.ort.Tensor(
      "float32", out.hyps.data.slice(best * K * D, (best + 1) * K * D),
      [1, K, D]);

    const ids = [BOS_ID];
    for (let step = 0; step < 255; step++) {
      const fed = ids.length < 2 ? [...ids, 0] : ids;
      const dec = await this.s.decoder.run({
        thoughts,
        ids: this.ids64(fed),
        pos: new this.ort.Tensor(
          "int64", BigInt64Array.from([BigInt(ids.length - 1)]), [1]),
      });
      const lg = dec.logits.data;
      if (ids.length >= 3) { // no_repeat_ngram=3
        const p0 = ids[ids.length - 2], p1 = ids[ids.length - 1];
        for (let k = 0; k < ids.length - 2; k++) {
          if (ids[k] === p0 && ids[k + 1] === p1) lg[ids[k + 2]] = -Infinity;
        }
      }
      let nxt = 0;
      for (let v = 1; v < lg.length; v++) if (lg[v] > lg[nxt]) nxt = v;
      ids.push(nxt);
      if (onToken) onToken(this.tok.decode(ids));
      if (nxt === EOS_ID) break;
    }
    const reply = this.tok.decode(ids);
    this.history.push(reply);
    return reply;
  }
}
