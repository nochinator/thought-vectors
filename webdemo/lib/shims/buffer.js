// Minimal Buffer for the sentencepiece.js bundle: it only needs
// Buffer.from(<base64 string>, "base64") (model loading) and
// Buffer.from(<string>) (utf-8), both returning a Uint8Array subclass.
export class Buffer extends Uint8Array {
  static from(data, encoding) {
    if (typeof data === "string") {
      if (encoding === "base64") {
        const bin = atob(data);
        const buf = new Buffer(bin.length);
        for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
        return buf;
      }
      return new Buffer(new TextEncoder().encode(data));
    }
    return new Buffer(data);
  }
}
export default { Buffer };
