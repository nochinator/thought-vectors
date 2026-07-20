// Browser stub for the sentencepiece.js bundle's `import * as fs from 'fs'`.
// Only Node code paths touch fs; in the browser these must never be called.
const unavailable = () => { throw new Error("fs is not available in the browser"); };
export const readFile = unavailable;
export const readFileSync = unavailable;
export const readSync = unavailable;
export default { readFile, readFileSync, readSync };
