#!/usr/bin/env bun
/** Read/write the schema embedded in a local .fig; no Figma service is called.
 * Dependency: kiwi-schema 0.5.0 (Evan Wallace, MIT).
 * Usage: bun fig_kiwi_io.cjs decode|encode REF_CANVAS JSON OUTPUT MODULES_DIR
 */
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');
const assert = require('node:assert/strict');
const [mode, reference, jsonPath, output, modules] = process.argv.slice(2);
const kiwi = require(modules ? path.join(modules, 'kiwi-schema') : 'kiwi-schema');
const bytes = fs.readFileSync(reference);
assert.equal(bytes.subarray(0, 8).toString(), 'fig-kiwi');
const chunks = [];
let offset = 12;
while (offset < bytes.length) {
  const length = bytes.readUInt32LE(offset); offset += 4;
  assert(offset + length <= bytes.length);
  chunks.push(bytes.subarray(offset, offset + length)); offset += length;
}
assert.equal(chunks.length, 2);
const schema = kiwi.decodeBinarySchema(zlib.inflateRawSync(chunks[0]));
const compiled = kiwi.compileSchema(schema);
const unpack = b => b.readUInt32LE(0) === 0xfd2fb528 ? zlib.zstdDecompressSync(b) : zlib.inflateRawSync(b);
const replacer = (key, value) => value instanceof Uint8Array ? { $bytes: Buffer.from(value).toString('base64') } : value;
const reviver = (key, value) => value && typeof value === 'object' && Object.keys(value).length === 1 && value.$bytes !== undefined ? Uint8Array.from(Buffer.from(value.$bytes, 'base64')) : value;
const plain = message => JSON.parse(JSON.stringify(message, replacer));
if (mode === 'decode') {
  const message = compiled.decodeMessage(unpack(chunks[1]));
  fs.writeFileSync(jsonPath, JSON.stringify(message, replacer));
  console.log(JSON.stringify({ version: bytes.readUInt32LE(8), nodes: message.nodeChanges.length, blobs: message.blobs.length }));
} else if (mode === 'encode') {
  const message = JSON.parse(fs.readFileSync(jsonPath, 'utf8'), reviver);
  const encoded = compiled.encodeMessage(message);
  // Kiwi floats are float32; compare after canonicalizing just those tiny roundings.
  const decoded = compiled.decodeMessage(encoded);
  function compare(a, b, at = '') {
    if (typeof a === 'number' && typeof b === 'number') {
      assert(Math.abs(a - b) <= Math.max(0.0002, Math.abs(a) * 1e-7), at + ': numeric mismatch'); return;
    }
    if (a && typeof a === 'object') {
      assert(b && typeof b === 'object', at);
      assert.deepEqual(Object.keys(a).sort(), Object.keys(b).sort(), at + ': lost fields');
      for (const key of Object.keys(a)) compare(a[key], b[key], at + '.' + key);
    } else assert.equal(a, b, at);
  }
  compare(plain(message), plain(decoded));
  const compressed = zlib.zstdCompressSync(encoded);
  const length = n => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
  const result = Buffer.concat([bytes.subarray(0, 12), length(chunks[0].length), chunks[0], length(compressed.length), compressed]);
  fs.writeFileSync(output, result);
  fs.writeFileSync(output + '.json', JSON.stringify(decoded, replacer));
  console.log(JSON.stringify({ round_trip: 'passed', nodes: decoded.nodeChanges.length, blobs: decoded.blobs.length, bytes: result.length }));
} else throw new Error('Expected decode or encode');
