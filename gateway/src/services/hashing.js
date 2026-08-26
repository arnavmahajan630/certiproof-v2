const crypto = require("crypto");

function sha256Hex(input) {
  return crypto.createHash("sha256").update(input).digest("hex");
}

// Numeric precision contract (plan Part B, item 1): witness_hash must be computed
// over the EXACT JSON string bytes the ML-Worker's response body contained — never
// a value re-serialized from a parsed JS object (JSON.stringify on a re-parsed float
// array is not guaranteed byte-identical: trailing zeros, exponent formatting, etc).
// Callers must pass the raw response text, not JSON.stringify(parsedObject).
function witnessHashFromRawJson(rawJsonString) {
  return sha256Hex(rawJsonString);
}

function sha256Buffer(buf) {
  return crypto.createHash("sha256").update(buf).digest("hex");
}

function randomToken(bytes = 16) {
  return crypto.randomBytes(bytes).toString("hex");
}

function newId(prefix) {
  return `${prefix}_${crypto.randomUUID()}`;
}

module.exports = { sha256Hex, witnessHashFromRawJson, sha256Buffer, randomToken, newId };
