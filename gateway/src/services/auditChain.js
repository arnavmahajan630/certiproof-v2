const db = require("../db/client");
const { sha256Hex } = require("./hashing");

function lastEntry() {
  return db.prepare("SELECT * FROM audit_chain ORDER BY entry_id DESC LIMIT 1").get();
}

function computeEntryHash(prevEntryHash, eventType, refId, payloadHash) {
  return sha256Hex(`${prevEntryHash || ""}|${eventType}|${refId}|${payloadHash}`);
}

function append(eventType, refId, payloadHash) {
  const prev = lastEntry();
  const prevEntryHash = prev ? prev.entry_hash : null;
  const entryHash = computeEntryHash(prevEntryHash, eventType, refId, payloadHash);
  const createdAt = new Date().toISOString();

  const info = db
    .prepare(
      `INSERT INTO audit_chain (event_type, ref_id, payload_hash, prev_entry_hash, entry_hash, created_at)
       VALUES (?, ?, ?, ?, ?, ?)`
    )
    .run(eventType, refId, payloadHash, prevEntryHash, entryHash, createdAt);

  return { entry_id: info.lastInsertRowid, entry_hash: entryHash, prev_entry_hash: prevEntryHash, created_at: createdAt };
}

// Walks the whole chain from the beginning and reports the first broken link, if any.
function verifyChainIntegrity() {
  const rows = db.prepare("SELECT * FROM audit_chain ORDER BY entry_id ASC").all();
  let prevEntryHash = null;
  for (const row of rows) {
    const expected = computeEntryHash(prevEntryHash, row.event_type, row.ref_id, row.payload_hash);
    if (expected !== row.entry_hash || row.prev_entry_hash !== prevEntryHash) {
      return { intact: false, brokenAtEntryId: row.entry_id };
    }
    prevEntryHash = row.entry_hash;
  }
  return { intact: true, brokenAtEntryId: null };
}

function entriesForRef(refId) {
  return db.prepare("SELECT * FROM audit_chain WHERE ref_id = ? ORDER BY entry_id ASC").all(refId);
}

module.exports = { append, verifyChainIntegrity, entriesForRef, lastEntry };
