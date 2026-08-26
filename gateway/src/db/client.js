const fs = require("fs");
const path = require("path");
const Database = require("better-sqlite3");

const DB_PATH = process.env.DB_PATH || path.join(__dirname, "..", "..", "data", "certiproof.db");

fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });

const db = new Database(DB_PATH);
db.pragma("journal_mode = WAL");
db.pragma("foreign_keys = ON");

const schema = fs.readFileSync(path.join(__dirname, "schema.sql"), "utf8");
db.exec(schema);

// Idempotent additive migrations for columns added after a DB already existed
// (CREATE TABLE IF NOT EXISTS above only applies schema.sql to brand-new tables).
// SQLite has no "ADD COLUMN IF NOT EXISTS", so guard with a pragma check instead
// of a bare try/catch swallowing unrelated errors.
function columnExists(table, column) {
  return db.prepare(`PRAGMA table_info(${table})`).all().some((c) => c.name === column);
}
function addColumnIfMissing(table, column, definition) {
  if (!columnExists(table, column)) {
    db.exec(`ALTER TABLE ${table} ADD COLUMN ${column} ${definition}`);
  }
}
addColumnIfMissing("evaluations", "poseidon_commitment", "TEXT");
// RCAJ-X migration: two new columns (see schema.sql's evaluations comments).
addColumnIfMissing("evaluations", "per_criterion_scores_json", "TEXT");
addColumnIfMissing("evaluations", "explanations_json", "TEXT");

module.exports = db;
