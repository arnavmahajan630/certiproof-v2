// Running the client module already applies schema.sql (CREATE TABLE IF NOT EXISTS).
// This script exists as an explicit, nameable migration step for the README/run instructions.
require("./client");
console.log("schema applied");
