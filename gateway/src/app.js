require("dotenv").config();
const express = require("express");

require("./db/client"); // applies schema on startup

const teacherRoutes = require("./routes/teacher.routes");
const studentRoutes = require("./routes/student.routes");
const auditorRoutes = require("./routes/auditor.routes");
const devTamperRoutes = require("./routes/devTamper.routes");

const app = express();
app.use(express.json({ limit: "5mb" }));

// CORS — permissive for local dev across the three services.
app.use((req, res, next) => {
  res.header("Access-Control-Allow-Origin", "*");
  res.header("Access-Control-Allow-Headers", "Content-Type, X-Role, X-Actor-Id");
  res.header("Access-Control-Allow-Methods", "GET,POST,PUT,DELETE,OPTIONS");
  if (req.method === "OPTIONS") return res.sendStatus(204);
  next();
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

app.use("/teacher", teacherRoutes);
app.use("/student", studentRoutes);
app.use("/", auditorRoutes); // /verify, /audit-chain
app.use("/dev/tamper", devTamperRoutes); // Auditor "Tests" panel — self-contained tamper vectors

app.use((err, req, res, next) => {
  console.error(err);
  res.status(500).json({ error: "internal_error", detail: err.message });
});

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`certiproof-gateway listening on :${PORT}`);
});
