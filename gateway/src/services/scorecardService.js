const fs = require("fs");
const path = require("path");
const PDFDocument = require("pdfkit");
const QRCode = require("qrcode");

const { sha256Buffer, randomToken } = require("./hashing");
const db = require("../db/client");
const auditChain = require("./auditChain");

const SCORECARDS_DIR = path.join(__dirname, "..", "..", "data", "scorecards");
fs.mkdirSync(SCORECARDS_DIR, { recursive: true });

const PUBLIC_BASE_URL = process.env.PUBLIC_BASE_URL || "http://localhost:5173";

// Machine-readable line embedded in the PDF body — parsed back out on re-upload
// instead of decoding the QR raster (deliberately avoided, see plan Part D).
function verifyLine(evaluationId, scorecardToken) {
  return `VERIFY-ID: ${evaluationId}:${scorecardToken}`;
}

function parseVerifyLine(text) {
  const match = text.match(/VERIFY-ID:\s*([A-Za-z0-9_-]+):([0-9a-f]+)/);
  if (!match) return null;
  return { evaluationId: match[1], scorecardToken: match[2] };
}

async function generateScorecard(evaluationId) {
  const evaluation = db.prepare("SELECT * FROM evaluations WHERE evaluation_id = ?").get(evaluationId);
  if (!evaluation) throw new Error(`evaluation ${evaluationId} not found`);
  const submission = db.prepare("SELECT * FROM submissions WHERE submission_id = ?").get(evaluation.submission_id);
  const batch = db.prepare("SELECT * FROM exam_batches WHERE batch_id = ?").get(submission.batch_id);
  const rubric = JSON.parse(batch.rubric_json);
  const criterionScores = JSON.parse(evaluation.per_criterion_scores_json || "[]");
  const explanations = JSON.parse(evaluation.explanations_json || "[]");

  const scorecardToken = randomToken(16);
  const verifyUrl = `${PUBLIC_BASE_URL}/verify?evaluation_id=${evaluationId}&token=${scorecardToken}`;
  const qrDataUrl = await QRCode.toDataURL(verifyUrl);
  const qrBuffer = Buffer.from(qrDataUrl.split(",")[1], "base64");

  const doc = new PDFDocument({ size: "A4", margin: 50 });
  const chunks = [];
  doc.on("data", (c) => chunks.push(c));

  const donePromise = new Promise((resolve) => doc.on("end", resolve));

  doc.fontSize(20).text("CertiProof — Scorecard", { align: "center" });
  doc.moveDown();
  doc.fontSize(11);
  doc.text(`Student ID: ${submission.student_id}`);
  doc.text(`Batch ID: ${batch.batch_id}`);
  doc.text(`Evaluation ID: ${evaluationId}`);
  doc.text(`Certified: ${batch.certified_at} by ${batch.certified_by}`);
  doc.text(`Evaluated: ${evaluation.evaluated_at}`);
  doc.moveDown();

  doc.fontSize(14).text("Per-Criterion Scores", { underline: true });
  doc.fontSize(11);
  (rubric.criteria || []).forEach((c, i) => {
    doc.text(`${i + 1}. ${c.criterion_text} — score: ${criterionScores[i]} / ${c.max_marks}, weight: ${c.weight}`);
    // Evidence summary (1-2 sentences), from the SAME attention computation the
    // proof is over — not a separate, post-hoc explanation (see Phase 6 of the
    // integration plan).
    const exp = explanations[i];
    if (exp) {
      const badge = exp.confidence === "review_recommended" ? "Flagged for review" : "High confidence";
      doc.fontSize(9).fillColor("#444").text(`   [${badge}] ${exp.reason_text}`, { width: 480 });
      doc.fontSize(11).fillColor("black");
    }
  });
  doc.moveDown();

  doc.fontSize(16).text(`Final Score: ${evaluation.final_score}`, { underline: true });
  doc.moveDown();

  doc.fontSize(10).text("Scan to verify this scorecard, or re-upload this PDF at the Student portal.");
  doc.image(qrBuffer, { width: 120 });
  doc.moveDown();

  // Machine-readable verification line — the primary path the re-upload flow uses.
  doc.fontSize(9).fillColor("#666").text(verifyLine(evaluationId, scorecardToken));

  doc.end();
  await donePromise;

  const pdfBuffer = Buffer.concat(chunks);
  const scorecardHash = sha256Buffer(pdfBuffer);
  const scorecardPath = path.join(SCORECARDS_DIR, `${evaluationId}.pdf`);
  fs.writeFileSync(scorecardPath, pdfBuffer);

  const generatedAt = new Date().toISOString();
  db.prepare(
    `UPDATE evaluations
     SET scorecard_hash = ?, scorecard_token = ?, scorecard_path = ?, scorecard_generated_at = ?
     WHERE evaluation_id = ?`
  ).run(scorecardHash, scorecardToken, scorecardPath, generatedAt, evaluationId);

  auditChain.append("scorecard_generated", evaluationId, scorecardHash);

  return { scorecardHash, scorecardToken, scorecardPath, generatedAt };
}

module.exports = { generateScorecard, parseVerifyLine, SCORECARDS_DIR };
