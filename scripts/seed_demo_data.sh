#!/usr/bin/env bash
# Builds the two pre-cached demo records for the Break-It act (docs/demo_script.md):
# one clean, verifiably valid record, and one deliberately tampered record whose
# specific failure reason is meant to be shown live. Proving happens here, ahead of
# time — the demo itself only ever calls the fast /verify path, per doc 00 Appendix 8.3.
#
# Usage: ./seed_demo_data.sh
# Requires: gateway on :4000, ml-worker on :8001, both already running.
set -euo pipefail

GW=http://127.0.0.1:4000
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HERE/fixtures/demo_records.json"

echo "=== Certifying demo batch (Biology — photosynthesis) ==="
BATCH=$(curl -s -X POST $GW/teacher/batches -H 'Content-Type: application/json' -H 'X-Role: teacher' \
  -d '{"rubric":{"criteria":[
        {"criterion_id":"c1","criterion_text":"Mentions sunlight as the energy source","max_marks":2,"weight":1.0},
        {"criterion_id":"c2","criterion_text":"Mentions water and carbon dioxide as inputs","max_marks":2,"weight":1.0},
        {"criterion_id":"c3","criterion_text":"Mentions glucose and oxygen as outputs","max_marks":2,"weight":1.0},
        {"criterion_id":"c4","criterion_text":"Mentions chlorophyll or the chloroplast as the site of the reaction","max_marks":2,"weight":0.5}
      ]},"certifiedBy":"demo_seed"}')
BATCH_ID=$(echo "$BATCH" | python3 -c "import json,sys;print(json.load(sys.stdin)['batch_id'])")
echo "batch_id=$BATCH_ID"

echo ""
echo "=== Record 1: CLEAN — genuine answer, untouched after proving ==="
CLEAN_ANSWER="Photosynthesis is the process by which plants use sunlight as their energy source to convert water absorbed from the soil and carbon dioxide taken in from the air into glucose, releasing oxygen as a byproduct. This reaction takes place inside the chloroplast, using the green pigment chlorophyll to capture light energy."
CLEAN_SUB=$(curl -s -X POST $GW/student/submissions -H 'Content-Type: application/json' -H 'X-Role: student' \
  -d "$(python3 -c "import json;print(json.dumps({'batch_id':'$BATCH_ID','student_id':'demo_clean_student','answer_text':'''$CLEAN_ANSWER'''}))")")
CLEAN_EVAL_ID=$(echo "$CLEAN_SUB" | python3 -c "import json,sys;print(json.load(sys.stdin)['evaluation_id'])")
CLEAN_SCORE=$(echo "$CLEAN_SUB" | python3 -c "import json,sys;print(json.load(sys.stdin)['evaluation']['final_score'])")
echo "clean evaluation_id=$CLEAN_EVAL_ID final_score=$CLEAN_SCORE"
curl -s "$GW/verify?evaluation_id=$CLEAN_EVAL_ID" | python3 -m json.tool

echo ""
echo "=== Record 2: TAMPERED — witness-substitution vector (the one worth naming in the demo) ==="
TAMPER_ANSWER="Plants take in sunlight and water to make food for themselves. They also release oxygen into the air while doing this."
TAMPER_SUB=$(curl -s -X POST $GW/student/submissions -H 'Content-Type: application/json' -H 'X-Role: student' \
  -d "$(python3 -c "import json;print(json.dumps({'batch_id':'$BATCH_ID','student_id':'demo_tampered_student','answer_text':'''$TAMPER_ANSWER'''}))")")
TAMPER_EVAL_ID=$(echo "$TAMPER_SUB" | python3 -c "import json,sys;print(json.load(sys.stdin)['evaluation_id'])")
ORIG_SCORE=$(echo "$TAMPER_SUB" | python3 -c "import json,sys;print(json.load(sys.stdin)['evaluation']['final_score'])")

DB="$HERE/../gateway/data/certiproof.db"
# fabricate per_criterion_scores_json (the displayed/scorecard scores) as if a
# compromised proving step substituted inflated scores -- the proof itself
# (generated against the ORIGINAL scores) stays untouched and still verifies
# structurally fine on its own; proof_validity's per_criterion_scores sub-check
# is what catches the mismatch against what's actually embedded in the proof.
sqlite3 "$DB" "UPDATE evaluations SET per_criterion_scores_json = '[1.98,1.97,1.95,1.8]' WHERE evaluation_id = '$TAMPER_EVAL_ID';"

echo "tampered evaluation_id=$TAMPER_EVAL_ID (original genuine final_score was $ORIG_SCORE; per_criterion_scores_json now fabricated)"
curl -s "$GW/verify?evaluation_id=$TAMPER_EVAL_ID" | python3 -m json.tool

python3 -c "
import json
json.dump({
    'batch_id': '$BATCH_ID',
    'clean': {'evaluation_id': '$CLEAN_EVAL_ID', 'final_score': $CLEAN_SCORE, 'note': 'untouched, verifies clean'},
    'tampered': {'evaluation_id': '$TAMPER_EVAL_ID', 'original_final_score': $ORIG_SCORE, 'note': 'witness-substitution vector: per_criterion_scores_json fabricated after proving; proof itself still structurally valid, proof_validity check catches it'}
}, open('$OUT', 'w'), indent=2)
"
echo ""
echo "=== Wrote demo record manifest -> $OUT ==="
echo "Use these two evaluation_ids directly in the Auditor view for Act 4 (Break-It) of docs/demo_script.md."
