# Demo Cases

Two ready-to-paste cases for the live walkthrough. Each block below maps directly onto a field in the **Teacher → Certify a Batch** form — copy the text straight in, no editing needed.

*Note: the frontend currently only accepts a student answer via the Teacher's OCR image upload (drag a photo in) — there's no typed-answer textarea yet. These two cases cover the rubric-certification step; bring your own answer sheet photo (or use `scripts/generate_test_sheets.py`) for the grading step.*

> **Gotcha if you've switched between Docker and local dev mid-session:** each evaluation's `proof_path` is recorded as an absolute path at creation time — a Docker-run evaluation stores a container path (`/app/app/models/proofs/...`), a local-run one stores a host path. If the ml-worker process that's *currently* answering `/verify` isn't the same one (Docker vs. local) that created the evaluation, it can't find the proof file at all, and you'll see `proof_validity: false` with a `proof_not_found` detail — that's a stale-path artifact, not a real tamper result. **Always create the batch and submission fresh, right before the demo, under whichever mode (Docker or local) is currently running** — don't reuse an evaluation ID left over from before a mode switch.

---

## Case 1 — Biology: Photosynthesis (clean, expect a strong score)

**Certified by:** `ms_kamble` *(or your own teacher id)*

**Criterion 1** — `c1`, max marks `2`, weight `1.0`
```
Mentions sunlight as the energy source
```

**Criterion 2** — `c2`, max marks `2`, weight `1.0`
```
Mentions water and carbon dioxide as inputs
```

**Criterion 3** — `c3`, max marks `2`, weight `1.0`
```
Mentions glucose and oxygen as outputs
```

**Criterion 4** — `c4`, max marks `2`, weight `0.5`
```
Mentions chlorophyll or the chloroplast as the site of the reaction
```

**A strong student answer to feed through OCR for this batch:**
```
Photosynthesis is the process by which plants use sunlight as their energy source to convert water absorbed from the soil and carbon dioxide taken in from the air into glucose, releasing oxygen as a byproduct. This reaction takes place inside the chloroplast, using the green pigment chlorophyll to capture light energy.
```
Expect a high score and every check in the Auditor view to pass clean.

---

## Case 2 — Newton's Laws (the Break-It case — this is the one to lead with)

**Certified by:** `ms_kamble` *(or your own teacher id)*

**Criterion 1** — `c1`, max marks `2`, weight `1.0`
```
States that an object at rest stays at rest unless acted on by a net force
```

**Criterion 2** — `c2`, max marks `2`, weight `1.0`
```
States that force equals mass times acceleration
```

**Criterion 3** — `c3`, max marks `2`, weight `1.0`
```
States that every action has an equal and opposite reaction
```

**A weaker (but genuine) student answer to feed through OCR for this batch:**
```
Objects don't move unless you push them. If you push something harder it speeds up faster. When two things push on each other they push back the same amount.
```

**What to do live:** after this evaluation comes back with a moderate score, open **Teacher → Review Evaluations**, load the batch, select this row, and note the `witness_hash`. Then go into the database and fabricate the stored per-criterion scores — as if someone edited the display record after proving:

```bash
sqlite3 gateway/data/certiproof.db \
  "UPDATE evaluations SET per_criterion_scores_json = '[1.98,1.97,1.95]' WHERE evaluation_id = 'PASTE_THE_EVALUATION_ID_HERE';"
```

Then paste that same evaluation ID into the **Auditor → Verify an Evaluation** field and hit Verify.

**What happens on screen:** the proof itself was generated against the real, original scores and is structurally untouched — but `ZK Proof` (the `proof_validity` check) fails, because it independently re-derives the true per-criterion scores from the proof's own public output and finds they don't match what's stored. Say aloud: *"The proof's public output IS the score — you can't quietly edit the displayed number without the proof itself disagreeing with you. That's the two-part design (proof + provenance) paying off in one screen."* For the provenance half specifically, run the `stage0_substitution` vector from the Auditor's Tests panel instead — that's the one that trips `witness_hash` rather than `proof_validity`.

---

See `docs/demo_script.md` for the full seven-act walkthrough these two cases slot into (Act 1 Certify → Act 4 Break-It), and `scripts/seed_demo_data.sh` for a scripted, non-interactive version of Case 2 that doesn't depend on the OCR step.
