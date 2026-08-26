import asyncio
import os
import uuid

import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.ocr.gemini_client import GeminiNotConfigured, GeminiRequestFailed
from app.ocr.gemini_client import transcribe as gemini_transcribe
from app.rcajx import model_hash as rcajx_model_hash
from app.rcajx.explain import generate_explanation
from app.rcajx.model import RCAJ_X
from app.rcajx.padded_model import MAX_CHUNKS, MAX_CRITERIA, pad_inputs
from app.rcajx.preprocessing import embed_answer, embed_criteria, negation_mismatch_flag
from app.rcajx.rubric_cache import get_rubric_R
from app.zk import rcajx_ezkl_pipeline as zk

router = APIRouter()

PROOF_DIR = os.path.join(os.path.dirname(__file__), "models", "rcajx", "proofs")
os.makedirs(PROOF_DIR, exist_ok=True)

_scoring_model: RCAJ_X | None = None


def _get_scoring_model() -> RCAJ_X:
    global _scoring_model
    if _scoring_model is None:
        checkpoint = torch.load(zk.CHECKPOINT_PATH, weights_only=False)
        config = {k: v for k, v in checkpoint["config"].items() if k in ["n_heads", "d_k", "d_v", "hidden"]}
        model = RCAJ_X(**config)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        _scoring_model = model
    return _scoring_model


class CriterionSpec(BaseModel):
    criterion_id: str
    criterion_text: str
    max_marks: float
    weight: float = 1.0


class EmbedRequest(BaseModel):
    answer_text: str
    criteria: list[CriterionSpec]
    question_id: str | None = None
    rubric_mode: str = "dev"  # "dev" | "locked" (see app.rcajx.rubric_cache)
    glossary: dict[str, str] = {}


class ScoreRequest(BaseModel):
    R: list[list[float]]
    A: list[list[float]]
    chunks: list[str]
    negation_flags: list[float]
    max_marks: list[float]
    criteria: list[CriterionSpec]
    criterion_weights: list[float] | None = None


class ProveRequest(BaseModel):
    R: list[list[float]]
    A: list[list[float]]
    negation_flags: list[float]
    max_marks: list[float]
    criterion_weights: list[float]
    tag: str | None = None


class VerifyRequest(BaseModel):
    proof_path: str


class RecomputeCommitmentRequest(BaseModel):
    R: list[list[float]]
    A: list[list[float]]
    negation_flags: list[float]
    max_marks: list[float]
    criterion_weights: list[float]
    tag: str | None = None


def _validate_shape(req_field: str, n: int, limit: int):
    if not (1 <= n <= limit):
        raise HTTPException(
            status_code=422,
            detail={"error": "bad_input_shape", "detail": f"{req_field} must have 1..{limit} entries, got {n}"},
        )


@router.post("/rcajx/embed")
def rcajx_embed(req: EmbedRequest):
    """Stage 0: rubric criteria + answer text -> embeddings. This response's raw
    bytes are what the Gateway commits as witness_hash BEFORE any Stage 1-3
    computation (numeric precision contract, see Phase 4 of the integration
    plan) — never re-derive these values from a later step."""
    _validate_shape("criteria", len(req.criteria), MAX_CRITERIA)
    try:
        criteria_texts = [c.criterion_text for c in req.criteria]
        if req.question_id is not None:
            rubric = {
                "question_id": req.question_id,
                "criteria": [
                    {"criterion_id": c.criterion_id, "text": c.criterion_text, "max_marks": c.max_marks}
                    for c in req.criteria
                ],
            }
            R = get_rubric_R(rubric, mode=req.rubric_mode)
        else:
            R = embed_criteria(criteria_texts)

        ans = embed_answer(req.answer_text, req.glossary)
        A, chunks = ans["A"], ans["chunks"]

        top_chunk = chunks[0] if chunks else ""
        negation_flags = [negation_mismatch_flag(c, top_chunk) for c in criteria_texts]
        max_marks = [c.max_marks for c in req.criteria]
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "embed_failure", "detail": str(e)})

    return {
        "R": R.tolist(),
        "A": A.tolist(),
        "chunks": chunks,
        "negation_flags": negation_flags,
        "max_marks": max_marks,
    }


@router.post("/rcajx/score")
def rcajx_score(req: ScoreRequest):
    """Stages 1-3, fast non-proving preview + explanations (unpadded — no circuit
    shape constraint here, that only applies to /ezkl/prove). The proof-bound
    final_score always comes from /ezkl/prove instead."""
    _validate_shape("R", len(req.R), MAX_CRITERIA)
    try:
        model = _get_scoring_model()
        R = torch.tensor(req.R, dtype=torch.float32)
        A = torch.tensor(req.A, dtype=torch.float32)
        negation_flags = torch.tensor(req.negation_flags, dtype=torch.float32)
        max_marks = torch.tensor(req.max_marks, dtype=torch.float32)
        criterion_weights = (
            torch.tensor(req.criterion_weights, dtype=torch.float32) if req.criterion_weights else None
        )

        with torch.no_grad():
            out = model(R, A, negation_flags, max_marks, criterion_weights=criterion_weights)

        criteria_dicts = [
            {"criterion_id": c.criterion_id, "text": c.criterion_text, "max_marks": c.max_marks}
            for c in req.criteria
        ]
        explanations = generate_explanation(model, R, A, req.chunks, criteria_dicts, negation_flags)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "score_failure", "detail": str(e)})

    return {
        "per_criterion_scores": out["per_criterion_scores"].tolist(),
        "final_score": out["final_score"].item() if out["final_score"] is not None else None,
        "spread": out["spread"].tolist(),
        "explanations": explanations,
    }


def _build_padded_inputs(req) -> dict:
    R = torch.tensor(req.R, dtype=torch.float32)
    A = torch.tensor(req.A, dtype=torch.float32)
    negation_flags = torch.tensor(req.negation_flags, dtype=torch.float32)
    max_marks = torch.tensor(req.max_marks, dtype=torch.float32)
    criterion_weights = torch.tensor(req.criterion_weights, dtype=torch.float32)
    return pad_inputs(R, A, negation_flags, max_marks, criterion_weights=criterion_weights)


@router.post("/ezkl/prove")
def ezkl_prove(req: ProveRequest):
    _validate_shape("R", len(req.R), MAX_CRITERIA)
    _validate_shape("A", len(req.A), MAX_CHUNKS)
    if not zk.circuit_is_ready():
        raise HTTPException(
            status_code=503,
            detail={"error": "circuit_not_ready", "detail": "EZKL circuit not built — see ml-worker/README.md 'Circuit setup'"},
        )
    tag = req.tag or uuid.uuid4().hex
    try:
        padded = _build_padded_inputs(req)
        result = zk.prove(padded, PROOF_DIR, tag=tag)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "proving_failure", "detail": str(e)})
    return result


@router.post("/ezkl/recompute-commitment")
def ezkl_recompute_commitment(req: RecomputeCommitmentRequest):
    _validate_shape("R", len(req.R), MAX_CRITERIA)
    _validate_shape("A", len(req.A), MAX_CHUNKS)
    if not zk.circuit_is_ready():
        raise HTTPException(
            status_code=503,
            detail={"error": "circuit_not_ready", "detail": "EZKL circuit not built"},
        )
    tag = req.tag or uuid.uuid4().hex
    try:
        padded = _build_padded_inputs(req)
        commitment = zk.recompute_poseidon_commitment(padded, tag=tag)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "commitment_recompute_failure", "detail": str(e)})
    return {"poseidon_commitment": commitment}


@router.post("/ezkl/verify")
def ezkl_verify(req: VerifyRequest):
    if not os.path.exists(req.proof_path):
        raise HTTPException(
            status_code=404,
            detail={"error": "proof_not_found", "detail": req.proof_path},
        )
    return zk.verify(req.proof_path)


@router.get("/rcajx/model-hash")
def rcajx_model_hash_route():
    """Returns the same zone1_model_hash / zone2_model_hash column names the
    Gateway DB schema already has (Phase 4: repointed meaning, not renamed) plus
    the compiled-circuit hash under rcajx_model_hash."""
    try:
        zone1_hash = rcajx_model_hash.encoder_model_hash()
        zone2_hash = rcajx_model_hash.scoring_model_hash()
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": "model_hash_failure", "detail": str(e)})
    result = {"zone1_model_hash": zone1_hash, "zone2_model_hash": zone2_hash}
    if zk.circuit_is_ready():
        result["rcajx_model_hash"] = zk.read_model_hash()
    return result


@router.post("/ocr/transcribe")
async def ocr_transcribe(file: UploadFile = File(...)):
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail={"error": "empty_file", "detail": "no image data received"})
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, gemini_transcribe, image_bytes, file.content_type or "image/jpeg"
        )
    except GeminiNotConfigured as e:
        raise HTTPException(status_code=503, detail={"error": "gemini_not_configured", "detail": str(e)})
    except GeminiRequestFailed as e:
        raise HTTPException(status_code=502, detail={"error": "ocr_failure", "detail": str(e)})
    except Exception as e:
        raise HTTPException(status_code=502, detail={"error": "ocr_failure", "detail": str(e)})
    return result
