"""Teacher-uploaded answer-sheet transcription via the Gemini API (free tier).

Explicitly out of scope, by design (see plan/hi-fancy-rain.md Part C): no
verification that this transcription came from the claimed model, or that it's
accurate. The answer text is trusted the same way a human-typed answer already
is — the ZK/witness_hash machinery begins after the text exists, regardless of
its origin. This is a disclosed demo mechanism, not a gap discovered later.
"""
import hashlib
import logging
import os

from google import genai
from google.genai import types

logger = logging.getLogger("certiproof.ml-worker.ocr")

TRANSCRIBE_PROMPT = (
    "You are transcribing a photographed or scanned handwritten exam answer sheet. "
    "Transcribe the student's written answer text exactly as written, correcting only "
    "obvious OCR artifacts (not spelling/grammar). Output only the transcribed answer "
    "text, no commentary, no markdown formatting."
)

_ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
    "image/gif",
}
_MIME_ALIASES = {
    "image/jpg": "image/jpeg",
    "image/pjpeg": "image/jpeg",
    "image/x-png": "image/png",
}


class GeminiNotConfigured(Exception):
    """Raised when GEMINI_API_KEY is missing or blank — distinct from a live API
    failure (bad key, quota, network) so callers can tell "not set up" apart from
    "configured but broken" and respond to each appropriately."""


class GeminiRequestFailed(Exception):
    """The API key is present but the live call itself failed (bad key, quota,
    network, model unavailable, etc). Wraps the SDK's raw error with a short,
    stable message instead of leaking its full exception text to callers."""


_client: "genai.Client | None" = None


def _model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"


def _get_client() -> "genai.Client":
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise GeminiNotConfigured("GEMINI_API_KEY is not set (or is blank)")
        _client = genai.Client(api_key=api_key)
    return _client


def _normalize_mime(image_bytes: bytes, mime_type: str | None) -> str:
    raw = (mime_type or "").split(";")[0].strip().lower()
    raw = _MIME_ALIASES.get(raw, raw)
    if raw in _ALLOWED_MIME:
        return raw
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(image_bytes) >= 12 and image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    if image_bytes.startswith(b"GIF8"):
        return "image/gif"
    return "image/jpeg"


def _extract_answer_text(response) -> str:
    try:
        direct = (response.text or "").strip()
        if direct:
            return direct
    except (ValueError, AttributeError):
        pass

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                continue
            text = getattr(part, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()


def _empty_response_reason(response) -> str:
    feedback = getattr(response, "prompt_feedback", None)
    block = getattr(feedback, "block_reason", None) if feedback is not None else None
    if block:
        return f"request blocked by safety filter ({block})"
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return "no candidates returned"
    reason = getattr(candidates[0], "finish_reason", None)
    return f"empty text (finish_reason={reason})"


def transcribe(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
    client = _get_client()
    model = _model_name()
    mime = _normalize_mime(image_bytes, mime_type)
    logger.info("gemini transcribe model=%s mime=%s bytes=%d", model, mime, len(image_bytes))
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime),
                        types.Part.from_text(text=TRANSCRIBE_PROMPT),
                    ],
                )
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
            ),
        )
    except GeminiNotConfigured:
        raise
    except Exception as e:
        raise GeminiRequestFailed(f"Gemini API call failed ({type(e).__name__}): {e}") from e

    answer_text = _extract_answer_text(response)
    if not answer_text:
        raise GeminiRequestFailed(
            f"Gemini returned an empty transcription — {_empty_response_reason(response)}"
        )
    raw_response_hash = hashlib.sha256(str(response).encode("utf-8", errors="ignore")).hexdigest()
    return {"answer_text": answer_text, "ocr_raw_response_hash": raw_response_hash}
