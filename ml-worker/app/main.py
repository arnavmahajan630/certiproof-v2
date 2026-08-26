import asyncio
import logging

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from app.zk import rcajx_ezkl_pipeline as zk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("certiproof.ml-worker")

app = FastAPI(title="CertiProof ML-Worker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.on_event("startup")
async def startup():
    # Unlike the old Zone 2 pipeline, the RCAJ-X circuit is NOT built at startup:
    # ezkl.setup() on this circuit needs a trusted-setup pass heavy enough to OOM
    # a 16GB machine during development. Setup is a one-time step meant to run
    # standalone (`python -m app.zk.rcajx_ezkl_pipeline`), ideally on a machine
    # with more RAM, with the resulting rcajx_circuit/ directory then copied onto
    # wherever this API actually runs. See ml-worker/README.md "Circuit setup".
    if zk.circuit_is_ready():
        logger.info("rcajx circuit ready, rcajx_model_hash = %s", zk.read_model_hash())
    else:
        logger.warning(
            "rcajx EZKL circuit artifacts not found in %s — /ezkl/prove and /ezkl/verify "
            "will 503 until it's built (see ml-worker/README.md 'Circuit setup'). "
            "/rcajx/embed and /rcajx/score (non-proving) still work.",
            zk.CIRCUIT_DIR,
        )

    logger.info("preloading RCAJ-X encoder + scoring model ...")
    from app.rcajx.preprocessing import encoder  # noqa: F401 — triggers model load
    from app.routes import _get_scoring_model

    _get_scoring_model()
    logger.info("rcajx models ready")


@app.get("/health")
def health():
    return {"status": "ok", "circuit_ready": zk.circuit_is_ready()}
