from fastapi import APIRouter, HTTPException, status
from models.schemas import GenerationRequest, Schema, GeneratedData
from pipeline import create_pipeline
import logging

router = APIRouter(
    prefix="/api/v1",
    tags=["synthetic-data"],
    responses={404: {"description": "Not found"}}
)

# ✅ Create graph instance
graph = create_pipeline()

logger = logging.getLogger(__name__)

@router.post(
    "/generate",
    response_model=GeneratedData,
    summary="Run full synthetic data generation pipeline",
    description="Run full pipeline: scenario → schema → approval → data → output",
    status_code=status.HTTP_200_OK
)
async def generate_data(request: GenerationRequest) -> GeneratedData:
    try:
        logger.info(f"🔁 Running full generation pipeline for scenario: {request.scenario[:60]}")
        # ✅ Use ainvoke for async pipeline
        result = await graph.ainvoke({"request": request})
        
        if result.get("error"):
            raise ValueError(result["error"])
        
        return result["output"]
    except Exception as e:
        logger.exception("Pipeline execution failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.post(
    "/schema/preview",
    response_model=Schema,
    summary="Preview inferred schema",
    description="Preprocess and infer schema without running full pipeline",
    status_code=status.HTTP_200_OK
)
async def preview_schema(request: GenerationRequest) -> Schema:
    try:
        logger.info(f"🧪 Previewing schema for scenario: {request.scenario[:60]}")
        state = await graph.ainvoke({"request": request})  # ✅ also async
        if state.get("error"):
            raise ValueError(state["error"])
        return state["schema"]
    except Exception as e:
        logger.exception("Schema preview failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.get("/health")
async def health_check():
    return {"status": "healthy"}
