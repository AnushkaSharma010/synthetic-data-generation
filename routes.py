from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional
from models.schemas import (
    GenerationRequest,
    Schema,
    GeneratedData,
    SchemaUpdateRequest
)
from modular_pipeline import Pipeline
import logging

router = APIRouter(
    prefix="/api/v1",
    tags=["synthetic-data"],
    responses={404: {"description": "Not found"}}
)

# Initialize the pipeline
pipeline = Pipeline()

# Configure logging
logger = logging.getLogger(__name__)

@router.post(
    "/schema/preview",
    response_model=Schema,
    summary="Preview inferred schema",
    description="Run preprocessing and field inference to generate a schema preview from the scenario",
    status_code=status.HTTP_200_OK
)
async def preview_schema(request: GenerationRequest) -> Schema:
    try:
        logger.info(f"Previewing schema for scenario: {request.scenario[:50]}...")
        return await pipeline.preview_schema(request)
    except Exception as e:
        logger.error(f"Schema preview failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.put(
    "/schema/update",
    response_model=Schema,
    summary="Update schema with modifications",
    description="Apply field edits, additions, or deletions to the schema",
    status_code=status.HTTP_200_OK
)
async def update_schema(update_request: SchemaUpdateRequest) -> Schema:
    try:
        logger.info(
            f"Updating schema with {len(update_request.field_updates)} field changes, "
            f"{len(update_request.deleted_fields)} deletions, "
            f"sample size: {update_request.sample_size}"
        )

        updates = {
            "fields": [
                {**field.model_dump(), "name": field.name}
                for field in update_request.field_updates
            ],
            "deleted_fields": update_request.deleted_fields,
            "sample_size": update_request.sample_size
        }

        return await pipeline.update_schema(update_request.current_schema, updates)

    except ValueError as e:
        logger.error(f"Schema validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Schema update failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.post(
    "/data/generate",
    response_model=GeneratedData,
    summary="Generate synthetic data",
    description="Generate data based on the finalized schema and output format",
    status_code=status.HTTP_200_OK
)
async def generate_data(
    schema: Schema,
    output_format: str = "json",
    sample_size: Optional[int] = None
) -> GeneratedData:
    try:
        logger.info(f"Generating {schema.sample_size} records in {output_format} format")
        
        # Override sample size if provided
        if sample_size is not None:
            schema.sample_size = sample_size
            
        return await pipeline.generate_data(schema, output_format)
    except Exception as e:
        logger.error(f"Data generation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.post(
    "/generate",
    response_model=GeneratedData,
    summary="Legacy full pipeline",
    description="Run the complete end-to-end generation pipeline (legacy support)",
    status_code=status.HTTP_200_OK,
    deprecated=True
)
async def full_generation(request: GenerationRequest) -> GeneratedData:
    try:
        logger.info(f"Running full pipeline for scenario: {request.scenario[:50]}...")
        return await pipeline.run_full_pipeline(request)
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )


@router.get("/health")
async def health_check():
    return {"status": "healthy"}