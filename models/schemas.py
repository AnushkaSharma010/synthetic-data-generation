from pydantic import BaseModel, Field, constr, field_validator
from typing import Annotated, List, Dict, Optional, Literal
from enum import Enum

class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    DATE = "date"
    DATETIME = "datetime"

class FieldDefinition(BaseModel):
    name: str = Field(
        ...,
        pattern=r'^[a-z][a-z0-9_]*$',
        description="Field name in snake_case"
    )
    type: FieldType
    description: Optional[str] = Field(
        ...,
        min_length=20,
        description="Detailed description of the field"
    )
    constraints: Optional[str] = Field(
        None,
        description="Validation rules or constraints"
    )
    example: Optional[str] = Field(
        default=None,
        description="Example value for the field"
    )
    @field_validator('description')
    def validate_description(cls, v):
        if v is not None and len(v) < 20:
            raise ValueError("Description must be at least 20 characters when provided")
        return v

class Schema(BaseModel):
    fields: List[FieldDefinition] = Field(
        ...,
        min_items=1,
        description="List of field definitions"
    )
    sample_size: int = Field(
        default=100,
        gt=0,
        le=10000,
        description="Number of records to generate (1-10,000)"
    )
    scenario: Annotated[str, Field(..., min_length=20, description="Description of the data generation scenario")]

class GenerationRequest(BaseModel):
    """Initial user request to start the process"""
    scenario: str = Field(
        ...,
        min_length=20,
        description="Detailed description of the data scenario"
    )
    sample_size: int = Field(
        default=100,
        description="Number of records to generate"
    )
    output_format: Literal["json", "csv", "excel"] = Field(
        default="json",
        description="Output file format"
    )

class GeneratedData(BaseModel):
    data: Optional[List[Dict]] = Field(
        None,
        description="Generated data in dictionary format"
    )
    file_content: Optional[bytes] = Field(
        None,
        description="Binary file content for CSV/Excel outputs"
    )
    format: str
    message: Optional[str] = Field(
        None,
        description="Status message or instructions"
    )

# Additional models for the update request
class FieldUpdate(BaseModel):
    name: str
    type: Optional[FieldType] = None
    description: Optional[str] = None
    constraints: Optional[str] = None

class SchemaUpdateRequest(BaseModel):
    current_schema: Schema
    field_updates: List[FieldUpdate]
    deleted_fields: List[str] = []
    sample_size: Optional[int] = None
