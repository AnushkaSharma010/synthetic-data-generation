import json
import logging
from typing import Optional
from models.schemas import Schema, FieldType,SchemaUpdateRequest, FieldDefinition
from pydantic_core import ValidationError
from utils.gemini_model import GeminiModel
from agents.PreprocessingAgent import PreprocessingAgent
logger = logging.getLogger(__name__)
class FieldInferenceAgent:
    def __init__(self):
        self.retry_limit = 2
        self.preprocessor = PreprocessingAgent()

    def infer_schema(self, scenario: str, sample_size: int = 100) -> Optional[dict]:
        prompt = f"""
        You are a data schema generator. Create a JSON schema for this scenario:
        {scenario}

        Output Requirements:
        1. MUST return ONLY JSON with fields array and sample_size
        2. DO NOT include scenario field in the response
        3. Field names must be snake_case
        4. Valid types: {[t.value for t in FieldType]}
        5. Each field must have description ≥20 characters
        6. Include constraints where applicable

        Example Output:
        {{
            "fields": [
                {{
                    "name": "user_id",
                    "type": "string",
                    "description": "Unique user identifier (UUID v4 format)",
                    "constraints": "Must be UUID v4 format"
                }}
            ],
            "sample_size": {sample_size}
        }}
        """

        for attempt in range(self.retry_limit):
            try:
                response = GeminiModel.generate(prompt, temperature=0.3, max_output_tokens=2000)
                cleaned = response.strip().replace("```json", "").replace("```", "")
                logger.info(f"[FieldInference] Gemini response attempt {attempt + 1}: {cleaned}")
                schema_data = json.loads(cleaned)

                if not isinstance(schema_data, dict):
                    raise ValueError("Response is not a JSON object")
                if "fields" not in schema_data:
                    raise ValueError("Missing 'fields' array")
                if "sample_size" not in schema_data:
                    raise ValueError("Missing 'sample_size'")
                for field in schema_data["fields"]:
                    if "description" in field and len(field["description"]) < 20:
                        field["description"] += " — additional details TBD."


                return schema_data

            except Exception as e:
                logger.warning(f"[FieldInference] Attempt {attempt + 1} failed: {e}")

        return None
    
    def enrich_updated_schema(self, update_request: SchemaUpdateRequest) -> Schema:
        original = update_request.current_schema
        updated_fields = []

        for field in original.fields:
            # If field is deleted, skip
            if field.name in update_request.deleted_fields:
                continue

            # Check for updated info
            matching_update = next((f for f in update_request.field_updates if f.name == field.name), None)
            if matching_update:
                name = field.name
                ftype = matching_update.type or field.type
                desc = matching_update.description or field.description
                cons = matching_update.constraints or field.constraints

                # Enrich if description/constraints are missing
                if not matching_update.description or not matching_update.constraints:
                    enriched = self.preprocessor.enrich_field_metadata(original.scenario, name, ftype.value)
                    if not matching_update.description:
                        desc = enriched.get("description", desc)
                    if not matching_update.constraints:
                        cons = enriched.get("constraints", cons)

                updated_fields.append(FieldDefinition(
                    name=name, type=ftype, description=desc, constraints=cons
                ))
            else:
                # Keep original field
                updated_fields.append(field)

        # Add any new fields from update request (not present in original)
        new_field_names = {f.name for f in updated_fields}
        for f in update_request.field_updates:
            if f.name not in new_field_names:
                enriched = self.preprocessor.enrich_field_metadata(original.scenario, f.name, f.type.value)
                updated_fields.append(FieldDefinition(
                    name=f.name,
                    type=f.type,
                    description=f.description or enriched["description"],
                    constraints=f.constraints or enriched["constraints"]
                ))

        # Apply new sample size if updated
        sample_size = update_request.sample_size or original.sample_size

        return Schema(fields=updated_fields, sample_size=sample_size, scenario=original.scenario)
