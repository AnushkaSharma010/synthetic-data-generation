import json
import logging
from json.decoder import JSONDecoder
from models.schemas import Schema
from utils.gemini_model import GeminiModel

logger = logging.getLogger(__name__)

class DataGeneratorAgent:
    def __init__(self):
        self.retry_limit = 2
        self.chunk_size = 20  # max records per Gemini call to avoid truncation

    def generate(self, schema: Schema) -> list:
        total = schema.sample_size
        all_data = []

        for start in range(0, total, self.chunk_size):
            chunk_count = min(self.chunk_size, total - start)

            prompt = f"""
            You are a JSON data generator for synthetic dataset creation.
            Generate **realistic** and **domain-specific** synthetic records.

            === SCHEMA ===
            {schema.model_dump_json(indent=2)}

            === RULES ===
            - Generate exactly {chunk_count} records.
            - Output only a **JSON array** of flat objects. Each object must strictly conform to the schema.
            - Each record must include **all fields** from the schema. No missing or null values.
            - Ensure strict type adherence: string, number, boolean, date (YYYY-MM-DD), datetime (ISO format).
            - Use realistic and consistent values inferred from field names, types, and constraints.
            - If a field represents a date, ensure the value is a valid and realistic date.
            - Do **not** include markdown, headers, comments, or explanations.
            - The output must be valid JSON and must start with `[` and end with `]`.

            === EXAMPLE OUTPUT FORMAT ===
            [
            {{
                "field_1": "value",
                "field_2": 123,
                ...
            }},
            ... (total {chunk_count} items)
            ]
            """
           
            for attempt in range(1, self.retry_limit + 1):
                try:
                    response = GeminiModel.generate(
                        prompt,
                        temperature=0.3,
                        max_output_tokens=2000
                    )

                    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
                    # Try to fix common Gemini issues before decoding
                    if cleaned.endswith(","):
                        cleaned = cleaned[:-1] + "]"  # Fix for trailing comma
                    if not cleaned.startswith("["):
                        cleaned = "[" + cleaned
                    if not cleaned.endswith("]"):
                        cleaned += "]"
                    if not cleaned:
                        raise ValueError("Gemini returned empty response")

                    data, idx = JSONDecoder().raw_decode(cleaned)

                    if not isinstance(data, list):
                        raise ValueError("Gemini output is not a JSON array")

                    if len(data) != chunk_count:
                        raise ValueError(f"Expected {chunk_count} records, got {len(data)}")

                    logger.info(f"[DataGenerator] Successfully generated {len(data)} records from offset {start}")
                    all_data.extend(data)
                    break

                except json.JSONDecodeError as e:
                    logger.warning(f"[DataGenerator] JSON decode error (attempt {attempt}) at offset {start}: {e}")
                except Exception as e:
                    logger.warning(f"[DataGenerator] Attempt {attempt} failed at offset {start}: {e}")
            else:
                raise RuntimeError(f"Data generation failed after {self.retry_limit} retries at offset {start}")

        return all_data
