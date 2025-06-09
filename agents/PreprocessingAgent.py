import re
from utils.gemini_model import GeminiModel

class PreprocessingAgent:
    def clean(self, scenario: str) -> str:
        """Cleans and clarifies the input scenario"""
        prompt = f"""
        Simplify and clean the following user scenario.
        Extract the core business domain and main data entities needed.
        Refine this data generation scenario:
        {scenario}

        Rules:
        1. Remove irrelevant details
        2. Expand abbreviations
        3. Clarify ambiguous terms
        4. Preserve technical requirements

        Return ONLY the cleaned text.
        """
        cleaned = GeminiModel.generate(prompt)
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    def enrich_field_metadata(self, scenario: str, field_name: str, field_type: str) -> dict:
        """Generates missing field description, constraints, and example"""
        prompt = f"""
        Given the scenario: "{scenario}"

        Provide a detailed description, sample constraint, and realistic example value
        for a data field named "{field_name}" of type "{field_type}".

        Respond in pure JSON like:
        {{
            "description": "...",
            "constraints": "...",
            "example": "..."
        }}
        """
        response = GeminiModel.generate(prompt).strip()
        response = response.replace("```json", "").replace("```", "").strip()

        # Basic fallback if Gemini misbehaves
        try:
            import json
            return json.loads(response)
        except:
            return {
                "description": f"{field_name.replace('_', ' ').capitalize()}",
                "constraints": None,
                "example": None
            }
