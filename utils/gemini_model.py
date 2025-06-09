import google.generativeai as genai
from config import config
from typing import Dict, Any
from google.generativeai.types import GenerationConfig

class GeminiModel:
    @staticmethod
    def configure():
        genai.configure(api_key=config.GEMINI_API_KEY)

    @staticmethod
    def get_model(model_name=config.DEFAULT_MODEL) -> genai.GenerativeModel:
        return genai.GenerativeModel(model_name)

    def generate(prompt: str, 
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_output_tokens: int = 3500) -> str:
        model = GeminiModel.get_model()
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens
        )

        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text