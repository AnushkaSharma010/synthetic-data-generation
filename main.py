from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from utils.gemini_model import GeminiModel
import uvicorn
import logging

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def create_app() -> FastAPI:
    app = FastAPI(
        title="Synthetic Data Generator",
        description="API for generating synthetic data using Gemini AI",
        version="1.0.0"
    )

    # CORS for frontend (optional)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Change to your frontend URL in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize Gemini
    GeminiModel.configure()

    # Mount routes
    app.include_router(router)

    return app

# Single instance
app = create_app()

if __name__ == "__main__":
    logging.info("Starting FastAPI server at http://localhost:8000 ...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
