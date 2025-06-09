from fastapi import FastAPI
from routes import router
from utils.gemini_model import GeminiModel
import uvicorn
import logging

logging.basicConfig(
    level=logging.INFO,  # or DEBUG to get more details
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),    # Log to a file named app.log
        logging.StreamHandler()            # Also log to console (optional)
    ]
)
def create_app() -> FastAPI:
    app = FastAPI(
        title="Synthetic Data Generator",
        description="API for generating synthetic data using Gemini AI",
        version="1.0.0"
    )
    
    # Initialize services
    GeminiModel.configure()
    
    # Include routes
    app.include_router(router)
    
    return app

app = create_app()

if __name__ == "__main__":
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)