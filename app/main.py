from fastapi import FastAPI
import logging
from pathlib import Path

from app.core.config import settings

#Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

@app.get("/")
async def root():
    """"Root endpoint."""
    return {"message": "IoMT  Attack Detection System", "status": "online"}

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting IoMT Attack Detection Service...")
    # Load model or perform other startup tasks here
    # Example: load_model(settings.MODEL_PATH)
    logger.info("IoMT Attack Detection Service started successfully.")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down IoMT Attack Detection Service...")
    # Perform cleanup tasks here
    logger.info("IoMT Attack Detection Service shut down successfully.")