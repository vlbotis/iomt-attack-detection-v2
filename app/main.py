# app/main.py
"""
IoMT Attack Detection System - Main FastAPI Application
Phase 2: Basic API Foundation with Health Check
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration
from app.core.config import settings, setup_logging

# Initialize logging
logger = setup_logging()

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="IoMT Attack Detection System - Real-time network traffic analysis and threat detection",
    docs_url="/docs" if settings.DEBUG else None,  # Disable in production
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Mock ML models: {settings.MOCK_ML_MODELS}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown."""
    logger.info("Shutting down IoMT Attack Detection System")

# Health check endpoint
@app.get("/health", tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the system is running.
    
    Returns:
        dict: System health status and basic information
    """
    try:
        # Basic system health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "service": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "debug": settings.DEBUG,
            "components": {
                "api": "operational",
                "config": "loaded",
                "logging": "active"
            }
        }
        
        # Check if ML models are configured
        if settings.MOCK_ML_MODELS:
            health_status["components"]["ml_models"] = "mocked"
        else:
            # Check if model files exist
            model_path = Path(settings.MODEL_PATH)
            if model_path.exists():
                health_status["components"]["ml_models"] = "ready"
            else:
                health_status["components"]["ml_models"] = "not_found"
                health_status["status"] = "degraded"
        
        logger.info("Health check completed successfully")
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "service": settings.PROJECT_NAME
            }
        )

# Root endpoint
@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint with basic system information.
    
    Returns:
        dict: Basic system information and available endpoints
    """
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }

# Add CORS middleware if needed
if settings.DEBUG:
    from fastapi.middleware.cors import CORSMiddleware
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower()
    )