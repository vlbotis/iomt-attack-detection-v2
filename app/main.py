# app/main.py
"""
IoMT Attack Detection System - Main FastAPI Application
Phase 4: Complete Integration with Prediction API
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
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# Try to import and include routers
try:
    from app.api.prediction import router as prediction_router
    app.include_router(
        prediction_router, 
        prefix=f"{settings.API_V1_STR}/predict", 
        tags=["Prediction API"]
    )
    PREDICTION_API_AVAILABLE = True
    logger.info("✅ Prediction API router loaded successfully")
except ImportError as e:
    PREDICTION_API_AVAILABLE = False
    logger.warning(f"⚠️ Prediction API not available: {e}")

try:
    from app.api.alerts import router as alerts_router
    app.include_router(
        alerts_router,
        prefix=f"{settings.API_V1_STR}/alerts",
        tags=["Alert API"]
    )
    ALERTS_API_AVAILABLE = True
    logger.info("✅ Alert API router loaded successfully")
except ImportError as e:
    ALERTS_API_AVAILABLE = False
    logger.warning(f"⚠️ Alert API not available: {e}")

# Try to import prediction service for initialization
try:
    from app.services.prediction import initialize_prediction_service
    PREDICTION_SERVICE_AVAILABLE = True
except ImportError as e:
    PREDICTION_SERVICE_AVAILABLE = False
    logger.warning(f"⚠️ Prediction service not available: {e}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Mock ML models: {settings.MOCK_ML_MODELS}")
    
    # Initialize prediction service if available
    if PREDICTION_SERVICE_AVAILABLE:
        logger.info("Initializing prediction service...")
        try:
            prediction_initialized = initialize_prediction_service()
            if prediction_initialized:
                logger.info("✅ Prediction service initialized successfully")
            else:
                logger.warning("⚠️ Prediction service initialization failed - using fallback mode")
        except Exception as e:
            logger.error(f"❌ Error initializing prediction service: {e}")
    else:
        logger.info("Prediction service not available - running in basic mode")

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
        
        # Check prediction API availability
        if PREDICTION_API_AVAILABLE:
            health_status["components"]["prediction_api"] = "available"
        else:
            health_status["components"]["prediction_api"] = "not_available"
            health_status["status"] = "degraded"
        
        # Check alerts API availability
        if ALERTS_API_AVAILABLE:
            health_status["components"]["alerts_api"] = "available"
        else:
            health_status["components"]["alerts_api"] = "not_available"
        
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
        
        # Check prediction service status if available
        if PREDICTION_SERVICE_AVAILABLE:
            try:
                from app.services.prediction import get_prediction_service
                service = get_prediction_service()
                if service.initialized:
                    health_status["components"]["prediction_service"] = "initialized"
                else:
                    health_status["components"]["prediction_service"] = "not_initialized"
                    health_status["status"] = "degraded"
            except Exception as e:
                health_status["components"]["prediction_service"] = f"error: {str(e)}"
                health_status["status"] = "degraded"
        else:
            health_status["components"]["prediction_service"] = "not_available"
        
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
    response = {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "docs": "/docs" if settings.DEBUG else "disabled",
        "health": "/health",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    # Add API info if available
    if PREDICTION_API_AVAILABLE:
        response["prediction_api"] = f"{settings.API_V1_STR}/predict"
        response["endpoints"] = {
            "single_prediction": f"{settings.API_V1_STR}/predict/single",
            "batch_prediction": f"{settings.API_V1_STR}/predict/batch",
            "quick_analysis": f"{settings.API_V1_STR}/predict/analyze",
            "service_status": f"{settings.API_V1_STR}/predict/status",
            "available_models": f"{settings.API_V1_STR}/predict/models"
        }
    else:
        response["prediction_api"] = "not_available"
    
    if ALERTS_API_AVAILABLE:
        response["alerts_api"] = f"{settings.API_V1_STR}/alerts"
        if "endpoints" not in response:
            response["endpoints"] = {}
        response["endpoints"].update({
            "alerts_list": f"{settings.API_V1_STR}/alerts/",
            "alert_statistics": f"{settings.API_V1_STR}/alerts/statistics/summary",
            "alert_trends": f"{settings.API_V1_STR}/alerts/statistics/trends",
            "create_test_alert": f"{settings.API_V1_STR}/alerts/test"
        })
    else:
        response["alerts_api"] = "not_available"
    
    if not PREDICTION_API_AVAILABLE and not ALERTS_API_AVAILABLE:
        response["note"] = "APIs not loaded - check logs for details"
    
    return response

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