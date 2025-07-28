# app/api/prediction.py
"""
Prediction API endpoints for IoMT Attack Detection System
Phase 4: Core Business Logic - Step 6
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from app.api.models import (
    PredictionRequest, 
    BatchPredictionRequest,
    PredictionResult, 
    BatchPredictionResult,
    ServiceStatus,
    ErrorResponse,
    create_error_response
)
from app.services.prediction import get_prediction_service, initialize_prediction_service
from app.core.config import settings

# Create router
router = APIRouter()

# Setup logging
logger = logging.getLogger(__name__)

@router.get("/status", response_model=ServiceStatus, tags=["Service"])
async def get_prediction_service_status():
    """
    Get the current status of the prediction service.
    
    Returns:
        ServiceStatus: Current service status and model information
    """
    try:
        service = get_prediction_service()
        status = service.get_service_status()
        
        return ServiceStatus(**status)
        
    except Exception as e:
        logger.error(f"Failed to get service status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get service status: {str(e)}"
        )

@router.post("/single", response_model=PredictionResult, tags=["Prediction"])
async def predict_single_traffic(request: PredictionRequest):
    """
    Analyze a single network traffic sample for attack detection.
    
    Args:
        request (PredictionRequest): Network traffic data to analyze
        
    Returns:
        PredictionResult: Attack prediction with confidence score
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    try:
        # Get prediction service
        service = get_prediction_service()
        
        if not service.initialized:
            raise HTTPException(
                status_code=503,
                detail="Prediction service not initialized"
            )
        
        # Convert Pydantic model to dict for processing
        network_data = request.network_data.dict()
        
        # Make prediction
        result = service.predict_single(network_data)
        
        # Check if prediction was successful
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {result.get('error_message', 'Unknown error')}"
            )
        
        # Filter response based on include_details flag
        include_details = getattr(request, 'include_details', True)
        if not include_details:
            # Return minimal response
            filtered_result = {
                "prediction": result["prediction"],
                "is_attack": result["is_attack"],
                "confidence": result["confidence"],
                "timestamp": result["timestamp"],
                "processing_time_ms": result["processing_time_ms"],
                "status": result["status"]
            }
        else:
            filtered_result = result
        
        return PredictionResult(**filtered_result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResult, tags=["Prediction"])
async def predict_batch_traffic(request: BatchPredictionRequest):
    """
    Analyze a batch of network traffic samples for attack detection.
    
    Args:
        request (BatchPredictionRequest): List of network traffic data to analyze
        
    Returns:
        BatchPredictionResult: List of predictions with summary statistics
        
    Raises:
        HTTPException: If prediction fails or service is unavailable
    """
    try:
        # Get prediction service
        service = get_prediction_service()
        
        if not service.initialized:
            raise HTTPException(
                status_code=503,
                detail="Prediction service not initialized"
            )
        
        # Convert Pydantic models to dicts
        network_data_list = [item.dict() for item in request.network_data_list]
        
        # Make batch prediction
        results = service.predict_batch(network_data_list)
        
        # Check for errors
        if not results or (len(results) == 1 and results[0].get("status") == "error"):
            raise HTTPException(
                status_code=500,
                detail=f"Batch prediction failed: {results[0].get('error_message', 'Unknown error') if results else 'No results'}"
            )
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.get("status") == "success"]
        attacks_detected = sum(1 for r in successful_results if r.get("is_attack", False))
        
        if successful_results:
            avg_confidence = sum(r.get("confidence", 0) for r in successful_results) / len(successful_results)
            total_processing_time = sum(r.get("processing_time_ms", 0) for r in successful_results)
        else:
            avg_confidence = 0.0
            total_processing_time = 0.0
        
        summary = {
            "total_samples": len(request.network_data_list),
            "successful_predictions": len(successful_results),
            "failed_predictions": len(results) - len(successful_results),
            "attacks_detected": attacks_detected,
            "attack_rate": attacks_detected / len(successful_results) if successful_results else 0.0,
            "average_confidence": round(avg_confidence, 4),
            "total_processing_time_ms": round(total_processing_time, 2)
        }
        
        # Filter results based on include_details flag
        include_details = getattr(request, 'include_details', True)
        if not include_details:
            # Return minimal response for each result
            filtered_results = []
            for result in results:
                filtered_result = {
                    "prediction": result.get("prediction", "unknown"),
                    "is_attack": result.get("is_attack", False),
                    "confidence": result.get("confidence", 0.0),
                    "timestamp": result.get("timestamp", ""),
                    "processing_time_ms": result.get("processing_time_ms", 0.0),
                    "status": result.get("status", "unknown")
                }
                if "batch_index" in result:
                    filtered_result["batch_index"] = result["batch_index"]
                filtered_results.append(PredictionResult(**filtered_result))
        else:
            filtered_results = [PredictionResult(**result) for result in results]
        
        return BatchPredictionResult(
            results=filtered_results,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.post("/analyze", response_model=PredictionResult, tags=["Prediction"])
async def analyze_network_traffic(
    network_data: Dict[str, Any],
    background_tasks: BackgroundTasks
):
    """
    Quick analysis endpoint for simple network traffic data.
    
    This is a simplified endpoint that accepts raw JSON data
    without strict validation for rapid prototyping.
    
    Args:
        network_data (Dict): Raw network traffic data
        background_tasks (BackgroundTasks): For logging/analytics
        
    Returns:
        PredictionResult: Attack prediction result
    """
    try:
        # Get prediction service
        service = get_prediction_service()
        
        if not service.initialized:
            raise HTTPException(
                status_code=503,
                detail="Prediction service not initialized"
            )
        
        # Make prediction with raw data
        result = service.predict_single(network_data)
        
        # Add background task for logging (example)
        background_tasks.add_task(log_prediction_request, network_data, result)
        
        # Check if prediction was successful
        if result.get("status") == "error":
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed: {result.get('error_message', 'Unknown error')}"
            )
        
        return PredictionResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/models", tags=["Service"])
async def get_available_models():
    """
    Get information about available ML models.
    
    Returns:
        Dict: Information about loaded models and their capabilities
    """
    try:
        service = get_prediction_service()
        
        if not service.initialized:
            return JSONResponse(
                status_code=503,
                content={"error": "Service not initialized", "models": []}
            )
        
        model_info = service.get_service_status()
        
        # Extract model information
        models_data = {
            "available_models": [],
            "mock_mode": model_info.get("mock_mode", False),
            "service_initialized": model_info.get("initialized", False)
        }
        
        if model_info.get("model_info") and model_info["model_info"].get("models"):
            for model_name, model_details in model_info["model_info"]["models"].items():
                models_data["available_models"].append({
                    "name": model_name,
                    "type": model_details.get("type", "unknown"),
                    "available": model_details.get("available", False),
                    "description": _get_model_description(model_name)
                })
        
        return models_data
        
    except Exception as e:
        logger.error(f"Failed to get model information: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )

def _get_model_description(model_name: str) -> str:
    """Get description for a model."""
    descriptions = {
        "lightgbm": "LightGBM gradient boosting model for attack classification",
        "feature_processor": "Feature preprocessing and selection component",
        "scaler": "Data normalization and scaling component"
    }
    return descriptions.get(model_name, f"ML component: {model_name}")

async def log_prediction_request(network_data: Dict[str, Any], result: Dict[str, Any]):
    """Background task to log prediction requests for analytics."""
    try:
        # This could be enhanced to log to a database, send to analytics service, etc.
        logger.info(f"Prediction logged - Attack: {result.get('is_attack', False)}, "
                   f"Type: {result.get('prediction', 'unknown')}, "
                   f"Confidence: {result.get('confidence', 0):.2%}")
    except Exception as e:
        logger.error(f"Failed to log prediction request: {str(e)}")

# Health check for the prediction API specifically
@router.get("/health", tags=["Service"])
async def prediction_api_health():
    """
    Health check specifically for the prediction API.
    
    Returns:
        Dict: Health status of the prediction API and its dependencies
    """
    try:
        service = get_prediction_service()
        
        health_status = {
            "status": "healthy" if service.initialized else "degraded",
            "api": "operational",
            "prediction_service": "initialized" if service.initialized else "not_initialized",
            "models_loaded": service.model_loader.is_loaded() if service.initialized else False,
            "mock_mode": settings.MOCK_ML_MODELS,
            "alert_threshold": settings.ALERT_THRESHOLD
        }
        
        status_code = 200 if service.initialized else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
        
    except Exception as e:
        logger.error(f"Prediction API health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "api": "error"
            }
        )