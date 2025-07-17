# app/api/models.py
"""
Pydantic models for IoMT Attack Detection API
Phase 4: Core Business Logic - Step 5
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

class NetworkTrafficData(BaseModel):
    """Model for network traffic input data."""
    
    # Core network features
    Duration: Optional[float] = Field(0.0, description="Connection duration in seconds")
    Rate: Optional[float] = Field(0.0, description="Data transmission rate")
    Srate: Optional[float] = Field(0.0, description="Source rate")
    
    # Protocol flags
    fin_flag_number: Optional[int] = Field(0, description="FIN flag count")
    syn_flag_number: Optional[int] = Field(0, description="SYN flag count") 
    rst_flag_number: Optional[int] = Field(0, description="RST flag count")
    psh_flag_number: Optional[int] = Field(0, description="PSH flag count")
    ack_flag_number: Optional[int] = Field(0, description="ACK flag count")
    
    # Protocol types (binary indicators)
    TCP: Optional[int] = Field(0, description="TCP protocol indicator (0 or 1)")
    UDP: Optional[int] = Field(0, description="UDP protocol indicator (0 or 1)")
    HTTP: Optional[int] = Field(0, description="HTTP protocol indicator (0 or 1)")
    HTTPS: Optional[int] = Field(0, description="HTTPS protocol indicator (0 or 1)")
    DNS: Optional[int] = Field(0, description="DNS protocol indicator (0 or 1)")
    
    # Packet information
    Header_Length: Optional[float] = Field(0.0, description="Header length")
    packet_size: Optional[float] = Field(0.0, description="Packet size in bytes")
    
    # Statistical features
    Min: Optional[float] = Field(0.0, description="Minimum value")
    Max: Optional[float] = Field(0.0, description="Maximum value")
    AVG: Optional[float] = Field(0.0, description="Average value")
    Std: Optional[float] = Field(0.0, description="Standard deviation")
    
    # Additional features (can be extended)
    additional_features: Optional[Dict[str, Union[int, float]]] = Field(
        default_factory=dict, 
        description="Additional network features"
    )
    
    @validator('TCP', 'UDP', 'HTTP', 'HTTPS', 'DNS')
    def validate_binary_indicators(cls, v):
        """Ensure protocol indicators are 0 or 1."""
        if v not in [0, 1]:
            raise ValueError('Protocol indicators must be 0 or 1')
        return v
    
    @validator('Duration', 'Rate', 'Srate', 'packet_size')
    def validate_non_negative(cls, v):
        """Ensure certain fields are non-negative."""
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "Duration": 0.5,
                "Rate": 1000.0,
                "TCP": 1,
                "UDP": 0,
                "HTTP": 1,
                "packet_size": 1500,
                "Header_Length": 20.0,
                "syn_flag_number": 1,
                "ack_flag_number": 2
            }
        }

class PredictionRequest(BaseModel):
    """Request model for single prediction."""
    
    network_data: NetworkTrafficData = Field(..., description="Network traffic data to analyze")
    include_details: Optional[bool] = Field(True, description="Include detailed prediction information")
    
    class Config:
        schema_extra = {
            "example": {
                "network_data": {
                    "Duration": 0.5,
                    "Rate": 1000.0,
                    "TCP": 1,
                    "HTTP": 1,
                    "packet_size": 1500
                },
                "include_details": True
            }
        }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    network_data_list: List[NetworkTrafficData] = Field(
        ..., 
        description="List of network traffic data to analyze",
        min_items=1,
        max_items=100  # Limit batch size
    )
    include_details: Optional[bool] = Field(True, description="Include detailed prediction information")
    
    @validator('network_data_list')
    def validate_batch_size(cls, v):
        """Validate batch size limits."""
        if len(v) > 100:
            raise ValueError('Batch size cannot exceed 100 samples')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "network_data_list": [
                    {
                        "Duration": 0.5,
                        "Rate": 1000.0,
                        "TCP": 1,
                        "HTTP": 1,
                        "packet_size": 1500
                    },
                    {
                        "Duration": 2.0,
                        "Rate": 50000.0,
                        "UDP": 1,
                        "packet_size": 64
                    }
                ],
                "include_details": True
            }
        }

class PredictionResult(BaseModel):
    """Response model for prediction results."""
    
    prediction: str = Field(..., description="Predicted attack type or 'benign'")
    is_attack: bool = Field(..., description="Whether an attack was detected")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="Prediction timestamp (ISO format)")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    
    # Optional detailed information
    model_version: Optional[str] = Field(None, description="Model version used")
    alert_generated: Optional[bool] = Field(None, description="Whether an alert was generated")
    status: Optional[str] = Field("success", description="Prediction status")
    error_message: Optional[str] = Field(None, description="Error message if status is error")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": "ddos",
                "is_attack": True,
                "confidence": 0.87,
                "timestamp": "2025-07-04T12:00:00.000Z",
                "processing_time_ms": 15.5,
                "model_version": "optimized_system",
                "alert_generated": True,
                "status": "success"
            }
        }

class BatchPredictionResult(BaseModel):
    """Response model for batch prediction results."""
    
    results: List[PredictionResult] = Field(..., description="List of prediction results")
    summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "prediction": "benign",
                        "is_attack": False,
                        "confidence": 0.95,
                        "timestamp": "2025-07-04T12:00:00.000Z",
                        "processing_time_ms": 12.3,
                        "batch_index": 0
                    }
                ],
                "summary": {
                    "total_samples": 2,
                    "attacks_detected": 1,
                    "average_confidence": 0.78,
                    "total_processing_time_ms": 25.8
                }
            }
        }

class ServiceStatus(BaseModel):
    """Model for service status information."""
    
    initialized: bool = Field(..., description="Whether service is initialized")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    mock_mode: bool = Field(..., description="Whether running in mock mode")
    alert_threshold: float = Field(..., description="Current alert threshold")
    timestamp: str = Field(..., description="Status timestamp")
    
    # Optional model information
    model_info: Optional[Dict[str, Any]] = Field(None, description="Detailed model information")
    
    class Config:
        schema_extra = {
            "example": {
                "initialized": True,
                "models_loaded": True,
                "mock_mode": True,
                "alert_threshold": 0.7,
                "timestamp": "2025-07-04T12:00:00.000Z",
                "model_info": {
                    "loaded": True,
                    "mock_mode": True,
                    "models": {
                        "lightgbm": {"type": "MockLightGBMModel", "available": True}
                    }
                }
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    
    class Config:
        schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data format",
                "timestamp": "2025-07-04T12:00:00.000Z",
                "details": {
                    "field": "packet_size",
                    "issue": "must be non-negative"
                }
            }
        }

# Helper function to create error responses
def create_error_response(error_type: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """Create a standardized error response."""
    return ErrorResponse(
        error=error_type,
        message=message,
        timestamp=datetime.utcnow().isoformat(),
        details=details or {}
    )