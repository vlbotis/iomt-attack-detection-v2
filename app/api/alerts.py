# app/api/alerts.py
"""
Alert API endpoints for IoMT Attack Detection System
Phase 5: Alert System - Step 8
"""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.services.alert import (
    get_alert_service, 
    AlertSeverity, 
    AlertStatus, 
    Alert,
    create_alert_from_prediction
)

# Create router
router = APIRouter()

# Setup logging
logger = logging.getLogger(__name__)

# Pydantic models for API
from pydantic import BaseModel, Field

class AlertResponse(BaseModel):
    """Response model for alert data."""
    id: str
    timestamp: str
    attack_type: str
    confidence: float
    severity: str
    status: str
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None
    packet_size: Optional[float] = None
    description: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    notes: Optional[str] = None

class AlertActionRequest(BaseModel):
    """Request model for alert actions."""
    action_by: str = Field(..., description="Username or identifier of who is performing the action")
    notes: Optional[str] = Field(None, description="Optional notes about the action")

class AlertStatistics(BaseModel):
    """Response model for alert statistics."""
    period_days: int
    total_alerts: int
    active_alerts: int
    acknowledged_alerts: int
    resolved_alerts: int
    false_positives: int
    attack_types: Dict[str, int]
    severity_distribution: Dict[str, int]
    average_confidence: float
    false_positive_rate: float

def _alert_to_response(alert: Alert) -> AlertResponse:
    """Convert Alert object to AlertResponse."""
    return AlertResponse(
        id=alert.id,
        timestamp=alert.timestamp,
        attack_type=alert.attack_type,
        confidence=alert.confidence,
        severity=alert.severity.value,
        status=alert.status.value,
        source_ip=alert.source_ip,
        destination_ip=alert.destination_ip,
        protocol=alert.protocol,
        packet_size=alert.packet_size,
        description=alert.description,
        acknowledged_by=alert.acknowledged_by,
        acknowledged_at=alert.acknowledged_at,
        resolved_at=alert.resolved_at,
        notes=alert.notes
    )

@router.get("/", response_model=List[AlertResponse], tags=["Alerts"])
async def get_alerts(
    status: Optional[str] = Query(None, description="Filter by status: active, acknowledged, resolved, false_positive"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
    attack_type: Optional[str] = Query(None, description="Filter by attack type"),
    limit: Optional[int] = Query(50, description="Maximum number of alerts to return", le=100),
    offset: int = Query(0, description="Number of alerts to skip", ge=0)
):
    """
    Get alerts with optional filtering and pagination.
    
    Returns:
        List[AlertResponse]: List of alerts matching the criteria
    """
    try:
        service = get_alert_service()
        
        # Convert string parameters to enums
        status_enum = None
        if status:
            try:
                status_enum = AlertStatus(status.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status: {status}. Valid values: active, acknowledged, resolved, false_positive"
                )
        
        severity_enum = None
        if severity:
            try:
                severity_enum = AlertSeverity(severity.lower())
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid severity: {severity}. Valid values: low, medium, high, critical"
                )
        
        # Get alerts
        alerts = service.get_alerts(
            status=status_enum,
            severity=severity_enum,
            attack_type=attack_type,
            limit=limit,
            offset=offset
        )
        
        # Convert to response format
        return [_alert_to_response(alert) for alert in alerts]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alerts: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )

@router.get("/{alert_id}", response_model=AlertResponse, tags=["Alerts"])
async def get_alert(alert_id: str):
    """
    Get a specific alert by ID.
    
    Args:
        alert_id (str): The alert ID
        
    Returns:
        AlertResponse: The alert details
    """
    try:
        service = get_alert_service()
        alert = service.get_alert(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
        return _alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alert: {str(e)}"
        )

@router.post("/{alert_id}/acknowledge", response_model=AlertResponse, tags=["Alert Actions"])
async def acknowledge_alert(alert_id: str, request: AlertActionRequest):
    """
    Acknowledge an alert.
    
    Args:
        alert_id (str): The alert ID
        request (AlertActionRequest): Action details
        
    Returns:
        AlertResponse: Updated alert details
    """
    try:
        service = get_alert_service()
        
        success = service.acknowledge_alert(
            alert_id=alert_id,
            acknowledged_by=request.action_by,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
        # Return updated alert
        alert = service.get_alert(alert_id)
        return _alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )

@router.post("/{alert_id}/resolve", response_model=AlertResponse, tags=["Alert Actions"])
async def resolve_alert(alert_id: str, request: AlertActionRequest):
    """
    Resolve an alert.
    
    Args:
        alert_id (str): The alert ID
        request (AlertActionRequest): Action details
        
    Returns:
        AlertResponse: Updated alert details
    """
    try:
        service = get_alert_service()
        
        success = service.resolve_alert(
            alert_id=alert_id,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
        # Return updated alert
        alert = service.get_alert(alert_id)
        return _alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resolve alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve alert: {str(e)}"
        )

@router.post("/{alert_id}/false-positive", response_model=AlertResponse, tags=["Alert Actions"])
async def mark_false_positive(alert_id: str, request: AlertActionRequest):
    """
    Mark an alert as a false positive.
    
    Args:
        alert_id (str): The alert ID
        request (AlertActionRequest): Action details
        
    Returns:
        AlertResponse: Updated alert details
    """
    try:
        service = get_alert_service()
        
        success = service.mark_false_positive(
            alert_id=alert_id,
            notes=request.notes
        )
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
        # Return updated alert
        alert = service.get_alert(alert_id)
        return _alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to mark alert {alert_id} as false positive: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to mark alert as false positive: {str(e)}"
        )

@router.get("/statistics/summary", response_model=AlertStatistics, tags=["Analytics"])
async def get_alert_statistics(
    days: int = Query(7, description="Number of days to include in statistics", ge=1, le=365)
):
    """
    Get alert statistics for the specified time period.
    
    Args:
        days (int): Number of days to include (default: 7)
        
    Returns:
        AlertStatistics: Alert statistics and analytics
    """
    try:
        service = get_alert_service()
        stats = service.get_alert_statistics(days=days)
        
        return AlertStatistics(**stats)
        
    except Exception as e:
        logger.error(f"Failed to get alert statistics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alert statistics: {str(e)}"
        )

@router.get("/statistics/trends", tags=["Analytics"])
async def get_alert_trends(
    days: int = Query(30, description="Number of days for trend analysis", ge=7, le=365)
):
    """
    Get alert trends and patterns over time.
    
    Args:
        days (int): Number of days to analyze
        
    Returns:
        Dict: Trend analysis data
    """
    try:
        service = get_alert_service()
        
        # Get alerts for the specified period
        alerts = service.get_alerts(limit=1000)  # Get recent alerts
        
        # Group by day
        daily_counts = {}
        attack_type_trends = {}
        severity_trends = {}
        
        for alert in alerts:
            # Extract date from timestamp
            alert_date = alert.timestamp[:10]  # YYYY-MM-DD format
            
            # Daily counts
            daily_counts[alert_date] = daily_counts.get(alert_date, 0) + 1
            
            # Attack type trends
            attack_type = alert.attack_type
            if attack_type not in attack_type_trends:
                attack_type_trends[attack_type] = {}
            attack_type_trends[attack_type][alert_date] = attack_type_trends[attack_type].get(alert_date, 0) + 1
            
            # Severity trends
            severity = alert.severity.value
            if severity not in severity_trends:
                severity_trends[severity] = {}
            severity_trends[severity][alert_date] = severity_trends[severity].get(alert_date, 0) + 1
        
        return {
            "period_days": days,
            "daily_alert_counts": daily_counts,
            "attack_type_trends": attack_type_trends,
            "severity_trends": severity_trends,
            "total_alerts_analyzed": len(alerts)
        }
        
    except Exception as e:
        logger.error(f"Failed to get alert trends: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve alert trends: {str(e)}"
        )

@router.delete("/{alert_id}", tags=["Alert Management"])
async def delete_alert(alert_id: str):
    """
    Delete an alert (admin only).
    
    Args:
        alert_id (str): The alert ID to delete
        
    Returns:
        Dict: Confirmation message
    """
    try:
        service = get_alert_service()
        
        if alert_id not in service.alerts:
            raise HTTPException(
                status_code=404,
                detail=f"Alert {alert_id} not found"
            )
        
        # Delete the alert
        del service.alerts[alert_id]
        service._save_alerts()
        
        logger.info(f"Alert {alert_id} deleted")
        
        return {
            "message": f"Alert {alert_id} deleted successfully",
            "deleted_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert {alert_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete alert: {str(e)}"
        )

@router.post("/test", response_model=AlertResponse, tags=["Testing"])
async def create_test_alert():
    """
    Create a test alert for development and testing purposes.
    
    Returns:
        AlertResponse: The created test alert
    """
    try:
        # Create test prediction result
        test_prediction = {
            "prediction": "intrusion",
            "is_attack": True,
            "confidence": 0.85,
            "timestamp": datetime.utcnow().isoformat(),
            "alert_generated": True,
            "model_version": "test"
        }
        
        test_network_data = {
            "source_ip": "192.168.1.100",
            "destination_ip": "10.0.0.1",
            "TCP": 1,
            "packet_size": 1500,
            "Duration": 2.5,
            "Rate": 50000.0,
            "syn_flag_number": 50
        }
        
        # Create alert
        alert = create_alert_from_prediction(test_prediction, test_network_data)
        
        if not alert:
            raise HTTPException(
                status_code=500,
                detail="Failed to create test alert"
            )
        
        logger.info(f"Test alert created: {alert.id}")
        return _alert_to_response(alert)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create test alert: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create test alert: {str(e)}"
        )

# Health check for alerts API
@router.get("/health", tags=["System"])
async def alerts_health():
    """
    Health check for the alerts API.
    
    Returns:
        Dict: Health status of the alerts system
    """
    try:
        service = get_alert_service()
        
        # Get basic statistics
        stats = service.get_alert_statistics(days=1)
        
        return {
            "status": "healthy",
            "alerts_api": "operational",
            "total_alerts": len(service.alerts),
            "recent_alerts_24h": stats["total_alerts"],
            "storage_path": str(service.storage_path),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Alerts health check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "alerts_api": "error"
            }
        )