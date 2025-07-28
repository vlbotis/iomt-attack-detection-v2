# app/services/alert.py
"""
Alert Service for IoMT Attack Detection System
Phase 5: Alert System - Step 7
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"

@dataclass
class Alert:
    """Alert data structure."""
    id: str
    timestamp: str
    attack_type: str
    confidence: float
    severity: AlertSeverity
    status: AlertStatus
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    protocol: Optional[str] = None
    packet_size: Optional[float] = None
    network_data: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    notes: Optional[str] = None

class AlertService:
    """Service for managing IoMT attack alerts."""
    
    def __init__(self, storage_path: str = "alerts.json"):
        """Initialize the alert service."""
        self.storage_path = Path(storage_path)
        self.alerts: Dict[str, Alert] = {}
        self.alert_counter = 0
        self._load_alerts()
        logger.info("Alert service initialized")
    
    def create_alert(self, prediction_result: Dict[str, Any], network_data: Dict[str, Any] = None) -> Alert:
        """
        Create a new alert from a prediction result.
        
        Args:
            prediction_result: The prediction result that triggered the alert
            network_data: Original network data that was analyzed
            
        Returns:
            Alert: The created alert
        """
        try:
            # Generate unique alert ID
            self.alert_counter += 1
            alert_id = f"ALERT_{datetime.now().strftime('%Y%m%d')}_{self.alert_counter:06d}"
            
            # Determine severity based on attack type and confidence
            severity = self._determine_severity(
                prediction_result.get("prediction", "unknown"),
                prediction_result.get("confidence", 0.0)
            )
            
            # Extract network information
            source_ip = None
            destination_ip = None
            protocol = None
            packet_size = None
            
            if network_data:
                source_ip = network_data.get("source_ip")
                destination_ip = network_data.get("destination_ip")
                packet_size = network_data.get("packet_size")
                
                # Determine protocol from network data
                if network_data.get("TCP", 0) == 1:
                    protocol = "TCP"
                elif network_data.get("UDP", 0) == 1:
                    protocol = "UDP"
                elif network_data.get("ICMP", 0) == 1:
                    protocol = "ICMP"
            
            # Create description
            attack_type = prediction_result.get("prediction", "unknown")
            confidence = prediction_result.get("confidence", 0.0)
            description = f"{attack_type.upper()} attack detected with {confidence:.1%} confidence"
            
            # Create alert
            alert = Alert(
                id=alert_id,
                timestamp=prediction_result.get("timestamp", datetime.utcnow().isoformat()),
                attack_type=attack_type,
                confidence=confidence,
                severity=severity,
                status=AlertStatus.ACTIVE,
                source_ip=source_ip,
                destination_ip=destination_ip,
                protocol=protocol,
                packet_size=packet_size,
                network_data=network_data,
                description=description
            )
            
            # Store alert
            self.alerts[alert_id] = alert
            self._save_alerts()
            
            logger.warning(f"🚨 ALERT CREATED: {alert_id} - {description}")
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {str(e)}")
            raise
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get a specific alert by ID."""
        return self.alerts.get(alert_id)
    
    def get_alerts(self, 
                   status: Optional[AlertStatus] = None,
                   severity: Optional[AlertSeverity] = None,
                   attack_type: Optional[str] = None,
                   limit: Optional[int] = None,
                   offset: int = 0) -> List[Alert]:
        """
        Get alerts with optional filtering.
        
        Args:
            status: Filter by alert status
            severity: Filter by severity level
            attack_type: Filter by attack type
            limit: Maximum number of alerts to return
            offset: Number of alerts to skip
            
        Returns:
            List[Alert]: Filtered list of alerts
        """
        alerts = list(self.alerts.values())
        
        # Apply filters
        if status:
            alerts = [a for a in alerts if a.status == status]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if attack_type:
            alerts = [a for a in alerts if a.attack_type.lower() == attack_type.lower()]
        
        # Sort by timestamp (newest first)
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply pagination
        if offset > 0:
            alerts = alerts[offset:]
        if limit:
            alerts = alerts[:limit]
        
        return alerts
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str, notes: Optional[str] = None) -> bool:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            acknowledged_by: Username or identifier of who acknowledged
            notes: Optional notes about the acknowledgment
            
        Returns:
            bool: True if successful, False if alert not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow().isoformat()
        if notes:
            alert.notes = notes
        
        self._save_alerts()
        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Mark an alert as resolved.
        
        Args:
            alert_id: ID of the alert to resolve
            notes: Optional resolution notes
            
        Returns:
            bool: True if successful, False if alert not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow().isoformat()
        if notes:
            alert.notes = notes
        
        self._save_alerts()
        logger.info(f"Alert {alert_id} resolved")
        return True
    
    def mark_false_positive(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """
        Mark an alert as a false positive.
        
        Args:
            alert_id: ID of the alert to mark
            notes: Optional notes about why it's a false positive
            
        Returns:
            bool: True if successful, False if alert not found
        """
        alert = self.alerts.get(alert_id)
        if not alert:
            return False
        
        alert.status = AlertStatus.FALSE_POSITIVE
        if notes:
            alert.notes = notes
        
        self._save_alerts()
        logger.info(f"Alert {alert_id} marked as false positive")
        return True
    
    def get_alert_statistics(self, days: int = 7) -> Dict[str, Any]:
        """
        Get alert statistics for the specified number of days.
        
        Args:
            days: Number of days to include in statistics
            
        Returns:
            Dict: Alert statistics
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        cutoff_str = cutoff_date.isoformat()
        
        # Filter alerts within the time range
        recent_alerts = [
            alert for alert in self.alerts.values()
            if alert.timestamp >= cutoff_str
        ]
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        active_alerts = len([a for a in recent_alerts if a.status == AlertStatus.ACTIVE])
        acknowledged_alerts = len([a for a in recent_alerts if a.status == AlertStatus.ACKNOWLEDGED])
        resolved_alerts = len([a for a in recent_alerts if a.status == AlertStatus.RESOLVED])
        false_positives = len([a for a in recent_alerts if a.status == AlertStatus.FALSE_POSITIVE])
        
        # Attack type distribution
        attack_types = {}
        for alert in recent_alerts:
            attack_type = alert.attack_type
            attack_types[attack_type] = attack_types.get(attack_type, 0) + 1
        
        # Severity distribution
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Average confidence
        if recent_alerts:
            avg_confidence = sum(alert.confidence for alert in recent_alerts) / len(recent_alerts)
        else:
            avg_confidence = 0.0
        
        return {
            "period_days": days,
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "acknowledged_alerts": acknowledged_alerts,
            "resolved_alerts": resolved_alerts,
            "false_positives": false_positives,
            "attack_types": attack_types,
            "severity_distribution": severity_counts,
            "average_confidence": round(avg_confidence, 3),
            "false_positive_rate": round(false_positives / total_alerts * 100, 1) if total_alerts > 0 else 0.0
        }
    
    def _determine_severity(self, attack_type: str, confidence: float) -> AlertSeverity:
        """Determine alert severity based on attack type and confidence."""
        
        # Critical attacks
        critical_attacks = ["ddos", "dos", "malware", "ransomware"]
        if attack_type.lower() in critical_attacks:
            if confidence >= 0.9:
                return AlertSeverity.CRITICAL
            elif confidence >= 0.8:
                return AlertSeverity.HIGH
            else:
                return AlertSeverity.MEDIUM
        
        # High severity attacks
        high_attacks = ["intrusion", "injection", "privilege_escalation"]
        if attack_type.lower() in high_attacks:
            if confidence >= 0.8:
                return AlertSeverity.HIGH
            elif confidence >= 0.7:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
        
        # Medium severity attacks
        medium_attacks = ["reconnaissance", "scanning", "probe"]
        if attack_type.lower() in medium_attacks:
            if confidence >= 0.8:
                return AlertSeverity.MEDIUM
            else:
                return AlertSeverity.LOW
        
        # Default based on confidence
        if confidence >= 0.9:
            return AlertSeverity.HIGH
        elif confidence >= 0.7:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW
    
    def _load_alerts(self):
        """Load alerts from storage."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    
                    # Load alerts
                    for alert_data in data.get('alerts', []):
                        alert = Alert(**alert_data)
                        # Convert string enums back to enum objects
                        alert.severity = AlertSeverity(alert.severity)
                        alert.status = AlertStatus(alert.status)
                        self.alerts[alert.id] = alert
                    
                    # Load counter
                    self.alert_counter = data.get('counter', 0)
                    
                logger.info(f"Loaded {len(self.alerts)} alerts from storage")
        except Exception as e:
            logger.error(f"Failed to load alerts: {str(e)}")
    
    def _save_alerts(self):
        """Save alerts to storage."""
        try:
            # Convert alerts to dict format
            alerts_data = []
            for alert in self.alerts.values():
                alert_dict = asdict(alert)
                # Convert enums to strings for JSON serialization
                alert_dict['severity'] = alert.severity.value
                alert_dict['status'] = alert.status.value
                alerts_data.append(alert_dict)
            
            data = {
                'alerts': alerts_data,
                'counter': self.alert_counter,
                'last_updated': datetime.utcnow().isoformat()
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save alerts: {str(e)}")

# Global service instance
_alert_service: Optional[AlertService] = None

def get_alert_service() -> AlertService:
    """Get the global alert service instance."""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service

def create_alert_from_prediction(prediction_result: Dict[str, Any], network_data: Dict[str, Any] = None) -> Optional[Alert]:
    """
    Convenience function to create an alert from a prediction result.
    
    Args:
        prediction_result: The prediction result
        network_data: Original network data
        
    Returns:
        Alert: Created alert or None if creation failed
    """
    try:
        # Only create alert if it's actually an attack and above threshold
        if not prediction_result.get("is_attack", False):
            return None
            
        if not prediction_result.get("alert_generated", False):
            return None
        
        service = get_alert_service()
        return service.create_alert(prediction_result, network_data)
        
    except Exception as e:
        logger.error(f"Failed to create alert from prediction: {str(e)}")
        return None


if __name__ == "__main__":
    # Test the alert service
    print("Testing Alert Service...")
    
    service = AlertService("test_alerts.json")
    
    # Test creating an alert
    test_prediction = {
        "prediction": "ddos",
        "is_attack": True,
        "confidence": 0.95,
        "timestamp": datetime.utcnow().isoformat(),
        "alert_generated": True
    }
    
    test_network_data = {
        "source_ip": "192.168.1.100",
        "destination_ip": "10.0.0.1",
        "TCP": 1,
        "packet_size": 1500,
        "Duration": 5.0,
        "Rate": 100000.0
    }
    
    alert = service.create_alert(test_prediction, test_network_data)
    print(f"✅ Created alert: {alert.id}")
    
    # Test getting alerts
    alerts = service.get_alerts(limit=5)
    print(f"✅ Retrieved {len(alerts)} alerts")
    
    # Test statistics
    stats = service.get_alert_statistics()
    print(f"✅ Alert statistics: {stats}")
    
    # Test acknowledgment
    success = service.acknowledge_alert(alert.id, "test_user", "Testing acknowledgment")
    print(f"✅ Alert acknowledged: {success}")
    
    print("Alert service test completed!")