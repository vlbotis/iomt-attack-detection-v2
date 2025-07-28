# app/services/prediction.py
# tested using python -m app.services.prediction
# Business logic! / different to app/api/prediction.py (REST endpoints)
"""
Prediction Service for IoMT Attack Detection System
Phase 4: Core Business Logic - Step 5
"""
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np

from app.models.loader import get_model_loader
from app.core.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    """Service for handling IoMT attack predictions."""

    def __init__(self):
        """Initialize the PredictionService."""
        self.model_loader = get_model_loader()
        self.initialized = False
        self._initialize()

    def _initialize(self):
        """Initialize models if not already loaded."""
        if not self.model_loader.is_loaded():
            logger.info("Initializing models for prediction service...")
            success = self.model_loader.load_models()
            if success:
                self.initialized = True
                logger.info("✅ Prediction service initialized successfully")
            else:
                logger.error("❌ Failed to initialize prediction service")
                self.initialized = False
        else:
            self.initialized = True
            logger.info("✅ Using already loaded models")
    
    def predict_single(self, network_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict attack type for a single network traffic sample.
        
        Args:
            network_data (Dict): Network traffic features
            
        Returns:
            Dict: Prediction result with confidence and metadata
        """
        start_time = time.time()

        try:
            if not self.initialized:
                raise RuntimeError("Prediction service not properly initialized")
            
            # Validate input
            self._validate_network_data(network_data)

            # Get the detection system
            detection_system = self.model_loader.get_detection_system()

            if detection_system:
                # Use the full detection system if available
                result = detection_system.predict_attack(network_data)

                # Standardize the response format
                prediction_result = {
                    "prediction": result.get("attack_type", "unknown"),
                    "is_attack": result.get("is_attack", False),
                    "confidence": float(result.get("confidence", 0.0)),
                    "timestamp": datetime.utcnow().isoformat(),
                    "processing_time_ms": (time.time() - start_time) * 1000,
                    "model_version": "optimized_system",
                    "status": "success"
                }
            else:
                # Fallback to individual model prediction
                prediction_result = self._predict_with_individual_models(network_data, start_time)

        # Generate alert if needed
            if prediction_result['is_attack'] and prediction_result['confidence'] >= settings.ALERT_THRESHOLD:
                prediction_result['alert_generated'] = True
                logger.warning(f"🚨 ATTACK DETECTED: {prediction_result['prediction']} "f"(confidence: {prediction_result['confidence']:.2%})")
            else:
                    prediction_result['alert_generated'] = False
            
            logger.info(f"Prediction completed: {prediction_result['prediction']} " f"(confidence: {prediction_result['confidence']:.2%}, "f"time: {prediction_result['processing_time_ms']:.1f}ms)")

            return prediction_result
    
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                "prediction": "error",
                "is_attack": False,
                "confidence": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "status": "error",
                "error_message": str(e)
            }
        
    def _predict_with_individual_models(self, network_data: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Fallback prediction using individual models."""
        
        # Convert network data to features (simplified for mock)
        features = self._extract_features(network_data)
        
        # Get models
        lgb_model = self.model_loader.get_model('lightgbm')
        feature_processor = self.model_loader.get_model('feature_processor')
        scaler = self.model_loader.get_model('scaler')
        
        if not all([lgb_model, feature_processor, scaler]):
            raise RuntimeError("Required models not available")
        
        # Process features
        processed_features = feature_processor.transform(features)
        scaled_features = scaler.transform(processed_features)
        
        # Make prediction
        if hasattr(lgb_model, 'predict_proba'):
            probabilities = lgb_model.predict_proba(scaled_features)
            confidence = float(np.max(probabilities))
            prediction_idx = np.argmax(probabilities)
        else:
            prediction = lgb_model.predict(scaled_features)
            confidence = float(prediction[0]) if hasattr(prediction, '__iter__') else float(prediction)
            prediction_idx = 1 if confidence > 0.5 else 0
        
        # Map prediction to attack type (simplified)
        attack_types = ["benign", "ddos", "dos", "malware", "intrusion"]
        if prediction_idx < len(attack_types):
            predicted_attack = attack_types[prediction_idx]
        else:
            predicted_attack = "unknown"
        
        is_attack = predicted_attack != "benign"
        
        return {
            "prediction": predicted_attack,
            "is_attack": is_attack,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "processing_time_ms": (time.time() - start_time) * 1000,
            "model_version": "lightgbm_individual",
            "status": "success"
        }
    
    def predict_batch(self, network_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict attack types for a batch of network traffic samples.
        
        Args:
            network_data_list (List[Dict]): List of network traffic features
            
        Returns:
            List[Dict]: List of prediction results
        """
        start_time = time.time()
        
        try:
            if not self.initialized:
                raise RuntimeError("Prediction service not properly initialized")
            
            logger.info(f"Processing batch of {len(network_data_list)} samples")
            
            results = []
            for i, network_data in enumerate(network_data_list):
                try:
                    result = self.predict_single(network_data)
                    result["batch_index"] = i
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process sample {i}: {str(e)}")
                    results.append({
                        "batch_index": i,
                        "prediction": "error",
                        "is_attack": False,
                        "confidence": 0.0,
                        "status": "error",
                        "error_message": str(e)
                    })
            
            # Summary statistics
            total_time = time.time() - start_time
            successful_predictions = sum(1 for r in results if r.get("status") == "success")
            attacks_detected = sum(1 for r in results if r.get("is_attack", False))
            
            logger.info(f"Batch processing complete: {successful_predictions}/{len(network_data_list)} "
                       f"successful, {attacks_detected} attacks detected, "
                       f"total time: {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            return [{"status": "error", "error_message": str(e)}]
        

    def _validate_network_data(self, network_data: Dict[str, Any]) -> None:
        """Validate input network data format."""
        if not isinstance(network_data, dict):
            raise ValueError("Network data must be a dictionary")
        
        if not network_data:
            raise ValueError("Network data cannot be empty")
        
        # Add more specific validation as needed
        # For now, just ensure we have some numeric-like values
        numeric_fields = 0
        for key, value in network_data.items():
            if isinstance(value, (int, float)):
                numeric_fields += 1
        
        if numeric_fields == 0:
            logger.warning("No numeric fields found in network data, using defaults")

    def _extract_features(self, network_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from network data for model input."""
        
        # Expected features based on your ML system
        expected_features = [
            'Header-Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'TCP', 'UDP', 'HTTP', 'HTTPS'
        ]
        
        # Extract features with defaults
        features = []
        for feature in expected_features:
            value = network_data.get(feature, 0.0)
            # Ensure numeric value
            try:
                features.append(float(value))
            except (ValueError, TypeError):
                features.append(0.0)
        
        # If we don't have enough features, pad with zeros
        while len(features) < 20:  # Minimum feature count
            features.append(0.0)
        
        return np.array([features])  # Return as 2D array for sklearn
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get current service status and statistics."""
        return {
            "initialized": self.initialized,
            "models_loaded": self.model_loader.is_loaded(),
            "mock_mode": settings.MOCK_ML_MODELS,
            "alert_threshold": settings.ALERT_THRESHOLD,
            "model_info": self.model_loader.get_model_info() if self.initialized else None,
            "timestamp": datetime.utcnow().isoformat()
        }

# Global service instance
_prediction_service: Optional[PredictionService] = None

def get_prediction_service() -> PredictionService:
    """Get the global prediction service instance."""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service

def initialize_prediction_service() -> bool:
    """Initialize the prediction service. Call this on application startup."""
    try:
        service = get_prediction_service()
        return service.initialized
    except Exception as e:
        logger.error(f"Failed to initialize prediction service: {e}")
        return False
    

if __name__ == "__main__":
    # Test the prediction service
    print("Testing Prediction Service...")
    
    service = get_prediction_service()
    
    if service.initialized:
        print("✅ Service initialized successfully")
        
        # Test single prediction
        test_data = {
            "Duration": 0.5,
            "Rate": 1000.0,
            "TCP": 1,
            "UDP": 0,
            "HTTP": 1,
            "packet_size": 1500
        }
        
        result = service.predict_single(test_data)
        print(f"✅ Single prediction: {result}")
        
        # Test batch prediction
        batch_data = [test_data, test_data.copy()]
        batch_results = service.predict_batch(batch_data)
        print(f"✅ Batch prediction: {len(batch_results)} results")
        
        # Test service status
        status = service.get_service_status()
        print(f"✅ Service status: {status}")
        
    else:
        print("❌ Service initialization failed")