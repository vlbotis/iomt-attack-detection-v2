# app/models/loader.py
"""
Model Loading Utilities for IoMT Attack Detection System
Phase 3: Model Integration - Step 3
"""

import logging
import pickle
import joblib
from pathlib import Path
from typing import Optional, Dict, Any
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelLoader:
    """Handles loading and managing ML models for the IoMT detection system."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_paths = {
            'lightgbm': settings.MODEL_PATH,
            'feature_processor': settings.FEATURE_PROCESSOR_PATH,
            'scaler': settings.SCALER_PATH
        }
        self.loaded = False
    
    def load_models(self) -> bool:
        """
        Load all required models for the IoMT detection system.
        
        Returns:
            bool: True if models loaded successfully, False otherwise
        """
        try:
            if settings.MOCK_ML_MODELS:
                logger.info("Loading mock models for development")
                self._load_mock_models()
                self.loaded = True
                return True
            
            logger.info("Loading production models")
            
            # Load each model
            for model_name, model_path in self.model_paths.items():
                if not self._load_single_model(model_name, model_path):
                    return False
            
            # Import and initialize optimized system if available
            try:
                # Add ml directory to Python path
                import sys
                import os
                ml_path = Path(__file__).parent.parent.parent / "ml"
                ml_path_str = str(ml_path.absolute())
                
                if ml_path_str not in sys.path:
                    sys.path.insert(0, ml_path_str)
                
                # Change to ml directory for imports to work
                old_cwd = os.getcwd()
                os.chdir(ml_path)
                
                try:
                    # Import your optimized system
                    from ml import optimized_system
                    
                    # Use the correct class name from your file
                    if hasattr(optimized_system, 'OptimizedIoMTClassificationSystem'):
                        self.detection_system = optimized_system.OptimizedIoMTClassificationSystem()
                        logger.info("✅ OptimizedIoMTClassificationSystem initialized")
                    else:
                        # List available classes for debugging
                        classes = [attr for attr in dir(optimized_system) 
                                 if not attr.startswith('_') and callable(getattr(optimized_system, attr))]
                        logger.info(f"Available classes in optimized_system: {classes}")
                        self.detection_system = None
                        
                finally:
                    # Restore original directory
                    os.chdir(old_cwd)
                    
            except ImportError as e:
                logger.warning(f"Could not import optimized system: {e}")
                self.detection_system = None
            except Exception as e:
                logger.error(f"Error initializing detection system: {e}")
                self.detection_system = None
            
            self.loaded = True
            logger.info("✅ All models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load models: {str(e)}")
            self.loaded = False
            return False
    
    def _load_single_model(self, model_name: str, model_path: str) -> bool:
        """Load a single model from file."""
        try:
            path = Path(model_path)
            
            if not path.exists():
                logger.error(f"❌ Model file not found: {model_path}")
                return False
            
            # Determine file type and load accordingly
            if path.suffix == '.pkl':
                with open(path, 'rb') as f:
                    model = pickle.load(f)
            elif path.suffix == '.joblib':
                model = joblib.load(path)
            else:
                logger.error(f"❌ Unsupported model file format: {path.suffix}")
                return False
            
            self.models[model_name] = model
            logger.info(f"✅ Loaded {model_name} from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {str(e)}")
            return False
    
    def _load_mock_models(self):
        """Load mock models for development and testing."""
        # Create mock models that simulate the real ones
        self.models = {
            'lightgbm': MockLightGBMModel(),
            'feature_processor': MockFeatureProcessor(),
            'scaler': MockScaler()
        }
        
        # Mock detection system
        self.detection_system = MockDetectionSystem()
        logger.info("✅ Mock models loaded for development")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name."""
        if not self.loaded:
            logger.warning("Models not loaded yet. Call load_models() first.")
            return None
        
        return self.models.get(model_name)
    
    def get_detection_system(self) -> Optional[Any]:
        """Get the main detection system."""
        if not self.loaded:
            logger.warning("Models not loaded yet. Call load_models() first.")
            return None
        
        return getattr(self, 'detection_system', None)
    
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self.loaded
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models."""
        info = {
            'loaded': self.loaded,
            'mock_mode': settings.MOCK_ML_MODELS,
            'models': {}
        }
        
        for model_name in self.models:
            model = self.models[model_name]
            info['models'][model_name] = {
                'type': type(model).__name__,
                'available': model is not None
            }
        
        return info


# Mock classes for development
class MockLightGBMModel:
    """Mock LightGBM model for development."""
    
    def predict(self, X):
        """Mock prediction."""
        import numpy as np
        # Return mock probabilities
        return np.random.random(len(X) if hasattr(X, '__len__') else 1)
    
    def predict_proba(self, X):
        """Mock probability prediction."""
        import numpy as np
        n_samples = len(X) if hasattr(X, '__len__') else 1
        # Return mock probabilities for binary classification
        probs = np.random.random((n_samples, 2))
        probs = probs / probs.sum(axis=1, keepdims=True)  # Normalize
        return probs


class MockFeatureProcessor:
    """Mock feature processor for development."""
    
    def transform(self, X):
        """Mock feature transformation."""
        # Just return the input (no actual processing)
        return X
    
    def fit_transform(self, X):
        """Mock fit and transform."""
        return self.transform(X)


class MockScaler:
    """Mock scaler for development."""
    
    def transform(self, X):
        """Mock scaling."""
        return X
    
    def fit_transform(self, X):
        """Mock fit and transform."""
        return X


class MockDetectionSystem:
    """Mock detection system for development."""
    
    def predict_attack(self, network_data):
        """Mock attack prediction."""
        import random
        return {
            'is_attack': random.choice([True, False]),
            'confidence': random.uniform(0.5, 0.95),
            'attack_type': random.choice(['DDoS', 'Malware', 'Intrusion', 'Normal']),
            'timestamp': '2025-05-24T18:45:21Z'
        }


# Global model loader instance
model_loader = ModelLoader()


def get_model_loader() -> ModelLoader:
    """Get the global model loader instance."""
    return model_loader


def initialize_models() -> bool:
    """Initialize all models. Call this on application startup."""
    logger.info("Initializing ML models...")
    return model_loader.load_models()


if __name__ == "__main__":
    # Test the model loader
    print("Testing Model Loader...")
    
    success = initialize_models()
    if success:
        print("✅ Models loaded successfully!")
        print("Model info:", model_loader.get_model_info())
        
        # Test getting a model
        lgb_model = model_loader.get_model('lightgbm')
        if lgb_model:
            print("✅ LightGBM model retrieved successfully")
    else:
        print("❌ Failed to load models")