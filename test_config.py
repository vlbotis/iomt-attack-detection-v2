# test_config.py - Script to test and validate configuration
import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'tests' else Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_configuration():
    """Test the configuration setup."""
    print("🔧 Testing IoMT Attack Detection Configuration")
    print("=" * 50)
    
    try:
        # Import settings
        from app.core.config import settings, setup_logging
        
        print("✅ Configuration loaded successfully")
        print(f"Environment: {settings.ENVIRONMENT}")
        print(f"Debug Mode: {settings.DEBUG}")
        print(f"Project: {settings.PROJECT_NAME} v{settings.VERSION}")
        
        # Test directory creation
        print("\n📁 Testing directory creation:")
        print(f"Models directory: {settings.models_dir} - {'✅ Created' if settings.models_dir.exists() else '❌ Failed'}")
        print(f"Logs directory: {settings.logs_dir} - {'✅ Created' if settings.logs_dir.exists() else '❌ Failed'}")
        
        # Test validation
        print("\n🔍 Testing validation:")
        print(f"Alert threshold: {settings.ALERT_THRESHOLD} - {'✅ Valid' if 0 <= settings.ALERT_THRESHOLD <= 1 else '❌ Invalid'}")
        
        # Test logging setup
        print("\n📝 Testing logging setup:")
        try:
            logger = setup_logging()
            logger.info("Test log message")
            print("✅ Logging configured successfully")
        except Exception as e:
            print(f"❌ Logging setup failed: {e}")
        
        # Display configuration (safe)
        print("\n⚙️  Current Configuration:")
        print("-" * 30)
        config_items = [
            ("Environment", settings.ENVIRONMENT),
            ("Debug", settings.DEBUG),
            ("Host:Port", f"{settings.HOST}:{settings.PORT}"),
            ("API Prefix", settings.API_V1_STR),
            ("Log Level", settings.LOG_LEVEL),
            ("Models Directory", settings.models_dir),
            ("Mock ML Models", settings.MOCK_ML_MODELS),
            ("Alert Threshold", settings.ALERT_THRESHOLD),
            ("Max Dashboard Points", settings.MAX_DASHBOARD_DATAPOINTS),
        ]
        
        for item, value in config_items:
            print(f"{item:<20}: {value}")
        
        print("\n🧪 Testing property methods:")
        print(f"Is Development: {settings.is_development}")
        print(f"Is Production: {settings.is_production}")
        
        # Test environment variables
        print("\n🌍 Environment Variables Check:")
        print("Direct os.getenv() vs Pydantic Settings:")
        env_vars = [
            ("ENVIRONMENT", settings.ENVIRONMENT),
            ("DEBUG", settings.DEBUG), 
            ("SECRET_KEY", "***HIDDEN***" if settings.SECRET_KEY else "Not Set"),
            ("LOG_LEVEL", settings.LOG_LEVEL),
            ("MODEL_PATH", settings.MODEL_PATH),
            ("ALERT_THRESHOLD", settings.ALERT_THRESHOLD),
            ("MOCK_ML_MODELS", settings.MOCK_ML_MODELS)
        ]
        
        for var_name, pydantic_value in env_vars:
            os_value = os.getenv(var_name, "Not Set")
            if "SECRET" in var_name:
                os_value = "***HIDDEN***" if os_value != "Not Set" else "Not Set"
            print(f"{var_name:<25}: os.getenv='{os_value}' | pydantic='{pydantic_value}'")
        
        # Check if .env file is being read
        env_file_path = Path(".env")
        if env_file_path.exists():
            print(f"\n📄 .env file found at: {env_file_path.absolute()}")
            with open(env_file_path) as f:
                lines = f.readlines()
            print(f"📝 .env file has {len(lines)} lines")
            print("First few lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  {i+1}: {line.rstrip()}")
        else:
            print("\n❌ .env file not found!")
        
        print("\n🎉 Configuration test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("pip install pydantic python-dotenv")
        return False
        
    except Exception as e:
        print(f"❌ Configuration Error: {e}")
        return False

def create_sample_env():
    """Create a sample .env file if it doesn't exist."""
    # Create .env in project root, not in tests directory
    project_root = Path(__file__).parent.parent if Path(__file__).parent.name == 'tests' else Path(__file__).parent
    env_file = project_root / ".env"
    
    if not env_file.exists():
        print("📝 Creating sample .env file...")
        env_content = """# Development Environment Configuration
ENVIRONMENT=development
DEBUG=true
SECRET_KEY=dev-secret-key-change-in-production-12345
LOG_LEVEL=DEBUG
MOCK_ML_MODELS=true
ALERT_THRESHOLD=0.7
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print(f"✅ Sample .env file created at {env_file}")
    else:
        print(f"✅ .env file already exists at {env_file}")

if __name__ == "__main__":
    print("IoMT Attack Detection - Configuration Test")
    print("=" * 50)
    
    # Create sample .env if needed
    create_sample_env()
    
    # Test configuration
    success = test_configuration()
    
    if success:
        print("\n🚀 Ready for development!")
        print("\nNext steps:")
        print("1. Review and modify .env file as needed")
        print("2. Install dependencies: pip install pydantic python-dotenv fastapi")
        print("3. Start building your FastAPI application")
    else:
        print("\n❌ Configuration test failed. Please check the errors above.")
        sys.exit(1)