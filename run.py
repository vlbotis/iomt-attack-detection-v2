#!/usr/bin/env python3
"""
IoMT Attack Detection System - Application Entry Point
Run the FastAPI application with proper configuration
"""

import uvicorn
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import configuration
from app.core.config import settings

def main():
    """Main entry point for the IoMT Attack Detection System."""
    print(f"🚀 Starting {settings.PROJECT_NAME} v{settings.VERSION}")
    print(f"📊 Environment: {settings.ENVIRONMENT}")
    print(f"🌐 Server: http://{settings.HOST}:{settings.PORT}")
    print(f"📚 API Docs: http://{settings.HOST}:{settings.PORT}/docs" if settings.DEBUG else "📚 API Docs: Disabled (Production)")
    print(f"❤️  Health Check: http://{settings.HOST}:{settings.PORT}/health")
    print("-" * 50)
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )

if __name__ == "__main__":
    main()