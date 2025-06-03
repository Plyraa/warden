#!/usr/bin/env python3
"""
Warden Headless - Production Audio Analysis API
Simple startup script for the headless audio analysis service
"""

import uvicorn
from config import Config

if __name__ == "__main__":
    print("🚀 Starting Warden Headless API...")
    print(f"   Server: http://{Config.HOST}:{Config.PORT}")
    print(f"   Health: http://{Config.HOST}:{Config.PORT}/health")
    print(f"   Docs:   http://{Config.HOST}:{Config.PORT}/docs")
    print()
    
    uvicorn.run(
        "app:app",
        host=Config.HOST,
        port=Config.PORT,
        reload=False,
        log_level="info"
    )
