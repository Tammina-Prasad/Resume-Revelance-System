"""
Main FastAPI application.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import API routers
from api.database import init_database
from api import job_descriptions, resumes, analyses, dashboard

# Create FastAPI application
app = FastAPI(
    title="Resume Relevance Analysis System",
    description="AI-powered system for analyzing resume relevance against job descriptions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(job_descriptions.router, prefix="/api/v1")
app.include_router(resumes.router, prefix="/api/v1")
app.include_router(analyses.router, prefix="/api/v1")
app.include_router(dashboard.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    # Initialize database
    init_database()
    
    # Ensure upload directory exists
    upload_folder = os.getenv("UPLOAD_FOLDER", "uploads")
    os.makedirs(upload_folder, exist_ok=True)
    
    print("Resume Relevance Analysis System started successfully!")


@app.get("/")
async def root():
    """Root endpoint with system information."""
    return {
        "message": "Resume Relevance Analysis System API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs",
        "endpoints": {
            "job_descriptions": "/api/v1/job-descriptions",
            "resumes": "/api/v1/resumes",
            "analyses": "/api/v1/analyses",
            "dashboard": "/api/v1/dashboard"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": "2024-01-01T00:00:00Z"}


@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "detail": str(exc)}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": "An unexpected error occurred"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )