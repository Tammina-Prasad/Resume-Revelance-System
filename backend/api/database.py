"""
Database configuration and session management.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator

from ..models.database import Base

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/resume_system.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False  # Set to True for SQL debugging
    )
else:
    engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency that provides a database session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_database():
    """Initialize the database with default data."""
    create_tables()
    
    # Add any default configuration or seed data here
    db = SessionLocal()
    try:
        from ..models.database import SystemConfiguration
        
        # Check if default configuration exists
        existing_config = db.query(SystemConfiguration).filter(
            SystemConfiguration.config_name == "default_analysis_config"
        ).first()
        
        if not existing_config:
            default_config = SystemConfiguration(
                config_name="default_analysis_config",
                config_value={
                    "hard_match_weight": 0.4,
                    "semantic_match_weight": 0.6,
                    "high_threshold": 75,
                    "medium_threshold": 50,
                    "fuzzy_threshold": 80,
                    "use_local_embeddings": False
                },
                description="Default configuration for resume relevance analysis"
            )
            db.add(default_config)
            db.commit()
            
    except Exception as e:
        print(f"Error initializing database: {e}")
        db.rollback()
    finally:
        db.close()