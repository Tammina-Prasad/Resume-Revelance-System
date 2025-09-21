#!/usr/bin/env python3
"""
Database initialization script for the Resume Relevance Analysis System
"""

import sys
import os
from pathlib import Path

# Add the current directory and parent to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models.database import Base

def init_database():
    """Initialize the database with all required tables"""
    
    # Create data directory if it doesn't exist
    data_dir = current_dir.parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    # Database file path
    db_path = data_dir / "resume_system.db"
    
    # Create database engine
    database_url = f"sqlite:///{db_path}"
    engine = create_engine(database_url, echo=True)
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    
    print(f"✅ Database initialized successfully at: {db_path}")
    print("📊 Created tables:")
    for table_name in Base.metadata.tables.keys():
        print(f"   • {table_name}")
    
    return engine

if __name__ == "__main__":
    try:
        init_database()
        print("\n🎉 Database setup complete!")
    except Exception as e:
        print(f"❌ Error initializing database: {str(e)}")
        import traceback
        traceback.print_exc()