"""
Resume API endpoints.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from sqlalchemy.orm import Session

from ..api.database import get_db
from ..models import (
    Resume, ResumeUpload, ResumeResponse, ResumeDetails,
    get_candidate_analysis_history
)
from ..core import ResumeParser, extract_contact_info, extract_education_info

router = APIRouter(prefix="/resumes", tags=["Resumes"])

# Configuration
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10485760))  # 10MB default


@router.post("/upload", response_model=ResumeResponse, status_code=status.HTTP_201_CREATED)
async def upload_resume(
    file: UploadFile = File(...),
    candidate_name: Optional[str] = Form(None),
    candidate_email: Optional[str] = Form(None)
):
    """
    Upload and process a resume file.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Check file extension
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ['.pdf', '.docx']:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF and DOCX files are supported"
        )
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {MAX_FILE_SIZE} bytes"
        )
    
    # Reset file pointer
    await file.seek(0)
    
    try:
        # Ensure upload directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Generate unique filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse resume
        parser = ResumeParser()
        parsed_data = parser.parse_resume(file_path)
        
        # Extract additional information
        contact_info = extract_contact_info(parsed_data['cleaned_text'])
        education_info = extract_education_info(parsed_data['cleaned_text'])
        
        # Create database record
        db = next(get_db())
        try:
            db_resume = Resume(
                candidate_name=candidate_name or contact_info.get('name'),
                candidate_email=candidate_email or contact_info.get('email'),
                file_name=file.filename,
                file_path=file_path,
                file_type=file_extension[1:],  # Remove the dot
                parsed_data=parsed_data,
                raw_text=parsed_data['raw_text'],
                cleaned_text=parsed_data['cleaned_text'],
                word_count=parsed_data['word_count'],
                char_count=parsed_data['char_count'],
                contact_info=contact_info,
                is_processed=True,
                processed_at=datetime.now()
            )
            
            db.add(db_resume)
            db.commit()
            db.refresh(db_resume)
            
            return db_resume
            
        except Exception as e:
            db.rollback()
            # Clean up file if database operation fails
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e
        finally:
            db.close()
            
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing resume: {str(e)}"
        )


@router.get("/", response_model=List[ResumeResponse])
async def list_resumes(
    skip: int = 0,
    limit: int = 100,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all resumes with basic information.
    """
    query = db.query(Resume)
    
    if search:
        query = query.filter(
            Resume.candidate_name.contains(search) |
            Resume.candidate_email.contains(search) |
            Resume.file_name.contains(search)
        )
    
    resumes = query.offset(skip).limit(limit).all()
    return resumes


@router.get("/{resume_id}", response_model=ResumeDetails)
async def get_resume(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed information about a specific resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    return resume


@router.put("/{resume_id}", response_model=ResumeResponse)
async def update_resume(
    resume_id: int,
    candidate_name: Optional[str] = None,
    candidate_email: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Update resume metadata.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    try:
        if candidate_name is not None:
            resume.candidate_name = candidate_name
        if candidate_email is not None:
            resume.candidate_email = candidate_email
        
        db.commit()
        db.refresh(resume)
        
        return resume
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating resume: {str(e)}"
        )


@router.delete("/{resume_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_resume(
    resume_id: int,
    delete_file: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete a resume record and optionally the file.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    try:
        file_path = resume.file_path
        
        # Delete database record
        db.delete(resume)
        db.commit()
        
        # Delete file if requested
        if delete_file and file_path and os.path.exists(file_path):
            os.remove(file_path)
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting resume: {str(e)}"
        )


@router.get("/{resume_id}/text")
async def get_resume_text(
    resume_id: int,
    text_type: str = "cleaned",  # raw, cleaned
    db: Session = Depends(get_db)
):
    """
    Get the text content of a resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    if text_type == "raw":
        text = resume.raw_text
    elif text_type == "cleaned":
        text = resume.cleaned_text
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid text_type. Use 'raw' or 'cleaned'"
        )
    
    return {"text": text, "type": text_type}


@router.get("/{resume_id}/parsed")
async def get_parsed_resume(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the parsed data for a resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    return resume.parsed_data or {}


@router.post("/{resume_id}/reprocess", response_model=ResumeResponse)
async def reprocess_resume(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Re-process a resume to update the parsed data.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    if not os.path.exists(resume.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume file not found on disk"
        )
    
    try:
        # Re-parse resume
        parser = ResumeParser()
        parsed_data = parser.parse_resume(resume.file_path)
        
        # Extract additional information
        contact_info = extract_contact_info(parsed_data['cleaned_text'])
        
        # Update database record
        resume.parsed_data = parsed_data
        resume.raw_text = parsed_data['raw_text']
        resume.cleaned_text = parsed_data['cleaned_text']
        resume.word_count = parsed_data['word_count']
        resume.char_count = parsed_data['char_count']
        resume.contact_info = contact_info
        resume.is_processed = True
        resume.processed_at = datetime.now()
        
        db.commit()
        db.refresh(resume)
        
        return resume
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reprocessing resume: {str(e)}"
        )


@router.get("/{resume_id}/analyses")
async def get_resume_analyses(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Get all analyses for a specific resume.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    analyses = get_candidate_analysis_history(resume_id, db)
    
    result = []
    for analysis in analyses:
        result.append({
            "analysis_id": analysis.id,
            "job_id": analysis.job_description_id,
            "job_title": analysis.job_description.title if analysis.job_description else None,
            "company_name": analysis.job_description.company_name if analysis.job_description else None,
            "final_score": analysis.final_score,
            "verdict": analysis.verdict,
            "created_at": analysis.created_at
        })
    
    return result


@router.get("/{resume_id}/download")
async def download_resume(
    resume_id: int,
    db: Session = Depends(get_db)
):
    """
    Download the original resume file.
    """
    resume = db.query(Resume).filter(Resume.id == resume_id).first()
    
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    if not os.path.exists(resume.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume file not found on disk"
        )
    
    from fastapi.responses import FileResponse
    
    return FileResponse(
        path=resume.file_path,
        filename=resume.file_name,
        media_type='application/octet-stream'
    )