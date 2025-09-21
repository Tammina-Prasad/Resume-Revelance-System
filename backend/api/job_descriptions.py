"""
Job Description API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from ..api.database import get_db
from ..models import (
    JobDescription, JobDescriptionCreate, JobDescriptionUpdate, 
    JobDescriptionResponse, JobDescriptionSummary,
    get_analysis_summary_by_job
)
from ..core import JobDescriptionParser

router = APIRouter(prefix="/job-descriptions", tags=["Job Descriptions"])


@router.post("/", response_model=JobDescriptionResponse, status_code=status.HTTP_201_CREATED)
async def create_job_description(
    jd_data: JobDescriptionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new job description and parse its content.
    """
    try:
        # Parse the job description
        parser = JobDescriptionParser()
        parsed_data = parser.parse_job_description(jd_data.description_text)
        
        # Create database record
        db_jd = JobDescription(
            title=jd_data.title,
            company_name=jd_data.company_name,
            description_text=jd_data.description_text,
            location=jd_data.location,
            experience_required=jd_data.experience_required,
            salary_range=jd_data.salary_range,
            created_by=jd_data.created_by,
            parsed_data=parsed_data
        )
        
        db.add(db_jd)
        db.commit()
        db.refresh(db_jd)
        
        return db_jd
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating job description: {str(e)}"
        )


@router.get("/", response_model=List[JobDescriptionSummary])
async def list_job_descriptions(
    skip: int = 0,
    limit: int = 100,
    include_inactive: bool = False,
    search: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    List all job descriptions with summary information.
    """
    query = db.query(JobDescription)
    
    if not include_inactive:
        query = query.filter(JobDescription.is_active == True)
    
    if search:
        query = query.filter(
            JobDescription.title.contains(search) |
            JobDescription.company_name.contains(search) |
            JobDescription.description_text.contains(search)
        )
    
    job_descriptions = query.offset(skip).limit(limit).all()
    
    # Add analysis summaries
    result = []
    for jd in job_descriptions:
        summary = get_analysis_summary_by_job(jd.id, db)
        result.append(JobDescriptionSummary(
            id=jd.id,
            title=jd.title,
            company_name=jd.company_name,
            location=jd.location,
            created_at=jd.created_at,
            analysis_count=summary["total"],
            avg_score=summary["avg_score"]
        ))
    
    return result


@router.get("/{jd_id}", response_model=JobDescriptionResponse)
async def get_job_description(
    jd_id: int,
    db: Session = Depends(get_db)
):
    """
    Get a specific job description by ID.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    return jd


@router.put("/{jd_id}", response_model=JobDescriptionResponse)
async def update_job_description(
    jd_id: int,
    jd_update: JobDescriptionUpdate,
    db: Session = Depends(get_db)
):
    """
    Update a job description.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    try:
        # Update fields
        for field, value in jd_update.dict(exclude_unset=True).items():
            setattr(jd, field, value)
        
        # Re-parse if description text was updated
        if jd_update.description_text:
            parser = JobDescriptionParser()
            parsed_data = parser.parse_job_description(jd_update.description_text)
            jd.parsed_data = parsed_data
        
        db.commit()
        db.refresh(jd)
        
        return jd
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating job description: {str(e)}"
        )


@router.delete("/{jd_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_job_description(
    jd_id: int,
    permanent: bool = False,
    db: Session = Depends(get_db)
):
    """
    Delete or deactivate a job description.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    try:
        if permanent:
            # Permanent deletion
            db.delete(jd)
        else:
            # Soft deletion (deactivation)
            jd.is_active = False
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting job description: {str(e)}"
        )


@router.get("/{jd_id}/parsed", response_model=dict)
async def get_parsed_job_description(
    jd_id: int,
    db: Session = Depends(get_db)
):
    """
    Get the parsed data for a job description.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    return jd.parsed_data or {}


@router.post("/{jd_id}/reparse", response_model=JobDescriptionResponse)
async def reparse_job_description(
    jd_id: int,
    db: Session = Depends(get_db)
):
    """
    Re-parse a job description to update the parsed data.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    try:
        # Re-parse the job description
        parser = JobDescriptionParser()
        parsed_data = parser.parse_job_description(jd.description_text)
        jd.parsed_data = parsed_data
        
        db.commit()
        db.refresh(jd)
        
        return jd
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error re-parsing job description: {str(e)}"
        )


@router.get("/{jd_id}/analytics")
async def get_job_analytics(
    jd_id: int,
    db: Session = Depends(get_db)
):
    """
    Get analytics for a specific job description.
    """
    jd = db.query(JobDescription).filter(JobDescription.id == jd_id).first()
    
    if not jd:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    summary = get_analysis_summary_by_job(jd_id, db)
    
    # Get top candidates
    from ..models.database import ResumeAnalysis, Resume
    top_candidates = db.query(ResumeAnalysis, Resume).join(Resume).filter(
        ResumeAnalysis.job_description_id == jd_id
    ).order_by(ResumeAnalysis.final_score.desc()).limit(10).all()
    
    top_candidates_data = []
    for analysis, resume in top_candidates:
        top_candidates_data.append({
            "analysis_id": analysis.id,
            "candidate_name": resume.candidate_name,
            "final_score": analysis.final_score,
            "verdict": analysis.verdict,
            "created_at": analysis.created_at
        })
    
    return {
        "job_id": jd_id,
        "job_title": jd.title,
        "company_name": jd.company_name,
        "summary": summary,
        "top_candidates": top_candidates_data
    }