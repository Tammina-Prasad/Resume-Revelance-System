"""
Analysis API endpoints.
"""

import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session

from ..api.database import get_db
from ..models import (
    Resume, JobDescription, ResumeAnalysis,
    AnalysisRequest, AnalysisResponse, AnalysisDetails, AnalysisSummary,
    AnalysisFilter, BulkAnalysisRequest, BulkAnalysisResponse
)
from ..core import create_relevance_engine

router = APIRouter(prefix="/analyses", tags=["Analysis"])


@router.post("/", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    analysis_request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create a new resume relevance analysis.
    """
    # Validate resume exists
    resume = db.query(Resume).filter(Resume.id == analysis_request.resume_id).first()
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found"
        )
    
    # Validate job description exists
    job_description = db.query(JobDescription).filter(
        JobDescription.id == analysis_request.job_description_id
    ).first()
    if not job_description:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    # Check if analysis already exists
    existing_analysis = db.query(ResumeAnalysis).filter(
        ResumeAnalysis.resume_id == analysis_request.resume_id,
        ResumeAnalysis.job_description_id == analysis_request.job_description_id
    ).first()
    
    if existing_analysis:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis already exists for this resume-job combination"
        )
    
    try:
        # Create placeholder analysis record
        analysis = ResumeAnalysis(
            resume_id=analysis_request.resume_id,
            job_description_id=analysis_request.job_description_id,
            final_score=0.0,
            hard_match_score=0.0,
            semantic_match_score=0.0,
            verdict="Low",
            analysis_results={},
            analysis_config=analysis_request.config_overrides or {}
        )
        
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        
        # Schedule background processing
        background_tasks.add_task(
            process_analysis_background,
            analysis.id,
            resume.file_path,
            job_description.description_text,
            analysis_request.config_overrides
        )
        
        return analysis
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating analysis: {str(e)}"
        )


async def process_analysis_background(
    analysis_id: int,
    resume_file_path: str,
    job_description_text: str,
    config_overrides: Optional[dict] = None
):
    """
    Background task to process the analysis.
    """
    db = next(get_db())
    try:
        # Get the analysis record
        analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
        if not analysis:
            return
        
        # Create relevance engine with config
        config = config_overrides or {}
        engine = create_relevance_engine(config)
        
        # Perform analysis
        results = engine.analyze_resume_relevance(
            resume_file_path=resume_file_path,
            jd_text=job_description_text
        )
        
        # Update analysis record with results
        analysis.final_score = results['relevance_analysis']['final_score']
        analysis.hard_match_score = results['detailed_scores']['hard_match']['overall_score']
        analysis.semantic_match_score = results['detailed_scores']['semantic_match']['overall_score']
        analysis.verdict = results['relevance_analysis']['verdict']
        analysis.confidence_level = results['relevance_analysis']['confidence_level']
        analysis.analysis_results = results
        analysis.processing_time_seconds = results['analysis_metadata']['processing_time_seconds']
        
        # Extract specific fields for easy querying
        analysis.component_scores = results['detailed_scores']
        analysis.matched_skills = results['matching_analysis']['matched_skills']
        analysis.missing_skills = results['gaps_and_recommendations']['missing_elements']
        analysis.matched_qualifications = results['matching_analysis']['matched_qualifications']
        analysis.executive_summary = results['feedback']['executive_summary']
        analysis.detailed_feedback = results['feedback']['detailed_feedback']
        analysis.recommendations = results['feedback']['actionable_recommendations']
        analysis.strengths = results['feedback']['strengths_highlighted']
        analysis.improvement_areas = results['feedback']['areas_for_improvement']
        
        db.commit()
        
    except Exception as e:
        # Update analysis with error status
        if analysis:
            analysis.analysis_results = {"error": str(e)}
            analysis.verdict = "Unknown"
            db.commit()
    finally:
        db.close()


@router.get("/", response_model=List[AnalysisSummary])
async def list_analyses(
    skip: int = 0,
    limit: int = 100,
    filter_params: AnalysisFilter = Depends(),
    db: Session = Depends(get_db)
):
    """
    List analyses with filtering options.
    """
    query = db.query(ResumeAnalysis, Resume, JobDescription).join(
        Resume, ResumeAnalysis.resume_id == Resume.id
    ).join(
        JobDescription, ResumeAnalysis.job_description_id == JobDescription.id
    )
    
    # Apply filters
    if filter_params.job_id:
        query = query.filter(ResumeAnalysis.job_description_id == filter_params.job_id)
    
    if filter_params.resume_id:
        query = query.filter(ResumeAnalysis.resume_id == filter_params.resume_id)
    
    if filter_params.verdict:
        query = query.filter(ResumeAnalysis.verdict == filter_params.verdict.value)
    
    if filter_params.min_score is not None:
        query = query.filter(ResumeAnalysis.final_score >= filter_params.min_score)
    
    if filter_params.max_score is not None:
        query = query.filter(ResumeAnalysis.final_score <= filter_params.max_score)
    
    if filter_params.date_from:
        query = query.filter(ResumeAnalysis.created_at >= filter_params.date_from)
    
    if filter_params.date_to:
        query = query.filter(ResumeAnalysis.created_at <= filter_params.date_to)
    
    if filter_params.candidate_name:
        query = query.filter(Resume.candidate_name.contains(filter_params.candidate_name))
    
    if filter_params.job_title:
        query = query.filter(JobDescription.title.contains(filter_params.job_title))
    
    # Order by creation date (newest first)
    query = query.order_by(ResumeAnalysis.created_at.desc())
    
    results = query.offset(skip).limit(limit).all()
    
    summaries = []
    for analysis, resume, job_desc in results:
        summaries.append(AnalysisSummary(
            id=analysis.id,
            candidate_name=resume.candidate_name,
            job_title=job_desc.title,
            company_name=job_desc.company_name,
            final_score=analysis.final_score,
            verdict=analysis.verdict,
            created_at=analysis.created_at
        ))
    
    return summaries


@router.get("/{analysis_id}", response_model=AnalysisDetails)
async def get_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Get detailed analysis results.
    """
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    # Build detailed response
    results = analysis.analysis_results or {}
    
    return AnalysisDetails(
        id=analysis.id,
        resume_id=analysis.resume_id,
        job_description_id=analysis.job_description_id,
        final_score=analysis.final_score,
        hard_match_score=analysis.hard_match_score,
        semantic_match_score=analysis.semantic_match_score,
        verdict=analysis.verdict,
        confidence_level=analysis.confidence_level,
        created_at=analysis.created_at,
        processing_time_seconds=analysis.processing_time_seconds,
        component_scores=results.get('detailed_scores', {}),
        matching_analysis=results.get('matching_analysis', {}),
        gaps_and_recommendations=results.get('gaps_and_recommendations', {}),
        feedback=results.get('feedback', {}),
        contextual_insights=results.get('contextual_insights', [])
    )


@router.delete("/{analysis_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_analysis(
    analysis_id: int,
    db: Session = Depends(get_db)
):
    """
    Delete an analysis.
    """
    analysis = db.query(ResumeAnalysis).filter(ResumeAnalysis.id == analysis_id).first()
    
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found"
        )
    
    try:
        db.delete(analysis)
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting analysis: {str(e)}"
        )


@router.post("/bulk", response_model=BulkAnalysisResponse)
async def create_bulk_analysis(
    bulk_request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create bulk analyses for multiple resumes against one job description.
    """
    # Validate job description exists
    job_description = db.query(JobDescription).filter(
        JobDescription.id == bulk_request.job_description_id
    ).first()
    if not job_description:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job description not found"
        )
    
    # Validate all resumes exist
    resumes = db.query(Resume).filter(Resume.id.in_(bulk_request.resume_ids)).all()
    if len(resumes) != len(bulk_request.resume_ids):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="One or more resumes not found"
        )
    
    try:
        from ..models.database import AnalysisSession
        from datetime import datetime, timedelta
        
        # Create analysis session
        session = AnalysisSession(
            session_name=bulk_request.session_name or f"Bulk Analysis {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            total_analyses=len(bulk_request.resume_ids),
            session_config=bulk_request.config_overrides or {}
        )
        
        db.add(session)
        db.commit()
        db.refresh(session)
        
        # Schedule background processing
        background_tasks.add_task(
            process_bulk_analysis_background,
            session.id,
            bulk_request.job_description_id,
            bulk_request.resume_ids,
            bulk_request.config_overrides
        )
        
        # Estimate completion time (rough estimate: 30 seconds per analysis)
        estimated_time = datetime.now() + timedelta(seconds=len(bulk_request.resume_ids) * 30)
        
        return BulkAnalysisResponse(
            session_id=session.id,
            total_requested=len(bulk_request.resume_ids),
            status="processing",
            estimated_completion_time=estimated_time
        )
        
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating bulk analysis: {str(e)}"
        )


async def process_bulk_analysis_background(
    session_id: int,
    job_description_id: int,
    resume_ids: List[int],
    config_overrides: Optional[dict] = None
):
    """
    Background task to process bulk analyses.
    """
    db = next(get_db())
    try:
        from ..models.database import AnalysisSession
        
        session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
        if not session:
            return
        
        job_description = db.query(JobDescription).filter(JobDescription.id == job_description_id).first()
        if not job_description:
            session.status = "failed"
            session.results_summary = {"error": "Job description not found"}
            db.commit()
            return
        
        # Create relevance engine
        config = config_overrides or {}
        engine = create_relevance_engine(config)
        
        completed = 0
        failed = 0
        results_summary = {"analyses": []}
        
        for resume_id in resume_ids:
            try:
                resume = db.query(Resume).filter(Resume.id == resume_id).first()
                if not resume:
                    failed += 1
                    continue
                
                # Check if analysis already exists
                existing = db.query(ResumeAnalysis).filter(
                    ResumeAnalysis.resume_id == resume_id,
                    ResumeAnalysis.job_description_id == job_description_id
                ).first()
                
                if existing:
                    # Skip existing analysis
                    continue
                
                # Perform analysis
                results = engine.analyze_resume_relevance(
                    resume_file_path=resume.file_path,
                    jd_text=job_description.description_text
                )
                
                # Create analysis record
                analysis = ResumeAnalysis(
                    resume_id=resume_id,
                    job_description_id=job_description_id,
                    final_score=results['relevance_analysis']['final_score'],
                    hard_match_score=results['detailed_scores']['hard_match']['overall_score'],
                    semantic_match_score=results['detailed_scores']['semantic_match']['overall_score'],
                    verdict=results['relevance_analysis']['verdict'],
                    confidence_level=results['relevance_analysis']['confidence_level'],
                    analysis_results=results,
                    processing_time_seconds=results['analysis_metadata']['processing_time_seconds'],
                    analysis_config=config
                )
                
                # Extract specific fields
                analysis.component_scores = results['detailed_scores']
                analysis.matched_skills = results['matching_analysis']['matched_skills']
                analysis.missing_skills = results['gaps_and_recommendations']['missing_elements']
                analysis.executive_summary = results['feedback']['executive_summary']
                analysis.detailed_feedback = results['feedback']['detailed_feedback']
                analysis.recommendations = results['feedback']['actionable_recommendations']
                
                db.add(analysis)
                db.commit()
                
                completed += 1
                
                # Add to summary
                results_summary["analyses"].append({
                    "resume_id": resume_id,
                    "candidate_name": resume.candidate_name,
                    "final_score": analysis.final_score,
                    "verdict": analysis.verdict
                })
                
            except Exception as e:
                failed += 1
                results_summary["analyses"].append({
                    "resume_id": resume_id,
                    "error": str(e)
                })
        
        # Update session
        session.completed_analyses = completed
        session.failed_analyses = failed
        session.status = "completed" if failed == 0 else "partial"
        session.results_summary = results_summary
        session.completed_at = datetime.now()
        
        db.commit()
        
    except Exception as e:
        if session:
            session.status = "failed"
            session.results_summary = {"error": str(e)}
            db.commit()
    finally:
        db.close()


@router.get("/sessions/{session_id}")
async def get_analysis_session(
    session_id: int,
    db: Session = Depends(get_db)
):
    """
    Get bulk analysis session status.
    """
    from ..models.database import AnalysisSession
    
    session = db.query(AnalysisSession).filter(AnalysisSession.id == session_id).first()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis session not found"
        )
    
    return {
        "id": session.id,
        "session_name": session.session_name,
        "status": session.status,
        "total_analyses": session.total_analyses,
        "completed_analyses": session.completed_analyses,
        "failed_analyses": session.failed_analyses,
        "started_at": session.started_at,
        "completed_at": session.completed_at,
        "results_summary": session.results_summary
    }