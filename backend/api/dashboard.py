"""
Dashboard API endpoints for analytics and statistics.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta

from ..api.database import get_db
from ..models import (
    Resume, JobDescription, ResumeAnalysis,
    DashboardStats, JobAnalyticsSummary, AnalysisSummary
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Get overall dashboard statistics.
    """
    try:
        # Count totals
        total_resumes = db.query(Resume).count()
        total_job_descriptions = db.query(JobDescription).filter(JobDescription.is_active == True).count()
        total_analyses = db.query(ResumeAnalysis).count()
        
        # Recent analyses (last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_analyses = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.created_at >= seven_days_ago
        ).count()
        
        # Average score
        avg_score_result = db.query(func.avg(ResumeAnalysis.final_score)).scalar()
        avg_score = round(avg_score_result, 2) if avg_score_result else 0.0
        
        # Verdict distribution
        verdict_counts = db.query(
            ResumeAnalysis.verdict,
            func.count(ResumeAnalysis.id)
        ).group_by(ResumeAnalysis.verdict).all()
        
        verdict_distribution = {verdict: count for verdict, count in verdict_counts}
        
        return DashboardStats(
            total_resumes=total_resumes,
            total_job_descriptions=total_job_descriptions,
            total_analyses=total_analyses,
            recent_analyses=recent_analyses,
            avg_score=avg_score,
            verdict_distribution=verdict_distribution
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching dashboard stats: {str(e)}"
        )


@router.get("/job-analytics", response_model=List[JobAnalyticsSummary])
async def get_job_analytics(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get analytics summary for jobs with the most analyses.
    """
    try:
        # Get job analytics with analysis counts
        job_analytics = db.query(
            JobDescription.id,
            JobDescription.title,
            JobDescription.company_name,
            func.count(ResumeAnalysis.id).label('analysis_count'),
            func.avg(ResumeAnalysis.final_score).label('avg_score')
        ).join(
            ResumeAnalysis, JobDescription.id == ResumeAnalysis.job_description_id
        ).filter(
            JobDescription.is_active == True
        ).group_by(
            JobDescription.id
        ).order_by(
            desc('analysis_count')
        ).limit(limit).all()
        
        result = []
        for job_id, title, company_name, analysis_count, avg_score in job_analytics:
            # Get verdict distribution for this job
            verdict_counts = db.query(
                ResumeAnalysis.verdict,
                func.count(ResumeAnalysis.id)
            ).filter(
                ResumeAnalysis.job_description_id == job_id
            ).group_by(ResumeAnalysis.verdict).all()
            
            verdict_distribution = {verdict: count for verdict, count in verdict_counts}
            
            # Get top candidates for this job
            top_candidates_query = db.query(
                ResumeAnalysis, Resume
            ).join(
                Resume, ResumeAnalysis.resume_id == Resume.id
            ).filter(
                ResumeAnalysis.job_description_id == job_id
            ).order_by(
                desc(ResumeAnalysis.final_score)
            ).limit(5).all()
            
            top_candidates = []
            for analysis, resume in top_candidates_query:
                top_candidates.append(AnalysisSummary(
                    id=analysis.id,
                    candidate_name=resume.candidate_name,
                    job_title=title,
                    company_name=company_name,
                    final_score=analysis.final_score,
                    verdict=analysis.verdict,
                    created_at=analysis.created_at
                ))
            
            result.append(JobAnalyticsSummary(
                job_id=job_id,
                job_title=title,
                company_name=company_name,
                total_candidates=analysis_count,
                avg_score=round(avg_score, 2) if avg_score else 0.0,
                verdict_distribution=verdict_distribution,
                top_candidates=top_candidates
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching job analytics: {str(e)}"
        )


@router.get("/recent-analyses", response_model=List[AnalysisSummary])
async def get_recent_analyses(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get the most recent analyses.
    """
    try:
        recent_analyses = db.query(
            ResumeAnalysis, Resume, JobDescription
        ).join(
            Resume, ResumeAnalysis.resume_id == Resume.id
        ).join(
            JobDescription, ResumeAnalysis.job_description_id == JobDescription.id
        ).order_by(
            desc(ResumeAnalysis.created_at)
        ).limit(limit).all()
        
        result = []
        for analysis, resume, job_desc in recent_analyses:
            result.append(AnalysisSummary(
                id=analysis.id,
                candidate_name=resume.candidate_name,
                job_title=job_desc.title,
                company_name=job_desc.company_name,
                final_score=analysis.final_score,
                verdict=analysis.verdict,
                created_at=analysis.created_at
            ))
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching recent analyses: {str(e)}"
        )


@router.get("/score-distribution")
async def get_score_distribution(db: Session = Depends(get_db)):
    """
    Get score distribution histogram data.
    """
    try:
        # Define score ranges
        score_ranges = [
            (0, 20), (20, 40), (40, 60), (60, 80), (80, 100)
        ]
        
        distribution = {}
        for min_score, max_score in score_ranges:
            count = db.query(ResumeAnalysis).filter(
                ResumeAnalysis.final_score >= min_score,
                ResumeAnalysis.final_score < max_score if max_score < 100 else ResumeAnalysis.final_score <= max_score
            ).count()
            
            range_label = f"{min_score}-{max_score}"
            distribution[range_label] = count
        
        return {"score_distribution": distribution}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching score distribution: {str(e)}"
        )


@router.get("/top-candidates")
async def get_top_candidates(
    job_id: int = None,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """
    Get top candidates overall or for a specific job.
    """
    try:
        query = db.query(
            ResumeAnalysis, Resume, JobDescription
        ).join(
            Resume, ResumeAnalysis.resume_id == Resume.id
        ).join(
            JobDescription, ResumeAnalysis.job_description_id == JobDescription.id
        )
        
        if job_id:
            query = query.filter(ResumeAnalysis.job_description_id == job_id)
        
        top_candidates = query.order_by(
            desc(ResumeAnalysis.final_score)
        ).limit(limit).all()
        
        result = []
        for analysis, resume, job_desc in top_candidates:
            result.append({
                "analysis_id": analysis.id,
                "candidate_name": resume.candidate_name,
                "candidate_email": resume.candidate_email,
                "job_title": job_desc.title,
                "company_name": job_desc.company_name,
                "final_score": analysis.final_score,
                "verdict": analysis.verdict,
                "hard_match_score": analysis.hard_match_score,
                "semantic_match_score": analysis.semantic_match_score,
                "created_at": analysis.created_at
            })
        
        return {"top_candidates": result}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching top candidates: {str(e)}"
        )


@router.get("/trends")
async def get_analysis_trends(
    days: int = 30,
    db: Session = Depends(get_db)
):
    """
    Get analysis trends over time.
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Daily analysis counts
        daily_counts = db.query(
            func.date(ResumeAnalysis.created_at).label('date'),
            func.count(ResumeAnalysis.id).label('count'),
            func.avg(ResumeAnalysis.final_score).label('avg_score')
        ).filter(
            ResumeAnalysis.created_at >= start_date
        ).group_by(
            func.date(ResumeAnalysis.created_at)
        ).order_by('date').all()
        
        trends = []
        for date, count, avg_score in daily_counts:
            trends.append({
                "date": date.isoformat(),
                "analysis_count": count,
                "avg_score": round(avg_score, 2) if avg_score else 0.0
            })
        
        return {"trends": trends, "period_days": days}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching analysis trends: {str(e)}"
        )


@router.get("/skills-analysis")
async def get_skills_analysis(
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """
    Get analysis of most commonly required and missing skills.
    """
    try:
        # This is a simplified version - in a real implementation,
        # you'd want to properly parse and aggregate the skills data
        analyses = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.missing_skills.isnot(None)
        ).all()
        
        missing_skills_count = {}
        matched_skills_count = {}
        
        for analysis in analyses:
            # Count missing skills
            missing_skills = analysis.missing_skills or {}
            critical_missing = missing_skills.get('critical_skills', [])
            for skill in critical_missing:
                missing_skills_count[skill] = missing_skills_count.get(skill, 0) + 1
            
            # Count matched skills
            matched_skills = analysis.matched_skills or {}
            must_have_matched = matched_skills.get('must_have', [])
            for match in must_have_matched:
                if isinstance(match, dict) and 'jd_skill' in match:
                    skill = match['jd_skill']
                    matched_skills_count[skill] = matched_skills_count.get(skill, 0) + 1
        
        # Sort and limit
        top_missing = sorted(missing_skills_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        top_matched = sorted(matched_skills_count.items(), key=lambda x: x[1], reverse=True)[:limit]
        
        return {
            "most_missing_skills": [{"skill": skill, "count": count} for skill, count in top_missing],
            "most_matched_skills": [{"skill": skill, "count": count} for skill, count in top_matched]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching skills analysis: {str(e)}"
        )


@router.get("/performance-metrics")
async def get_performance_metrics(db: Session = Depends(get_db)):
    """
    Get system performance metrics.
    """
    try:
        # Average processing time
        avg_processing_time = db.query(
            func.avg(ResumeAnalysis.processing_time_seconds)
        ).scalar()
        
        # Processing time distribution
        fast_analyses = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.processing_time_seconds < 30
        ).count()
        
        medium_analyses = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.processing_time_seconds >= 30,
            ResumeAnalysis.processing_time_seconds < 60
        ).count()
        
        slow_analyses = db.query(ResumeAnalysis).filter(
            ResumeAnalysis.processing_time_seconds >= 60
        ).count()
        
        # Confidence level distribution
        confidence_counts = db.query(
            ResumeAnalysis.confidence_level,
            func.count(ResumeAnalysis.id)
        ).group_by(ResumeAnalysis.confidence_level).all()
        
        confidence_distribution = {level: count for level, count in confidence_counts if level}
        
        return {
            "avg_processing_time_seconds": round(avg_processing_time, 2) if avg_processing_time else 0.0,
            "processing_time_distribution": {
                "fast": fast_analyses,
                "medium": medium_analyses,
                "slow": slow_analyses
            },
            "confidence_distribution": confidence_distribution
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching performance metrics: {str(e)}"
        )