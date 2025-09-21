"""
Relevance Analysis Engine
Main orchestrator that combines hard and semantic matching to generate 
final scores, verdicts, and comprehensive analysis results.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from enum import Enum

# Import our core modules
from .resume_parser import ResumeParser, extract_contact_info, extract_education_info
from .jd_parser import JobDescriptionParser
from .hard_match import HardMatchEngine
from .semantic_match import SemanticMatchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RelevanceVerdict(Enum):
    """Enumeration for relevance verdicts."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"


class RelevanceAnalysisEngine:
    """
    Main engine that orchestrates the complete relevance analysis workflow.
    Combines hard matching and semantic matching to generate comprehensive results.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Relevance Analysis Engine.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration parameters
        """
        # Set default configuration
        self.config = {
            'hard_match_weight': 0.4,
            'semantic_match_weight': 0.6,
            'high_threshold': 75,
            'medium_threshold': 50,
            'fuzzy_threshold': 80,
            'openai_api_key': os.getenv('OPENAI_API_KEY'),
            'use_local_embeddings': False
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Initialize components
        self.resume_parser = ResumeParser()
        self.jd_parser = JobDescriptionParser()
        self.hard_match_engine = HardMatchEngine(fuzzy_threshold=self.config['fuzzy_threshold'])
        self.semantic_match_engine = SemanticMatchEngine(
            openai_api_key=self.config['openai_api_key'],
            use_local_embeddings=self.config['use_local_embeddings']
        )
        
        logger.info("Relevance Analysis Engine initialized successfully")
    
    def analyze_resume_relevance(self, resume_file_path: str, jd_text: str, 
                                candidate_info: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Perform complete relevance analysis between a resume file and job description.
        
        Args:
            resume_file_path (str): Path to the resume file (.pdf or .docx)
            jd_text (str): Job description text
            candidate_info (Optional[Dict[str, str]]): Additional candidate information
            
        Returns:
            Dict[str, Any]: Complete analysis results
        """
        try:
            analysis_start_time = datetime.now()
            logger.info(f"Starting relevance analysis for resume: {resume_file_path}")
            
            # Step 1: Parse resume
            logger.info("Parsing resume...")
            resume_data = self.resume_parser.parse_resume(resume_file_path)
            resume_text = resume_data['cleaned_text']
            
            # Step 2: Parse job description
            logger.info("Parsing job description...")
            jd_data = self.jd_parser.parse_job_description(jd_text)
            
            # Step 3: Extract additional resume information
            contact_info = extract_contact_info(resume_text)
            education_info = extract_education_info(resume_text)
            
            # Step 4: Perform hard matching
            logger.info("Performing hard match analysis...")
            hard_match_results = self.hard_match_engine.calculate_hard_match_score(resume_text, jd_data)
            
            # Step 5: Perform semantic matching
            logger.info("Performing semantic match analysis...")
            semantic_match_results = self.semantic_match_engine.calculate_semantic_match_score(resume_text, jd_data)
            
            # Step 6: Calculate final score and verdict
            final_score, verdict = self._calculate_final_score_and_verdict(
                hard_match_results['overall_score'],
                semantic_match_results['overall_score']
            )
            
            # Step 7: Generate missing elements
            missing_elements = self._identify_missing_elements(hard_match_results, jd_data)
            
            # Step 8: Generate comprehensive feedback
            feedback = self._generate_comprehensive_feedback(
                resume_data, jd_data, hard_match_results, semantic_match_results, 
                final_score, verdict, missing_elements
            )
            
            # Compile final results
            analysis_end_time = datetime.now()
            processing_time = (analysis_end_time - analysis_start_time).total_seconds()
            
            results = {
                'analysis_metadata': {
                    'timestamp': analysis_start_time.isoformat(),
                    'processing_time_seconds': round(processing_time, 2),
                    'resume_file': resume_file_path,
                    'engine_version': '1.0.0'
                },
                'candidate_info': {
                    'contact_information': contact_info,
                    'education_background': education_info,
                    'resume_stats': {
                        'word_count': resume_data['word_count'],
                        'char_count': resume_data['char_count'],
                        'sections_identified': list(resume_data['sections'].keys())
                    },
                    'additional_info': candidate_info or {}
                },
                'job_analysis': {
                    'job_title': jd_data['job_title'],
                    'company_name': jd_data['company_name'],
                    'required_skills': jd_data['must_have_skills'],
                    'preferred_skills': jd_data['good_to_have_skills'],
                    'qualifications': jd_data['qualifications'],
                    'experience_required': jd_data['experience_required']
                },
                'relevance_analysis': {
                    'final_score': round(final_score, 2),
                    'verdict': verdict.value,
                    'confidence_level': self._calculate_confidence_level(hard_match_results, semantic_match_results)
                },
                'detailed_scores': {
                    'hard_match': {
                        'overall_score': hard_match_results['overall_score'],
                        'component_scores': hard_match_results['component_scores'],
                        'weight_applied': self.config['hard_match_weight']
                    },
                    'semantic_match': {
                        'overall_score': semantic_match_results['overall_score'],
                        'vector_similarity': semantic_match_results.get('vector_similarity', 0),
                        'llm_reasoning_score': semantic_match_results.get('llm_reasoning', {}).get('score', 0),
                        'weight_applied': self.config['semantic_match_weight']
                    }
                },
                'matching_analysis': {
                    'matched_skills': hard_match_results['matched_skills'],
                    'matched_qualifications': hard_match_results['matched_qualifications'],
                    'skill_transferability': semantic_match_results.get('skill_analysis', {}),
                    'experience_alignment': semantic_match_results.get('experience_analysis', {})
                },
                'gaps_and_recommendations': {
                    'missing_elements': missing_elements,
                    'development_priorities': self._prioritize_skill_gaps(missing_elements, jd_data),
                    'experience_gaps': self._identify_experience_gaps(semantic_match_results, jd_data),
                    'improvement_suggestions': self._generate_improvement_suggestions(
                        missing_elements, semantic_match_results, jd_data
                    )
                },
                'feedback': {
                    'executive_summary': feedback['executive_summary'],
                    'detailed_feedback': feedback['detailed_feedback'],
                    'actionable_recommendations': feedback['actionable_recommendations'],
                    'strengths_highlighted': feedback['strengths'],
                    'areas_for_improvement': feedback['areas_for_improvement']
                },
                'contextual_insights': semantic_match_results.get('contextual_insights', [])
            }
            
            logger.info(f"Relevance analysis completed. Final score: {final_score}%, Verdict: {verdict.value}")
            return results
            
        except Exception as e:
            logger.error(f"Error in relevance analysis: {str(e)}")
            raise
    
    def _calculate_final_score_and_verdict(self, hard_score: float, semantic_score: float) -> Tuple[float, RelevanceVerdict]:
        """
        Calculate weighted final score and determine verdict.
        
        Args:
            hard_score (float): Hard match score
            semantic_score (float): Semantic match score
            
        Returns:
            Tuple[float, RelevanceVerdict]: Final score and verdict
        """
        # Calculate weighted score
        final_score = (
            hard_score * self.config['hard_match_weight'] +
            semantic_score * self.config['semantic_match_weight']
        )
        
        # Determine verdict based on thresholds
        if final_score >= self.config['high_threshold']:
            verdict = RelevanceVerdict.HIGH
        elif final_score >= self.config['medium_threshold']:
            verdict = RelevanceVerdict.MEDIUM
        else:
            verdict = RelevanceVerdict.LOW
        
        return final_score, verdict
    
    def _calculate_confidence_level(self, hard_results: Dict[str, Any], semantic_results: Dict[str, Any]) -> str:
        """Calculate confidence level in the analysis."""
        hard_score = hard_results['overall_score']
        semantic_score = semantic_results['overall_score']
        
        # Calculate score agreement
        score_difference = abs(hard_score - semantic_score)
        
        if score_difference <= 10:
            return "High"
        elif score_difference <= 25:
            return "Medium"
        else:
            return "Low"
    
    def _identify_missing_elements(self, hard_match_results: Dict[str, Any], jd_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify missing skills, qualifications, and other elements."""
        missing_elements = {
            'critical_skills': hard_match_results['missing_skills']['must_have'],
            'preferred_skills': hard_match_results['missing_skills']['good_to_have'],
            'qualifications': hard_match_results['missing_qualifications'],
            'certifications': [],
            'experience_areas': []
        }
        
        # Identify missing certifications from qualifications
        for qual in jd_data.get('qualifications', []):
            if any(cert_keyword in qual.lower() for cert_keyword in ['certified', 'certification', 'license']):
                if qual not in [mq for mq in hard_match_results['matched_qualifications']]:
                    missing_elements['certifications'].append(qual)
        
        # Identify experience areas that might be missing
        responsibilities = jd_data.get('responsibilities', [])
        for resp in responsibilities:
            if len(resp) > 20:  # Filter meaningful responsibilities
                missing_elements['experience_areas'].append(resp[:100])  # Truncate for brevity
        
        return missing_elements
    
    def _prioritize_skill_gaps(self, missing_elements: Dict[str, List[str]], jd_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize skill gaps based on importance."""
        priorities = []
        
        # Critical skills (must-have) get highest priority
        for skill in missing_elements['critical_skills']:
            priorities.append({
                'skill': skill,
                'priority': 'Critical',
                'category': 'Technical',
                'learning_effort': 'High',
                'business_impact': 'High'
            })
        
        # Preferred skills get medium priority
        for skill in missing_elements['preferred_skills']:
            priorities.append({
                'skill': skill,
                'priority': 'Medium',
                'category': 'Technical',
                'learning_effort': 'Medium',
                'business_impact': 'Medium'
            })
        
        # Certifications get specific priority based on type
        for cert in missing_elements['certifications']:
            priority = 'High' if any(keyword in cert.lower() for keyword in ['aws', 'azure', 'google', 'cisco']) else 'Medium'
            priorities.append({
                'skill': cert,
                'priority': priority,
                'category': 'Certification',
                'learning_effort': 'High',
                'business_impact': priority
            })
        
        return priorities
    
    def _identify_experience_gaps(self, semantic_results: Dict[str, Any], jd_data: Dict[str, Any]) -> List[str]:
        """Identify experience-related gaps."""
        gaps = []
        
        experience_analysis = semantic_results.get('experience_analysis', {})
        concerns = experience_analysis.get('concerns', [])
        
        for concern in concerns:
            gaps.append(concern)
        
        # Add experience requirements if not met
        exp_required = jd_data.get('experience_required', '')
        if exp_required and exp_required != 'Not specified':
            exp_match = experience_analysis.get('experience_match', 0)
            if exp_match < 70:
                gaps.append(f"May not fully meet experience requirement: {exp_required}")
        
        return gaps
    
    def _generate_improvement_suggestions(self, missing_elements: Dict[str, List[str]], 
                                        semantic_results: Dict[str, Any], jd_data: Dict[str, Any]) -> List[str]:
        """Generate actionable improvement suggestions."""
        suggestions = []
        
        # Skill development suggestions
        critical_skills = missing_elements['critical_skills']
        if critical_skills:
            suggestions.append(f"Focus on developing these critical skills: {', '.join(critical_skills[:3])}")
        
        # Certification suggestions
        certifications = missing_elements['certifications']
        if certifications:
            suggestions.append(f"Consider pursuing relevant certifications: {', '.join(certifications[:2])}")
        
        # Project experience suggestions
        skill_analysis = semantic_results.get('skill_analysis', {})
        if skill_analysis:
            tech_fit = skill_analysis.get('overall_technical_fit', 0)
            if tech_fit < 60:
                suggestions.append("Build portfolio projects demonstrating the required technical skills")
        
        # Experience level suggestions
        experience_analysis = semantic_results.get('experience_analysis', {})
        if experience_analysis:
            exp_match = experience_analysis.get('experience_match', 0)
            if exp_match < 50:
                suggestions.append("Gain more relevant work experience through internships or projects")
        
        return suggestions
    
    def _generate_comprehensive_feedback(self, resume_data: Dict[str, Any], jd_data: Dict[str, Any],
                                       hard_results: Dict[str, Any], semantic_results: Dict[str, Any],
                                       final_score: float, verdict: RelevanceVerdict,
                                       missing_elements: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate comprehensive feedback for the candidate."""
        
        # Executive Summary
        executive_summary = self._create_executive_summary(final_score, verdict, jd_data)
        
        # Detailed Feedback
        detailed_feedback = self._create_detailed_feedback(hard_results, semantic_results, missing_elements)
        
        # Actionable Recommendations
        actionable_recommendations = self._create_actionable_recommendations(missing_elements, semantic_results)
        
        # Strengths
        strengths = self._identify_strengths(hard_results, semantic_results)
        
        # Areas for Improvement
        areas_for_improvement = self._identify_improvement_areas(missing_elements, semantic_results)
        
        return {
            'executive_summary': executive_summary,
            'detailed_feedback': detailed_feedback,
            'actionable_recommendations': actionable_recommendations,
            'strengths': strengths,
            'areas_for_improvement': areas_for_improvement
        }
    
    def _create_executive_summary(self, final_score: float, verdict: RelevanceVerdict, jd_data: Dict[str, Any]) -> str:
        """Create executive summary of the analysis."""
        job_title = jd_data.get('job_title', 'the position')
        
        summary_templates = {
            RelevanceVerdict.HIGH: f"This candidate shows strong alignment with the {job_title} role, scoring {final_score:.1f}%. "
                                  f"Their background demonstrates solid competency in most required areas with good potential for success.",
            
            RelevanceVerdict.MEDIUM: f"This candidate shows moderate alignment with the {job_title} role, scoring {final_score:.1f}%. "
                                    f"While they have relevant experience, there are some skill gaps that would need to be addressed.",
            
            RelevanceVerdict.LOW: f"This candidate shows limited alignment with the {job_title} role, scoring {final_score:.1f}%. "
                                 f"Significant skill development and experience would be needed to meet the role requirements."
        }
        
        return summary_templates.get(verdict, f"Analysis completed with a score of {final_score:.1f}%.")
    
    def _create_detailed_feedback(self, hard_results: Dict[str, Any], semantic_results: Dict[str, Any],
                                missing_elements: Dict[str, List[str]]) -> str:
        """Create detailed feedback paragraph."""
        feedback_parts = []
        
        # Hard match analysis
        hard_score = hard_results['overall_score']
        if hard_score >= 70:
            feedback_parts.append("The candidate demonstrates strong technical skill alignment with exact matches for most required competencies.")
        elif hard_score >= 50:
            feedback_parts.append("The candidate has relevant technical skills, though some areas would benefit from further development.")
        else:
            feedback_parts.append("The technical skill profile requires significant enhancement to meet the role requirements.")
        
        # Semantic analysis
        semantic_score = semantic_results['overall_score']
        if semantic_score >= 70:
            feedback_parts.append("Their overall experience and background show strong conceptual alignment with the role.")
        elif semantic_score >= 50:
            feedback_parts.append("Their experience demonstrates good foundational knowledge with some transferable skills.")
        else:
            feedback_parts.append("The experience profile would benefit from more direct exposure to the role's core responsibilities.")
        
        # Missing elements
        critical_missing = len(missing_elements.get('critical_skills', []))
        if critical_missing > 0:
            feedback_parts.append(f"Key development areas include {critical_missing} critical technical skills that are essential for role success.")
        
        return " ".join(feedback_parts)
    
    def _create_actionable_recommendations(self, missing_elements: Dict[str, List[str]], 
                                         semantic_results: Dict[str, Any]) -> List[str]:
        """Create actionable recommendations."""
        recommendations = []
        
        # Skill development
        critical_skills = missing_elements.get('critical_skills', [])
        if critical_skills:
            recommendations.append(f"Immediate priority: Develop expertise in {', '.join(critical_skills[:3])}")
        
        # Experience building
        experience_analysis = semantic_results.get('experience_analysis', {})
        if experience_analysis.get('experience_match', 0) < 60:
            recommendations.append("Build relevant project experience through personal projects or contributions to open source")
        
        # Certification pursuit
        certifications = missing_elements.get('certifications', [])
        if certifications:
            recommendations.append(f"Consider pursuing industry certifications: {', '.join(certifications[:2])}")
        
        # Portfolio enhancement
        recommendations.append("Create a portfolio showcasing projects that demonstrate problem-solving in the target domain")
        
        return recommendations
    
    def _identify_strengths(self, hard_results: Dict[str, Any], semantic_results: Dict[str, Any]) -> List[str]:
        """Identify candidate strengths."""
        strengths = []
        
        # Technical strengths
        matched_skills = hard_results.get('matched_skills', {})
        must_have_matches = matched_skills.get('must_have', [])
        if must_have_matches:
            exact_matches = [m['jd_skill'] for m in must_have_matches if m['match_type'] == 'exact']
            if exact_matches:
                strengths.append(f"Strong technical foundation in: {', '.join(exact_matches[:3])}")
        
        # Experience strengths
        experience_analysis = semantic_results.get('experience_analysis', {})
        growth_indicators = experience_analysis.get('growth_indicators', [])
        for indicator in growth_indicators[:2]:
            strengths.append(indicator)
        
        # Semantic strengths
        similarity_analysis = semantic_results.get('similarity_analysis', {})
        analysis_strengths = similarity_analysis.get('strengths', [])
        for strength in analysis_strengths[:2]:
            strengths.append(strength)
        
        return strengths
    
    def _identify_improvement_areas(self, missing_elements: Dict[str, List[str]], 
                                  semantic_results: Dict[str, Any]) -> List[str]:
        """Identify areas for improvement."""
        improvements = []
        
        # Critical skill gaps
        critical_skills = missing_elements.get('critical_skills', [])
        if critical_skills:
            improvements.append(f"Develop proficiency in: {', '.join(critical_skills[:3])}")
        
        # Experience gaps
        experience_analysis = semantic_results.get('experience_analysis', {})
        concerns = experience_analysis.get('concerns', [])
        for concern in concerns[:2]:
            improvements.append(concern)
        
        # Semantic gaps
        similarity_analysis = semantic_results.get('similarity_analysis', {})
        gaps = similarity_analysis.get('gaps', [])
        for gap in gaps[:2]:
            improvements.append(gap)
        
        return improvements


# Factory function for easy initialization
def create_relevance_engine(config: Optional[Dict[str, Any]] = None) -> RelevanceAnalysisEngine:
    """
    Factory function to create a configured RelevanceAnalysisEngine.
    
    Args:
        config (Optional[Dict[str, Any]]): Configuration parameters
        
    Returns:
        RelevanceAnalysisEngine: Configured engine instance
    """
    return RelevanceAnalysisEngine(config)


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'hard_match_weight': 0.4,
        'semantic_match_weight': 0.6,
        'high_threshold': 75,
        'medium_threshold': 50,
        'use_local_embeddings': True  # For testing without OpenAI API
    }
    
    # Create engine
    engine = create_relevance_engine(config)
    
    # Example analysis (uncomment when testing with actual files)
    # result = engine.analyze_resume_relevance(
    #     resume_file_path="path/to/resume.pdf",
    #     jd_text="Job description text here...",
    #     candidate_info={'name': 'John Doe', 'email': 'john@example.com'}
    # )
    # 
    # print(f"Final Score: {result['relevance_analysis']['final_score']}%")
    # print(f"Verdict: {result['relevance_analysis']['verdict']}")
    # print(f"Executive Summary: {result['feedback']['executive_summary']}")
    
    logger.info("Relevance Analysis Engine example completed")