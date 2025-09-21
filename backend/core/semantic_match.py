"""
Semantic Match Engine Module
Performs semantic comparison between resumes and job descriptions using 
LangChain, vector embeddings, and LLM reasoning.
"""

import os
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import Document

# Sentence transformers for local embeddings (fallback)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticMatchEngine:
    """
    A class to perform semantic matching between resume and job description
    using LangChain, vector embeddings, and LLM reasoning.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, use_local_embeddings: bool = False):
        """
        Initialize the Semantic Match Engine.
        
        Args:
            openai_api_key (Optional[str]): OpenAI API key
            use_local_embeddings (bool): Whether to use local sentence transformers
        """
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        self.use_local_embeddings = use_local_embeddings
        
        # Initialize components
        self._setup_embeddings()
        self._setup_llm()
        self._setup_prompts()
        self._setup_text_splitter()
        
        # Initialize vector store (will be created per analysis)
        self.vector_store = None
    
    def _setup_embeddings(self):
        """Setup embedding model."""
        try:
            if self.use_local_embeddings or not self.openai_api_key:
                if SentenceTransformer:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    self.embeddings = None
                    logger.info("Using local sentence transformers for embeddings")
                else:
                    raise ImportError("sentence-transformers not available")
            else:
                self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
                self.embedding_model = None
                logger.info("Using OpenAI embeddings")
        except Exception as e:
            logger.warning(f"Embedding setup failed: {str(e)}. Falling back to basic similarity.")
            self.embeddings = None
            self.embedding_model = None
    
    def _setup_llm(self):
        """Setup LLM for reasoning."""
        try:
            if self.openai_api_key:
                self.llm = ChatOpenAI(
                    openai_api_key=self.openai_api_key,
                    model_name="gpt-3.5-turbo",
                    temperature=0.3,
                    max_tokens=1000
                )
                logger.info("Using OpenAI ChatGPT for LLM reasoning")
            else:
                # Fallback to basic text analysis
                self.llm = None
                logger.warning("No OpenAI API key provided. LLM reasoning will be limited.")
        except Exception as e:
            logger.error(f"LLM setup failed: {str(e)}")
            self.llm = None
    
    def _setup_prompts(self):
        """Setup prompt templates for LLM chains."""
        # Semantic similarity analysis prompt
        self.similarity_prompt = ChatPromptTemplate.from_template("""
        You are an expert HR analyst. Compare the following resume and job description for semantic similarity.
        
        Job Description:
        {job_description}
        
        Resume:
        {resume_text}
        
        Analyze the semantic similarity focusing on:
        1. Overall role alignment and responsibilities match
        2. Technical competency alignment (even if exact skills differ)
        3. Experience level and domain expertise
        4. Career progression and project complexity
        5. Soft skills and leadership indicators
        
        Provide your analysis as a JSON response with:
        - similarity_score: A score from 0-100 indicating semantic similarity
        - reasoning: Brief explanation of the score
        - strengths: Key alignment areas
        - gaps: Areas where the resume doesn't align with JD requirements
        
        Focus on conceptual and contextual understanding, not just keyword matching.
        
        Response format:
        {{
            "similarity_score": <score>,
            "reasoning": "<explanation>",
            "strengths": ["<strength1>", "<strength2>", ...],
            "gaps": ["<gap1>", "<gap2>", ...]
        }}
        """)
        
        # Contextual skill analysis prompt
        self.skill_analysis_prompt = ChatPromptTemplate.from_template("""
        You are an expert technical recruiter. Analyze if the candidate's background demonstrates 
        the required competencies, even if they don't use the exact same technologies.
        
        Required Skills/Technologies:
        {required_skills}
        
        Candidate Background:
        {resume_text}
        
        For each required skill, determine if the candidate has:
        1. Direct experience (exact match)
        2. Transferable experience (similar/related technologies)
        3. Foundational knowledge (could learn quickly)
        4. No relevant experience
        
        Consider:
        - Similar technologies in the same domain
        - Foundational skills that transfer across tools
        - Learning ability demonstrated through diverse tech stack
        - Project complexity that indicates deeper understanding
        
        Provide analysis as JSON:
        {{
            "skill_analysis": [
                {{
                    "required_skill": "<skill>",
                    "match_level": "<direct|transferable|foundational|none>",
                    "evidence": "<explanation>",
                    "confidence": <0-100>
                }}
            ],
            "overall_technical_fit": <0-100>
        }}
        """)
        
        # Experience level analysis prompt
        self.experience_prompt = ChatPromptTemplate.from_template("""
        Analyze the candidate's experience level and responsibilities compared to job requirements.
        
        Job Requirements:
        {job_requirements}
        
        Candidate Experience:
        {resume_text}
        
        Evaluate:
        1. Years of experience alignment
        2. Responsibility level (junior, mid, senior, lead)
        3. Project scale and complexity
        4. Leadership and mentoring experience
        5. Industry domain knowledge
        
        Provide analysis as JSON:
        {{
            "experience_match": <0-100>,
            "level_assessment": "<junior|mid|senior|lead>",
            "responsibility_alignment": <0-100>,
            "growth_indicators": ["<indicator1>", "<indicator2>", ...],
            "concerns": ["<concern1>", "<concern2>", ...]
        }}
        """)
    
    def _setup_text_splitter(self):
        """Setup text splitter for document chunking."""
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
    
    def calculate_semantic_match_score(self, resume_text: str, jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate semantic match score between resume and job description.
        
        Args:
            resume_text (str): Cleaned resume text
            jd_data (Dict[str, Any]): Parsed job description data
            
        Returns:
            Dict[str, Any]: Semantic match analysis results
        """
        try:
            results = {
                'overall_score': 0.0,
                'similarity_analysis': {},
                'skill_analysis': {},
                'experience_analysis': {},
                'vector_similarity': 0.0,
                'llm_reasoning': {},
                'contextual_insights': []
            }
            
            jd_text = jd_data.get('cleaned_text', '')
            
            # 1. Vector similarity analysis
            if self.embeddings or self.embedding_model:
                vector_score = self._calculate_vector_similarity(resume_text, jd_text)
                results['vector_similarity'] = vector_score
            
            # 2. LLM-based semantic analysis
            if self.llm:
                # Overall similarity analysis
                similarity_analysis = self._analyze_semantic_similarity(resume_text, jd_text)
                results['similarity_analysis'] = similarity_analysis
                
                # Skill-specific analysis
                required_skills = jd_data.get('must_have_skills', []) + jd_data.get('good_to_have_skills', [])
                if required_skills:
                    skill_analysis = self._analyze_skill_transferability(resume_text, required_skills)
                    results['skill_analysis'] = skill_analysis
                
                # Experience level analysis
                experience_analysis = self._analyze_experience_fit(resume_text, jd_text)
                results['experience_analysis'] = experience_analysis
                
                # Calculate overall score from LLM analysis
                llm_score = self._calculate_llm_score(similarity_analysis, skill_analysis, experience_analysis)
                results['llm_reasoning'] = {
                    'score': llm_score,
                    'components': {
                        'similarity': similarity_analysis.get('similarity_score', 0),
                        'technical_fit': skill_analysis.get('overall_technical_fit', 0),
                        'experience_match': experience_analysis.get('experience_match', 0)
                    }
                }
            
            # 3. Combine scores
            final_score = self._combine_semantic_scores(results)
            results['overall_score'] = round(final_score, 2)
            
            # 4. Generate contextual insights
            results['contextual_insights'] = self._generate_insights(results, jd_data)
            
            logger.info(f"Semantic match analysis completed. Score: {final_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic match calculation: {str(e)}")
            # Return default results on error
            return {
                'overall_score': 0.0,
                'error': str(e),
                'similarity_analysis': {},
                'skill_analysis': {},
                'experience_analysis': {},
                'vector_similarity': 0.0,
                'llm_reasoning': {},
                'contextual_insights': ['Analysis failed due to technical error']
            }
    
    def _calculate_vector_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate vector similarity using embeddings."""
        try:
            if self.embedding_model:  # Local embeddings
                resume_embedding = self.embedding_model.encode([resume_text])
                jd_embedding = self.embedding_model.encode([jd_text])
                
                # Calculate cosine similarity
                similarity = np.dot(resume_embedding[0], jd_embedding[0]) / (
                    np.linalg.norm(resume_embedding[0]) * np.linalg.norm(jd_embedding[0])
                )
                return float(similarity * 100)
                
            elif self.embeddings:  # OpenAI embeddings
                # Create vector store
                documents = [
                    Document(page_content=resume_text, metadata={"type": "resume"}),
                    Document(page_content=jd_text, metadata={"type": "job_description"})
                ]
                
                vector_store = Chroma.from_documents(documents, self.embeddings)
                
                # Perform similarity search
                similar_docs = vector_store.similarity_search_with_score(jd_text, k=1)
                
                if similar_docs:
                    # Convert distance to similarity (assuming cosine distance)
                    distance = similar_docs[0][1]
                    similarity = 1 - distance  # Convert distance to similarity
                    return float(similarity * 100)
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Vector similarity calculation failed: {str(e)}")
            return 0.0
    
    def _analyze_semantic_similarity(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Analyze semantic similarity using LLM."""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.similarity_prompt)
            
            response = chain.run(
                job_description=jd_text,
                resume_text=resume_text
            )
            
            # Parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_llm_response(response, 'similarity')
                
        except Exception as e:
            logger.warning(f"Semantic similarity analysis failed: {str(e)}")
            return {'similarity_score': 0, 'reasoning': 'Analysis failed', 'strengths': [], 'gaps': []}
    
    def _analyze_skill_transferability(self, resume_text: str, required_skills: List[str]) -> Dict[str, Any]:
        """Analyze skill transferability using LLM."""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.skill_analysis_prompt)
            
            response = chain.run(
                required_skills=', '.join(required_skills),
                resume_text=resume_text
            )
            
            # Parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._parse_llm_response(response, 'skills')
                
        except Exception as e:
            logger.warning(f"Skill analysis failed: {str(e)}")
            return {'skill_analysis': [], 'overall_technical_fit': 0}
    
    def _analyze_experience_fit(self, resume_text: str, jd_text: str) -> Dict[str, Any]:
        """Analyze experience level fit using LLM."""
        try:
            chain = LLMChain(llm=self.llm, prompt=self.experience_prompt)
            
            response = chain.run(
                job_requirements=jd_text,
                resume_text=resume_text
            )
            
            # Parse JSON response
            import json
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                return self._parse_llm_response(response, 'experience')
                
        except Exception as e:
            logger.warning(f"Experience analysis failed: {str(e)}")
            return {
                'experience_match': 0,
                'level_assessment': 'unknown',
                'responsibility_alignment': 0,
                'growth_indicators': [],
                'concerns': []
            }
    
    def _parse_llm_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback parser for LLM responses that aren't valid JSON."""
        import re
        
        if analysis_type == 'similarity':
            # Extract score
            score_match = re.search(r'(\d+)', response)
            score = int(score_match.group(1)) if score_match else 0
            
            return {
                'similarity_score': min(score, 100),
                'reasoning': response[:200],
                'strengths': [],
                'gaps': []
            }
        
        elif analysis_type == 'skills':
            return {
                'skill_analysis': [],
                'overall_technical_fit': 0
            }
        
        elif analysis_type == 'experience':
            return {
                'experience_match': 0,
                'level_assessment': 'unknown',
                'responsibility_alignment': 0,
                'growth_indicators': [],
                'concerns': []
            }
        
        return {}
    
    def _calculate_llm_score(self, similarity_analysis: Dict, skill_analysis: Dict, experience_analysis: Dict) -> float:
        """Calculate overall LLM-based score."""
        similarity_score = similarity_analysis.get('similarity_score', 0)
        technical_fit = skill_analysis.get('overall_technical_fit', 0)
        experience_match = experience_analysis.get('experience_match', 0)
        
        # Weighted average
        weights = {'similarity': 0.4, 'technical': 0.35, 'experience': 0.25}
        
        llm_score = (
            similarity_score * weights['similarity'] +
            technical_fit * weights['technical'] +
            experience_match * weights['experience']
        )
        
        return llm_score
    
    def _combine_semantic_scores(self, results: Dict[str, Any]) -> float:
        """Combine different semantic analysis scores."""
        vector_score = results.get('vector_similarity', 0)
        llm_score = results.get('llm_reasoning', {}).get('score', 0)
        
        if llm_score > 0 and vector_score > 0:
            # Combine both scores with LLM weighted higher
            final_score = (llm_score * 0.7) + (vector_score * 0.3)
        elif llm_score > 0:
            final_score = llm_score
        elif vector_score > 0:
            final_score = vector_score
        else:
            final_score = 0
        
        return min(final_score, 100.0)
    
    def _generate_insights(self, results: Dict[str, Any], jd_data: Dict[str, Any]) -> List[str]:
        """Generate contextual insights based on analysis."""
        insights = []
        
        # Overall score insights
        score = results.get('overall_score', 0)
        if score >= 80:
            insights.append("Strong semantic alignment between resume and job requirements")
        elif score >= 60:
            insights.append("Good conceptual fit with some areas for improvement")
        elif score >= 40:
            insights.append("Moderate alignment with significant gaps to address")
        else:
            insights.append("Limited semantic similarity - may require substantial skill development")
        
        # Skill-specific insights
        skill_analysis = results.get('skill_analysis', {})
        if skill_analysis:
            technical_fit = skill_analysis.get('overall_technical_fit', 0)
            if technical_fit >= 70:
                insights.append("Strong technical competency alignment")
            elif technical_fit >= 50:
                insights.append("Adequate technical background with some transferable skills")
            else:
                insights.append("Limited technical skill alignment - training may be needed")
        
        # Experience insights
        experience_analysis = results.get('experience_analysis', {})
        if experience_analysis:
            exp_match = experience_analysis.get('experience_match', 0)
            level = experience_analysis.get('level_assessment', 'unknown')
            if exp_match >= 70:
                insights.append(f"Experience level ({level}) aligns well with job requirements")
            else:
                insights.append(f"Experience level ({level}) may not fully match requirements")
        
        return insights


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = SemanticMatchEngine(use_local_embeddings=True)
    
    # Sample data
    sample_resume = """
    Senior Software Engineer with 8 years of experience in building scalable web applications.
    Expertise in Python, Django, and cloud technologies. Led teams of 5+ developers on complex projects.
    Master's degree in Computer Science. Experience with microservices architecture and DevOps practices.
    """
    
    sample_jd_data = {
        'cleaned_text': 'Looking for a Senior Software Engineer to lead our backend development team. Must have experience with Java, Spring Boot, and AWS.',
        'must_have_skills': ['Java', 'Spring Boot', 'AWS'],
        'good_to_have_skills': ['Microservices', 'Leadership'],
        'qualifications': ['Bachelor in Computer Science'],
        'job_title': 'Senior Software Engineer'
    }
    
    # Analyze
    result = engine.calculate_semantic_match_score(sample_resume, sample_jd_data)
    print(f"Semantic Match Score: {result['overall_score']}%")
    print(f"Vector Similarity: {result['vector_similarity']}%")
    print(f"Contextual Insights: {result['contextual_insights']}")