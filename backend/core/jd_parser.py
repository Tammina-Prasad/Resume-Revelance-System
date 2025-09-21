"""
Job Description Parser Module
Handles extraction and analysis of key entities from job descriptions using NLP.
"""

import re
import logging
from typing import Dict, List, Any, Optional
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JobDescriptionParser:
    """
    A class to parse job descriptions and extract key information using NLP.
    """
    
    def __init__(self):
        self.setup_nltk()
        self.setup_spacy()
        self.skill_keywords = self._load_skill_keywords()
        self.qualification_keywords = self._load_qualification_keywords()
    
    def setup_nltk(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')
        
        try:
            nltk.data.find('chunkers/maxent_ne_chunker')
        except LookupError:
            nltk.download('maxent_ne_chunker')
        
        try:
            nltk.data.find('corpora/words')
        except LookupError:
            nltk.download('words')
    
    def setup_spacy(self):
        """Load spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    def parse_job_description(self, jd_text: str) -> Dict[str, Any]:
        """
        Parse job description and extract key entities.
        
        Args:
            jd_text (str): Job description text
            
        Returns:
            Dict[str, Any]: Parsed job description data
        """
        try:
            # Clean the text
            cleaned_text = self._clean_text(jd_text)
            
            # Extract key information
            parsed_data = {
                'original_text': jd_text,
                'cleaned_text': cleaned_text,
                'job_title': self._extract_job_title(cleaned_text),
                'company_name': self._extract_company_name(cleaned_text),
                'must_have_skills': self._extract_must_have_skills(cleaned_text),
                'good_to_have_skills': self._extract_good_to_have_skills(cleaned_text),
                'qualifications': self._extract_qualifications(cleaned_text),
                'experience_required': self._extract_experience_requirements(cleaned_text),
                'responsibilities': self._extract_responsibilities(cleaned_text),
                'benefits': self._extract_benefits(cleaned_text),
                'location': self._extract_location(cleaned_text),
                'keywords': self._extract_keywords(cleaned_text),
                'entities': self._extract_entities(cleaned_text)
            }
            
            logger.info("Successfully parsed job description")
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing job description: {str(e)}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters except important punctuation
        text = re.sub(r'[^\w\s.,()-]', ' ', text)
        return text.strip()
    
    def _extract_job_title(self, text: str) -> str:
        """Extract job title from the job description."""
        # Common patterns for job titles
        patterns = [
            r'(?:job title|position|role)[:]\s*([^\n]+)',
            r'(?:hiring for|looking for)[:]\s*([^\n]+)',
            r'^([A-Z][^.!?]*(?:engineer|developer|analyst|manager|specialist|coordinator|assistant|executive|director|lead|senior|junior))',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for capitalized title-like phrases in first few lines
        lines = text.split('\n')[:5]
        for line in lines:
            line = line.strip()
            if len(line.split()) <= 6 and any(keyword in line.lower() for keyword in 
                ['engineer', 'developer', 'analyst', 'manager', 'specialist', 'coordinator']):
                return line
        
        return "Not specified"
    
    def _extract_company_name(self, text: str) -> str:
        """Extract company name from job description."""
        # Look for company patterns
        patterns = [
            r'(?:company|organization)[:]\s*([^\n]+)',
            r'(?:join|work at|working for)[:]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Use spaCy for organization entity recognition
        if self.nlp:
            doc = self.nlp(text[:500])  # Check first 500 characters
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    return ent.text
        
        return "Not specified"
    
    def _extract_must_have_skills(self, text: str) -> List[str]:
        """Extract must-have skills from job description."""
        must_have_skills = []
        
        # Look for explicit must-have sections
        must_have_patterns = [
            r'(?:must have|required|essential|mandatory).*?skills?[:]\s*([^.]*)',
            r'(?:requirements?|qualifications?)[:]\s*([^.]*)',
            r'(?:must|should).*?(?:have|know|familiar).*?([^.]*)',
        ]
        
        for pattern in must_have_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                must_have_skills.extend(skills)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in must_have_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        return unique_skills[:10]  # Limit to top 10
    
    def _extract_good_to_have_skills(self, text: str) -> List[str]:
        """Extract good-to-have skills from job description."""
        good_to_have_skills = []
        
        # Look for explicit good-to-have sections
        patterns = [
            r'(?:good to have|nice to have|preferred|bonus|plus|additional).*?skills?[:]\s*([^.]*)',
            r'(?:preferred|bonus).*?(?:experience|knowledge).*?([^.]*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                skills = self._extract_skills_from_text(match)
                good_to_have_skills.extend(skills)
        
        # Remove duplicates
        seen = set()
        unique_skills = []
        for skill in good_to_have_skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        return unique_skills[:10]  # Limit to top 10
    
    def _extract_skills_from_text(self, text: str) -> List[str]:
        """Extract individual skills from a text segment."""
        skills = []
        
        # Common skill patterns
        for skill_pattern in self.skill_keywords:
            if re.search(r'\b' + re.escape(skill_pattern) + r'\b', text, re.IGNORECASE):
                skills.append(skill_pattern)
        
        # Extract programming languages, frameworks, and tools
        tech_patterns = [
            r'\b(python|java|javascript|c\+\+|c#|ruby|php|go|rust|scala|kotlin)\b',
            r'\b(react|angular|vue|node\.?js|express|django|flask|spring|laravel)\b',
            r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|linux|windows)\b',
            r'\b(mysql|postgresql|mongodb|redis|elasticsearch|cassandra)\b',
            r'\b(machine learning|ml|ai|deep learning|tensorflow|pytorch|scikit-learn)\b',
        ]
        
        for pattern in tech_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            skills.extend(matches)
        
        return skills
    
    def _extract_qualifications(self, text: str) -> List[str]:
        """Extract educational and professional qualifications."""
        qualifications = []
        
        # Educational qualifications
        edu_patterns = [
            r'\b(bachelor|master|phd|doctorate|b\.?tech|m\.?tech|mba|ms|bs)\b.*?(?:in|of)\s+([^.,\n]+)',
            r'\b(degree|diploma|certification)\s+in\s+([^.,\n]+)',
        ]
        
        for pattern in edu_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    qualifications.append(f"{match[0]} in {match[1]}".strip())
                else:
                    qualifications.append(match.strip())
        
        # Professional certifications
        cert_keywords = [
            'aws certified', 'azure certified', 'google cloud certified',
            'pmp', 'cissp', 'cisa', 'cism', 'agile', 'scrum master'
        ]
        
        for keyword in cert_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                qualifications.append(keyword)
        
        return qualifications
    
    def _extract_experience_requirements(self, text: str) -> str:
        """Extract experience requirements."""
        exp_patterns = [
            r'(\d+)\s*(?:\+|or more)?\s*years?\s*(?:of\s*)?(?:experience|exp)',
            r'(?:minimum|min|at least)\s*(\d+)\s*years?',
            r'(\d+)\s*to\s*(\d+)\s*years?',
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return "Not specified"
    
    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract job responsibilities."""
        responsibilities = []
        
        # Look for responsibility sections
        resp_patterns = [
            r'(?:responsibilities|duties|what you.?ll do)[:]\s*([^.]*)',
            r'(?:role|position)\s+involves[:]\s*([^.]*)',
        ]
        
        for pattern in resp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Split by bullet points or line breaks
                resp_items = re.split(r'[•\-*]\s*|\n\s*', match)
                for item in resp_items:
                    item = item.strip()
                    if len(item) > 10:  # Filter out short fragments
                        responsibilities.append(item)
        
        return responsibilities[:10]  # Limit to top 10
    
    def _extract_benefits(self, text: str) -> List[str]:
        """Extract benefits and perks."""
        benefits = []
        
        benefit_patterns = [
            r'(?:benefits|perks|what we offer)[:]\s*([^.]*)',
            r'(?:compensation|salary|package)[:]\s*([^.]*)',
        ]
        
        for pattern in benefit_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                benefit_items = re.split(r'[•\-*]\s*|\n\s*', match)
                for item in benefit_items:
                    item = item.strip()
                    if len(item) > 5:
                        benefits.append(item)
        
        return benefits
    
    def _extract_location(self, text: str) -> str:
        """Extract job location."""
        location_patterns = [
            r'(?:location|based in|office in)[:]\s*([^.\n]+)',
            r'(?:remote|work from home|wfh)',
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip() if match.groups() else match.group(0)
        
        return "Not specified"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from the job description."""
        # Use NLTK for keyword extraction
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Filter out stop words and short words
        keywords = [word for word in tokens if word.isalnum() and 
                   len(word) > 3 and word not in stop_words]
        
        # Get word frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:20]]
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy."""
        entities = {}
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                entity_type = ent.label_
                if entity_type not in entities:
                    entities[entity_type] = []
                entities[entity_type].append(ent.text)
        
        return entities
    
    def _load_skill_keywords(self) -> List[str]:
        """Load predefined skill keywords."""
        return [
            # Programming Languages
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'Ruby', 'PHP', 'Go', 'Rust',
            'Scala', 'Kotlin', 'Swift', 'R', 'MATLAB', 'SQL',
            
            # Web Technologies
            'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'Django', 'Flask',
            'Spring Boot', 'Laravel', 'WordPress', 'jQuery',
            
            # Databases
            'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'Oracle',
            'SQLite', 'DynamoDB',
            
            # Cloud and DevOps
            'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'CI/CD',
            'Terraform', 'Ansible',
            
            # Data Science/ML
            'Machine Learning', 'Deep Learning', 'TensorFlow', 'PyTorch', 'Scikit-learn',
            'Pandas', 'NumPy', 'Data Analysis', 'Statistics',
            
            # Other Technologies
            'Linux', 'Windows', 'API', 'REST', 'GraphQL', 'Microservices', 'Agile', 'Scrum'
        ]
    
    def _load_qualification_keywords(self) -> List[str]:
        """Load predefined qualification keywords."""
        return [
            'Bachelor', 'Master', 'PhD', 'Doctorate', 'B.Tech', 'M.Tech', 'MBA', 'MS', 'BS',
            'Computer Science', 'Software Engineering', 'Information Technology',
            'AWS Certified', 'Azure Certified', 'Google Cloud Certified', 'PMP', 'Scrum Master'
        ]


# Example usage
if __name__ == "__main__":
    parser = JobDescriptionParser()
    
    # Test with sample job description
    sample_jd = """
    Job Title: Senior Python Developer
    
    We are looking for a Senior Python Developer to join our team.
    
    Requirements:
    - Bachelor's degree in Computer Science
    - 5+ years of experience in Python development
    - Must have: Python, Django, PostgreSQL, AWS
    - Good to have: React, Docker, Kubernetes
    
    Responsibilities:
    - Develop and maintain web applications
    - Work with cross-functional teams
    - Write clean, maintainable code
    """
    
    result = parser.parse_job_description(sample_jd)
    print(f"Job Title: {result['job_title']}")
    print(f"Must-have skills: {result['must_have_skills']}")
    print(f"Good-to-have skills: {result['good_to_have_skills']}")