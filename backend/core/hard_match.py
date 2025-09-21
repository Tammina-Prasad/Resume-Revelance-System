"""
Hard Match Engine Module
Performs keyword and skill matching between resumes and job descriptions using 
exact and fuzzy matching techniques.
"""

import re
import logging
from typing import Dict, List, Tuple, Any, Set
from fuzzywuzzy import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HardMatchEngine:
    """
    A class to perform hard matching between resume and job description.
    Uses exact matching, fuzzy matching, and TF-IDF similarity.
    """
    
    def __init__(self, fuzzy_threshold: int = 80):
        """
        Initialize the Hard Match Engine.
        
        Args:
            fuzzy_threshold (int): Minimum similarity score for fuzzy matching (0-100)
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000
        )
    
    def calculate_hard_match_score(self, resume_text: str, jd_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate hard match score between resume and job description.
        
        Args:
            resume_text (str): Cleaned resume text
            jd_data (Dict[str, Any]): Parsed job description data
            
        Returns:
            Dict[str, Any]: Hard match analysis results
        """
        try:
            # Extract skills and keywords from resume
            resume_skills = self._extract_skills_from_resume(resume_text)
            resume_keywords = self._extract_keywords_from_text(resume_text)
            
            # Get JD requirements
            must_have_skills = jd_data.get('must_have_skills', [])
            good_to_have_skills = jd_data.get('good_to_have_skills', [])
            qualifications = jd_data.get('qualifications', [])
            jd_keywords = jd_data.get('keywords', [])
            
            # Perform matching
            must_have_matches = self._match_skills(resume_skills, must_have_skills)
            good_to_have_matches = self._match_skills(resume_skills, good_to_have_skills)
            qualification_matches = self._match_qualifications(resume_text, qualifications)
            keyword_matches = self._match_keywords(resume_keywords, jd_keywords)
            
            # Calculate scores
            must_have_score = self._calculate_skill_score(must_have_matches, must_have_skills)
            good_to_have_score = self._calculate_skill_score(good_to_have_matches, good_to_have_skills)
            qualification_score = self._calculate_qualification_score(qualification_matches, qualifications)
            keyword_score = self._calculate_keyword_score(keyword_matches, jd_keywords)
            
            # Calculate overall TF-IDF similarity
            tfidf_score = self._calculate_tfidf_similarity(resume_text, jd_data.get('cleaned_text', ''))
            
            # Weighted overall score
            overall_score = self._calculate_weighted_score({
                'must_have': must_have_score,
                'good_to_have': good_to_have_score,
                'qualifications': qualification_score,
                'keywords': keyword_score,
                'tfidf': tfidf_score
            })
            
            # Compile results
            results = {
                'overall_score': round(overall_score, 2),
                'component_scores': {
                    'must_have_skills': round(must_have_score, 2),
                    'good_to_have_skills': round(good_to_have_score, 2),
                    'qualifications': round(qualification_score, 2),
                    'keywords': round(keyword_score, 2),
                    'tfidf_similarity': round(tfidf_score, 2)
                },
                'matched_skills': {
                    'must_have': must_have_matches,
                    'good_to_have': good_to_have_matches
                },
                'matched_qualifications': qualification_matches,
                'matched_keywords': keyword_matches,
                'missing_skills': {
                    'must_have': [skill for skill in must_have_skills 
                                if not any(match['jd_skill'].lower() == skill.lower() 
                                         for match in must_have_matches)],
                    'good_to_have': [skill for skill in good_to_have_skills 
                                   if not any(match['jd_skill'].lower() == skill.lower() 
                                            for match in good_to_have_matches)]
                },
                'missing_qualifications': [qual for qual in qualifications 
                                         if not any(match['jd_qualification'].lower() in qual.lower() 
                                                  for match in qualification_matches)]
            }
            
            logger.info(f"Hard match analysis completed. Overall score: {overall_score}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hard match calculation: {str(e)}")
            raise
    
    def _extract_skills_from_resume(self, resume_text: str) -> List[str]:
        """Extract skills from resume text."""
        # Predefined skill lists
        technical_skills = [
            # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 'go', 'rust',
            'scala', 'kotlin', 'swift', 'r', 'matlab', 'sql', 'perl', 'bash', 'powershell',
            
            # Web Technologies
            'html', 'css', 'react', 'angular', 'vue.js', 'vue', 'node.js', 'nodejs', 'express',
            'django', 'flask', 'spring', 'spring boot', 'laravel', 'wordpress', 'jquery',
            
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'oracle',
            'sqlite', 'dynamodb', 'neo4j', 'couchdb',
            
            # Cloud and DevOps
            'aws', 'azure', 'google cloud', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
            'ci/cd', 'terraform', 'ansible', 'vagrant', 'puppet', 'chef',
            
            # Data Science/ML
            'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'data analysis',
            'statistics', 'data visualization', 'tableau', 'power bi',
            
            # Other Technologies
            'linux', 'windows', 'api', 'rest', 'restful', 'graphql', 'microservices',
            'agile', 'scrum', 'kanban', 'jira', 'confluence'
        ]
        
        found_skills = []
        resume_lower = resume_text.lower()
        
        for skill in technical_skills:
            if re.search(r'\b' + re.escape(skill.lower()) + r'\b', resume_lower):
                found_skills.append(skill)
        
        return found_skills
    
    def _extract_keywords_from_text(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Remove common words and extract meaningful terms
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        common_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was',
            'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now',
            'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she',
            'too', 'use', 'work', 'experience', 'year', 'years', 'university', 'college'
        }
        
        keywords = [word for word in words if word not in common_words and len(word) > 3]
        
        # Get unique keywords with frequency
        keyword_freq = {}
        for keyword in keywords:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:50]]
    
    def _match_skills(self, resume_skills: List[str], jd_skills: List[str]) -> List[Dict[str, Any]]:
        """Match skills between resume and job description."""
        matches = []
        
        for jd_skill in jd_skills:
            best_match = None
            best_score = 0
            match_type = 'none'
            
            # Exact match
            for resume_skill in resume_skills:
                if jd_skill.lower() == resume_skill.lower():
                    best_match = resume_skill
                    best_score = 100
                    match_type = 'exact'
                    break
            
            # Fuzzy match if no exact match found
            if best_match is None:
                for resume_skill in resume_skills:
                    score = fuzz.ratio(jd_skill.lower(), resume_skill.lower())
                    if score >= self.fuzzy_threshold and score > best_score:
                        best_match = resume_skill
                        best_score = score
                        match_type = 'fuzzy'
            
            if best_match:
                matches.append({
                    'jd_skill': jd_skill,
                    'resume_skill': best_match,
                    'score': best_score,
                    'match_type': match_type
                })
        
        return matches
    
    def _match_qualifications(self, resume_text: str, jd_qualifications: List[str]) -> List[Dict[str, Any]]:
        """Match qualifications between resume and job description."""
        matches = []
        resume_lower = resume_text.lower()
        
        for jd_qual in jd_qualifications:
            # Check for direct mentions
            if jd_qual.lower() in resume_lower:
                matches.append({
                    'jd_qualification': jd_qual,
                    'match_type': 'exact',
                    'score': 100
                })
                continue
            
            # Check for partial matches
            qual_words = jd_qual.lower().split()
            matched_words = []
            for word in qual_words:
                if len(word) > 3 and word in resume_lower:
                    matched_words.append(word)
            
            if matched_words:
                score = (len(matched_words) / len(qual_words)) * 100
                if score >= 50:  # At least 50% of words matched
                    matches.append({
                        'jd_qualification': jd_qual,
                        'matched_words': matched_words,
                        'match_type': 'partial',
                        'score': score
                    })
        
        return matches
    
    def _match_keywords(self, resume_keywords: List[str], jd_keywords: List[str]) -> List[Dict[str, Any]]:
        """Match keywords between resume and job description."""
        matches = []
        
        for jd_keyword in jd_keywords:
            for resume_keyword in resume_keywords:
                # Exact match
                if jd_keyword.lower() == resume_keyword.lower():
                    matches.append({
                        'jd_keyword': jd_keyword,
                        'resume_keyword': resume_keyword,
                        'score': 100,
                        'match_type': 'exact'
                    })
                    break
                
                # Fuzzy match
                score = fuzz.ratio(jd_keyword.lower(), resume_keyword.lower())
                if score >= self.fuzzy_threshold:
                    matches.append({
                        'jd_keyword': jd_keyword,
                        'resume_keyword': resume_keyword,
                        'score': score,
                        'match_type': 'fuzzy'
                    })
                    break
        
        return matches
    
    def _calculate_skill_score(self, matches: List[Dict[str, Any]], total_skills: List[str]) -> float:
        """Calculate skill matching score."""
        if not total_skills:
            return 0.0
        
        total_score = 0
        for match in matches:
            # Weight exact matches higher than fuzzy matches
            weight = 1.0 if match['match_type'] == 'exact' else 0.8
            total_score += (match['score'] / 100) * weight
        
        return (total_score / len(total_skills)) * 100
    
    def _calculate_qualification_score(self, matches: List[Dict[str, Any]], total_qualifications: List[str]) -> float:
        """Calculate qualification matching score."""
        if not total_qualifications:
            return 0.0
        
        total_score = 0
        for match in matches:
            weight = 1.0 if match['match_type'] == 'exact' else 0.7
            total_score += (match['score'] / 100) * weight
        
        return (total_score / len(total_qualifications)) * 100
    
    def _calculate_keyword_score(self, matches: List[Dict[str, Any]], total_keywords: List[str]) -> float:
        """Calculate keyword matching score."""
        if not total_keywords:
            return 0.0
        
        unique_matched_keywords = set()
        total_score = 0
        
        for match in matches:
            if match['jd_keyword'] not in unique_matched_keywords:
                unique_matched_keywords.add(match['jd_keyword'])
                weight = 1.0 if match['match_type'] == 'exact' else 0.8
                total_score += (match['score'] / 100) * weight
        
        # Use min to avoid scores over 100%
        return min((total_score / len(total_keywords)) * 100, 100.0)
    
    def _calculate_tfidf_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate TF-IDF similarity between resume and job description."""
        try:
            if not resume_text or not jd_text:
                return 0.0
            
            # Prepare documents
            documents = [resume_text, jd_text]
            
            # Fit and transform
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Return similarity between resume and JD (scale to 0-100)
            return similarity_matrix[0, 1] * 100
            
        except Exception as e:
            logger.warning(f"TF-IDF calculation failed: {str(e)}")
            return 0.0
    
    def _calculate_weighted_score(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted overall score."""
        weights = {
            'must_have': 0.35,      # Most important
            'good_to_have': 0.15,   # Less critical
            'qualifications': 0.25,  # Important for role fit
            'keywords': 0.15,       # General relevance
            'tfidf': 0.10          # Overall similarity
        }
        
        weighted_score = 0
        for component, score in component_scores.items():
            if component in weights:
                weighted_score += score * weights[component]
        
        return min(weighted_score, 100.0)  # Cap at 100


# Utility functions for advanced matching
def calculate_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    return intersection / union if union > 0 else 0.0


def extract_n_grams(text: str, n: int = 2) -> Set[str]:
    """Extract n-grams from text."""
    words = text.lower().split()
    n_grams = set()
    
    for i in range(len(words) - n + 1):
        n_gram = ' '.join(words[i:i + n])
        n_grams.add(n_gram)
    
    return n_grams


# Example usage
if __name__ == "__main__":
    # Sample data for testing
    engine = HardMatchEngine()
    
    sample_resume = """
    John Doe
    Software Engineer with 5 years of experience in Python, Django, and AWS.
    Bachelor's degree in Computer Science.
    Experience with machine learning and data analysis.
    """
    
    sample_jd_data = {
        'must_have_skills': ['Python', 'Django', 'AWS'],
        'good_to_have_skills': ['Machine Learning', 'Docker'],
        'qualifications': ['Bachelor in Computer Science'],
        'keywords': ['software', 'engineer', 'python', 'aws'],
        'cleaned_text': 'Looking for a software engineer with Python and AWS experience'
    }
    
    result = engine.calculate_hard_match_score(sample_resume, sample_jd_data)
    print(f"Overall Hard Match Score: {result['overall_score']}%")
    print(f"Component Scores: {result['component_scores']}")
    print(f"Matched Skills: {result['matched_skills']}")