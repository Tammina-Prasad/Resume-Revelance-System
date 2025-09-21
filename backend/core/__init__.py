"""
Core module initialization file.
Provides easy imports for all core functionality.
"""

from .resume_parser import ResumeParser, extract_contact_info, extract_education_info
from .jd_parser import JobDescriptionParser
from .hard_match import HardMatchEngine
from .semantic_match import SemanticMatchEngine
from .relevance_engine import RelevanceAnalysisEngine, RelevanceVerdict, create_relevance_engine

__all__ = [
    'ResumeParser',
    'extract_contact_info',
    'extract_education_info',
    'JobDescriptionParser',
    'HardMatchEngine',
    'SemanticMatchEngine',
    'RelevanceAnalysisEngine',
    'RelevanceVerdict',
    'create_relevance_engine'
]

__version__ = "1.0.0"