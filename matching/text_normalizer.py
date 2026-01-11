"""
Text normalization module for preprocessing JDs and Resumes.
Handles cleaning, standardization, and preparation for NLP.
"""

import re
import unicodedata
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Normalize and clean text for consistent processing."""
    
    # Common abbreviation expansions
    ABBREVIATIONS = {
        "yrs": "years",
        "yr": "year",
        "exp": "experience",
        "mgmt": "management",
        "mgr": "manager",
        "sr": "senior",
        "jr": "junior",
        "dev": "developer",
        "eng": "engineer",
        "swe": "software engineer",
        "sde": "software development engineer",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "db": "database",
        "dba": "database administrator",
        "ui": "user interface",
        "ux": "user experience",
        "qa": "quality assurance",
        "devops": "development operations",
        "ci/cd": "continuous integration continuous deployment",
        "api": "application programming interface",
        "sdk": "software development kit",
        "aws": "amazon web services",
        "gcp": "google cloud platform",
        "k8s": "kubernetes",
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "rb": "ruby",
        "sql": "structured query language",
        "nosql": "non relational database",
        "oop": "object oriented programming",
        "fp": "functional programming",
        "tdd": "test driven development",
        "bdd": "behavior driven development",
        "agile": "agile methodology",
        "scrum": "scrum methodology",
        "pm": "project manager",
        "po": "product owner",
        "ba": "business analyst",
        "hr": "human resources",
        "b2b": "business to business",
        "b2c": "business to consumer",
        "saas": "software as a service",
        "paas": "platform as a service",
        "iaas": "infrastructure as a service",
        "roi": "return on investment",
        "kpi": "key performance indicator",
        "etc": "",
        "e.g.": "for example",
        "i.e.": "that is",
    }
    
    # Technical terms that should not be lowercased (preserve case for matching)
    PRESERVE_CASE_TERMS = {
        "JavaScript", "TypeScript", "Python", "Java", "C++", "C#",
        "React", "Angular", "Vue", "Node.js", "Express",
        "PostgreSQL", "MySQL", "MongoDB", "Redis", "Elasticsearch",
        "Docker", "Kubernetes", "AWS", "GCP", "Azure",
        "TensorFlow", "PyTorch", "Keras", "Scikit-learn",
        "GraphQL", "REST", "gRPC", "WebSocket",
        "Git", "GitHub", "GitLab", "Bitbucket",
        "Linux", "Windows", "macOS", "Ubuntu",
        "Jira", "Confluence", "Slack", "Trello"
    }
    
    def __init__(self):
        self.preserve_case_pattern = self._build_preserve_pattern()
    
    def _build_preserve_pattern(self) -> re.Pattern:
        """Build regex pattern for terms to preserve."""
        terms = sorted(self.PRESERVE_CASE_TERMS, key=len, reverse=True)
        pattern = '|'.join(re.escape(term) for term in terms)
        return re.compile(f'({pattern})', re.IGNORECASE)
    
    def normalize_text(self, text: str, preserve_structure: bool = False) -> str:
        """
        Main normalization function.
        
        Args:
            text: Raw input text
            preserve_structure: If True, keeps paragraph breaks
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Step 1: Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Step 2: Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Step 3: Normalize whitespace
        if preserve_structure:
            # Keep paragraph structure
            paragraphs = text.split('\n\n')
            paragraphs = [' '.join(p.split()) for p in paragraphs]
            text = '\n\n'.join(p for p in paragraphs if p.strip())
        else:
            text = ' '.join(text.split())
        
        # Step 4: Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Step 5: Normalize punctuation
        text = self._normalize_punctuation(text)
        
        # Step 6: Handle numbers and dates
        text = self._normalize_numbers(text)
        
        return text.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        words = text.split()
        expanded = []
        
        for word in words:
            # Check lowercase version
            word_lower = word.lower().strip('.,;:!?()')
            if word_lower in self.ABBREVIATIONS:
                expansion = self.ABBREVIATIONS[word_lower]
                if expansion:
                    expanded.append(expansion)
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def _normalize_punctuation(self, text: str) -> str:
        """Normalize punctuation marks."""
        # Replace various dash types with standard hyphen
        text = re.sub(r'[–—−]', '-', text)
        
        # Replace multiple punctuation with single
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Standardize quotes
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        # Remove bullet points and list markers
        text = re.sub(r'^[\s]*[•●○◦▪▸►◆★☆✓✔✗✘→⇒➔➜]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*[\d]+[.)]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[\s]*[a-zA-Z][.)]\s*', '', text, flags=re.MULTILINE)
        
        return text
    
    def _normalize_numbers(self, text: str) -> str:
        """Normalize number formats."""
        # Convert "5+" to "5 or more"
        text = re.sub(r'(\d+)\+', r'\1 or more', text)
        
        # Convert "3-5 years" to standard format
        text = re.sub(r'(\d+)\s*[-–—to]+\s*(\d+)\s*(years?)', r'\1 to \2 \3', text)
        
        return text
    
    def extract_years_of_experience(self, text: str) -> List[Tuple[int, int]]:
        """
        Extract years of experience mentions.
        
        Returns:
            List of (min_years, max_years) tuples found
        """
        patterns = [
            # "5+ years", "5 years+"
            r'(\d+)\s*\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)?',
            # "3-5 years"
            r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)',
            # "minimum 3 years"
            r'(?:minimum|min|at least)\s+(\d+)\s*(?:years?|yrs?)',
            # "3 years minimum"
            r'(\d+)\s*(?:years?|yrs?)\s+(?:minimum|min)',
        ]
        
        results = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                if len(groups) == 1:
                    years = int(groups[0])
                    results.append((years, years + 3))  # Assume range for single value
                elif len(groups) == 2 and groups[1]:
                    results.append((int(groups[0]), int(groups[1])))
        
        return results
    
    def tokenize_for_skills(self, text: str) -> List[str]:
        """
        Tokenize text specifically for skill extraction.
        Preserves multi-word skills and technical terms.
        """
        # First, protect known technical terms
        protected = {}
        counter = 0
        
        for term in self.PRESERVE_CASE_TERMS:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            if pattern.search(text):
                placeholder = f"__PROTECTED_{counter}__"
                protected[placeholder] = term
                text = pattern.sub(placeholder, text)
                counter += 1
        
        # Split on common delimiters
        tokens = re.split(r'[,;/\|]|\band\b|\bor\b', text)
        
        # Clean each token
        cleaned_tokens = []
        for token in tokens:
            token = token.strip()
            # Restore protected terms
            for placeholder, original in protected.items():
                token = token.replace(placeholder, original)
            
            if token and len(token) > 1:
                cleaned_tokens.append(token)
        
        return cleaned_tokens


# Singleton instance
text_normalizer = TextNormalizer()
