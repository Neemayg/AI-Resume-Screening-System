"""
Text normalization module for preprocessing JDs and Resumes.
Handles cleaning, standardization, and preparation for NLP.
"""

import re
import unicodedata
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)

class TextNormalizer:
    """Normalize and clean text for consistent processing."""
    
    # Common abbreviation expansions
    ABBREVIATIONS = {
        "yrs": "years", "yr": "year", "exp": "experience",
        "sr": "senior", "jr": "junior", "dev": "developer",
        "eng": "engineer", "ml": "machine learning",
        "ai": "artificial intelligence", "dl": "deep learning",
        "nlp": "natural language processing",
        "db": "database", "api": "application programming interface",
        "aws": "amazon web services", "gcp": "google cloud platform",
        "k8s": "kubernetes", "sql": "structured query language",
    }
    
    def normalize_text(self, text: str, preserve_structure: bool = False) -> str:
        """Main normalization function."""
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
        
        # Normalize whitespace
        if preserve_structure:
            paragraphs = text.split('\n\n')
            paragraphs = [' '.join(p.split()) for p in paragraphs]
            text = '\n\n'.join(p for p in paragraphs if p.strip())
        else:
            text = ' '.join(text.split())
        
        # Expand abbreviations
        text = self._expand_abbreviations(text)
        
        # Normalize punctuation
        text = self._normalize_punctuation(text)
        
        return text.strip()
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        words = text.split()
        expanded = []
        
        for word in words:
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
        text = re.sub(r'[–—−]', '-', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        text = re.sub(r'^[\s]*[•●○◦▪▸►]\s*', '', text, flags=re.MULTILINE)
        return text
    
    def extract_years_of_experience(self, text: str) -> List[Tuple[int, int]]:
        """Extract years of experience mentions."""
        patterns = [
            r'(\d+)\s*\+?\s*(?:years?|yrs?)',
            r'(\d+)\s*[-–to]+\s*(\d+)\s*(?:years?|yrs?)',
        ]
        
        results = []
        text_lower = text.lower()
        
        for pattern in patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                if len(groups) == 1:
                    years = int(groups[0])
                    results.append((years, years + 3))
                elif len(groups) == 2 and groups[1]:
                    results.append((int(groups[0]), int(groups[1])))
        
        return results


# Singleton instance
text_normalizer = TextNormalizer()
