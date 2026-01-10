"""
Semantic similarity scorer using sentence embeddings.
Provides contextual understanding beyond keyword matching.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import sentence-transformers, provide fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Using fallback similarity.")

class SemanticScorer:
    """
    Calculate semantic similarity between JD and Resume sections.
    Uses sentence embeddings for contextual understanding.
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic scorer.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                        'all-MiniLM-L6-v2' is fast and effective.
                        'all-mpnet-base-v2' is more accurate but slower.
        """
        self.model = None
        self.model_name = model_name
        self._embedding_cache = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        use_cache: bool = True
    ) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Args:
            text1: First text (e.g., JD requirements)
            text2: Second text (e.g., Resume experience)
            use_cache: Whether to cache embeddings
            
        Returns:
            Similarity score 0-1
        """
        if not text1 or not text2:
            return 0.0
        
        if self.model is None:
            return self._fallback_similarity(text1, text2)
        
        # Get embeddings
        emb1 = self._get_embedding(text1, use_cache)
        emb2 = self._get_embedding(text2, use_cache)
        
        # Calculate cosine similarity
        similarity = self._cosine_similarity(emb1, emb2)
        
        # Normalize to 0-1 range (cosine can be -1 to 1)
        return (similarity + 1) / 2
    
    def calculate_section_similarities(
        self,
        jd_sections: Dict[str, str],
        resume_sections: Dict[str, str]
    ) -> Dict[str, float]:
        """
        Calculate similarity scores between corresponding sections.
        
        Args:
            jd_sections: Dictionary of JD sections
                {"requirements": "...", "responsibilities": "...", ...}
            resume_sections: Dictionary of Resume sections
                {"experience": "...", "skills": "...", ...}
                
        Returns:
            Dictionary of section-to-section similarity scores
        """
        results = {}
        
        # Define which sections to compare
        comparisons = [
            ('requirements', 'experience', 'requirements_to_experience'),
            ('requirements', 'skills', 'requirements_to_skills'),
            ('responsibilities', 'experience', 'responsibilities_to_experience'),
            ('skills_required', 'skills', 'skills_alignment'),
        ]
        
        for jd_section, resume_section, result_key in comparisons:
            jd_text = jd_sections.get(jd_section, '')
            resume_text = resume_sections.get(resume_section, '')
            
            if jd_text and resume_text:
                results[result_key] = self.calculate_similarity(jd_text, resume_text)
            else:
                results[result_key] = 0.5  # Neutral if section missing
        
        # Overall similarity
        jd_full = ' '.join(str(v) for v in jd_sections.values() if v)
        resume_full = ' '.join(str(v) for v in resume_sections.values() if v)
        results['overall'] = self.calculate_similarity(jd_full, resume_full)
        
        return results
    
    def score_resume_for_jd(
        self,
        jd_text: str,
        resume_text: str,
        jd_sections: Optional[Dict] = None,
        resume_sections: Optional[Dict] = None
    ) -> Dict:
        """
        Calculate comprehensive semantic score for resume-JD match.
        
        Returns:
            {
                "overall_score": 0.75,
                "section_scores": {...},
                "key_phrase_matches": [...],
                "explanation": "..."
            }
        """
        result = {
            "overall_score": 0.0,
            "section_scores": {},
            "key_phrase_matches": [],
            "explanation": ""
        }
        
        # Overall similarity
        result["overall_score"] = self.calculate_similarity(jd_text, resume_text)
        
        # Section-wise similarities if available
        if jd_sections and resume_sections:
            result["section_scores"] = self.calculate_section_similarities(
                jd_sections, resume_sections
            )
        
        # Find key phrase matches
        result["key_phrase_matches"] = self._find_key_phrase_matches(
            jd_text, resume_text
        )
        
        # Generate explanation
        score = result["overall_score"]
        if score >= 0.7:
            result["explanation"] = "Strong semantic alignment between resume and JD"
        elif score >= 0.5:
            result["explanation"] = "Moderate semantic alignment"
        elif score >= 0.3:
            result["explanation"] = "Weak semantic alignment, may need review"
        else:
            result["explanation"] = "Low semantic alignment, likely not a good match"
        
        return result
    
    def _get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Get or compute embedding for text."""
        if use_cache and text in self._embedding_cache:
            return self._embedding_cache[text]
        
        embedding = self.model.encode(text, convert_to_numpy=True)
        
        if use_cache:
            # Limit cache size
            if len(self._embedding_cache) > 1000:
                # Remove oldest entries
                keys = list(self._embedding_cache.keys())[:500]
                for key in keys:
                    del self._embedding_cache[key]
            
            self._embedding_cache[text] = embedding
        
        return embedding
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback similarity calculation when sentence-transformers unavailable.
        Uses word overlap with TF-IDF-like weighting.
        """
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                      'does', 'did', 'will', 'would', 'could', 'should', 'may',
                      'might', 'must', 'shall', 'can', 'need', 'this', 'that',
                      'these', 'those', 'i', 'you', 'we', 'they', 'it', 'its'}
        
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.5
        
        # Jaccard similarity
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _find_key_phrase_matches(
        self,
        jd_text: str,
        resume_text: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Find key phrases from JD that appear in resume.
        
        Returns:
            List of matching phrases with context
        """
        matches = []
        
        # Extract potential key phrases from JD (simple approach)
        # In production, use proper keyphrase extraction
        jd_lower = jd_text.lower()
        resume_lower = resume_text.lower()
        
        # Look for multi-word phrases
        words = jd_text.split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3]).lower()
            # Clean phrase
            phrase = ''.join(c for c in phrase if c.isalnum() or c.isspace())
            
            if len(phrase) > 10 and phrase in resume_lower:
                matches.append({
                    "phrase": phrase,
                    "type": "exact_match"
                })
        
        # Limit and deduplicate
        seen = set()
        unique_matches = []
        for m in matches:
            if m["phrase"] not in seen:
                seen.add(m["phrase"])
                unique_matches.append(m)
                if len(unique_matches) >= top_k:
                    break
        
        return unique_matches
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()


# Singleton instance
semantic_scorer = SemanticScorer()
