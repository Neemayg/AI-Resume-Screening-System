"""
Semantic similarity scorer using sentence embeddings.
Provides contextual understanding beyond keyword matching.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available. Using fallback similarity.")


class SemanticScorer:
    """Calculate semantic similarity using embeddings."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize scorer with embedding model."""
        self.model = None
        self.model_name = model_name
        self._embedding_cache = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded semantic model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load semantic model: {e}")
                self.model = None
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Returns:
            Similarity score 0-1
        """
        if not text1 or not text2:
            return 0.0
        
        if self.model is None:
            return self._fallback_similarity(text1, text2)
        
        try:
            # Get embeddings
            emb1 = self.model.encode(text1, convert_to_numpy=True)
            emb2 = self.model.encode(text2, convert_to_numpy=True)
            
            # Cosine similarity
            similarity = self._cosine_similarity(emb1, emb2)
            
            # Normalize to 0-1
            return (similarity + 1) / 2
        except Exception as e:
            logger.error(f"Semantic similarity failed: {e}")
            return self._fallback_similarity(text1, text2)
    
    def score_resume_for_jd(
        self,
        jd_text: str,
        resume_text: str
    ) -> Dict:
        """
        Calculate comprehensive semantic score.
        
        Returns:
            {
                "overall_score": 0.75,
                "explanation": "..."
            }
        """
        score = self.calculate_similarity(jd_text, resume_text)
        
        if score >= 0.7:
            explanation = "Strong semantic alignment"
        elif score >= 0.5:
            explanation = "Moderate semantic alignment"
        elif score >= 0.3:
            explanation = "Weak semantic alignment"
        else:
            explanation = "Low semantic alignment"
        
        return {
            "overall_score": score,
            "explanation": explanation
        }
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """Fallback word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are'}
        
        words1 -= stop_words
        words2 -= stop_words
        
        if not words1 or not words2:
            return 0.5
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._embedding_cache.clear()


# Singleton
semantic_scorer = SemanticScorer()
