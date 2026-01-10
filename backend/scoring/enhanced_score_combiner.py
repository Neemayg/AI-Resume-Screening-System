"""
Enhanced Score Combiner - Combines all scoring signals
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class ScoreComponents:
    """All score components for a resume."""
    skill_match_score: float = 0.0
    experience_score: float = 0.0
    semantic_similarity_score: float = 0.0
    role_alignment_score: float = 0.0
    education_score: float = 0.0
    
    # Penalties
    missing_core_skills_penalty: float = 0.0
    experience_gap_penalty: float = 0.0
    
    # Bonuses
    extra_relevant_skills_bonus: float = 0.0


@dataclass
class ScoreResult:
    final_score: float
    breakdown: Dict

class EnhancedScoreCombiner:
    """Combines multiple scoring signals into final score."""
    
    def __init__(self, weights: Optional[Dict] = None):
        """Initialize with configurable weights."""
        self.weights = weights or {
            "skill_match_weight": 0.40,  # Increased
            "experience_weight": 0.10,
            "semantic_similarity_weight": 0.30, # Increased
            "role_alignment_weight": 0.10,
            "education_weight": 0.10
        }
    
    def combine_scores(self, 
                      tfidf_score: float, 
                      semantic_score: float, 
                      skill_match_data: object,
                      custom_weights: Optional[Dict] = None) -> ScoreResult:
        """
        Bridge method to combine scores from similarity engine.
        """
        # Create components from inputs
        components = ScoreComponents(
            skill_match_score=getattr(skill_match_data, 'score', 0.0),
            semantic_similarity_score=semantic_score if semantic_score > 0 else tfidf_score, # Fallback to TF-IDF if semantic missing
            experience_score=0.8, # Default assumption for now
            role_alignment_score=tfidf_score, # Use TF-IDF as proxy for role alignment
            education_score=0.8, # Default assumption
            
            # Extract penalties/bonuses from skill data
            missing_core_skills_penalty=0.0, # Handled in skill score already
            extra_relevant_skills_bonus=0.0
        )
        
        # Calculate
        result_dict = self.calculate_final_score(components)
        
        return ScoreResult(
            final_score=result_dict['final_score'],
            breakdown=result_dict['breakdown']
        )

    def calculate_final_score(self, components: ScoreComponents) -> Dict:
        """
        Calculate final score with breakdown.
        
        Returns:
            {
                "final_score": 75.5,
                "breakdown": {...}
            }
        """
        # Calculate base score
        base_score = (
            components.skill_match_score * self.weights["skill_match_weight"] +
            components.experience_score * self.weights["experience_weight"] +
            components.semantic_similarity_score * self.weights["semantic_similarity_weight"] +
            components.role_alignment_score * self.weights["role_alignment_weight"] +
            components.education_score * self.weights["education_weight"]
        )
        
        # Apply penalties (capped at 30%)
        penalties = min(0.3, 
            components.missing_core_skills_penalty +
            components.experience_gap_penalty
        )
        
        # Apply bonuses (capped at 15%)
        bonuses = min(0.15, components.extra_relevant_skills_bonus)
        
        # Final calculation
        final_score = base_score - penalties + bonuses
        final_score_normalized = max(0, min(100, final_score * 100))
        
        return {
            "final_score": round(final_score_normalized, 2),
            "breakdown": {
                "base_score": round(base_score * 100, 2),
                "penalties": round(penalties * 100, 2),
                "bonuses": round(bonuses * 100, 2),
                "components": {
                    "skill_match": round(components.skill_match_score * 100, 2),
                    "experience": round(components.experience_score * 100, 2),
                    "semantic": round(components.semantic_similarity_score * 100, 2),
                    "role_alignment": round(components.role_alignment_score * 100, 2),
                    "education": round(components.education_score * 100, 2)
                }
            }
        }
    
    def get_score_category(self, final_score: float) -> str:
        """Categorize score."""
        if final_score >= 85:
            return "Excellent Match"
        elif final_score >= 70:
            return "Strong Match"
        elif final_score >= 55:
            return "Good Match"
        elif final_score >= 40:
            return "Partial Match"
        else:
            return "Weak Match"
    
    def get_recommendation(self, final_score: float, missing_critical: int) -> str:
        """Generate recommendation."""
        if final_score >= 70 and missing_critical == 0:
            return "Recommended for interview"
        elif final_score >= 55:
            return "Consider for interview with reservations"
        else:
            return "Does not meet minimum requirements"


# Singleton instance match for import
score_combiner = EnhancedScoreCombiner()

