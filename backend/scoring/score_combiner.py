"""
Score Combiner Module
Combines all individual scores into a final weighted score
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json


@dataclass
class ScoreComponents:
    """All individual score components for a resume"""
    skill_match_score: float = 0.0
    core_skill_score: float = 0.0
    optional_skill_score: float = 0.0
    experience_score: float = 0.0
    education_score: float = 0.0
    semantic_similarity_score: float = 0.0
    section_match_scores: Dict[str, float] = field(default_factory=dict)
    keyword_density_score: float = 0.0
    role_alignment_score: float = 0.0
    
    # Penalty scores (negative impact)
    missing_core_skills_penalty: float = 0.0
    experience_gap_penalty: float = 0.0
    education_mismatch_penalty: float = 0.0
    
    # Bonus scores (positive impact)
    exact_role_match_bonus: float = 0.0
    certification_bonus: float = 0.0
    extra_relevant_skills_bonus: float = 0.0


@dataclass
class WeightConfiguration:
    """Configurable weights for score combination"""
    # Primary weights (should sum to ~1.0 for base score)
    skill_match_weight: float = 0.30
    core_skill_weight: float = 0.20
    experience_weight: float = 0.20
    semantic_similarity_weight: float = 0.15
    role_alignment_weight: float = 0.10
    education_weight: float = 0.05
    
    # Penalty multipliers
    missing_core_skill_penalty_weight: float = 0.15
    experience_gap_penalty_weight: float = 0.10
    education_mismatch_penalty_weight: float = 0.05
    
    # Bonus multipliers
    exact_role_bonus_weight: float = 0.05
    certification_bonus_weight: float = 0.03
    extra_skills_bonus_weight: float = 0.02
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'WeightConfiguration':
        """Load weight configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config = json.load(f)
            return cls(**config)
        except FileNotFoundError:
            return cls()  # Return defaults
    
    def save_to_file(self, filepath: str):
        """Save current configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(self.__dict__, f, indent=2)


class ScoreCombiner:
    """
    Combines multiple scoring signals into a final score
    Uses weighted combination with penalties and bonuses
    """
    
    def __init__(self, weight_config: Optional[WeightConfiguration] = None):
        self.weights = weight_config or WeightConfiguration()
    
    def calculate_base_score(self, components: ScoreComponents) -> float:
        """
        Calculate the base weighted score from positive signals
        """
        base_score = (
            components.skill_match_score * self.weights.skill_match_weight +
            components.core_skill_score * self.weights.core_skill_weight +
            components.experience_score * self.weights.experience_weight +
            components.semantic_similarity_score * self.weights.semantic_similarity_weight +
            components.role_alignment_score * self.weights.role_alignment_weight +
            components.education_score * self.weights.education_weight
        )
        return min(base_score, 1.0)  # Cap at 1.0
    
    def calculate_penalties(self, components: ScoreComponents) -> float:
        """
        Calculate total penalty to subtract from base score
        """
        total_penalty = (
            components.missing_core_skills_penalty * self.weights.missing_core_skill_penalty_weight +
            components.experience_gap_penalty * self.weights.experience_gap_penalty_weight +
            components.education_mismatch_penalty * self.weights.education_mismatch_penalty_weight
        )
        return min(total_penalty, 0.5)  # Cap penalty at 50%
    
    def calculate_bonuses(self, components: ScoreComponents) -> float:
        """
        Calculate total bonus to add to base score
        """
        total_bonus = (
            components.exact_role_match_bonus * self.weights.exact_role_bonus_weight +
            components.certification_bonus * self.weights.certification_bonus_weight +
            components.extra_relevant_skills_bonus * self.weights.extra_skills_bonus_weight
        )
        return min(total_bonus, 0.15)  # Cap bonus at 15%
    
    def calculate_final_score(self, components: ScoreComponents) -> Dict:
        """
        Calculate final score with full breakdown
        
        Returns:
            Dict containing final_score and complete breakdown
        """
        base_score = self.calculate_base_score(components)
        penalties = self.calculate_penalties(components)
        bonuses = self.calculate_bonuses(components)
        
        # Final calculation
        final_score = base_score - penalties + bonuses
        
        # Normalize to 0-100 scale
        final_score_normalized = max(0, min(100, final_score * 100))
        
        return {
            'final_score': round(final_score_normalized, 2),
            'breakdown': {
                'base_score': round(base_score * 100, 2),
                'penalties': round(penalties * 100, 2),
                'bonuses': round(bonuses * 100, 2),
                'component_scores': {
                    'skill_match': round(components.skill_match_score * 100, 2),
                    'core_skills': round(components.core_skill_score * 100, 2),
                    'experience': round(components.experience_score * 100, 2),
                    'semantic_similarity': round(components.semantic_similarity_score * 100, 2),
                    'role_alignment': round(components.role_alignment_score * 100, 2),
                    'education': round(components.education_score * 100, 2)
                },
                'penalty_breakdown': {
                    'missing_core_skills': round(components.missing_core_skills_penalty * 100, 2),
                    'experience_gap': round(components.experience_gap_penalty * 100, 2),
                    'education_mismatch': round(components.education_mismatch_penalty * 100, 2)
                },
                'bonus_breakdown': {
                    'exact_role_match': round(components.exact_role_match_bonus * 100, 2),
                    'certifications': round(components.certification_bonus * 100, 2),
                    'extra_skills': round(components.extra_relevant_skills_bonus * 100, 2)
                }
            }
        }
    
    def get_score_category(self, final_score: float) -> str:
        """
        Categorize the final score into human-readable categories
        """
        if final_score >= 85:
            return "Excellent Match"
        elif final_score >= 70:
            return "Strong Match"
        elif final_score >= 55:
            return "Good Match"
        elif final_score >= 40:
            return "Partial Match"
        elif final_score >= 25:
            return "Weak Match"
        else:
            return "Poor Match"
