"""
Simplified Score Combiner - Integrates with existing system
Adds weighted scoring to current NLP-based matching
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from config.weights_config import COMPONENT_WEIGHTS, SKILL_WEIGHTS, PENALTIES, BONUSES

@dataclass
class ScoreBreakdown:
    """Detailed score breakdown"""
    skill_match: float = 0.0
    experience_match: float = 0.0
    semantic_similarity: float = 0.0
    role_alignment: float = 0.0
    education_match: float = 0.0
    penalties: Dict[str, float] = field(default_factory=dict)
    bonuses: Dict[str, float] = field(default_factory=dict)
    final_score: float = 0.0
    
class WeightedScorer:
    """
    Applies weighted scoring to existing resume analysis results
    """
    
    def __init__(self):
        self.weights = COMPONENT_WEIGHTS
        self.skill_weights = SKILL_WEIGHTS
        self.penalties = PENALTIES
        self.bonuses = BONUSES
    
    def calculate_weighted_score(
        self,
        skill_match_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
        total_required_skills: int,
        experience_years: float = 0,
        required_experience: float = 0
    ) -> ScoreBreakdown:
        """
        Calculate weighted score from analysis results
        
        Args:
            skill_match_score: Raw TF-IDF similarity score (0-1)
            matched_skills: List of matched skills
            missing_skills: List of missing required skills
            total_required_skills: Total number of required skills
            experience_years: Candidate's experience in years
            required_experience: Required experience in years
            
        Returns:
            ScoreBreakdown with weighted final score
        """
        breakdown = ScoreBreakdown()
        
        # Skill match component (weighted)
        breakdown.skill_match = skill_match_score * 100
        
        # Calculate experience match
        if required_experience > 0:
            if experience_years >= required_experience:
                breakdown.experience_match = 100
            else:
                breakdown.experience_match = (experience_years / required_experience) * 100
        else:
            breakdown.experience_match = 80  # Default if not specified
        
        # Semantic similarity (from TF-IDF score)
        breakdown.semantic_similarity = skill_match_score * 100
        
        # Role alignment (simplified - based on skill match)
        breakdown.role_alignment = breakdown.skill_match
        
        # Education default
        breakdown.education_match = 75
        
        # Calculate penalties
        if missing_skills:
            core_skills_missing = len(missing_skills)
            penalty_per_skill = abs(self.penalties['missing_core_skill']) * 100
            total_penalty = min(
                core_skills_missing * penalty_per_skill,
                abs(self.penalties['missing_core_skill_max']) * 100
            )
            breakdown.penalties['missing_core_skills'] = total_penalty
        
        if experience_years < required_experience and required_experience > 0:
            exp_gap = required_experience - experience_years
            exp_penalty = min(
                exp_gap * abs(self.penalties['experience_gap_per_year']) * 100,
                abs(self.penalties['experience_gap_max']) * 100
            )
            breakdown.penalties['experience_gap'] = exp_penalty
        
        # Calculate bonuses
        if experience_years > required_experience and required_experience > 0:
            excess = min(experience_years - required_experience, 5)
            breakdown.bonuses['experience_bonus'] = excess * self.bonuses['exceeds_experience'] * 100
        
        extra_skills = max(0, len(matched_skills) - total_required_skills)
        if extra_skills > 0:
            breakdown.bonuses['extra_skills'] = min(
                extra_skills * self.bonuses['extra_relevant_skill'] * 100,
                self.bonuses['extra_skill_max'] * 100
            )
        
        # Calculate final weighted score
        base_score = (
            breakdown.skill_match * self.weights['skill_match_score'] +
            breakdown.experience_match * self.weights['experience_match_score'] +
            breakdown.semantic_similarity * self.weights['semantic_similarity_score'] +
            breakdown.role_alignment * self.weights['role_alignment_score'] +
            breakdown.education_match * self.weights['education_match_score']
        )
        
        # Apply penalties and bonuses
        total_penalties = sum(breakdown.penalties.values())
        total_bonuses = sum(breakdown.bonuses.values())
        
        breakdown.final_score = max(0, min(100, base_score - total_penalties + total_bonuses))
        
        return breakdown
    
    def get_score_category(self, score: float) -> str:
        """Categorize score into human-readable labels"""
        if score >= 85:
            return "Excellent Match"
        elif score >= 70:
            return "Strong Match"
        elif score >= 55:
            return "Good Match"
        elif score >= 40:
            return "Partial Match"
        elif score >= 25:
            return "Weak Match"
        else:
            return "Poor Match"
    
    def generate_recommendation(self, score: float, missing_core_skills: int) -> str:
        """Generate hiring recommendation"""
        if score >= 70 and missing_core_skills == 0:
            return "Recommended for interview"
        elif score >= 55:
            return "Consider for interview with reservations"
        else:
            return "Does not meet minimum requirements"


# Global instance
weighted_scorer = WeightedScorer()
