"""
Ranking Module
Handles the ranking logic and result generation for resume screening.
"""

import random
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import SCORE_THRESHOLDS, SUMMARY_TEMPLATES
from models import ResumeResult, FitCategory

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Data class for holding candidate scoring data."""
    resume_id: str
    resume_name: str
    original_text: str
    preprocessed_text: str
    similarity_data: Dict


class RankingEngine:
    """
    Engine for ranking and scoring candidates based on similarity results.
    Generates final results with rankings, fit categories, and summaries.
    """
    
    def __init__(self):
        """Initialize the ranking engine."""
        logger.info("Ranking Engine initialized")
    
    def determine_fit_category(self, score: float) -> FitCategory:
        """
        Determine fit category based on score.
        
        Args:
            score: Match score (0-100)
            
        Returns:
            FitCategory enum value
        """
        if score >= SCORE_THRESHOLDS['high']:
            return FitCategory.HIGH
        elif score >= SCORE_THRESHOLDS['medium']:
            return FitCategory.MEDIUM
        else:
            return FitCategory.LOW
    
    def generate_summary(self, 
                         score: float, 
                         matched_skills: List[str], 
                         missing_skills: List[str],
                         skill_breakdown: Dict) -> str:
        """
        Generate a personalized summary for the candidate.
        
        Args:
            score: Match score
            matched_skills: List of matched skills
            missing_skills: List of missing skills
            skill_breakdown: Skill category breakdown
            
        Returns:
            Generated summary string
        """
        fit = self.determine_fit_category(score)
        
        # Select base template
        templates = SUMMARY_TEMPLATES.get(fit.value.lower(), SUMMARY_TEMPLATES['medium'])
        base_summary = random.choice(templates)
        
        # Add specific details
        details = []
        
        # Highlight strong categories
        strong_categories = [
            cat for cat, data in skill_breakdown.items()
            if data.get('percentage', 0) >= 70
        ]
        
        if strong_categories and fit != FitCategory.LOW:
            category_names = [cat.replace('_', ' ').title() for cat in strong_categories[:2]]
            details.append(f"Strong proficiency in {', '.join(category_names)}.")
        
        # Mention key matched skills
        if matched_skills and len(matched_skills) >= 3:
            top_skills = matched_skills[:3]
            details.append(f"Key skills include {', '.join(top_skills)}.")
        
        # Mention improvement areas for medium/low fits
        if fit in [FitCategory.MEDIUM, FitCategory.LOW] and missing_skills:
            if len(missing_skills) <= 3:
                details.append(f"May benefit from experience in {', '.join(missing_skills[:2])}.")
            else:
                details.append(f"Additional development in {missing_skills[0]} and {missing_skills[1]} would strengthen candidacy.")
        
        # Combine summary
        if details:
            return f"{base_summary} {' '.join(details)}"
        return base_summary
    
    def rank_candidates(self, 
                        candidates: List[CandidateScore],
                        job_title: Optional[str] = None) -> List[ResumeResult]:
        """
        Rank candidates based on their scores and generate final results.
        
        Args:
            candidates: List of CandidateScore objects
            job_title: Detected job title for context
            
        Returns:
            Sorted list of ResumeResult objects
        """
        results = []
        
        for candidate in candidates:
            sim_data = candidate.similarity_data
            
            # Get the final score
            final_score = sim_data.get('final_score', 0)
            
            # Determine fit category
            fit = self.determine_fit_category(final_score)
            
            # Generate summary
            summary = self.generate_summary(
                final_score,
                sim_data.get('matched_skills', []),
                sim_data.get('missing_skills', []),
                sim_data.get('skill_breakdown', {})
            )
            
            # Create result object
            result = ResumeResult(
                rank=1,  # ✅ Must be >= 1 for Pydantic validation (will be updated after sorting)
                id=candidate.resume_id,
                resume_name=candidate.resume_name,
                match_score=int(round(final_score)),
                fit=fit,
                matched_skills=sim_data.get('matched_skills', [])[:8],  # Limit to 8
                missing_skills=sim_data.get('missing_skills', [])[:5],  # Limit to 5
                summary=summary,
                skill_breakdown=sim_data.get('skill_breakdown', {})
            )
            
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Assign ranks
        for i, result in enumerate(results, 1):
            result.rank = i
        
        logger.info(f"Ranked {len(results)} candidates")
        
        return results
    
    def get_ranking_stats(self, results: List[ResumeResult]) -> Dict:
        """
        Get statistics about the ranking results.
        
        Args:
            results: List of ranked results
            
        Returns:
            Dictionary with ranking statistics
        """
        if not results:
            return {}
        
        scores = [r.match_score for r in results]
        
        fit_counts = {
            'high': sum(1 for r in results if r.fit == FitCategory.HIGH),
            'medium': sum(1 for r in results if r.fit == FitCategory.MEDIUM),
            'low': sum(1 for r in results if r.fit == FitCategory.LOW)
        }
        
        # Collect all matched skills across candidates
        all_matched_skills = []
        for r in results:
            all_matched_skills.extend(r.matched_skills)
        
        skill_frequency = {}
        for skill in all_matched_skills:
            skill_frequency[skill] = skill_frequency.get(skill, 0) + 1
        
        # Sort skills by frequency
        common_skills = sorted(
            skill_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:10]
        
        return {
            'total_candidates': len(results),
            'average_score': round(sum(scores) / len(scores), 1),
            'highest_score': max(scores),
            'lowest_score': min(scores),
            'score_range': max(scores) - min(scores),
            'fit_distribution': fit_counts,
            'common_matched_skills': [s[0] for s in common_skills],
            'recommendation': self._generate_recommendation(results, fit_counts)
        }
    
    def _generate_recommendation(self, 
                                  results: List[ResumeResult], 
                                  fit_counts: Dict) -> str:
        """
        Generate a recommendation based on the results.
        
        Args:
            results: Ranked results
            fit_counts: Distribution of fit categories
            
        Returns:
            Recommendation string
        """
        if fit_counts['high'] >= 2:
            return f"Strong candidate pool with {fit_counts['high']} excellent matches. Consider scheduling interviews with top {min(3, fit_counts['high'])} candidates."
        elif fit_counts['high'] >= 1:
            return f"One standout candidate identified. Consider also reviewing the {fit_counts['medium']} medium-fit candidates for potential."
        elif fit_counts['medium'] >= 2:
            return f"No perfect matches, but {fit_counts['medium']} candidates show good potential. Consider skills-based interview assessments."
        else:
            return "Limited matches found. Consider expanding search criteria or revisiting job requirements."
    
    def compare_candidates(self, 
                           result1: ResumeResult, 
                           result2: ResumeResult) -> Dict:
        """
        Compare two candidates side by side.
        
        Args:
            result1: First candidate result
            result2: Second candidate result
            
        Returns:
            Comparison dictionary
        """
        skills1 = set(result1.matched_skills)
        skills2 = set(result2.matched_skills)
        
        return {
            'score_difference': result1.match_score - result2.match_score,
            'common_skills': list(skills1 & skills2),
            'unique_to_first': list(skills1 - skills2),
            'unique_to_second': list(skills2 - skills1),
            'recommendation': (
                f"{result1.resume_name} is recommended" 
                if result1.match_score > result2.match_score 
                else f"{result2.resume_name} is recommended"
            )
        }


# Global ranking engine instance
ranking_engine = RankingEngine()