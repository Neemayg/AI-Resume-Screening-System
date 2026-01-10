"""
Explanation Generator
Generates human-readable explanations for ranking decisions
"""

from typing import Dict, List
from ranking.ranking_engine import RankedResume


class ExplanationGenerator:
    """
    Generates detailed explanations for why resumes are ranked the way they are
    """
    
    def __init__(self):
        self.templates = {
            'skill_match_high': "Candidate demonstrates strong proficiency in {count} of {total} required skills, including critical skills: {skills}.",
            'skill_match_low': "Candidate is missing {count} critical skills: {skills}. This significantly impacts the ranking.",
            'experience_exceeds': "With {years} years of experience (exceeding the {required} years required), the candidate brings substantial expertise.",
            'experience_meets': "Candidate meets the experience requirement with {years} years (requirement: {required} years).",
            'experience_under': "Candidate has {years} years of experience, which is below the {required} years required.",
            'role_alignment_high': "Previous roles closely align with the target position, suggesting a smooth transition.",
            'role_alignment_low': "Limited direct experience in similar roles may require additional onboarding.",
            'semantic_high': "Resume content strongly aligns with the job description language and requirements.",
            'semantic_low': "Resume content has limited alignment with job description terminology."
        }
    
    def generate_full_explanation(
        self,
        ranked_resume: RankedResume,
        jd_analysis: Dict
    ) -> Dict:
        """
        Generate a comprehensive text explanation
        """
        # Stub implementation
        return {
            "summary": f"Ranked #{ranked_resume.rank} with a score of {ranked_resume.final_score:.1f}.",
            "details": ranked_resume.recommendation
        }
