"""
Role-Specific Weight Adjuster
Adjusts scoring weights based on the detected role type
"""

from typing import Dict, Optional
from scoring.score_combiner import WeightConfiguration
import json


class RoleWeightAdjuster:
    """
    Adjusts scoring weights based on role characteristics
    Different roles prioritize different aspects
    """
    
    # Role type weight profiles
    ROLE_PROFILES = {
        'technical_senior': {
            'skill_match_weight': 0.35,
            'core_skill_weight': 0.25,
            'experience_weight': 0.20,
            'semantic_similarity_weight': 0.10,
            'role_alignment_weight': 0.05,
            'education_weight': 0.05,
            'missing_core_skill_penalty_weight': 0.20
        },
        'technical_junior': {
            'skill_match_weight': 0.25,
            'core_skill_weight': 0.20,
            'experience_weight': 0.10,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.15,
            'education_weight': 0.15,
            'missing_core_skill_penalty_weight': 0.10
        },
        'management': {
            'skill_match_weight': 0.15,
            'core_skill_weight': 0.15,
            'experience_weight': 0.30,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.20,
            'education_weight': 0.05,
            'missing_core_skill_penalty_weight': 0.10
        },
        'data_science': {
            'skill_match_weight': 0.30,
            'core_skill_weight': 0.25,
            'experience_weight': 0.15,
            'semantic_similarity_weight': 0.10,
            'role_alignment_weight': 0.10,
            'education_weight': 0.10,
            'missing_core_skill_penalty_weight': 0.18
        },
        'design': {
            'skill_match_weight': 0.25,
            'core_skill_weight': 0.20,
            'experience_weight': 0.20,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.15,
            'education_weight': 0.05,
            'missing_core_skill_penalty_weight': 0.12
        },
        'default': {
            'skill_match_weight': 0.30,
            'core_skill_weight': 0.20,
            'experience_weight': 0.20,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.10,
            'education_weight': 0.05,
            'missing_core_skill_penalty_weight': 0.15
        }
    }
    
    def __init__(self, job_roles_data: Dict):
        self.job_roles = job_roles_data
        self._build_role_to_profile_mapping()
    
    def _build_role_to_profile_mapping(self):
        """Map specific roles to weight profiles"""
        self.role_mapping = {}
        
        for role_id, role_data in self.job_roles.items():
            role_title = role_data.get('title', '').lower()
            seniority = role_data.get('seniority_level', 'mid')
            category = role_data.get('category', 'general')
            
            # Determine profile based on role characteristics
            if category in ['data_science', 'machine_learning', 'ai']:
                profile = 'data_science'
            elif category in ['design', 'ux', 'ui']:
                profile = 'design'
            elif category in ['management', 'leadership', 'executive']:
                profile = 'management'
            elif category in ['engineering', 'development', 'technical']:
                if seniority in ['senior', 'lead', 'principal', 'staff']:
                    profile = 'technical_senior'
                else:
                    profile = 'technical_junior'
            else:
                profile = 'default'
            
            self.role_mapping[role_id] = profile
    
    def get_adjusted_weights(self, role_id: str) -> WeightConfiguration:
        """
        Get weight configuration adjusted for the specific role
        """
        profile_name = self.role_mapping.get(role_id, 'default')
        profile = self.ROLE_PROFILES.get(profile_name, self.ROLE_PROFILES['default'])
        
        return WeightConfiguration(**profile)
    
    def get_weights_for_jd_analysis(self, jd_analysis: Dict) -> WeightConfiguration:
        """
        Get weights based on JD analysis results
        """
        detected_role_id = jd_analysis.get('detected_role_id')
        
        if detected_role_id:
            return self.get_adjusted_weights(detected_role_id)
        
        # Fallback: Analyze JD characteristics directly
        required_experience = jd_analysis.get('required_experience_years', 0)
        skill_count = len(jd_analysis.get('required_skills', []))
        
        if required_experience >= 7:
            return WeightConfiguration(**self.ROLE_PROFILES['technical_senior'])
        elif skill_count > 15:
            return WeightConfiguration(**self.ROLE_PROFILES['technical_senior'])
        else:
            return WeightConfiguration(**self.ROLE_PROFILES['default'])
