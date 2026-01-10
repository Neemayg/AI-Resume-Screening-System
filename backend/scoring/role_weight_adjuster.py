"""
Role-Specific Weight Adjuster
Adjusts scoring weights based on detected role type
"""

from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RoleWeightAdjuster:
    """Adjusts weights based on role characteristics."""
    
    # Role-specific weight profiles
    ROLE_PROFILES = {
        'technical_senior': {
            'skill_match_weight': 0.35,
            'experience_weight': 0.25,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.15,
            'education_weight': 0.10
        },
        'technical_junior': {
            'skill_match_weight': 0.30,
            'experience_weight': 0.15,
            'semantic_similarity_weight': 0.20,
            'role_alignment_weight': 0.20,
            'education_weight': 0.15
        },
        'management': {
            'skill_match_weight': 0.20,
            'experience_weight': 0.35,
            'semantic_similarity_weight': 0.20,
            'role_alignment_weight': 0.20,
            'education_weight': 0.05
        },
        'data_science': {
            'skill_match_weight': 0.35,
            'experience_weight': 0.20,
            'semantic_similarity_weight': 0.15,
            'role_alignment_weight': 0.15,
            'education_weight': 0.15
        },
        'default': {
            'skill_match_weight': 0.30,
            'experience_weight': 0.20,
            'semantic_similarity_weight': 0.20,
            'role_alignment_weight': 0.15,
            'education_weight': 0.10
        }
    }
    
    def __init__(self, job_roles_data: Optional[Dict] = None):
        """Initialize with job roles data."""
        self.job_roles = job_roles_data or {}
        self._build_role_to_profile_mapping()
    
    def _build_role_to_profile_mapping(self):
        """Map specific roles to weight profiles."""
        self.role_mapping = {}
        
        # Map from job_roles.json
        job_roles_dict = self.job_roles.get('job_roles', {})
        
        for role_id, role_data in job_roles_dict.items():
            if not isinstance(role_data, dict):
                continue
                
            role_title = role_data.get('title', '').lower()
            
            # Determine profile
            if any(keyword in role_title for keyword in ['senior', 'lead', 'principal', 'staff', 'architect']):
                profile = 'technical_senior'
            elif any(keyword in role_title for keyword in ['junior', 'entry', 'intern', 'graduate']):
                profile = 'technical_junior'
            elif any(keyword in role_title for keyword in ['manager', 'director', 'vp', 'head', 'chief']):
                profile = 'management'
            elif any(keyword in role_title for keyword in ['data', 'scientist', 'analyst', 'ml', 'ai']):
                profile = 'data_science'
            else:
                profile = 'default'
            
            self.role_mapping[role_id] = profile
    
    def get_adjusted_weights(self, role_id: str) -> Dict:
        """Get weights adjusted for specific role."""
        profile_name = self.role_mapping.get(role_id, 'default')
        return self.ROLE_PROFILES.get(profile_name, self.ROLE_PROFILES['default'])
    
    def get_weights_for_role_detection(self, role_detection: Dict) -> Dict:
        """Get weights based on role detection results."""
        role_id = role_detection.get('detected_role_key', 'unknown')
        
        if role_id in self.role_mapping:
            return self.get_adjusted_weights(role_id)
        
        # Fallback to default
        return self.ROLE_PROFILES['default']


# Singleton
role_weight_adjuster = RoleWeightAdjuster()
