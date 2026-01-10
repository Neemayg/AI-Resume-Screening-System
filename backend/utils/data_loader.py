"""
Utility to load and cache job_roles.json and skills_dataset.json.
Provides fast access to structured data throughout the pipeline.
"""

import json
import os
from functools import lru_cache
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """Singleton-style loader for reference datasets."""
    
    _instance = None
    _job_roles: Dict = None
    _skills_dataset: Dict = None
    _skill_lookup: Dict = None
    _role_lookup: Dict = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Load all datasets on first instantiation."""
        # Adjusted path to match backend structure: backend/utils -> backend/datasets (assumed data location)
        # Original code used os.path.dirname(os.path.dirname(__file__)) -> data
        # In this project, data is in backend/datasets usually
        
        base_path = os.path.dirname(os.path.dirname(__file__)) # Should be backend/
        data_path = os.path.join(base_path, 'datasets') # Adjusted to likely location
        
        # Load job roles
        job_roles_path = os.path.join(data_path, 'job_roles.json')
        try:
            with open(job_roles_path, 'r', encoding='utf-8') as f:
                self._job_roles = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load job_roles.json: {e}")
            self._job_roles = {}
        
        # Load skills dataset
        skills_path = os.path.join(data_path, 'skills_dataset.json')
        try:
            with open(skills_path, 'r', encoding='utf-8') as f:
                self._skills_dataset = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load skills_dataset.json: {e}")
            self._skills_dataset = {}
        
        # Build lookup indices
        self._build_skill_lookup()
        self._build_role_lookup()
        
        logger.info(f"Loaded {len(self._job_roles)} job roles")
        logger.info(f"Loaded skills dataset with {len(self._skill_lookup)} skills")
    
    def _build_skill_lookup(self):
        """
        Build a normalized skill lookup dictionary.
        Maps all variations/aliases to canonical skill names.
        """
        self._skill_lookup = {}
        
        for category, data in self._skills_dataset.items():
            # Handle dictionary structure (e.g., {"skills": [...]})
            if isinstance(data, dict):
                skills = data.get('skills', [])
            elif isinstance(data, list):
                skills = data
            else:
                continue
                
            if not isinstance(skills, list):
                continue
                
            for skill_entry in skills:
                if isinstance(skill_entry, dict):
                    canonical = skill_entry.get('name', '')
                    aliases = skill_entry.get('aliases', [])
                    weight = skill_entry.get('weight', 0.5)
                    related = skill_entry.get('related', [])
                elif isinstance(skill_entry, str):
                    canonical = skill_entry
                    aliases = []
                    weight = 0.5
                    related = []
                else:
                    continue
                
                # Add canonical name
                normalized = self._normalize_skill_name(canonical)
                self._skill_lookup[normalized] = {
                    "canonical": canonical,
                    "category": category,
                    "weight": weight,
                    "related": related
                }
                
                # Add all aliases
                for alias in aliases:
                    normalized_alias = self._normalize_skill_name(alias)
                    self._skill_lookup[normalized_alias] = {
                        "canonical": canonical,
                        "category": category,
                        "weight": weight,
                        "related": related
                    }
    
    def _build_role_lookup(self):
        """
        Build role lookup with aliases.
        """
        self._role_lookup = {}
        
        for role_key, role_data in self._job_roles.items():
            canonical = role_data.get('title', role_key)
            aliases = role_data.get('aliases', [])
            
            # Add canonical
            normalized = self._normalize_role_name(canonical)
            self._role_lookup[normalized] = {
                "canonical": canonical,
                "role_key": role_key,
                "role_data": role_data
            }
            
            # Add key itself
            normalized_key = self._normalize_role_name(role_key)
            self._role_lookup[normalized_key] = {
                "canonical": canonical,
                "role_key": role_key,
                "role_data": role_data
            }
            
            # Add all aliases
            for alias in aliases:
                normalized_alias = self._normalize_role_name(alias)
                self._role_lookup[normalized_alias] = {
                    "canonical": canonical,
                    "role_key": role_key,
                    "role_data": role_data
                }
    
    @staticmethod
    def _normalize_skill_name(name: str) -> str:
        """Normalize skill name for lookup."""
        return name.lower().strip().replace('-', '').replace('.', '').replace(' ', '')
    
    @staticmethod
    def _normalize_role_name(name: str) -> str:
        """Normalize role name for lookup."""
        return name.lower().strip().replace('-', ' ').replace('_', ' ')
    
    # =========================================================================
    # PUBLIC METHODS
    # =========================================================================
    
    def get_skill_info(self, skill_name: str) -> Optional[Dict]:
        """Look up skill information by name or alias."""
        normalized = self._normalize_skill_name(skill_name)
        return self._skill_lookup.get(normalized)
    
    def get_canonical_skill(self, skill_name: str) -> Optional[str]:
        """Get canonical skill name from any variant."""
        info = self.get_skill_info(skill_name)
        return info['canonical'] if info else None
    
    def get_role_info(self, role_name: str) -> Optional[Dict]:
        """Look up role information by name or alias."""
        normalized = self._normalize_role_name(role_name)
        return self._role_lookup.get(normalized)
    
    def get_all_roles(self) -> Dict:
        """Get all role definitions."""
        return self._job_roles
    
    def get_skills_by_category(self, category: str) -> List[Dict]:
        """Get all skills in a category."""
        return self._skills_dataset.get(category, [])
    
    def get_role_core_skills(self, role_key: str) -> List[str]:
        """Get core required skills for a role."""
        role_data = self._job_roles.get(role_key, {})
        return role_data.get('core_skills', [])
    
    def get_role_secondary_skills(self, role_key: str) -> List[str]:
        """Get secondary/preferred skills for a role."""
        role_data = self._job_roles.get(role_key, {})
        return role_data.get('secondary_skills', [])
    
    def find_similar_skills(self, skill_name: str) -> List[str]:
        """Find related/similar skills."""
        info = self.get_skill_info(skill_name)
        if info:
            return info.get('related', [])
        return []


# Global instance for easy access
data_loader = DataLoader()
