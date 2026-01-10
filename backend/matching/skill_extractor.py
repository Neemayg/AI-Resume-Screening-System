"""
Skill extraction module that identifies and normalizes skills
from JDs and resumes using the skills_dataset.json.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import logging

from utils.data_loader import data_loader
from matching.text_normalizer import text_normalizer

logger = logging.getLogger(__name__)

class SkillExtractor:
    """Extract and normalize skills from text."""
    
    def __init__(self):
        self.data = data_loader
        self._build_skill_patterns()
    
    def _build_skill_patterns(self):
        """Build regex patterns for skill matching."""
        # Get all skill names and aliases
        all_skills = []
        
        # Iterate through the skill lookup
        for normalized, info in self.data._skill_lookup.items():
            canonical = info['canonical']
            if canonical not in all_skills:
                all_skills.append(canonical)
        
        # Sort by length (longest first) for greedy matching
        all_skills.sort(key=len, reverse=True)
        
        # Build pattern (escape special regex characters)
        escaped = [re.escape(skill) for skill in all_skills]
        self.skill_pattern = re.compile(
            r'\b(' + '|'.join(escaped) + r')\b',
            re.IGNORECASE
        )
    
    def extract_skills(self, text: str) -> Dict[str, List[Dict]]:
        """
        Extract all skills from text.
        
        Args:
            text: Input text (JD or Resume)
            
        Returns:
            Dictionary with categorized skills:
            {
                "programming_languages": [
                    {"name": "Python", "canonical": "Python", "weight": 0.9, "count": 3}
                ],
                "frameworks": [...],
                ...
            }
        """
        if not text:
            return {}
        
        # Normalize text
        normalized_text = text_normalizer.normalize_text(text)
        
        # Track found skills
        found_skills = defaultdict(list)
        skill_counts = defaultdict(int)
        seen_canonical = set()
        
        # Method 1: Pattern matching against known skills
        matches = self.skill_pattern.finditer(normalized_text)
        for match in matches:
            skill_text = match.group(1)
            skill_info = self.data.get_skill_info(skill_text)
            
            if skill_info:
                canonical = skill_info['canonical']
                skill_counts[canonical] += 1
                
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    found_skills[skill_info['category']].append({
                        "name": skill_text,
                        "canonical": canonical,
                        "weight": skill_info['weight'],
                        "count": 0  # Will be updated later
                    })
        
        # Method 2: Token-based extraction for edge cases
        tokens = text_normalizer.tokenize_for_skills(normalized_text)
        for token in tokens:
            # Check each word in token
            words = token.split()
            for word in words:
                skill_info = self.data.get_skill_info(word)
                if skill_info:
                    canonical = skill_info['canonical']
                    skill_counts[canonical] += 1
                    
                    if canonical not in seen_canonical:
                        seen_canonical.add(canonical)
                        found_skills[skill_info['category']].append({
                            "name": word,
                            "canonical": canonical,
                            "weight": skill_info['weight'],
                            "count": 0
                        })
            
            # Check full token (for multi-word skills)
            skill_info = self.data.get_skill_info(token)
            if skill_info:
                canonical = skill_info['canonical']
                skill_counts[canonical] += 1
                
                if canonical not in seen_canonical:
                    seen_canonical.add(canonical)
                    found_skills[skill_info['category']].append({
                        "name": token,
                        "canonical": canonical,
                        "weight": skill_info['weight'],
                        "count": 0
                    })
        
        # Update counts
        for category, skills in found_skills.items():
            for skill in skills:
                skill['count'] = skill_counts[skill['canonical']]
        
        return dict(found_skills)
    
    def extract_skills_flat(self, text: str) -> List[str]:
        """
        Extract skills as a flat list of canonical names.
        
        Returns:
            List of canonical skill names
        """
        categorized = self.extract_skills(text)
        flat = []
        for skills in categorized.values():
            for skill in skills:
                flat.append(skill['canonical'])
        return flat
    
    def normalize_skill_list(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skill names to their canonical forms.
        
        Args:
            skills: List of skill strings (possibly with variants)
            
        Returns:
            List of canonical skill names
        """
        normalized = []
        seen = set()
        
        for skill in skills:
            canonical = self.data.get_canonical_skill(skill)
            if canonical and canonical not in seen:
                normalized.append(canonical)
                seen.add(canonical)
            elif not canonical and skill not in seen:
                # Keep original if not in dataset
                normalized.append(skill)
                seen.add(skill)
        
        return normalized
    
    def categorize_skills_by_importance(
        self,
        skills: List[str],
        role_key: Optional[str] = None
    ) -> Dict[str, List[str]]:
        """
        Categorize skills by importance level.
        
        Args:
            skills: List of skill names
            role_key: Optional role key for role-specific categorization
            
        Returns:
            {
                "core_required": [...],
                "core_preferred": [...],
                "secondary": [...],
                "other": [...]
            }
        """
        categorized = {
            "core_required": [],
            "core_preferred": [],
            "secondary": [],
            "other": []
        }
        
        if role_key:
            # Use role-specific definitions
            role_data = self.data._job_roles.get(role_key, {})
            core_skills = set(self.normalize_skill_list(
                role_data.get('core_skills', [])
            ))
            preferred_skills = set(self.normalize_skill_list(
                role_data.get('preferred_skills', [])
            ))
            secondary_skills = set(self.normalize_skill_list(
                role_data.get('secondary_skills', [])
            ))
            
            for skill in skills:
                canonical = self.data.get_canonical_skill(skill) or skill
                if canonical in core_skills:
                    categorized["core_required"].append(canonical)
                elif canonical in preferred_skills:
                    categorized["core_preferred"].append(canonical)
                elif canonical in secondary_skills:
                    categorized["secondary"].append(canonical)
                else:
                    categorized["other"].append(canonical)
        else:
            # Use general weight-based categorization
            for skill in skills:
                skill_info = self.data.get_skill_info(skill)
                if skill_info:
                    weight = skill_info['weight']
                    canonical = skill_info['canonical']
                    
                    if weight >= 0.8:
                        categorized["core_required"].append(canonical)
                    elif weight >= 0.6:
                        categorized["core_preferred"].append(canonical)
                    elif weight >= 0.4:
                        categorized["secondary"].append(canonical)
                    else:
                        categorized["other"].append(canonical)
                else:
                    categorized["other"].append(skill)
        
        return categorized
    
    def find_skill_gaps(
        self,
        required_skills: List[str],
        candidate_skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Find gaps between required and candidate skills.
        
        Returns:
            {
                "missing": [...],          # Required but not in candidate
                "matched": [...],          # Both required and in candidate
                "extra": [...],            # In candidate but not required
                "partial_match": [...]     # Related skills found
            }
        """
        required_normalized = set(self.normalize_skill_list(required_skills))
        candidate_normalized = set(self.normalize_skill_list(candidate_skills))
        
        result = {
            "missing": [],
            "matched": [],
            "extra": [],
            "partial_match": []
        }
        
        # Find direct matches and missing
        for skill in required_normalized:
            if skill in candidate_normalized:
                result["matched"].append(skill)
            else:
                # Check for related skills
                related = self.data.find_similar_skills(skill)
                related_in_candidate = [
                    r for r in related if r in candidate_normalized
                ]
                
                if related_in_candidate:
                    result["partial_match"].append({
                        "required": skill,
                        "found_related": related_in_candidate
                    })
                else:
                    result["missing"].append(skill)
        
        # Find extra skills
        result["extra"] = list(candidate_normalized - required_normalized)
        
        return result


# Singleton instance
skill_extractor = SkillExtractor()
