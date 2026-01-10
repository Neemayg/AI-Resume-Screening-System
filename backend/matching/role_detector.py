"""
Role detection module that identifies the most relevant role
from a Job Description using job_roles.json.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import logging

from utils.data_loader import data_loader
from matching.text_normalizer import text_normalizer
from matching.skill_extractor import skill_extractor

logger = logging.getLogger(__name__)

class RoleDetector:
    """Detect and match job roles from JD text."""
    
    def __init__(self):
        self.data = data_loader
        self._build_role_patterns()
    
    def _build_role_patterns(self):
        """Build patterns for role detection."""
        # Collect all role titles and aliases
        all_titles = []
        
        for role_key, role_data in self.data._job_roles.items():
            title = role_data.get('title', role_key)
            aliases = role_data.get('aliases', [])
            
            all_titles.append((title, role_key))
            for alias in aliases:
                all_titles.append((alias, role_key))
        
        # Sort by length (longest first)
        all_titles.sort(key=lambda x: len(x[0]), reverse=True)
        
        self.role_titles = all_titles
    
    def detect_role(self, jd_text: str) -> Dict:
        """
        Detect the primary role from a Job Description.
        
        Args:
            jd_text: Full job description text
            
        Returns:
            {
                "detected_role_key": "software_engineer",
                "detected_role_title": "Software Engineer",
                "confidence": 0.85,
                "match_reasons": [...],
                "alternative_roles": [...],
                "role_data": { ... full role definition ... }
            }
        """
        normalized_jd = text_normalizer.normalize_text(jd_text).lower()
        
        # Score each role
        role_scores = {}
        
        for role_key, role_data in self.data._job_roles.items():
            score, reasons = self._score_role_match(
                normalized_jd, jd_text, role_key, role_data
            )
            role_scores[role_key] = {
                "score": score,
                "reasons": reasons,
                "role_data": role_data
            }
        
        # Sort by score
        sorted_roles = sorted(
            role_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )
        
        if not sorted_roles:
            return self._create_unknown_role_response(jd_text)
        
        # Get top role
        top_role_key, top_role_info = sorted_roles[0]
        
        # Calculate confidence based on score gap
        if len(sorted_roles) > 1:
            score_gap = top_role_info['score'] - sorted_roles[1][1]['score']
            confidence = min(0.95, 0.5 + score_gap)
        else:
            confidence = 0.9 if top_role_info['score'] > 0.5 else 0.6
        
        return {
            "detected_role_key": top_role_key,
            "detected_role_title": top_role_info['role_data'].get('title', top_role_key),
            "confidence": round(confidence, 2),
            "match_reasons": top_role_info['reasons'],
            "alternative_roles": [
                {
                    "role_key": key,
                    "title": info['role_data'].get('title', key),
                    "score": round(info['score'], 2)
                }
                for key, info in sorted_roles[1:4]  # Top 3 alternatives
            ],
            "role_data": top_role_info['role_data']
        }
    
    def _score_role_match(
        self,
        normalized_jd: str,
        original_jd: str,
        role_key: str,
        role_data: Dict
    ) -> Tuple[float, List[str]]:
        """
        Score how well a JD matches a specific role.
        
        Returns:
            (score, list_of_reasons)
        """
        score = 0.0
        reasons = []
        
        # 1. Title Match (weight: 30%)
        title_score = self._score_title_match(normalized_jd, role_data)
        score += title_score * 0.30
        if title_score > 0.5:
            reasons.append(f"Title match: {title_score:.0%}")
        
        # 2. Core Skills Match (weight: 35%)
        skill_score = self._score_skill_match(original_jd, role_data)
        score += skill_score * 0.35
        if skill_score > 0.3:
            reasons.append(f"Core skills match: {skill_score:.0%}")
        
        # 3. Keywords Match (weight: 20%)
        keyword_score = self._score_keyword_match(normalized_jd, role_data)
        score += keyword_score * 0.20
        if keyword_score > 0.3:
            reasons.append(f"Keywords match: {keyword_score:.0%}")
        
        # 4. Description Similarity (weight: 15%)
        desc_score = self._score_description_match(normalized_jd, role_data)
        score += desc_score * 0.15
        if desc_score > 0.3:
            reasons.append(f"Description similarity: {desc_score:.0%}")
        
        return score, reasons
    
    def _score_title_match(self, normalized_jd: str, role_data: Dict) -> float:
        """Score based on title/alias presence in JD."""
        title = role_data.get('title', '').lower()
        aliases = [a.lower() for a in role_data.get('aliases', [])]
        
        # Check for exact title match
        if title and title in normalized_jd:
            return 1.0
        
        # Check aliases
        for alias in aliases:
            if alias in normalized_jd:
                return 0.9
        
        # Partial title match
        title_words = title.split()
        if len(title_words) > 1:
            matches = sum(1 for w in title_words if w in normalized_jd)
            if matches >= len(title_words) * 0.6:
                return 0.6
        
        return 0.0
    
    def _score_skill_match(self, original_jd: str, role_data: Dict) -> float:
        """Score based on core skills presence in JD."""
        core_skills = role_data.get('core_skills', [])
        if not core_skills:
            return 0.5  # Neutral if no core skills defined
        
        # Extract skills from JD
        jd_skills = set(skill_extractor.extract_skills_flat(original_jd))
        
        # Normalize role core skills
        normalized_core = set(skill_extractor.normalize_skill_list(core_skills))
        
        # Calculate overlap
        matched = jd_skills.intersection(normalized_core)
        
        if normalized_core:
            return len(matched) / len(normalized_core)
        return 0.0
    
    def _score_keyword_match(self, normalized_jd: str, role_data: Dict) -> float:
        """Score based on role-specific keywords."""
        keywords = role_data.get('keywords', [])
        if not keywords:
            return 0.5
        
        matched = sum(1 for kw in keywords if kw.lower() in normalized_jd)
        return matched / len(keywords)
    
    def _score_description_match(self, normalized_jd: str, role_data: Dict) -> float:
        """Score based on description similarity."""
        role_desc = role_data.get('description', '').lower()
        if not role_desc:
            return 0.5
        
        # Simple word overlap
        jd_words = set(normalized_jd.split())
        desc_words = set(role_desc.split())
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                      'to', 'for', 'of', 'with', 'by', 'as', 'is', 'are', 'was',
                      'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                      'does', 'did', 'will', 'would', 'could', 'should', 'may',
                      'might', 'must', 'shall', 'can', 'need', 'dare', 'ought',
                      'used', 'this', 'that', 'these', 'those'}
        
        jd_words -= stop_words
        desc_words -= stop_words
        
        if not desc_words:
            return 0.5
        
        overlap = jd_words.intersection(desc_words)
        return min(1.0, len(overlap) / (len(desc_words) * 0.3))
    
    def _create_unknown_role_response(self, jd_text: str) -> Dict:
        """Create response for when no role is confidently detected."""
        # Extract skills to help identify
        skills = skill_extractor.extract_skills_flat(jd_text)
        
        return {
            "detected_role_key": "unknown",
            "detected_role_title": "General Position",
            "confidence": 0.3,
            "match_reasons": ["No strong role match found"],
            "alternative_roles": [],
            "role_data": {
                "core_skills": skills[:10],
                "secondary_skills": skills[10:20],
                "description": "Role not in database"
            }
        }
    
    def get_role_requirements(self, role_key: str) -> Dict:
        """
        Get complete requirements for a role.
        
        Returns:
            {
                "core_skills": [...],
                "preferred_skills": [...],
                "secondary_skills": [...],
                "experience_range": (min, max),
                "education": [...],
                "certifications": [...],
                "responsibilities": [...],
                "keywords": [...]
            }
        """
        role_data = self.data._job_roles.get(role_key, {})
        
        return {
            "core_skills": role_data.get('core_skills', []),
            "preferred_skills": role_data.get('preferred_skills', []),
            "secondary_skills": role_data.get('secondary_skills', []),
            "experience_range": (
                role_data.get('min_experience', 0),
                role_data.get('max_experience', 10)
            ),
            "education": role_data.get('education', []),
            "certifications": role_data.get('certifications', []),
            "responsibilities": role_data.get('responsibilities', []),
            "keywords": role_data.get('keywords', [])
        }


# Singleton instance
role_detector = RoleDetector()
