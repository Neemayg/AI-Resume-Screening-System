"""
Skill matching module that compares JD requirements with Resume skills.
Implements weighted scoring based on skill importance.
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import logging

from utils.data_loader import data_loader
from matching.skill_extractor import skill_extractor
from config.weights_config import SKILL_WEIGHTS

logger = logging.getLogger(__name__)

@dataclass
class SkillMatchResult:
    """Result of skill matching between JD and Resume."""
    score: float                           # Overall skill match score (0-1)
    matched_core: List[str]                # Core skills found in resume
    missing_core: List[str]                # Core skills NOT in resume
    matched_preferred: List[str]           # Preferred skills found
    missing_preferred: List[str]           # Preferred skills missing
    matched_secondary: List[str]           # Secondary skills found
    extra_relevant: List[str]              # Extra skills that are relevant
    extra_irrelevant: List[str]            # Extra skills not relevant
    partial_matches: List[Dict]            # Skills with related alternatives
    skill_coverage: float                  # % of required skills covered
    weighted_score: float                  # Weighted by skill importance
    explanation: List[str]                 # Human-readable explanation


class SkillMatcher:
    """Match and score skills between JD and Resume."""
    
    def __init__(self):
        self.data = data_loader
        self.extractor = skill_extractor
    
    def match_skills(
        self,
        jd_skills: Dict[str, List[str]],
        resume_skills: List[str],
        role_key: Optional[str] = None
    ) -> SkillMatchResult:
        """
        Match resume skills against JD requirements.
        
        Args:
            jd_skills: Categorized JD skills:
                {
                    "core_required": [...],
                    "core_preferred": [...],
                    "secondary": [...],
                    "tools": [...]
                }
            resume_skills: List of skills from resume
            role_key: Optional role key for role-specific matching
            
        Returns:
            SkillMatchResult with detailed matching information
        """
        # Normalize all skills
        resume_normalized = set(self.extractor.normalize_skill_list(resume_skills))
        
        core_required = set(self.extractor.normalize_skill_list(
            jd_skills.get('core_required', [])
        ))
        core_preferred = set(self.extractor.normalize_skill_list(
            jd_skills.get('core_preferred', [])
        ))
        secondary = set(self.extractor.normalize_skill_list(
            jd_skills.get('secondary', [])
        ))
        tools = set(self.extractor.normalize_skill_list(
            jd_skills.get('tools', [])
        ))
        
        # If role_key provided, supplement with role-specific skills
        if role_key:
            role_requirements = self._get_role_skill_requirements(role_key)
            core_required.update(role_requirements.get('core', set()))
            secondary.update(role_requirements.get('secondary', set()))
        
        # Calculate matches
        matched_core = list(resume_normalized.intersection(core_required))
        missing_core = list(core_required - resume_normalized)
        
        matched_preferred = list(resume_normalized.intersection(core_preferred))
        missing_preferred = list(core_preferred - resume_normalized)
        
        matched_secondary = list(resume_normalized.intersection(secondary))
        
        # Check for partial matches (related skills)
        partial_matches = self._find_partial_matches(
            missing_core + missing_preferred,
            resume_normalized
        )
        
        # Classify extra skills
        all_required = core_required | core_preferred | secondary | tools
        extra_skills = resume_normalized - all_required
        extra_relevant, extra_irrelevant = self._classify_extra_skills(
            extra_skills, role_key
        )
        
        # Calculate scores
        coverage_score = self._calculate_coverage_score(
            core_required, core_preferred, secondary,
            matched_core, matched_preferred, matched_secondary
        )
        
        weighted_score = self._calculate_weighted_score(
            core_required, core_preferred, secondary,
            matched_core, matched_preferred, matched_secondary,
            partial_matches, extra_relevant
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            matched_core, missing_core,
            matched_preferred, missing_preferred,
            matched_secondary, extra_relevant,
            partial_matches
        )
        
        return SkillMatchResult(
            score=weighted_score,
            matched_core=matched_core,
            missing_core=missing_core,
            matched_preferred=matched_preferred,
            missing_preferred=missing_preferred,
            matched_secondary=matched_secondary,
            extra_relevant=list(extra_relevant),
            extra_irrelevant=list(extra_irrelevant),
            partial_matches=partial_matches,
            skill_coverage=coverage_score,
            weighted_score=weighted_score,
            explanation=explanation
        )
    
    def _get_role_skill_requirements(self, role_key: str) -> Dict[str, Set[str]]:
        """Get skill requirements from role definition."""
        role_data = self.data._job_roles.get(role_key, {})
        
        return {
            'core': set(self.extractor.normalize_skill_list(
                role_data.get('core_skills', [])
            )),
            'preferred': set(self.extractor.normalize_skill_list(
                role_data.get('preferred_skills', [])
            )),
            'secondary': set(self.extractor.normalize_skill_list(
                role_data.get('secondary_skills', [])
            ))
        }
    
    def _find_partial_matches(
        self,
        missing_skills: List[str],
        resume_skills: Set[str]
    ) -> List[Dict]:
        """
        Find cases where resume has related skills to missing requirements.
        
        Returns:
            List of {"required": skill, "found": [related skills], "coverage": 0-1}
        """
        partial_matches = []
        
        for skill in missing_skills:
            related = self.data.find_similar_skills(skill)
            found_related = [r for r in related if r in resume_skills]
            
            if found_related:
                # Calculate how well related skills cover the requirement
                coverage = min(1.0, len(found_related) * 0.4)
                
                partial_matches.append({
                    "required": skill,
                    "found": found_related,
                    "coverage": coverage
                })
        
        return partial_matches
    
    def _classify_extra_skills(
        self,
        extra_skills: Set[str],
        role_key: Optional[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Classify extra skills as relevant or irrelevant.
        
        Returns:
            (relevant_skills, irrelevant_skills)
        """
        relevant = set()
        irrelevant = set()
        
        # Get role-related categories if role_key provided
        relevant_categories = set()
        if role_key:
            role_data = self.data._job_roles.get(role_key, {})
            relevant_categories = set(role_data.get('relevant_skill_categories', []))
        
        for skill in extra_skills:
            skill_info = self.data.get_skill_info(skill)
            
            if skill_info:
                category = skill_info.get('category', '')
                weight = skill_info.get('weight', 0.5)
                
                # Relevant if: high weight, OR in relevant category for role
                if weight >= 0.6 or category in relevant_categories:
                    relevant.add(skill)
                else:
                    irrelevant.add(skill)
            else:
                # Unknown skill - consider irrelevant
                irrelevant.add(skill)
        
        return relevant, irrelevant
    
    def _calculate_coverage_score(
        self,
        core_required: Set[str],
        core_preferred: Set[str],
        secondary: Set[str],
        matched_core: List[str],
        matched_preferred: List[str],
        matched_secondary: List[str]
    ) -> float:
        """
        Calculate simple coverage percentage.
        
        Returns:
            Float 0-1 representing % of skills covered
        """
        total_required = len(core_required) + len(core_preferred) + len(secondary)
        
        if total_required == 0:
            return 1.0
        
        total_matched = len(matched_core) + len(matched_preferred) + len(matched_secondary)
        
        return total_matched / total_required
    
    def _calculate_weighted_score(
        self,
        core_required: Set[str],
        core_preferred: Set[str],
        secondary: Set[str],
        matched_core: List[str],
        matched_preferred: List[str],
        matched_secondary: List[str],
        partial_matches: List[Dict],
        extra_relevant: Set[str]
    ) -> float:
        """
        Calculate weighted skill match score.
        
        Core required skills are weighted heavily,
        secondary skills less so, etc.
        
        Returns:
            Float 0-1 weighted score
        """
        score = 0.0
        max_score = 0.0
        
        # Core required (weight: 1.0)
        weight = SKILL_WEIGHTS['core_required']
        if core_required:
            max_score += len(core_required) * weight
            score += len(matched_core) * weight
        
        # Core preferred (weight: 0.7)
        weight = SKILL_WEIGHTS['core_preferred']
        if core_preferred:
            max_score += len(core_preferred) * weight
            score += len(matched_preferred) * weight
        
        # Secondary (weight: 0.4)
        weight = SKILL_WEIGHTS['secondary']
        if secondary:
            max_score += len(secondary) * weight
            score += len(matched_secondary) * weight
        
        # Add partial match credit
        for partial in partial_matches:
            coverage = partial['coverage']
            # Give partial credit (up to 50% of full skill weight)
            if partial['required'] in core_required:
                score += coverage * 0.5 * SKILL_WEIGHTS['core_required']
            elif partial['required'] in core_preferred:
                score += coverage * 0.5 * SKILL_WEIGHTS['core_preferred']
        
        # Add small bonus for extra relevant skills
        bonus = min(0.1, len(extra_relevant) * 0.02)
        score += bonus
        
        # Normalize to 0-1
        if max_score == 0:
            return 1.0
        
        return min(1.0, score / max_score)
    
    def _generate_explanation(
        self,
        matched_core: List[str],
        missing_core: List[str],
        matched_preferred: List[str],
        missing_preferred: List[str],
        matched_secondary: List[str],
        extra_relevant: Set[str],
        partial_matches: List[Dict]
    ) -> List[str]:
        """Generate human-readable explanation of match."""
        explanation = []
        
        # Core skills summary
        total_core = len(matched_core) + len(missing_core)
        if total_core > 0:
            pct = len(matched_core) / total_core * 100
            explanation.append(
                f"Core skills: {len(matched_core)}/{total_core} matched ({pct:.0f}%)"
            )
            if missing_core:
                explanation.append(f"Missing core: {', '.join(missing_core[:5])}")
        
        # Preferred skills
        total_preferred = len(matched_preferred) + len(missing_preferred)
        if total_preferred > 0:
            pct = len(matched_preferred) / total_preferred * 100
            explanation.append(
                f"Preferred skills: {len(matched_preferred)}/{total_preferred} ({pct:.0f}%)"
            )
        
        # Partial matches
        if partial_matches:
            for pm in partial_matches[:3]:
                explanation.append(
                    f"Has {', '.join(pm['found'][:2])} instead of {pm['required']}"
                )
        
        # Extra relevant skills
        if extra_relevant:
            explanation.append(
                f"Additional relevant: {', '.join(list(extra_relevant)[:5])}"
            )
        
        return explanation


# Singleton instance
skill_matcher = SkillMatcher()
