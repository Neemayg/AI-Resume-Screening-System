"""
JD (Job Description) Analyzer Module
Implements weighted analysis pipeline for role classification and skill categorization.

Analysis Weights:
- Title Analysis: 40%
- Explicit Requirements: 35%
- Preferred Skills: 15%
- Experience Level: 10%
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from backend.utils.data_loader import data_loader
from backend.matching.text_normalizer import text_normalizer
from backend.matching.skill_extractor import skill_extractor

logger = logging.getLogger(__name__)


class ExperienceLevel(Enum):
    """Experience level classifications."""
    INTERN = "intern"
    ENTRY = "entry"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


@dataclass
class SkillCategories:
    """Categorized skills with weights for a specific role/level."""
    core_required: List[str] = field(default_factory=list)
    core_preferred: List[str] = field(default_factory=list)
    secondary: List[str] = field(default_factory=list)
    downweighted: List[str] = field(default_factory=list)
    
    # Weights for scoring
    core_weight: float = 0.60
    preferred_weight: float = 0.25
    secondary_weight: float = 0.15


@dataclass
class JDAnalysisResult:
    """Complete JD analysis result."""
    # Role detection
    detected_role_key: str = ""
    detected_role_title: str = ""
    role_confidence: float = 0.0
    
    # Experience level
    experience_level: ExperienceLevel = ExperienceLevel.MID
    experience_years_required: Optional[int] = None
    
    # Skill categories
    skill_categories: SkillCategories = field(default_factory=SkillCategories)
    
    # Raw extractions
    raw_required_skills: List[str] = field(default_factory=list)
    raw_preferred_skills: List[str] = field(default_factory=list)
    
    # Analysis metadata
    analysis_weights_used: Dict[str, float] = field(default_factory=dict)
    match_reasons: List[str] = field(default_factory=list)


class JDAnalyzer:
    """
    Weighted JD analysis with role classification.
    
    Implements the Phase 1 pipeline:
    1. Title Analysis (40% weight)
    2. Explicit Requirements (35% weight)
    3. Preferred Skills (15% weight)
    4. Experience Level (10% weight)
    """
    
    # Analysis component weights
    ANALYSIS_WEIGHTS = {
        "title": 0.40,
        "requirements": 0.35,
        "preferred": 0.15,
        "experience": 0.10
    }
    
    # Experience level keywords
    EXPERIENCE_KEYWORDS = {
        ExperienceLevel.INTERN: [
            "intern", "internship", "trainee", "fresher", "student",
            "co-op", "placement", "apprentice"
        ],
        ExperienceLevel.ENTRY: [
            "entry level", "entry-level", "graduate", "fresher",
            "junior", "associate", "0-2 years", "1-2 years"
        ],
        ExperienceLevel.JUNIOR: [
            "junior", "jr", "associate", "1-3 years", "2-3 years"
        ],
        ExperienceLevel.MID: [
            "mid-level", "mid level", "intermediate", "3-5 years",
            "4-6 years", "3+ years"
        ],
        ExperienceLevel.SENIOR: [
            "senior", "sr", "experienced", "5+ years", "5-8 years",
            "7+ years", "lead"
        ],
        ExperienceLevel.LEAD: [
            "lead", "team lead", "tech lead", "technical lead",
            "8+ years", "10+ years"
        ],
        ExperienceLevel.PRINCIPAL: [
            "principal", "staff", "architect", "distinguished",
            "director", "10+ years", "15+ years"
        ]
    }
    
    # Section header patterns
    REQUIRED_SECTION_PATTERNS = [
        r'required\s*(?:skills?|qualifications?)?:?',
        r'must\s*have:?',
        r'requirements?:?',
        r'essential\s*(?:skills?|qualifications?)?:?',
        r'minimum\s*(?:requirements?|qualifications?):?',
        r'what\s*(?:you\'?ll|we)\s*need:?',
        r'basic\s*qualifications?:?'
    ]
    
    PREFERRED_SECTION_PATTERNS = [
        r'preferred\s*(?:skills?|qualifications?)?:?',
        r'nice\s*to\s*have:?',
        r'bonus(?:\s*points?)?:?',
        r'additional\s*(?:skills?|qualifications?)?:?',
        r'desired\s*(?:skills?|qualifications?)?:?',
        r'plus(?:es)?:?',
        r'good\s*to\s*have:?'
    ]
    
    # Skills to downweight for junior/intern roles
    SENIOR_ONLY_SKILLS = [
        "system design", "architecture", "system architecture",
        "production deployment", "team lead", "leadership",
        "team management", "mentoring", "strategic planning",
        "budgeting", "stakeholder management", "cross-functional",
        "high level design", "hld", "distributed systems design"
    ]
    
    # Semantic skill equivalents
    SKILL_EQUIVALENTS = {
        # Python ecosystem
        "python": ["python3", "py", "pycharm", "python programming"],
        "machine learning": ["ml", "sklearn", "scikit-learn", "modeling", "ml models"],
        "pandas": ["dataframe", "data wrangling", "pd"],
        "numpy": ["numerical computing", "np", "array computing"],
        "scikit-learn": ["sklearn", "sk-learn"],
        "tensorflow": ["tf", "tf2"],
        "pytorch": ["torch"],
        
        # Data skills
        "eda": ["exploratory data analysis", "data analysis", "data exploration"],
        "nlp": ["natural language processing", "text processing", "text analytics"],
        "deep learning": ["dl", "neural networks", "nn"],
        
        # Tools
        "jupyter": ["jupyter notebook", "notebooks", "colab", "google colab"],
        "sql": ["mysql", "postgresql", "postgres", "sqlite", "database queries"],
        "git": ["github", "version control", "gitlab", "bitbucket"],
        
        # General
        "statistics": ["stats", "probability", "statistical analysis"],
        "data visualization": ["viz", "matplotlib", "seaborn", "plotly"],
    }
    
    def __init__(self):
        self.data = data_loader
        self.extractor = skill_extractor
        self._build_reverse_equivalents()
    
    def _build_reverse_equivalents(self):
        """Build reverse lookup for skill equivalents."""
        self._equivalent_to_canonical = {}
        for canonical, equivalents in self.SKILL_EQUIVALENTS.items():
            for equiv in equivalents:
                self._equivalent_to_canonical[equiv.lower()] = canonical.lower()
    
    def analyze_jd(self, jd_text: str) -> JDAnalysisResult:
        """
        Perform comprehensive JD analysis.
        
        Args:
            jd_text: Full job description text
            
        Returns:
            JDAnalysisResult with categorized skills and role info
        """
        result = JDAnalysisResult()
        result.analysis_weights_used = self.ANALYSIS_WEIGHTS.copy()
        
        # Step 1: Title Analysis (40%)
        role_info = self._extract_title_and_role(jd_text)
        result.detected_role_key = role_info["role_key"]
        result.detected_role_title = role_info["role_title"]
        result.role_confidence = role_info["confidence"]
        result.match_reasons.extend(role_info.get("reasons", []))
        
        # Step 2: Experience Level Detection (10%)
        level_info = self._detect_experience_level(jd_text)
        result.experience_level = level_info["level"]
        result.experience_years_required = level_info.get("years")
        result.match_reasons.append(f"Experience: {result.experience_level.value}")
        
        # Step 3: Parse Explicit Requirements (35%)
        result.raw_required_skills = self._parse_explicit_requirements(jd_text)
        
        # Step 4: Parse Preferred Skills (15%)
        result.raw_preferred_skills = self._parse_preferred_skills(jd_text)
        
        # Step 5: Categorize skills based on role and level
        result.skill_categories = self._categorize_skills_for_role(
            role_key=result.detected_role_key,
            experience_level=result.experience_level,
            required_skills=result.raw_required_skills,
            preferred_skills=result.raw_preferred_skills,
            jd_text=jd_text
        )
        
        logger.info(
            f"JD Analysis: Role={result.detected_role_title}, "
            f"Level={result.experience_level.value}, "
            f"Core={len(result.skill_categories.core_required)}, "
            f"Preferred={len(result.skill_categories.core_preferred)}"
        )
        
        return result
    
    def _extract_title_and_role(self, jd_text: str) -> Dict:
        """
        Extract job title and match to role definition.
        Uses 40% weight in final scoring.
        """
        normalized = text_normalizer.normalize_text(jd_text).lower()
        lines = jd_text.strip().split('\n')
        
        # Try to find explicit title
        title = ""
        for line in lines[:10]:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for explicit title pattern
            title_match = re.match(r'(?:job\s*)?title\s*:\s*(.+)', line, re.IGNORECASE)
            if title_match:
                title = title_match.group(1).strip()
                break
            
            # Check for position pattern
            pos_match = re.match(r'position\s*:\s*(.+)', line, re.IGNORECASE)
            if pos_match:
                title = pos_match.group(1).strip()
                break
            
            # Check for role pattern
            role_match = re.match(r'role\s*:\s*(.+)', line, re.IGNORECASE)
            if role_match:
                title = role_match.group(1).strip()
                break
            
            # First meaningful line might be title
            if len(line) < 100 and not line.endswith('.'):
                title = line
                break
        
        # Score against job_roles.json
        best_role_key = "unknown"
        best_role_data = {}
        best_score = 0.0
        reasons = []
        
        # Check both standard and additional roles
        all_roles = {}
        all_roles.update(self.data._job_roles.get('job_roles', {}))
        all_roles.update(self.data._job_roles.get('additional_job_roles', {}))
        
        for role_key, role_data in all_roles.items():
            if not isinstance(role_data, dict) or 'title' not in role_data:
                continue
            
            score = 0.0
            role_title = role_data.get('title', '').lower()
            aliases = [a.lower() for a in role_data.get('aliases', [])]
            
            # Title exact match
            if role_title and role_title in title.lower():
                score += 1.0
            elif title.lower() in role_title:
                score += 0.8
            
            # Alias match
            for alias in aliases:
                if alias in title.lower() or alias in normalized:
                    score += 0.5
                    break
            
            # Core skills presence
            core_skills = role_data.get('core_skills', [])
            skill_matches = sum(1 for s in core_skills if s.lower() in normalized)
            if core_skills:
                score += (skill_matches / len(core_skills)) * 0.3
            
            if score > best_score:
                best_score = score
                best_role_key = role_key
                best_role_data = role_data
        
        confidence = min(0.95, best_score / 1.5)
        
        if best_score > 0.3:
            reasons.append(f"Role matched: {best_role_data.get('title', best_role_key)}")
        
        return {
            "role_key": best_role_key,
            "role_title": best_role_data.get('title', title or 'Unknown'),
            "confidence": confidence,
            "role_data": best_role_data,
            "extracted_title": title,
            "reasons": reasons
        }
    
    def _detect_experience_level(self, jd_text: str) -> Dict:
        """
        Detect experience level from JD.
        Uses 10% weight in final scoring.
        """
        text_lower = jd_text.lower()
        
        # Try to extract years
        years = None
        year_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
            r'(?:experience|exp)(?:\s+of)?\s*[:;]?\s*(\d+)\+?\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?'
        ]
        
        for pattern in year_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                break
        
        # Determine level from keywords (prioritize explicit matches)
        detected_level = ExperienceLevel.MID  # Default
        max_matches = 0
        
        # Check from most junior to most senior
        level_order = [
            ExperienceLevel.INTERN,
            ExperienceLevel.ENTRY,
            ExperienceLevel.JUNIOR,
            ExperienceLevel.MID,
            ExperienceLevel.SENIOR,
            ExperienceLevel.LEAD,
            ExperienceLevel.PRINCIPAL
        ]
        
        for level in level_order:
            keywords = self.EXPERIENCE_KEYWORDS.get(level, [])
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > max_matches:
                max_matches = matches
                detected_level = level
        
        # Override with years if explicit
        if years is not None:
            if years == 0:
                detected_level = ExperienceLevel.INTERN
            elif years <= 2:
                detected_level = ExperienceLevel.ENTRY
            elif years <= 3:
                detected_level = ExperienceLevel.JUNIOR
            elif years <= 5:
                detected_level = ExperienceLevel.MID
            elif years <= 8:
                detected_level = ExperienceLevel.SENIOR
            else:
                detected_level = ExperienceLevel.LEAD
        
        return {
            "level": detected_level,
            "years": years
        }
    
    def _parse_explicit_requirements(self, jd_text: str) -> List[str]:
        """
        Parse "Required" or "Must have" sections.
        Uses 35% weight in final scoring.
        """
        skills = []
        
        # Find required section
        for pattern in self.REQUIRED_SECTION_PATTERNS:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                # Extract content after the header until next section
                start = match.end()
                # Find next section header
                end = len(jd_text)
                for next_pattern in (self.REQUIRED_SECTION_PATTERNS + 
                                     self.PREFERRED_SECTION_PATTERNS):
                    next_match = re.search(next_pattern, jd_text[start:], re.IGNORECASE)
                    if next_match:
                        end = min(end, start + next_match.start())
                
                section_text = jd_text[start:end]
                section_skills = self.extractor.extract_skills_flat(section_text)
                skills.extend(section_skills)
                break
        
        # If no explicit section found, extract from full text
        if not skills:
            skills = self.extractor.extract_skills_flat(jd_text)
        
        return list(set(skills))
    
    def _parse_preferred_skills(self, jd_text: str) -> List[str]:
        """
        Parse "Preferred" or "Nice to have" sections.
        Uses 15% weight in final scoring.
        """
        skills = []
        
        for pattern in self.PREFERRED_SECTION_PATTERNS:
            match = re.search(pattern, jd_text, re.IGNORECASE)
            if match:
                start = match.end()
                # Find next section header
                end = len(jd_text)
                for next_pattern in self.REQUIRED_SECTION_PATTERNS:
                    next_match = re.search(next_pattern, jd_text[start:], re.IGNORECASE)
                    if next_match:
                        end = min(end, start + next_match.start())
                
                section_text = jd_text[start:end]
                section_skills = self.extractor.extract_skills_flat(section_text)
                skills.extend(section_skills)
                break
        
        return list(set(skills))
    
    def _categorize_skills_for_role(
        self,
        role_key: str,
        experience_level: ExperienceLevel,
        required_skills: List[str],
        preferred_skills: List[str],
        jd_text: str
    ) -> SkillCategories:
        """
        Categorize skills with appropriate weights based on role and level.
        """
        categories = SkillCategories()
        
        # Get role-specific core skills if available
        all_roles = {}
        all_roles.update(self.data._job_roles.get('job_roles', {}))
        all_roles.update(self.data._job_roles.get('additional_job_roles', {}))
        
        role_data = all_roles.get(role_key, {})
        role_core_skills = set(s.lower() for s in role_data.get('core_skills', []))
        
        # Normalize skills
        required_normalized = set(s.lower() for s in required_skills)
        preferred_normalized = set(s.lower() for s in preferred_skills)
        
        # Combine with role-defined core skills
        all_core = required_normalized | role_core_skills
        
        # Apply experience level adjustments
        is_junior = experience_level in [
            ExperienceLevel.INTERN, 
            ExperienceLevel.ENTRY, 
            ExperienceLevel.JUNIOR
        ]
        
        # Categorize
        for skill in all_core:
            # Check if should be downweighted for junior roles
            if is_junior and self._is_senior_skill(skill):
                categories.downweighted.append(skill)
            else:
                categories.core_required.append(skill)
        
        for skill in preferred_normalized:
            if skill not in all_core:
                if is_junior and self._is_senior_skill(skill):
                    categories.downweighted.append(skill)
                else:
                    categories.core_preferred.append(skill)
        
        # Extract any remaining skills from JD as secondary
        all_jd_skills = set(s.lower() for s in self.extractor.extract_skills_flat(jd_text))
        categorized = (
            set(categories.core_required) | 
            set(categories.core_preferred) | 
            set(categories.downweighted)
        )
        categories.secondary = list(all_jd_skills - categorized)
        
        # Adjust weights based on experience level
        if is_junior:
            categories.core_weight = 0.60
            categories.preferred_weight = 0.25
            categories.secondary_weight = 0.15
        else:
            categories.core_weight = 0.50
            categories.preferred_weight = 0.30
            categories.secondary_weight = 0.20
        
        return categories
    
    def _is_senior_skill(self, skill: str) -> bool:
        """Check if a skill should be downweighted for junior roles."""
        skill_lower = skill.lower()
        for senior_skill in self.SENIOR_ONLY_SKILLS:
            if senior_skill in skill_lower or skill_lower in senior_skill:
                return True
        return False
    
    def get_canonical_skill(self, skill: str) -> str:
        """Get canonical form of a skill (handles semantic equivalents)."""
        skill_lower = skill.lower()
        
        # Check if it's an equivalent
        if skill_lower in self._equivalent_to_canonical:
            return self._equivalent_to_canonical[skill_lower]
        
        # Check if it's already canonical
        if skill_lower in self.SKILL_EQUIVALENTS:
            return skill_lower
        
        return skill_lower
    
    def skills_match(self, skill1: str, skill2: str) -> bool:
        """Check if two skills match (including semantic equivalents)."""
        canonical1 = self.get_canonical_skill(skill1)
        canonical2 = self.get_canonical_skill(skill2)
        return canonical1 == canonical2


# Singleton instance
jd_analyzer = JDAnalyzer()
