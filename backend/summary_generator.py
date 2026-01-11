
# backend/summary_generator.py
"""
RULE-BASED NATURAL LANGUAGE SUMMARY GENERATOR
Produces human-like recruiter notes without external LLM APIs
"""

import random
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


# ═══════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

class MatchTier(Enum):
    """Score-based candidate tiers"""
    EXCELLENT = "excellent"    # 85-100%
    STRONG = "strong"          # 70-84%
    MODERATE = "moderate"      # 50-69%
    DEVELOPING = "developing"  # 30-49%
    WEAK = "weak"              # 0-29%


class ExperienceLevel(Enum):
    """Experience-based categorization"""
    STUDENT = "student"           # 0 years, currently studying
    FRESH_GRAD = "fresh_grad"     # 0-0.5 years
    EARLY_CAREER = "early_career" # 0.5-2 years
    MID_LEVEL = "mid_level"       # 2-5 years
    SENIOR = "senior"             # 5-10 years
    EXPERT = "expert"             # 10+ years


@dataclass
class CandidateProfile:
    """Structured candidate data for summary generation"""
    name: str
    match_score: float
    role_fit: str
    matched_core: List[str]
    missing_core: List[str]
    matched_preferred: List[str]
    missing_preferred: List[str]
    matched_soft: List[str]
    exp_years: float
    is_student: bool = False
    projects_count: int = 0
    education_match: bool = True
    experience_data: Optional[Dict] = None  # Added for FIX 3 integration
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CandidateProfile':
        """Create profile from dictionary input"""
        return cls(
            name=data.get("name", "Candidate"),
            match_score=data.get("match_score", 0),
            role_fit=data.get("role_fit", "position"),
            matched_core=data.get("matched_core", []),
            missing_core=data.get("missing_core", []),
            matched_preferred=data.get("matched_preferred", []),
            missing_preferred=data.get("missing_preferred", []),
            matched_soft=data.get("matched_soft", []),
            exp_years=data.get("exp_years", 0),
            is_student=data.get("is_student", False),
            projects_count=data.get("projects_count", 0),
            education_match=data.get("education_match", True),
            experience_data=data.get("experience_data", None)
        )


# ═══════════════════════════════════════════════════════════════════════════
# PHRASE LIBRARIES - The "Vocabulary" of Our Generator
# ═══════════════════════════════════════════════════════════════════════════

class PhraseLibrary:
    """
    Contains all phrase variations for natural language generation.
    Multiple options per category prevent repetitive output.
    """
    
    # ─────────────────────────────────────────────────────────────────────
    # OPENING PHRASES (Score-based)
    # ─────────────────────────────────────────────────────────────────────
    
    OPENINGS = {
        MatchTier.EXCELLENT: [
            "{name} is an exceptional candidate for the {role} position with an outstanding {score}% match.",
            "{name} stands out as a top-tier applicant for the {role} role, achieving an impressive {score}% alignment.",
            "With a remarkable {score}% match score, {name} emerges as an excellent fit for the {role} position.",
            "{name} demonstrates exceptional qualifications for the {role} role with a {score}% match rate.",
            "Our analysis identifies {name} as a prime candidate for {role}, scoring an excellent {score}%.",
        ],
        MatchTier.STRONG: [
            "{name} is a strong contender for the {role} role with a solid {score}% match.",
            "{name} presents a compelling profile for the {role} position, achieving {score}% alignment.",
            "With a {score}% match score, {name} shows strong potential for the {role} role.",
            "{name} demonstrates considerable promise for the {role} position with {score}% compatibility.",
            "Our screening identifies {name} as a highly suitable candidate for {role} at {score}%.",
        ],
        MatchTier.MODERATE: [
            "{name} shows moderate alignment with the {role} position at {score}%.",
            "{name} presents a reasonable match for the {role} role with a {score}% score.",
            "At {score}% compatibility, {name} demonstrates partial alignment with {role} requirements.",
            "{name} meets several key criteria for the {role} position with {score}% match.",
            "With a {score}% score, {name} shows foundational fit for the {role} role.",
        ],
        MatchTier.DEVELOPING: [
            "{name} shows developing potential for the {role} role at {score}%.",
            "At {score}%, {name} demonstrates emerging alignment with {role} requirements.",
            "{name} presents a developing profile for the {role} position with {score}% match.",
            "With room for growth, {name} scores {score}% for the {role} position.",
            "{name} shows foundational qualities for {role}, currently at {score}% alignment.",
        ],
        MatchTier.WEAK: [
            "{name} currently shows limited alignment with the {role} position at {score}%.",
            "At {score}%, {name}'s profile indicates gaps relative to {role} requirements.",
            "{name} demonstrates {score}% match for {role}, suggesting significant upskilling needed.",
            "Our analysis indicates {name} at {score}% may require substantial substantial development for {role}.",
            "With a {score}% score, {name} shows notable gaps for the {role} position.",
        ],
    }
    
    # ─────────────────────────────────────────────────────────────────────
    # SKILL DESCRIPTION PHRASES
    # ─────────────────────────────────────────────────────────────────────
    
    CORE_SKILLS_POSITIVE = {
        "strong": [
            "demonstrates solid proficiency in core technical areas like {skills}",
            "shows strong command of essential skills including {skills}",
            "brings proven expertise in {skills}",
            "exhibits comprehensive knowledge of {skills}",
            "possesses robust capabilities in {skills}",
        ],
        "moderate": [
            "has working knowledge of {skills}",
            "shows familiarity with {skills}",
            "demonstrates foundational skills in {skills}",
            "brings experience with {skills}",
            "has exposure to {skills}",
        ],
        "few": [
            "shows some relevant experience with {skills}",
            "has touched on {skills}",
            "demonstrates initial exposure to {skills}",
        ]
    }
    
    CORE_SKILLS_GAPS = {
        "gentle": [
            "would benefit from developing expertise in {skills}",
            "could strengthen their profile by upskilling in {skills}",
            "has opportunity to grow in {skills}",
            "would be well-served by gaining experience in {skills}",
            "shows room for development in {skills}",
        ],
        "direct": [
            "lacks experience in critical areas: {skills}",
            "is missing key technical requirements: {skills}",
            "needs to acquire skills in {skills}",
            "has gaps in required competencies: {skills}",
        ],
        "single_skill": [
            "while {skills} represents a growth opportunity",
            "though gaining {skills} proficiency would strengthen their profile",
            "with {skills} being an area for development",
            "where {skills} knowledge could be enhanced",
        ]
    }
    
    PREFERRED_SKILLS = {
        "positive": [
            "Additionally, {name} brings valuable bonus skills in {skills}",
            "As a plus, expertise in {skills} adds depth to their profile",
            "Their knowledge of {skills} provides additional value",
            "Bonus competencies in {skills} enhance their candidacy",
            "They also offer proficiency in preferred areas like {skills}",
        ],
    }
    
    SOFT_SKILLS = {
        "strong": [
            "A unique strength is their {skills} ability, adding team value",
            "Their {skills} capabilities bring valuable interpersonal dynamics",
            "Notable soft skills in {skills} complement their technical profile",
            "{skills} abilities suggest strong collaborative potential",
            "Beyond technical skills, {skills} indicates well-rounded capabilities",
        ],
        "multiple": [
            "Demonstrates strong interpersonal qualities including {skills}",
            "Brings valuable soft skills: {skills}",
            "Shows well-developed {skills} capabilities",
        ],
    }
    
    # NOTE: Experience phrases are now handled dynamically by _generate_experience_section
    # but we keep this as fallback structure if needed
    EXPERIENCE = {} 
    
    RECOMMENDATIONS = {
        MatchTier.EXCELLENT: [
            "Strongly recommended for immediate interview consideration.",
            "Highly recommended to advance to interview stage.",
            "Suggest prioritizing for first-round interviews.",
            "Top candidate - recommend expedited interview process.",
        ],
        MatchTier.STRONG: [
            "Recommended for interview consideration.",
            "Good candidate for the interview shortlist.",
            "Suggest including in interview pool.",
            "Merits interview opportunity to assess cultural fit.",
        ],
        MatchTier.MODERATE: [
            "Consider for interview if top candidates unavailable.",
            "Potential backup candidate - interview at discretion.",
            "May warrant interview to assess trainability.",
            "Consider if willing to invest in skill development.",
        ],
        MatchTier.DEVELOPING: [
            "Recommend holding for future opportunities.",
            "Better suited for more junior openings.",
            "Consider for talent pipeline, not immediate role.",
            "May revisit after candidate gains more experience.",
        ],
        MatchTier.WEAK: [
            "Not recommended for current position.",
            "Significant gaps suggest poor fit for this role.",
            "Consider redirecting to more suitable openings.",
            "Profile does not meet minimum requirements.",
        ]
    }
    
    TRANSITIONS = {
        "addition": ["Additionally,", "Furthermore,", "Moreover,", "Also,", "Beyond this,"],
        "contrast": ["However,", "That said,", "On the other hand,", "While", "Although"],
        "emphasis": ["Notably,", "Importantly,", "Significantly,", "Particularly,"],
        "conclusion": ["Overall,", "In summary,", "To conclude,", "All things considered,"],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SUMMARY GENERATOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

class SummaryGenerator:
    """
    Generates human-like recruiter summaries using rule-based NLG.
    """
    
    def __init__(self, variety_mode: bool = True, seed: Optional[int] = None):
        self.variety_mode = variety_mode
        self.phrases = PhraseLibrary()
        if seed is not None:
            random.seed(seed)
    
    def generate_narrative(self, data: Dict) -> str:
        profile = CandidateProfile.from_dict(data)
        tier = self._determine_tier(profile.match_score)
        exp_level = self._determine_experience_level(profile.exp_years, profile.is_student)
        
        sections = []
        sections.append(self._generate_opening(profile, tier))
        
        core_section = self._generate_core_skills_section(profile, tier)
        if core_section: sections.append(core_section)
        
        preferred_section = self._generate_preferred_skills_section(profile)
        if preferred_section: sections.append(preferred_section)
        
        soft_section = self._generate_soft_skills_section(profile)
        if soft_section: sections.append(soft_section)
        
        # New robust experience section
        exp_section = self._generate_experience_section(profile, exp_level)
        if exp_section: sections.append(exp_section)
        
        sections.append(self._generate_recommendation(tier))
        
        return self._combine_sections(sections)
    
    def generate_detailed_narrative(self, data: Dict) -> Dict:
        profile = CandidateProfile.from_dict(data)
        tier = self._determine_tier(profile.match_score)
        
        return {
            "summary": self.generate_narrative(data),
            "headline": self._generate_headline(profile, tier),
            "tier": tier.value,
            "key_strengths": self._extract_key_strengths(profile),
            "key_gaps": self._extract_key_gaps(profile),
            "action": self._get_action_item(tier),
            "confidence": self._calculate_confidence(profile)
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # FIX 3: CLEAN EXPERIENCE SECTION GENERATION
    # ═══════════════════════════════════════════════════════════════════
    
    def _generate_experience_section(self, profile: CandidateProfile, exp_level: ExperienceLevel) -> str:
        """Generate experience section with clean formatting."""
        
        # Get formatted experience string
        exp_data = getattr(profile, 'experience_data', None)
        
        if exp_data:
            formatted = exp_data.get("formatted_experience", "")
            exp_level_label = exp_data.get("experience_level", "")
        else:
            formatted = self._format_experience_value(
                profile.exp_years,
                profile.is_student
            )
            # Implied level logic for fallback
            # exp_level_label = self._get_level_label(profile.exp_years, profile.is_student)
        
        first_name = profile.name.split()[0]
        
        # ─────────────────────────────────────────────────────────────
        # STUDENT / ENTRY LEVEL HANDLING
        # ─────────────────────────────────────────────────────────────
        if profile.is_student:
            if profile.exp_years == 0:
                return f"As a current student, {first_name} brings fresh academic perspective and enthusiasm for learning."
            elif profile.exp_years < 1:
                months = int(profile.exp_years * 12)
                return f"As a student with {months} months of practical experience, {first_name} combines academic knowledge with hands-on exposure."
            else:
                return f"Despite being a student, {first_name} has gained {formatted} of relevant experience."
        
        # ─────────────────────────────────────────────────────────────
        # ENTRY LEVEL (non-student)
        # ─────────────────────────────────────────────────────────────
        if profile.exp_years == 0 and not profile.is_student:
            return f"{first_name} is an entry-level candidate eager to begin their professional journey."
        
        # ─────────────────────────────────────────────────────────────
        # STANDARD EXPERIENCE PHRASES
        # ─────────────────────────────────────────────────────────────
        if profile.exp_years < 1:
            months = int(profile.exp_years * 12)
            if months <= 1:
                 return f"Just starting their career, {first_name} is building foundational expertise."
            return f"With {months} months of experience, {first_name} is building foundational expertise."
        
        if profile.exp_years < 2:
            return f"{first_name}'s {formatted} of experience demonstrates early career progression."
        
        if profile.exp_years < 5:
            return f"With {formatted} of experience, {first_name} brings solid professional maturity."
        
        if profile.exp_years < 10:
            return f"{first_name}'s {formatted} of experience reflects established industry expertise."
        
        return f"With {formatted} of experience, {first_name} brings exceptional depth of knowledge."
    
    def _format_experience_value(self, years: float, is_student: bool) -> str:
        """Format experience value cleanly (Fallback logic)."""
        if is_student and years == 0:
            return "current student status"
        
        if years == 0:
            return "entry level"
        
        months = int(years * 12)
        
        if months < 12:
            if months == 1:
                return "1 month"
            return f"{months} months"
        
        if years == 1.0:
            return "1 year"
        
        if years < 2:
            remaining = months % 12
            if remaining == 0:
                return "1 year"
            return f"1 year and {remaining} months"
        
        # Round to clean number for 2+ years
        if years % 1 == 0:
            return f"{int(years)} years"
        
        return f"{years:.1f} years"
    
    # ═══════════════════════════════════════════════════════════════════
    # OTHER GENERATORS (Unchanged)
    # ═══════════════════════════════════════════════════════════════════
    
    def _determine_tier(self, score: float) -> MatchTier:
        if score >= 85: return MatchTier.EXCELLENT
        elif score >= 70: return MatchTier.STRONG
        elif score >= 50: return MatchTier.MODERATE
        elif score >= 30: return MatchTier.DEVELOPING
        else: return MatchTier.WEAK
    
    def _determine_experience_level(self, years: float, is_student: bool) -> ExperienceLevel:
        if is_student: return ExperienceLevel.STUDENT
        elif years < 0.5: return ExperienceLevel.FRESH_GRAD
        elif years < 2: return ExperienceLevel.EARLY_CAREER
        elif years < 5: return ExperienceLevel.MID_LEVEL
        elif years < 10: return ExperienceLevel.SENIOR
        else: return ExperienceLevel.EXPERT

    def _generate_opening(self, profile: CandidateProfile, tier: MatchTier) -> str:
        template = self._select_phrase(self.phrases.OPENINGS[tier])
        first_name = profile.name.split()[0] if profile.name else "Candidate"
        return template.format(name=first_name, role=profile.role_fit, score=int(profile.match_score))
    
    def _generate_core_skills_section(self, profile: CandidateProfile, tier: MatchTier) -> str:
        parts = []
        if profile.matched_core:
            intensity = "strong" if len(profile.matched_core) >= 4 else "moderate" if len(profile.matched_core) >= 2 else "few"
            template = self._select_phrase(self.phrases.CORE_SKILLS_POSITIVE[intensity])
            skills_str = self._format_skill_list(profile.matched_core)
            parts.append(f"They {template.format(skills=skills_str)}")
        
        if profile.missing_core:
            style = "single_skill" if len(profile.missing_core) == 1 else "gentle" if tier in [MatchTier.EXCELLENT, MatchTier.STRONG] else "direct"
            template = self._select_phrase(self.phrases.CORE_SKILLS_GAPS[style])
            skills_str = self._format_skill_list(profile.missing_core)
            if style == "single_skill":
                parts.append(template.format(skills=skills_str))
            else:
                transition = self._select_phrase(self.phrases.TRANSITIONS["contrast"])
                parts.append(f"{transition} they {template.format(skills=skills_str)}")
        
        if not profile.matched_core and not profile.missing_core: return ""
        return self._connect_parts(parts)
    
    def _generate_preferred_skills_section(self, profile: CandidateProfile) -> str:
        if not profile.matched_preferred: return ""
        first_name = profile.name.split()[0]
        template = self._select_phrase(self.phrases.PREFERRED_SKILLS["positive"])
        skills_str = self._format_skill_list(profile.matched_preferred[:3])
        return template.format(name=first_name, skills=skills_str)
    
    def _generate_soft_skills_section(self, profile: CandidateProfile) -> str:
        if not profile.matched_soft: return ""
        template = self._select_phrase(self.phrases.SOFT_SKILLS["strong" if len(profile.matched_soft) == 1 else "multiple"])
        skills_str = self._format_skill_list(profile.matched_soft)
        return template.format(skills=skills_str)
    
    def _generate_recommendation(self, tier: MatchTier) -> str:
        return self._select_phrase(self.phrases.RECOMMENDATIONS[tier])
    
    # Helpers
    def _select_phrase(self, phrases: List[str]) -> str:
        if not phrases: return ""
        return random.choice(phrases) if self.variety_mode else phrases[0]
    
    def _format_skill_list(self, skills: List[str], max_items: int = 4) -> str:
        if not skills: return ""
        skills = skills[:max_items]
        if len(skills) == 1: return skills[0]
        elif len(skills) == 2: return f"{skills[0]} and {skills[1]}"
        else: return f"{', '.join(skills[:-1])}, and {skills[-1]}"
    
    def _connect_parts(self, parts: List[str]) -> str:
        if not parts: return ""
        result = parts[0]
        for part in parts[1:]:
            if part[0].isupper(): result += ". " + part
            else: result += ", " + part
        return result
    
    def _combine_sections(self, sections: List[str]) -> str:
        sections = [s.strip() for s in sections if s and s.strip()]
        if not sections: return "Unable to generate summary."
        formatted = []
        for section in sections:
            section = section.strip()
            if section and not section.endswith(('.', '!', '?')): section += '.'
            formatted.append(section)
        return ' '.join(formatted)
    
    def _generate_headline(self, profile: CandidateProfile, tier: MatchTier) -> str:
        first_name = profile.name.split()[0]
        headlines = {
            MatchTier.EXCELLENT: f"⭐ {first_name}: Exceptional {profile.role_fit} Candidate ({profile.match_score}%)",
            MatchTier.STRONG: f"✓ {first_name}: Strong {profile.role_fit} Candidate ({profile.match_score}%)",
            MatchTier.MODERATE: f"● {first_name}: Moderate {profile.role_fit} Match ({profile.match_score}%)",
            MatchTier.DEVELOPING: f"○ {first_name}: Developing Candidate ({profile.match_score}%)",
            MatchTier.WEAK: f"✗ {first_name}: Below Requirements ({profile.match_score}%)",
        }
        return headlines.get(tier, f"{first_name}: {profile.match_score}% Match")
    
    def _extract_key_strengths(self, profile: CandidateProfile) -> List[str]:
        strengths = []
        if len(profile.matched_core) >= 3: strengths.append(f"Strong core skills ({len(profile.matched_core)} matched)")
        if profile.matched_soft: strengths.append(f"Good soft skills ({', '.join(profile.matched_soft[:2])})")
        if profile.matched_preferred: strengths.append(f"Has preferred skills ({len(profile.matched_preferred)})")
        if profile.projects_count >= 3: strengths.append(f"Project experience ({profile.projects_count} projects)")
        return strengths
    
    def _extract_key_gaps(self, profile: CandidateProfile) -> List[str]:
        gaps = []
        if profile.missing_core: gaps.append(f"Missing: {', '.join(profile.missing_core[:3])}")
        if profile.exp_years < 1 and "senior" in profile.role_fit.lower(): gaps.append("Limited experience")
        return gaps
    
    def _get_action_item(self, tier: MatchTier) -> str:
        actions = {
            MatchTier.EXCELLENT: "Schedule interview immediately",
            MatchTier.STRONG: "Add to interview shortlist",
            MatchTier.MODERATE: "Review with hiring manager",
            MatchTier.DEVELOPING: "Hold for future openings",
            MatchTier.WEAK: "Send rejection notice",
        }
        return actions.get(tier, "Review manually")
    
    def _calculate_confidence(self, profile: CandidateProfile) -> str:
        data_points = 0
        if profile.matched_core or profile.missing_core: data_points += 2
        if profile.matched_preferred: data_points += 1
        if profile.matched_soft: data_points += 1
        return "High" if data_points >= 3 else "Medium"


class InternSummaryGenerator(SummaryGenerator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # We can still append phrases if we want to use them for other sections
        self.phrases.CORE_SKILLS_GAPS["gentle"] = [
            "has opportunity to learn {skills} during the internship",
            "can develop {skills} through mentorship",
        ]


class TechnicalSummaryGenerator(SummaryGenerator):
    pass # Use default

class BatchSummaryGenerator:
    def __init__(self, generator: Optional[SummaryGenerator] = None):
        self.generator = generator or SummaryGenerator()
    
    def generate_batch(self, candidates: List[Dict]) -> List[Dict]:
        sorted_candidates = sorted(candidates, key=lambda x: x.get("match_score", 0), reverse=True)
        results = []
        for rank, candidate in enumerate(sorted_candidates, 1):
            result = self.generator.generate_detailed_narrative(candidate)
            result["rank"] = rank
            results.append(result)
        return results
