"""
Centralized configuration for all scoring weights and thresholds.
Modify these values to tune the matching accuracy.
"""

# =============================================================================
# SCORING COMPONENT WEIGHTS (must sum to 1.0)
# =============================================================================
COMPONENT_WEIGHTS = {
    "skill_match_score": 0.35,        # How well skills match
    "experience_match_score": 0.20,   # Experience level alignment
    "semantic_similarity_score": 0.20, # Semantic understanding
    "role_alignment_score": 0.15,     # Role context matching
    "education_match_score": 0.10     # Education requirements
}

# =============================================================================
# SKILL CATEGORY WEIGHTS
# =============================================================================
SKILL_WEIGHTS = {
    "core_required": 1.0,      # Must-have skills
    "core_preferred": 0.7,     # Strongly preferred
    "secondary": 0.4,          # Nice to have
    "tools": 0.3,              # Tools and platforms
    "soft_skills": 0.2         # Communication, leadership, etc.
}

# =============================================================================
# PENALTY CONFIGURATION
# =============================================================================
PENALTIES = {
    "missing_core_skill": -0.15,      # Per missing core skill
    "missing_core_skill_max": -0.45,  # Maximum penalty for missing core
    "experience_gap_per_year": -0.05, # Per year below required
    "experience_gap_max": -0.20,      # Maximum experience penalty
    "missing_education": -0.10,       # Missing required degree
    "missing_certification": -0.05    # Missing preferred cert
}

# =============================================================================
# BONUS CONFIGURATION
# =============================================================================
BONUSES = {
    "extra_relevant_skill": 0.02,     # Per additional relevant skill
    "extra_skill_max": 0.10,          # Maximum bonus
    "exceeds_experience": 0.05,       # More experience than required
    "higher_education": 0.05,         # Higher degree than required
    "leadership_keywords": 0.03       # Leadership experience
}

# =============================================================================
# THRESHOLDS
# =============================================================================
THRESHOLDS = {
    "min_skill_match_ratio": 0.3,     # Minimum to be considered
    "high_confidence_score": 0.75,    # Strong match threshold
    "medium_confidence_score": 0.50,  # Moderate match
    "semantic_similarity_min": 0.4    # Minimum semantic relevance
}

# =============================================================================
# EXPERIENCE LEVEL MAPPING
# =============================================================================
EXPERIENCE_LEVELS = {
    "intern": {"min": 0, "max": 0},
    "entry": {"min": 0, "max": 2},
    "junior": {"min": 1, "max": 3},
    "mid": {"min": 3, "max": 5},
    "senior": {"min": 5, "max": 8},
    "lead": {"min": 6, "max": 10},
    "staff": {"min": 8, "max": 12},
    "principal": {"min": 10, "max": 15},
    "director": {"min": 10, "max": 20}
}

# =============================================================================
# EDUCATION HIERARCHY
# =============================================================================
EDUCATION_HIERARCHY = {
    "phd": 5,
    "doctorate": 5,
    "masters": 4,
    "mba": 4,
    "bachelors": 3,
    "associate": 2,
    "diploma": 1,
    "certification": 1,
    "high_school": 0
}
