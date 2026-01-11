
# backend/config.py
"""
CONFIGURATION SETTINGS
Easily adjustable parameters for tuning the system
"""

class ScoringConfig:
    """Scoring weights and thresholds - adjust these to tune results"""
    
    # ═══════════════════════════════════════════════════════════════
    # WEIGHT CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    # These must sum to 1.0
    WEIGHTS = {
        "core_skills": 0.50,      # Most important - must-have skills
        "preferred_skills": 0.25, # Nice-to-have skills  
        "text_similarity": 0.15,  # Overall context match
        "experience_fit": 0.10    # Experience level fit
    }
    
    # For different role levels, adjust weights:
    INTERN_WEIGHTS = {
        "core_skills": 0.55,      # Skills matter most
        "preferred_skills": 0.25,
        "text_similarity": 0.15,
        "experience_fit": 0.05    # Experience barely matters
    }
    
    SENIOR_WEIGHTS = {
        "core_skills": 0.40,
        "preferred_skills": 0.20,
        "text_similarity": 0.15,
        "experience_fit": 0.25    # Experience matters more
    }
    
    # ═══════════════════════════════════════════════════════════════
    # THRESHOLD CONFIGURATION  
    # ═══════════════════════════════════════════════════════════════
    
    THRESHOLDS = {
        "excellent": 80,  # 80%+ = Excellent Match
        "high": 65,       # 65-79% = High Match
        "medium": 45,     # 45-64% = Medium Match
        "low": 0          # Below 45% = Low Match
    }
    
    # ═══════════════════════════════════════════════════════════════
    # SKILL MATCHING CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    # Minimum core skills to extract if auto-detection fails
    MIN_CORE_SKILLS = 3
    MAX_CORE_SKILLS = 8
    
    # Semantic matching confidence threshold
    SEMANTIC_MATCH_THRESHOLD = 0.7
    
    # ═══════════════════════════════════════════════════════════════
    # EXPERIENCE CONFIGURATION
    # ═══════════════════════════════════════════════════════════════
    
    # Maximum years to consider (prevents parsing errors)
    MAX_EXPERIENCE_YEARS = 50
    
    # Intern candidates: maximum years before "overqualified"
    INTERN_MAX_YEARS = 2