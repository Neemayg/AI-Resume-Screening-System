
# backend/anti_hallucination_rules.py
"""
RULES TO PREVENT SCORE/DATA HALLUCINATION
"""

class AntiHallucinationGuard:
    """
    Guardrails to prevent the system from producing
    unrealistic or fabricated data.
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 1: NEVER USE PARSED EXPERIENCE YEARS FOR SCORING
    # ═══════════════════════════════════════════════════════════════════════
    
    EXPERIENCE_RULES = """
    ❌ DO NOT: Parse "2022-2026" as "4 years experience"
    ❌ DO NOT: Use experience_years in score calculation
    ❌ DO NOT: Penalize candidates with 0 years
    
    ✅ DO: Ignore experience entirely for intern roles
    ✅ DO: Use project count as BONUS only
    ✅ DO: Accept student/intern status as valid
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 2: FIXED DENOMINATORS ONLY
    # ═══════════════════════════════════════════════════════════════════════
    
    DENOMINATOR_RULES = """
    ❌ DO NOT: Count all JD keywords as denominator
    ❌ DO NOT: Parse JD text to dynamically set skill requirements
    ❌ DO NOT: Allow denominator > 10 for core skills
    
    ✅ DO: Use FIXED skill lists (6 core, 7 preferred)
    ✅ DO: Keep denominators constant across all resumes
    ✅ DO: Only update skill lists through config changes
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 3: BOUNDED SCORE COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    SCORE_BOUNDS = {
        "core_score": {"min": 0, "max": 55},
        "preferred_score": {"min": 0, "max": 20},
        "text_similarity": {"min": 0, "max": 15},
        "bonus_score": {"min": 0, "max": 10},
        "final_score": {"min": 0, "max": 100}
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 4: SEMANTIC MATCH LIMITS
    # ═══════════════════════════════════════════════════════════════════════
    
    SEMANTIC_RULES = """
    ❌ DO NOT: Give semantic matches full credit (1.0)
    ❌ DO NOT: Allow cascading semantic inference
    ❌ DO NOT: Match parent concepts to child requirements
    
    ✅ DO: Give semantic matches exactly 0.5 credit
    ✅ DO: Only allow child→parent semantic matching
    ✅ DO: Require explicit indicators for semantic credit
    """
    
    # ═══════════════════════════════════════════════════════════════════════
    # RULE 5: CONSISTENT RANKING
    # ═══════════════════════════════════════════════════════════════════════
    
    RANKING_RULES = """
    ❌ DO NOT: Use random factors in scoring
    ❌ DO NOT: Allow ties at same score to flip order
    ❌ DO NOT: Change scores between runs
    
    ✅ DO: Use deterministic calculations only
    ✅ DO: Sort by score descending, then by name alphabetically
    ✅ DO: Cache and reuse extracted skills
    """
    
    @staticmethod
    def validate_score(score_result: dict) -> bool:
        """Validate score result against hallucination rules."""
        
        errors = []
        
        # Check final score bounds
        if not (0 <= score_result["final_score"] <= 100):
            errors.append(f"Final score out of bounds: {score_result['final_score']}")
        
        # Check component bounds
        breakdown = score_result.get("breakdown", {})
        
        core_score = breakdown.get("core_skills", {}).get("score", 0)
        if not (0 <= core_score <= 55):
            errors.append(f"Core score out of bounds: {core_score}")
        
        pref_score = breakdown.get("preferred_skills", {}).get("score", 0)
        if not (0 <= pref_score <= 20):
            errors.append(f"Preferred score out of bounds: {pref_score}")
        
        # Check for experience-based penalties
        if "experience" in str(breakdown).lower():
            if "penalty" in str(breakdown).lower():
                errors.append("Experience penalty detected - should not exist for interns")
        
        if errors:
            print("⚠️ HALLUCINATION DETECTED:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
