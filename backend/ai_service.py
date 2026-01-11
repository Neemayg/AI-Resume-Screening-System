
# backend/ai_service.py
"""
ENHANCED AI RESUME SCREENING SERVICE
Integrated with InternScorer (Fixed Denominators) and Human-Like Summaries.
"""

import re
from typing import Dict, List, Tuple, Optional
import traceback

# Import the new Fixed Engine
from intern_scorer import InternScorer
# Import the Summary Generator
from summary_generator import SummaryGenerator, InternSummaryGenerator


class AIResumeScreener:
    """
    Wrapper for the InternScorer to maintain compatibility with the API.
    """
    
    def __init__(self):
        self.scorer = InternScorer()
        # Initialize natural language generators
        self.summary_generator = SummaryGenerator(variety_mode=True)
        self.intern_generator = InternSummaryGenerator(variety_mode=True)
    
    def screen_resume(self, resume_text: str, jd_text: str, resume_name: str = "Candidate") -> Dict:
        """
        Screen resume using the fixed InternScorer engine.
        Generate rich summaries using SummaryGenerator.
        """
        # 1. SCORE using Fixed Engine
        score_result = self.scorer.score_resume(resume_text, jd_text)
        
        # 2. Extract Data for Summary
        breakdown = score_result["breakdown"]
        final_score = score_result["final_score"]
        match_category = score_result["match_category"]
        
        # Prepare data for SummaryGenerator
        # InternScorer returns 'explicit_matches', 'semantic_matches' (list of strings or dicts?)
        # InternScorer returns lists of strings for 'explicit_matches' and 'missing'
        # 'semantic_matches' might be list of strings. Let's check intern_scorer.py.
        # Yes, they are lists of strings.
        
        summary_input = {
            "name": resume_name,
            "match_score": final_score,
            "role_fit": "Data Science Intern", # Fixed scope as per InternScorer
            "matched_core": breakdown["core_skills"]["explicit_matches"] + breakdown["core_skills"]["semantic_matches"],
            "missing_core": breakdown["core_skills"]["missing"],
            "matched_preferred": breakdown["preferred_skills"]["explicit_matches"] + breakdown["preferred_skills"]["semantic_matches"],
            "missing_preferred": [], # InternScorer doesn't explicitly return missing preferred in the same list structure easily, but we can assume empty or ignored for summary
            "matched_soft": [], # Not handled by InternScorer
            "exp_years": 0, # Ignored for interns as per Anti-Hallucination Rule 1
            "is_student": True, # Assume student for Intern Scorer context
            "experience_data": {
                "formatted_experience": "Calculated based on projects", 
                "experience_level": "Intern"
            }
        }
        
        # 3. GENERATE SUMMARY
        # Always use Intern generator for this fixed context
        narrative_result = self.intern_generator.generate_detailed_narrative(summary_input)
        
        # 4. FORMAT RESPONSE
        return {
            "final_score": final_score,
            "match_category": match_category,
            "breakdown": {
                "core_skills": breakdown["core_skills"],
                "preferred_skills": breakdown["preferred_skills"],
                "text_similarity": breakdown["text_similarity"],
                "experience_fit": breakdown["intern_bonus"] # Map Bonus to Exp Fit slot for frontend compat
            },
            "recommendation": narrative_result["action"],
            "narrative_summary": narrative_result["summary"],
            "headline": narrative_result["headline"],
            "detailed_analysis": {
                "resume_skills_found": summary_input["matched_core"] + summary_input["matched_preferred"],
                "jd_skills_found": [], # Not dynamic anymore
                "experience_data": summary_input["experience_data"]
            }
        }

# ═══════════════════════════════════════════════════════════════════
# ADAPTER
# ═══════════════════════════════════════════════════════════════════

class AIAdapter:
    def __init__(self):
        self.screener = AIResumeScreener()

    def analyze_resume(self, resume_text: str, jd_text: str, name: str = "Unknown") -> Dict:
        try:
            result = self.screener.screen_resume(resume_text, jd_text, name)
            
            rich_summary = result.get("narrative_summary", "Summary unavailable.")
            # Add breakdown to summary text for visibility
            core_score = result['breakdown']['core_skills']['score']
            pref_score = result['breakdown']['preferred_skills']['score']
            bonus_score = result['breakdown']['experience_fit']['score']
            
            score_breakdown = f"Score Breakdown: Core {core_score:.1f} | Pref {pref_score:.1f} | Bonus {bonus_score:.1f}"
            combined_summary = f"{rich_summary}\n\n({score_breakdown})"

            return {
                "candidate_name": name,
                "match_score": int(result["final_score"]),
                "fit_level": result["match_category"].split(" ")[0], 
                "matched_skills": 
                    result["breakdown"]["core_skills"]["explicit_matches"] + 
                    result["breakdown"]["core_skills"]["semantic_matches"],
                "missing_skills": result["breakdown"]["core_skills"]["missing"],
                "summary": combined_summary,
                "recommendation": result["recommendation"],
                "experience_years": 0.5 # Placeholder for frontend display
            }
        except Exception as e:
            print(f"Error in analyze_resume: {e}")
            traceback.print_exc()
            return {
                "candidate_name": name,
                "match_score": 0,
                "fit_level": "Error",
                "matched_skills": [],
                "missing_skills": [],
                "summary": "An error occurred during analysis.",
                "recommendation": "Review manually",
                "experience_years": 0
            }

    def improve_job_description(self, text: str) -> Dict:
        return {"improved_text": text, "suggestions": []}
        
    def compare_candidates(self, candidates: List[Dict], jd_text: str) -> Dict:
        """
        Compare two candidates and generate a recommendation.
        """
        if len(candidates) < 2:
            return {"comparison": {"winner": "N/A", "summary": "Need at least 2 candidates to compare."}}
            
        c1 = candidates[0]
        c2 = candidates[1]
        
        # Determine winner
        if c1["match_score"] > c2["match_score"]:
            winner = c1
            loser = c2
        else:
            winner = c2
            loser = c1
            
        diff = winner["match_score"] - loser["match_score"]
        
        # Generate Summary
        if diff < 5:
            summary = (
                f"Both candidates are very similar in profile. {winner['candidate_name']} is slightly ahead "
                f"({winner['match_score']} vs {loser['match_score']}). Consider interviewing both."
            )
        else:
            summary = (
                f"{winner['candidate_name']} is the clear recommendation with a {diff:.1f} point lead. "
                f"They demonstrate stronger alignment with the core requirements. "
                f"({winner['match_score']} vs {loser['match_score']})."
            )
            
        return {
            "comparison": {
                "winner": winner["candidate_name"],
                "summary": summary
            }
        }
        
    def generate_full_report(self, results: List[Dict], jd_text: str) -> Dict:
        return {"report": "N/A"}

ai_service = AIAdapter()
