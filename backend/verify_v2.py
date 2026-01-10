"""
Verification script for the V2 Advanced Scoring System.
Tests the integration of all new modules and runs a sample ranking.
"""

import sys
import os
import logging
import json

# Add backend to path so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, 'backend')
sys.path.append(backend_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify_system():
    logger.info("Initializing V2 System Modules...")

    try:
        # 1. Import Dependencies
        from utils.data_loader import data_loader
        from matching.text_normalizer import text_normalizer
        from matching.skill_extractor import skill_extractor
        from matching.role_detector import role_detector
        from matching.section_parser import section_parser
        from matching.skill_matcher import skill_matcher
        from matching.semantic_scorer import semantic_scorer
        from scoring.score_combiner import ScoreCombiner
        from scoring.role_weight_adjuster import RoleWeightAdjuster
        from ranking.ranking_engine import RankingEngine
        from ranking.result_formatter import RankingResultFormatter

        logger.info("✅ All modules imported successfully.")
        
        # 2. Check Data Loading
        if not data_loader._job_roles:
            logger.warning("⚠️ Data loader has no job roles. Check datasets/job_roles.json")
        else:
            logger.info(f"✅ Data loader initialized with {len(data_loader._job_roles)} roles.")

        # 3. Initialize Ranking Engine
        # The RankingEngine expects instances/modules for its components
        # In our implementation, we used singletons for most utils
        
        # We also need to pass the raw data dicts to the engine as per its init signature:
        # __init__(self, skill_normalizer, role_detector, skill_matcher, experience_analyzer, 
        #          semantic_analyzer, penalty_calculator, job_roles_data, skills_data)
        
        # Note: We didn't explicitly implement 'experience_analyzer' or 'penalty_calculator' 
        # as separate modules in the user's list, but RankingEngine uses them.
        # Looking at RankingEngine code provided:
        # self.experience_analyzer = experience_analyzer
        # ...
        # if self.experience_analyzer: experience_requirements = ...
        
        # So we can pass None for now if they aren't critical or use matching placeholders.
        # But wait, we DID define logic for experience inside RankingEngine's score_single_resume 
        # via _calculate_experience_score? 
        # Actually typical design would be modular. Ideally section_parser handles extraction.
        # Let's check RankingEngine again... it calls self.experience_analyzer.extract_requirements(jd_text)
        
        # We'll create a simple Mock for components we might have missed or that are integrated elsewhere
        class MockAnalyzer:
            def extract_requirements(self, text):
                return {'min_years': 3, 'preferred_years': 5}
                
        ranking_engine = RankingEngine(
            skill_normalizer=skill_extractor,  # skill_extractor has normalize_skill_list
            role_detector=role_detector,
            skill_matcher=skill_matcher,
            experience_analyzer=MockAnalyzer(), # Placeholder
            semantic_analyzer=semantic_scorer,
            penalty_calculator=None, # Will use defaults
            job_roles_data=data_loader._job_roles,
            skills_data=data_loader._skills_dataset
        )
        
        logger.info("✅ Ranking Engine initialized.")

        # 4. Run Sample Analysis
        logger.info("\n--- Running Sample Analysis ---")
        
        # Mock JD
        jd_text = """
        Job Title: Senior Software Engineer
        
        We are looking for a Senior Python Developer with 5+ years of experience.
        Must have strong skills in Python, Django, and React.
        Experience with AWS and Docker is required.
        Knowledge of Machine Learning is a plus.
        """
        
        # Mock Resumes
        resumes = [
            {
                "id": "1",
                "name": "Alice Python",
                "text": "Senior Python Developer. 6 years experience. Expert in Python, Django, Flask. Familiar with AWS and Docker.",
                "extracted_skills": ["Python", "Django", "Flask", "AWS", "Docker", "Git"],
                "experience": [{"title": "Dev", "years": 6}],
                "total_experience": 6
            },
            {
                "id": "2",
                "name": "Bob Java",
                "text": "Java Developer. 2 years experience. Knowing Java, Spring. Learning Python.",
                "extracted_skills": ["Java", "Spring", "Python"],
                "experience": [{"title": "Jr Dev", "years": 2}],
                "total_experience": 2
            }
        ]
        
        # Run Ranking
        results = ranking_engine.rank_resumes(resumes, jd_text, parallel=False)
        
        logger.info(f"✅ Analysis complete. Ranked {len(results)} resumes.")
        
        # 5. Output Results
        formatted = RankingResultFormatter.to_api_response(
            results, 
            ranking_engine.analyze_job_description(jd_text)
        )
        
        print("\n=== Ranking Results ===")
        print(json.dumps(formatted['summary'], indent=2))
        
        print("\n=== Top Candidate Details ===")
        if results:
            top = results[0]
            print(f"Name: {top.candidate_name}")
            print(f"Score: {top.final_score}")
            print(f"Category: {top.category}")
            print(f"Breakdown: {json.dumps(top.score_breakdown, indent=2)}")
            
    except ImportError as e:
        logger.error(f"❌ Import Error: {e}")
    except Exception as e:
        logger.error(f"❌ Runtime Error: {e}", exc_info=True)

if __name__ == "__main__":
    verify_system()
