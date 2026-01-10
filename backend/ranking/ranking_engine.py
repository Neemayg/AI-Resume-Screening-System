"""
Resume Ranking Engine
Orchestrates the entire ranking process and produces final rankings
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Adjusted imports for backend structure (removing 'app.')
from scoring.score_combiner import ScoreCombiner, ScoreComponents, WeightConfiguration
from scoring.role_weight_adjuster import RoleWeightAdjuster


@dataclass
class RankedResume:
    """Represents a ranked resume with all scoring details"""
    resume_id: str
    candidate_name: str
    final_score: float
    rank: int
    category: str
    score_breakdown: Dict
    matched_skills: List[str]
    missing_critical_skills: List[str]
    experience_match: Dict
    strengths: List[str]
    weaknesses: List[str]
    recommendation: str
    processing_time_ms: float


class RankingEngine:
    """
    Main engine for ranking resumes against a job description
    """
    
    def __init__(
        self,
        skill_normalizer,
        role_detector,
        skill_matcher,
        experience_analyzer,
        semantic_analyzer,
        penalty_calculator,
        job_roles_data: Dict,
        skills_data: Dict
    ):
        self.skill_normalizer = skill_normalizer
        self.role_detector = role_detector
        self.skill_matcher = skill_matcher
        self.experience_analyzer = experience_analyzer
        self.semantic_analyzer = semantic_analyzer
        self.penalty_calculator = penalty_calculator
        
        self.job_roles = job_roles_data
        self.skills_data = skills_data
        
        self.weight_adjuster = RoleWeightAdjuster(job_roles_data)
        # Initialize default scorer, will be overridden often
        self.score_combiner = ScoreCombiner()
    
    def analyze_job_description(self, jd_text: str) -> Dict:
        """
        Comprehensive JD analysis
        """
        # Detect role
        role_detection = self.role_detector.detect_role(jd_text)
        detected_role_id = role_detection.get('detected_role_key') # Adjusted key from role_detector
        detected_role_info = role_detection.get('role_data', {}) # Adjusted key
        
        # Extract and normalize skills from JD (using skill_matcher logic or extractor directly)
        # Assuming skill_matcher has extract_skills_from_text or similar,
        # but in previous steps skill_matcher.match_skills takes extracted lists.
        # So we use self.skill_matcher.extractor probably
        jd_skills_raw = self.skill_matcher.extractor.extract_skills_flat(jd_text)
        jd_skills_normalized = self.skill_normalizer.normalize_skill_list(jd_skills_raw)
        
        # Get role-specific core skills
        core_skills = detected_role_info.get('core_skills', [])
        secondary_skills = detected_role_info.get('secondary_skills', []) # 'optional' -> 'secondary'
        
        # Merge JD skills with role-defined skills
        all_required_skills = list(set(jd_skills_normalized + core_skills))
        
        # Categorize skills
        categorized_skills = self._categorize_skills(all_required_skills, core_skills)
        
        # Extract experience requirements
        # Assuming experience_analyzer exists and has extract_requirements
        # If not passed in init, we might need a placeholder or import
        if self.experience_analyzer:
            experience_requirements = self.experience_analyzer.extract_requirements(jd_text)
        else:
            # Fallback if module missing
            experience_requirements = {'min_years': 0, 'preferred_years': 0}
            
        # Get adjusted weights for this role
        adjusted_weights = self.weight_adjuster.get_adjusted_weights(detected_role_id)
        
        return {
            'detected_role_id': detected_role_id,
            'detected_role': role_detection,
            'role_info': detected_role_info,
            'all_required_skills': all_required_skills,
            'core_skills': categorized_skills['core'],
            'important_skills': categorized_skills['important'],
            'nice_to_have_skills': categorized_skills['nice_to_have'],
            'experience_requirements': experience_requirements,
            'adjusted_weights': adjusted_weights,
            'jd_text': jd_text
        }
    
    def _categorize_skills(
        self, 
        all_skills: List[str], 
        core_skills: List[str]
    ) -> Dict[str, List[str]]:
        """
        Categorize skills into core, important, and nice-to-have
        """
        core_set = set(skill.lower() for skill in core_skills)
        
        categorized = {
            'core': [],
            'important': [],
            'nice_to_have': []
        }
        
        for skill in all_skills:
            skill_lower = skill.lower()
            # skill_info from self.skill_matcher.data (DataLoader instance)
            skill_info = self.skill_matcher.data.get_skill_info(skill_lower) or {}
            skill_weight = skill_info.get('weight', 0.5)
            
            if skill_lower in core_set:
                categorized['core'].append(skill)
            elif skill_weight >= 0.7:
                categorized['important'].append(skill)
            else:
                categorized['nice_to_have'].append(skill)
        
        return categorized
    
    def score_single_resume(
        self, 
        resume_data: Dict, 
        jd_analysis: Dict
    ) -> RankedResume:
        """
        Score a single resume against the analyzed JD
        """
        start_time = datetime.now()
        
        resume_id = resume_data.get('id', 'unknown')
        candidate_name = resume_data.get('name', 'Unknown Candidate')
        resume_text = resume_data.get('text', '')
        resume_skills = resume_data.get('extracted_skills', []) or []
        resume_experience = resume_data.get('experience', []) # List of dicts usually
        if isinstance(resume_experience, list):
             # Convert to dict format expected by _calculate_experience_score
             # Or adjust _calculate_experience_score to handle list
             # For now, let's assume we calculate total years here or helper does it
             total_years = 0 # Placeholder, logic in SectionParser
             # We might need to rely on pre-calculated 'total_experience' if available
             total_years = resume_data.get('total_experience', 0)
             resume_experience_dict = {'total_years': total_years, 'entries': resume_experience}
        else:
            resume_experience_dict = resume_experience

        resume_education = resume_data.get('education', [])
        # Similar simple conversion for education
        resume_education_dict = {'highest_degree': 'Unknown', 'field': ''} # Simplification for now
        
        # Normalize resume skills
        resume_skills_normalized = self.skill_normalizer.normalize_skill_list(resume_skills)
        
        # Calculate individual scores
        skill_scores = self._calculate_skill_scores(
            resume_skills_normalized, 
            jd_analysis
        )
        
        experience_score = self._calculate_experience_score(
            resume_experience_dict,
            jd_analysis['experience_requirements']
        )
        
        semantic_score = self.semantic_analyzer.calculate_similarity(
            resume_text,
            jd_analysis['jd_text']
        )
        
        role_alignment_score = self._calculate_role_alignment(
            resume_data,
            jd_analysis
        )
        
        education_score = self._calculate_education_score(
            resume_education_dict,
            jd_analysis
        )
        
        # Calculate penalties
        # Assume penalty_calculator has calculate_all_penalties
        # If not provided, use default 0s
        if self.penalty_calculator:
             penalties = self.penalty_calculator.calculate_all_penalties(
                resume_skills_normalized,
                jd_analysis['core_skills'],
                resume_experience_dict,
                jd_analysis['experience_requirements']
            )
        else:
            penalties = {'missing_core_skills': 0.0, 'experience_gap': 0.0}

        
        # Calculate bonuses
        bonuses = self._calculate_bonuses(resume_data, jd_analysis)
        
        # Build score components
        components = ScoreComponents(
            skill_match_score=skill_scores['overall'],
            core_skill_score=skill_scores['core'],
            optional_skill_score=skill_scores['optional'],
            experience_score=experience_score['score'],
            education_score=education_score,
            semantic_similarity_score=semantic_score,
            role_alignment_score=role_alignment_score,
            missing_core_skills_penalty=penalties.get('missing_core_skills', 0.0),
            experience_gap_penalty=penalties.get('experience_gap', 0.0),
            education_mismatch_penalty=penalties.get('education_mismatch', 0),
            exact_role_match_bonus=bonuses['exact_role'],
            certification_bonus=bonuses['certifications'],
            extra_relevant_skills_bonus=bonuses['extra_skills']
        )
        
        # Get role-adjusted weights
        adjusted_weights = jd_analysis.get('adjusted_weights', WeightConfiguration())
        combiner = ScoreCombiner(adjusted_weights)
        
        # Calculate final score
        final_result = combiner.calculate_final_score(components)
        
        # Generate insights
        insights = self._generate_insights(
            resume_skills_normalized,
            jd_analysis,
            skill_scores,
            experience_score
        )
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return RankedResume(
            resume_id=resume_id,
            candidate_name=candidate_name,
            final_score=final_result['final_score'],
            rank=0,  # Will be set during ranking
            category=combiner.get_score_category(final_result['final_score']),
            score_breakdown=final_result['breakdown'],
            matched_skills=skill_scores['matched_skills'],
            missing_critical_skills=skill_scores['missing_core'],
            experience_match=experience_score,
            strengths=insights['strengths'],
            weaknesses=insights['weaknesses'],
            recommendation=insights['recommendation'],
            processing_time_ms=processing_time
        )
    
    def _calculate_skill_scores(
        self, 
        resume_skills: List[str], 
        jd_analysis: Dict
    ) -> Dict:
        """Calculate all skill-related scores"""
        resume_skills_set = set(skill.lower() for skill in resume_skills)
        
        core_skills = jd_analysis['core_skills']
        important_skills = jd_analysis['important_skills']
        nice_to_have = jd_analysis['nice_to_have_skills']
        
        # Core skill matching
        matched_core = []
        missing_core = []
        for skill in core_skills:
            if self._skill_matches(skill, resume_skills_set):
                matched_core.append(skill)
            else:
                missing_core.append(skill)
        
        core_score = len(matched_core) / max(len(core_skills), 1)
        
        # Important skill matching
        matched_important = [
            skill for skill in important_skills
            if self._skill_matches(skill, resume_skills_set)
        ]
        important_score = len(matched_important) / max(len(important_skills), 1)
        
        # Nice-to-have matching
        matched_nice = [
            skill for skill in nice_to_have
            if self._skill_matches(skill, resume_skills_set)
        ]
        nice_score = len(matched_nice) / max(len(nice_to_have), 1)
        
        # Overall skill score (weighted)
        overall_score = (
            core_score * 0.5 +
            important_score * 0.35 +
            nice_score * 0.15
        )
        
        # All matched skills
        all_matched = matched_core + matched_important + matched_nice
        
        return {
            'overall': overall_score,
            'core': core_score,
            'important': important_score,
            'optional': nice_score,
            'matched_skills': all_matched,
            'missing_core': missing_core,
            'matched_core': matched_core,
            'matched_important': matched_important,
            'matched_nice_to_have': matched_nice
        }
    
    def _skill_matches(self, skill: str, resume_skills_set: set) -> bool:
        """Check if a skill matches considering aliases"""
        skill_lower = skill.lower()
        
        # Direct match
        if skill_lower in resume_skills_set:
            return True
        
        # Check aliases
        # Assuming skills_data is a dict or using data_loader
        # self.skills_data passed in init is likely data_loader._skills_dataset
        # But structure is category -> list of skills. This is hard to lookup directly.
        # Better to use data_loader.get_skill_info if available (which is self.skill_matcher.data)
        
        skill_info = self.skill_matcher.data.get_skill_info(skill_lower)
        if not skill_info:
             return False

        aliases = [a.lower() for a in self.skill_matcher.data.get_skill_info(skill_lower).get('related', [])] # Wait 'related' is not aliases. 'aliases' is not stored in lookup directly in previous implementation?
        # Actually data_loader._build_skill_lookup stores 'canonical'.
        # If skill_lower is canonical, we only check it.
        # The lookup keys ARE the aliases too.
        # So simpler logic:
        # Check if canonical of skill matches canonical of anything in resume_skills_set
        
        target_canonical = self.skill_matcher.data.get_canonical_skill(skill_lower)
        if not target_canonical:
            return False
            
        # This is expensive 0(N*M), optimized via normalized set check
        # But we already normalized resume_skills to canonicals in score_single_resume!
        # So simple Check is enough:
        return target_canonical.lower() in resume_skills_set

    
    def _calculate_experience_score(
        self, 
        resume_experience: Dict, 
        requirements: Dict
    ) -> Dict:
        """Calculate experience match score"""
        resume_years = resume_experience.get('total_years', 0)
        # Handle cases where min_years might be string or None
        try:
             required_years = float(requirements.get('min_years', 0))
        except:
             required_years = 0
             
        preferred_years = requirements.get('preferred_years', required_years)
        
        if resume_years >= preferred_years:
            score = 1.0
            status = 'exceeds'
        elif resume_years >= required_years:
            # Partial score for meeting minimum but not preferred
            score = 0.7 + (0.3 * (resume_years - required_years) / max(preferred_years - required_years, 1))
            status = 'meets'
        elif resume_years >= required_years * 0.7:
            # Slightly under but close
            score = 0.4 + (0.3 * resume_years / max(required_years, 1))
            status = 'slightly_under'
        else:
            score = max(0.1, resume_years / max(required_years, 1) * 0.4)
            status = 'under'
        
        return {
            'score': score,
            'resume_years': resume_years,
            'required_years': required_years,
            'preferred_years': preferred_years,
            'status': status
        }
    
    def _calculate_role_alignment(
        self, 
        resume_data: Dict, 
        jd_analysis: Dict
    ) -> float:
        """Calculate how well past roles align with target role"""
        past_roles = resume_data.get('past_roles', []) # Might need extraction
        target_role_id = jd_analysis.get('detected_role_id')
        target_role_info = jd_analysis.get('role_info', {})
        
        if not past_roles or not target_role_id:
            return 0.5  # Neutral score
        
        target_title = target_role_info.get('title', '').lower()
        # target_aliases = [a.lower() for a in target_role_info.get('aliases', [])]
        # related_roles = [r.lower() for r in target_role_info.get('related_roles', [])]
        
        # Simplified for now as extraction logic for past_roles is complex
        return 0.5 
    
    def _calculate_education_score(
        self, 
        resume_education: Dict, 
        jd_analysis: Dict
    ) -> float:
        """Calculate education match score"""
        # role_info = jd_analysis.get('role_info', {})
        # required_education = role_info.get('education_requirements', {})
        
        # Simplified placeholder logic
        return 0.8  
    
    def _calculate_bonuses(
        self, 
        resume_data: Dict, 
        jd_analysis: Dict
    ) -> Dict:
        """Calculate bonus scores"""
        bonuses = {
            'exact_role': 0.0,
            'certifications': 0.0,
            'extra_skills': 0.0
        }
        return bonuses
    
    def _generate_insights(
        self,
        resume_skills: List[str],
        jd_analysis: Dict,
        skill_scores: Dict,
        experience_score: Dict
    ) -> Dict:
        """Generate human-readable insights"""
        strengths = []
        weaknesses = []
        
        # Skill-based insights
        if skill_scores['core'] >= 0.8:
            strengths.append(f"Strong match on core skills ({len(skill_scores['matched_core'])}/{len(jd_analysis['core_skills'])})")
        elif skill_scores['core'] < 0.5:
            weaknesses.append(f"Missing critical skills: {', '.join(skill_scores['missing_core'][:3])}")
        
        if skill_scores['important'] >= 0.7:
            strengths.append("Good coverage of important technical skills")
        
        # Experience insights
        if experience_score['status'] == 'exceeds':
            strengths.append(f"Experience exceeds requirements ({experience_score['resume_years']} years)")
        elif experience_score['status'] == 'under':
            weaknesses.append(f"Experience below requirements ({experience_score['resume_years']}/{experience_score['required_years']} years)")
        
        # Generate recommendation
        if len(strengths) > len(weaknesses) and skill_scores['core'] >= 0.6:
            recommendation = "Recommended for interview"
        elif skill_scores['core'] >= 0.5:
            recommendation = "Consider for interview with reservations"
        else:
            recommendation = "Does not meet minimum requirements"
        
        return {
            'strengths': strengths,
            'weaknesses': weaknesses,
            'recommendation': recommendation
        }
    
    def rank_resumes(
        self,
        resumes: List[Dict],
        jd_text: str,
        parallel: bool = True,
        max_workers: int = 4
    ) -> List[RankedResume]:
        """
        Main method to rank all resumes against a JD
        """
        # Analyze JD once
        jd_analysis = self.analyze_job_description(jd_text)
        
        # Score all resumes
        if parallel and len(resumes) > 1:
            scored_resumes = self._score_resumes_parallel(
                resumes, jd_analysis, max_workers
            )
        else:
            scored_resumes = [
                self.score_single_resume(resume, jd_analysis)
                for resume in resumes
            ]
        
        # Sort by final score (descending)
        scored_resumes.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign ranks
        for i, resume in enumerate(scored_resumes):
            resume.rank = i + 1
        
        return scored_resumes
    
    def _score_resumes_parallel(
        self,
        resumes: List[Dict],
        jd_analysis: Dict,
        max_workers: int
    ) -> List[RankedResume]:
        """Score resumes in parallel"""
        scored_resumes = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.score_single_resume, resume, jd_analysis): resume
                for resume in resumes
            }
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    scored_resumes.append(result)
                except Exception as e:
                    resume = futures[future]
                    # print(f"Error scoring resume {resume.get('id')}: {e}")
                    # Silently skip or log
        
        return scored_resumes
