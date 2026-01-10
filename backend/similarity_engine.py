"""
Similarity Engine Module
Implements TF-IDF vectorization and cosine similarity for resume matching.
"""

import logging
from typing import List, Dict, Tuple, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import TFIDF_CONFIG, SKILL_WEIGHTS
from nlp_engine import nlp_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityEngine:
    """
    Engine for computing similarity between job descriptions and resumes.
    Uses TF-IDF vectorization and cosine similarity with skill-based boosting.
    """
    
    def __init__(self):
        """Initialize the similarity engine."""
        self.vectorizer = None
        self.jd_vector = None
        self.jd_skills = []
        self.jd_text = ""
        
        logger.info("Similarity Engine initialized")
    
    def initialize_vectorizer(self, documents: List[str]) -> None:
        """
        Initialize and fit the TF-IDF vectorizer on documents.
        
        Args:
            documents: List of documents (JD + all resumes) to fit vectorizer
        """
        self.vectorizer = TfidfVectorizer(
            max_features=TFIDF_CONFIG['max_features'],
            ngram_range=TFIDF_CONFIG['ngram_range'],
            min_df=TFIDF_CONFIG['min_df'],
            max_df=TFIDF_CONFIG['max_df'],
            sublinear_tf=TFIDF_CONFIG['sublinear_tf'],
            lowercase=True,
            stop_words='english'
        )
        
        # Fit on all documents
        self.vectorizer.fit(documents)
        logger.info(f"Vectorizer fitted with vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def set_job_description(self, jd_text: str, preprocessed_jd: str) -> Dict:
        """
        Set the job description for comparison.
        
        Args:
            jd_text: Original job description text
            preprocessed_jd: Preprocessed job description text
            
        Returns:
            Dictionary with JD analysis results
        """
        self.jd_text = jd_text
        self.jd_preprocessed = preprocessed_jd
        
        # Extract skills from JD
        self.jd_skills = nlp_engine.extract_skills_flat(jd_text)
        
        # Detect job role
        self.detected_role, self.role_confidence = nlp_engine.detect_job_role(jd_text)
        
        # Detect experience level
        self.experience_level, self.years = nlp_engine.detect_experience_level(jd_text)
        
        # Extract keywords
        self.jd_keywords = nlp_engine.extract_keywords(jd_text, top_n=30)
        
        logger.info(f"JD processed - Role: {self.detected_role}, Skills found: {len(self.jd_skills)}")
        
        return {
            "detected_role": self.detected_role,
            "role_confidence": self.role_confidence,
            "experience_level": self.experience_level,
            "years_required": self.years,
            "skills_required": self.jd_skills,
            "top_keywords": [kw for kw, _ in self.jd_keywords[:10]]
        }
    
    def compute_similarity(self, resume_text: str, preprocessed_resume: str) -> Dict:
        """
        Compute comprehensive similarity between JD and a resume.
        
        Args:
            resume_text: Original resume text
            preprocessed_resume: Preprocessed resume text
            
        Returns:
            Dictionary with similarity metrics
        """
        if self.vectorizer is None:
            # Fallback: create vectorizer on the fly
            self.initialize_vectorizer([self.jd_preprocessed, preprocessed_resume])
        
        # Transform texts to TF-IDF vectors
        jd_vector = self.vectorizer.transform([self.jd_preprocessed])
        resume_vector = self.vectorizer.transform([preprocessed_resume])
        
        # Compute cosine similarity
        tfidf_similarity = cosine_similarity(jd_vector, resume_vector)[0][0]
        
        # Compute skill-based similarity
        resume_skills = nlp_engine.extract_skills_flat(resume_text)
        skill_similarity = self._compute_skill_similarity(resume_skills)
        
        # Compute keyword overlap
        keyword_similarity = self._compute_keyword_similarity(resume_text)
        
        # Get matched and missing skills
        matched_skills = list(set(self.jd_skills) & set(resume_skills))
        missing_skills = list(set(self.jd_skills) - set(resume_skills))
        
        # Compute weighted final score
        final_score = self._compute_weighted_score(
            tfidf_similarity,
            skill_similarity,
            keyword_similarity
        )
        
        # Get skill breakdown by category
        skill_breakdown = self._get_skill_breakdown(resume_text)
        
        return {
            "tfidf_score": round(tfidf_similarity * 100, 2),
            "skill_score": round(skill_similarity * 100, 2),
            "keyword_score": round(keyword_similarity * 100, 2),
            "final_score": round(final_score * 100, 2),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills[:10],  # Limit to top 10
            "resume_skills": resume_skills,
            "skill_breakdown": skill_breakdown
        }
    
    def _compute_skill_similarity(self, resume_skills: List[str]) -> float:
        """
        Compute skill-based similarity score.
        
        Args:
            resume_skills: List of skills found in resume
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.jd_skills:
            return 0.5  # Default if no skills in JD
        
        jd_skills_lower = set(s.lower() for s in self.jd_skills)
        resume_skills_lower = set(s.lower() for s in resume_skills)
        
        # Exact matches
        exact_matches = jd_skills_lower & resume_skills_lower
        
        # Partial matches (one is substring of another)
        partial_matches = 0
        for jd_skill in jd_skills_lower:
            for resume_skill in resume_skills_lower:
                if jd_skill != resume_skill:
                    if jd_skill in resume_skill or resume_skill in jd_skill:
                        partial_matches += 0.5
        
        # Calculate score
        exact_score = len(exact_matches) / len(jd_skills_lower)
        partial_score = min(partial_matches / len(jd_skills_lower), 0.3)  # Cap partial contribution
        
        return min(exact_score + partial_score, 1.0)
    
    def _compute_keyword_similarity(self, resume_text: str) -> float:
        """
        Compute keyword overlap similarity.
        
        Args:
            resume_text: Resume text to compare
            
        Returns:
            Similarity score between 0 and 1
        """
        if not self.jd_keywords:
            return 0.5
        
        resume_lower = resume_text.lower()
        matches = 0
        
        for keyword, weight in self.jd_keywords:
            if keyword in resume_lower:
                matches += 1
        
        return matches / len(self.jd_keywords)
    
    def _compute_weighted_score(self, 
                                tfidf: float, 
                                skills: float, 
                                keywords: float) -> float:
        """
        Compute weighted final score from component scores.
        
        Weights (adjusted to be less strict):
        - TF-IDF: 30% (general content similarity)
        - Skills: 50% (specific skill matching - increased weight)
        - Keywords: 20% (keyword presence - increased)
        """
        weights = {
            'tfidf': 0.30,
            'skills': 0.50,  # Increased from 0.45
            'keywords': 0.20  # Increased from 0.15
        }
        
        weighted_score = (
            tfidf * weights['tfidf'] +
            skills * weights['skills'] +
            keywords * weights['keywords']
        )
        
        # Apply boost for decent skill matches (lowered threshold)
        if skills > 0.4:  # Changed from 0.8
            weighted_score = min(weighted_score * 1.15, 1.0)  # Increased boost
        
        # Apply bonus for any keyword matches
        if keywords > 0.3:
            weighted_score = min(weighted_score * 1.10, 1.0)
        
        return weighted_score
    
    def _get_skill_breakdown(self, resume_text: str) -> Dict[str, Dict]:
        """
        Get detailed skill breakdown by category.
        
        Args:
            resume_text: Resume text to analyze
            
        Returns:
            Dictionary with category breakdown
        """
        resume_skills_by_category = nlp_engine.extract_skills(resume_text)
        jd_skills_by_category = nlp_engine.extract_skills(self.jd_text)
        
        breakdown = {}
        
        for category in nlp_engine.get_skill_categories():
            jd_cat_skills = set(s.lower() for s in jd_skills_by_category.get(category, []))
            resume_cat_skills = set(s.lower() for s in resume_skills_by_category.get(category, []))
            
            if jd_cat_skills:
                matched = jd_cat_skills & resume_cat_skills
                breakdown[category] = {
                    "required": len(jd_cat_skills),
                    "matched": len(matched),
                    "percentage": round(len(matched) / len(jd_cat_skills) * 100, 1) if jd_cat_skills else 0,
                    "matched_skills": list(matched),
                    "missing_skills": list(jd_cat_skills - resume_cat_skills)
                }
        
        return breakdown

    def enhanced_compute_similarity(self, 
                                  resume_text: str, 
                                  resume_preprocessed: str,
                                  jd_role_detection: Optional[Dict] = None) -> Dict:
        """
        Compute similarity using advanced matching modules.
        Integrates Skill Matcher, Semantic Scorer, and Role Detection.
        """
        # 1. Base TF-IDF Score
        base_result = self.compute_similarity(resume_text, resume_preprocessed)
        
        try:
            # 2. Advanced Scoring Components
            from matching.skill_matcher import skill_matcher
            from matching.semantic_scorer import semantic_scorer
            from scoring.enhanced_score_combiner import score_combiner
            from scoring.role_weight_adjuster import role_weight_adjuster
            
            # Semantic Score
            semantic_result = semantic_scorer.score_resume_for_jd(
                resume_text, self.jd_text
            )
            semantic_score_val = semantic_result.get('overall_score', 0.0)
            
            # Skill Match Score
            
            # Skill Match Score
            # Extract skills first
            resume_skills_dict = nlp_engine.extract_skills(resume_text)
            resume_skill_list = resume_skills_dict.get('all', [])
            
            # Use pre-extracted JD skills if available, otherwise extract
            jd_skill_list = getattr(self, 'jd_skills', [])
            if not jd_skill_list:
                 jd_skill_list = nlp_engine.extract_skills(self.jd_text).get('all', [])

            skill_match_result = skill_matcher.match_skills(
                jd_skill_list, resume_skill_list
            )
            
            # Detect Resume Role
            resume_role = None
            if jd_role_detection:
                from matching.role_detector import role_detector
                resume_role = role_detector.detect_role(resume_text)
            
            # Adjust Weights based on Role
            weights = None
            if jd_role_detection and resume_role:
                weights = role_weight_adjuster.adjust_weights(
                    jd_role_detection, resume_role
                )
            
            # Combine Scores
            final_score_data = score_combiner.combine_scores(
                tfidf_score=base_result['tfidf_score'],
                semantic_score=semantic_score_val,
                skill_match_data=skill_match_result,
                custom_weights=weights
            )
            
            # Merge results
            base_result.update({
                'final_score': final_score_data.final_score,
                'semantic_score': semantic_score_val * 100,
                'skill_match_score': skill_match_result.match_score,
                'matched_skills': skill_match_result.matched_skills,
                'missing_skills': skill_match_result.missing_skills,
                'enhanced_breakdown': final_score_data.breakdown,
                'skill_match_detail': {
                    'matched_core': skill_match_result.matched_skills, # Simplified for now
                    'missing_core': skill_match_result.missing_skills,
                    'coverage': skill_match_result.match_score / 100
                }
            })
            
        except Exception as e:
            logger.error(f"Enhanced scoring failed: {e}")
            # Fallback will use base_result as is
            
        return base_result
    
    def batch_compute_similarity(self, 
                                 resumes: List[Dict[str, str]]) -> List[Dict]:
        """
        Compute similarity for multiple resumes.
        
        Args:
            resumes: List of dictionaries with 'original' and 'preprocessed' keys
            
        Returns:
            List of similarity results
        """
        # Initialize vectorizer with all documents
        all_docs = [self.jd_preprocessed] + [r['preprocessed'] for r in resumes]
        self.initialize_vectorizer(all_docs)
        
        results = []
        for resume in resumes:
            similarity = self.compute_similarity(
                resume['original'],
                resume['preprocessed']
            )
            results.append(similarity)
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get the most important features from the TF-IDF vectorizer.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of (feature, importance) tuples
        """
        if self.vectorizer is None:
            return []
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Transform JD and get feature weights
        jd_tfidf = self.vectorizer.transform([self.jd_preprocessed])
        feature_weights = jd_tfidf.toarray()[0]
        
        # Sort by weight
        sorted_indices = np.argsort(feature_weights)[::-1][:top_n]
        
        return [
            (feature_names[i], round(feature_weights[i], 4))
            for i in sorted_indices
            if feature_weights[i] > 0
        ]


# Global similarity engine instance
similarity_engine = SimilarityEngine()