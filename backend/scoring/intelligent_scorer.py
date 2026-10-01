"""
Intelligent Resume Scoring Engine
Role-aware weighted scoring with semantic matching for AI/ML internship evaluation.
Fixes: score compression, improper skill weighting, hallucination prevention.
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.matching.jd_analyzer import jd_analyzer, JDAnalysisResult, ExperienceLevel
from backend.config.weights_config import EXPERIENCE_LEVEL_WEIGHTS, SKILL_EQUIVALENTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RecommendationTier(Enum):
    """Tiered recommendation system instead of binary."""
    STRONG_FIT = "Strong Fit"
    GOOD_FIT = "Good Fit"
    PARTIAL_FIT = "Partial Fit"
    WEAK_FIT = "Weak Fit"
    NOT_RECOMMENDED = "Not Recommended"


@dataclass
class ScoringBreakdown:
    """Transparent scoring breakdown for explainability."""
    core_skill_score: float = 0.0
    preferred_skill_score: float = 0.0
    secondary_skill_score: float = 0.0
    semantic_similarity_score: float = 0.0
    learning_potential_bonus: float = 0.0
    
    # Matched skills for transparency
    matched_core: List[str] = field(default_factory=list)
    matched_preferred: List[str] = field(default_factory=list)
    matched_secondary: List[str] = field(default_factory=list)
    missing_core: List[str] = field(default_factory=list)
    
    # Final computed values
    raw_score: float = 0.0
    normalized_score: float = 0.0


@dataclass
class ResumeEvaluation:
    """Complete evaluation result with explainability."""
    candidate_name: str
    match_score: int  # 0-100
    recommendation: RecommendationTier
    fit_level: str
    
    # Skill analysis
    matched_skills: List[str]
    missing_core_skills: List[str]
    extra_relevant_skills: List[str]
    
    # Breakdown for transparency
    scoring_breakdown: ScoringBreakdown
    
    # Explanation
    explanation: str
    strengths: List[str]
    improvement_areas: List[str]
    
    # Raw data (extraction-only, no hallucination)
    extracted_experience_years: int
    extracted_education: str
    extracted_current_title: str


class IntelligentScoringEngine:
    """
    Role-aware scoring engine with semantic matching.
    Designed for fair intern/entry-level evaluation.
    """
    
    # Semantic skill equivalents (concept-level matching)
    SKILL_SYNONYMS = {
        # Python ecosystem
        "python": {"python3", "py", "python programming", "pycharm"},
        "pandas": {"dataframe", "data wrangling", "pd"},
        "numpy": {"np", "numerical computing", "array operations"},
        "scikit-learn": {"sklearn", "sk-learn", "scikit learn"},
        "machine learning": {"ml", "ml models", "modeling", "predictive modeling"},
        "deep learning": {"dl", "neural networks", "nn", "neural nets"},
        "tensorflow": {"tf", "tf2", "tensorflow 2"},
        "pytorch": {"torch", "py torch"},
        
        # Data skills
        "eda": {"exploratory data analysis", "data exploration", "data profiling"},
        "data analysis": {"data analytics", "analytical skills", "analysis"},
        "nlp": {"natural language processing", "text processing", "text analytics", "text mining"},
        "computer vision": {"cv", "image processing", "image recognition"},
        "data visualization": {"viz", "data viz", "visualization", "charts", "dashboards"},
        
        # Tools
        "jupyter": {"jupyter notebook", "notebooks", "ipython", "colab", "google colab"},
        "sql": {"mysql", "postgresql", "postgres", "sqlite", "database queries", "querying"},
        "git": {"github", "version control", "gitlab", "bitbucket"},
        "statistics": {"stats", "probability", "statistical analysis", "stat"},
        
        # ML/AI specific
        "classification": {"classifier", "classifiers", "categorization"},
        "regression": {"regressor", "predictive models"},
        "clustering": {"cluster analysis", "k-means", "unsupervised"},
        "feature engineering": {"feature extraction", "feature selection"},
    }
    
    # Learning potential indicators for interns
    LEARNING_INDICATORS = [
        r"coursework",
        r"academic\s+project",
        r"learning",
        r"studying",
        r"certification",
        r"online\s+course",
        r"bootcamp",
        r"currently\s+learning",
        r"self[-\s]taught",
        r"personal\s+project",
        r"hackathon",
        r"competition",
        r"kaggle",
    ]
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 3),  # Include trigrams for multi-word skills
            max_features=8000
        )
        self._build_synonym_lookup()
    
    def _build_synonym_lookup(self):
        """Build reverse lookup for synonyms."""
        self._synonym_to_canonical = {}
        for canonical, synonyms in self.SKILL_SYNONYMS.items():
            for syn in synonyms:
                self._synonym_to_canonical[syn.lower()] = canonical.lower()
            self._synonym_to_canonical[canonical.lower()] = canonical.lower()
    
    def get_canonical_skill(self, skill: str) -> str:
        """Get canonical form of a skill."""
        skill_lower = skill.lower().strip()
        return self._synonym_to_canonical.get(skill_lower, skill_lower)
    
    def skills_match(self, resume_skill: str, jd_skill: str) -> bool:
        """Check if two skills match semantically."""
        canonical_resume = self.get_canonical_skill(resume_skill)
        canonical_jd = self.get_canonical_skill(jd_skill)
        
        # Exact canonical match
        if canonical_resume == canonical_jd:
            return True
        
        # Check if one contains the other
        if canonical_resume in canonical_jd or canonical_jd in canonical_resume:
            return True
        
        return False
    
    def extract_skills_from_text(self, text: str) -> Set[str]:
        """
        Extract skills from text - EXTRACTION ONLY, no inference.
        Prevents hallucination by only returning explicitly mentioned skills.
        """
        text_lower = text.lower()
        found_skills = set()
        
        # All known skills from our synonym map
        all_skills = set()
        for canonical, synonyms in self.SKILL_SYNONYMS.items():
            all_skills.add(canonical.lower())
            all_skills.update(s.lower() for s in synonyms)
        
        # Add common tech skills
        tech_skills = {
            'python', 'java', 'javascript', 'c++', 'r', 'scala',
            'pandas', 'numpy', 'scipy', 'matplotlib', 'seaborn', 'plotly',
            'scikit-learn', 'sklearn', 'tensorflow', 'pytorch', 'keras',
            'machine learning', 'deep learning', 'neural networks',
            'nlp', 'natural language processing', 'computer vision',
            'eda', 'data analysis', 'data visualization', 'statistics',
            'sql', 'mysql', 'postgresql', 'mongodb', 'nosql',
            'git', 'github', 'jupyter', 'colab',
            'flask', 'fastapi', 'django', 'rest api', 'api',
            'aws', 'gcp', 'azure', 'docker', 'kubernetes',
            'linux', 'bash', 'shell scripting',
            'html', 'css', 'react', 'node.js',
            'data science', 'data engineering', 'analytics',
            'regression', 'classification', 'clustering',
            'feature engineering', 'model training', 'model evaluation',
            'tf-idf', 'word embeddings', 'transformers', 'bert', 'gpt',
            'random forest', 'xgboost', 'gradient boosting', 'svm',
            'cross-validation', 'hyperparameter tuning',
            'excel', 'tableau', 'power bi',
        }
        all_skills.update(tech_skills)
        
        # Extract only explicitly mentioned skills
        for skill in all_skills:
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                # Use canonical form
                canonical = self.get_canonical_skill(skill)
                found_skills.add(canonical)
        
        return found_skills
    
    def calculate_learning_potential(self, resume_text: str, experience_level: ExperienceLevel) -> float:
        """
        Calculate learning potential bonus for interns/entry-level.
        Only applies for junior roles.
        """
        if experience_level not in [ExperienceLevel.INTERN, ExperienceLevel.ENTRY, ExperienceLevel.JUNIOR]:
            return 0.0
        
        text_lower = resume_text.lower()
        matches = 0
        
        for pattern in self.LEARNING_INDICATORS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                matches += 1
        
        # Bonus up to 10% for strong learning indicators
        return min(0.10, matches * 0.02)
    
    def extract_candidate_info(self, resume_text: str) -> Dict:
        """
        Extract candidate info - STRICTLY from text, no inference.
        """
        lines = resume_text.split('\n')
        
        # Name extraction
        name = "Unknown Candidate"
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 40:
                skip_words = ['email', 'phone', '@', 'http', 'github', 'linkedin', 
                             'resume', 'cv', 'objective', 'summary', 'address']
                if not any(w in line.lower() for w in skip_words):
                    if re.match(r'^[A-Za-z\s\.\-]+$', line):
                        name = line.title()
                        break
        
        # Experience years - only explicit mentions
        exp_years = 0
        exp_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?experience',
            r'experience[:\s]+(\d+)\+?\s*years?',
        ]
        for pattern in exp_patterns:
            match = re.search(pattern, resume_text.lower())
            if match:
                exp_years = int(match.group(1))
                break
        
        # Education - only explicit mentions
        education = ""
        text_lower = resume_text.lower()
        if 'phd' in text_lower or 'doctorate' in text_lower:
            education = "PhD"
        elif any(x in text_lower for x in ["master's", 'masters', 'm.s.', 'mba', 'm.tech']):
            education = "Master's Degree"
        elif any(x in text_lower for x in ["bachelor's", 'bachelors', 'b.s.', 'b.tech', 'b.e.']):
            education = "Bachelor's Degree"
        elif any(x in text_lower for x in ['pursuing', 'student', 'currently studying']):
            education = "Currently Pursuing Degree"
        
        # Title extraction
        title = "Candidate"
        title_patterns = [
            r'(?:data\s+science\s+intern)',
            r'(?:ml\s+intern)',
            r'(?:ai\s+intern)',
            r'(?:software\s+(?:developer|engineer))',
            r'(?:data\s+(?:scientist|analyst|engineer))',
        ]
        for pattern in title_patterns:
            match = re.search(pattern, resume_text, re.IGNORECASE)
            if match:
                title = match.group().title()
                break
        
        return {
            "name": name,
            "experience_years": exp_years,
            "education": education,
            "title": title
        }
    
    def evaluate_resume(
        self,
        resume_text: str,
        jd_text: str,
        resume_name: str = "Candidate"
    ) -> ResumeEvaluation:
        """
        Main evaluation function with proper weighted scoring.
        """
        # Step 1: Analyze JD using our JD analyzer
        jd_analysis = jd_analyzer.analyze_jd(jd_text)
        
        # Step 2: Extract resume skills (no hallucination)
        resume_skills = self.extract_skills_from_text(resume_text)
        
        # Step 3: Extract candidate info
        candidate_info = self.extract_candidate_info(resume_text)
        if candidate_info["name"] == "Unknown Candidate":
            candidate_info["name"] = resume_name.replace('.pdf', '').replace('.docx', '').replace('_', ' ').title()
        
        # Step 4: Calculate skill matches with semantic matching
        breakdown = self._calculate_skill_scores(
            resume_skills=resume_skills,
            jd_analysis=jd_analysis
        )
        
        # Step 5: Calculate semantic similarity
        breakdown.semantic_similarity_score = self._calculate_semantic_similarity(resume_text, jd_text)
        
        # Step 6: Calculate learning potential (for interns)
        breakdown.learning_potential_bonus = self.calculate_learning_potential(
            resume_text, jd_analysis.experience_level
        )
        
        # Step 7: Calculate final score with proper weighting
        final_score = self._calculate_final_score(breakdown, jd_analysis)
        
        # Step 8: Determine recommendation tier
        recommendation = self._get_recommendation_tier(final_score, breakdown)
        
        # Step 9: Generate explanation
        explanation, strengths, improvements = self._generate_explanation(
            breakdown, jd_analysis, final_score
        )
        
        return ResumeEvaluation(
            candidate_name=candidate_info["name"],
            match_score=final_score,
            recommendation=recommendation,
            fit_level=self._score_to_fit_level(final_score),
            matched_skills=breakdown.matched_core + breakdown.matched_preferred,
            missing_core_skills=breakdown.missing_core,
            extra_relevant_skills=breakdown.matched_secondary,
            scoring_breakdown=breakdown,
            explanation=explanation,
            strengths=strengths,
            improvement_areas=improvements,
            extracted_experience_years=candidate_info["experience_years"],
            extracted_education=candidate_info["education"],
            extracted_current_title=candidate_info["title"]
        )
    
    def _calculate_skill_scores(
        self,
        resume_skills: Set[str],
        jd_analysis: JDAnalysisResult
    ) -> ScoringBreakdown:
        """Calculate skill match scores with semantic matching."""
        breakdown = ScoringBreakdown()
        
        core_required = set(s.lower() for s in jd_analysis.skill_categories.core_required)
        core_preferred = set(s.lower() for s in jd_analysis.skill_categories.core_preferred)
        secondary = set(s.lower() for s in jd_analysis.skill_categories.secondary)
        
        resume_skills_lower = set(s.lower() for s in resume_skills)
        
        # Match core required skills (semantic matching)
        for jd_skill in core_required:
            matched = False
            for resume_skill in resume_skills_lower:
                if self.skills_match(resume_skill, jd_skill):
                    breakdown.matched_core.append(jd_skill)
                    matched = True
                    break
            if not matched:
                breakdown.missing_core.append(jd_skill)
        
        # Match preferred skills
        for jd_skill in core_preferred:
            for resume_skill in resume_skills_lower:
                if self.skills_match(resume_skill, jd_skill):
                    breakdown.matched_preferred.append(jd_skill)
                    break
        
        # Match secondary skills
        for jd_skill in secondary:
            for resume_skill in resume_skills_lower:
                if self.skills_match(resume_skill, jd_skill):
                    breakdown.matched_secondary.append(jd_skill)
                    break
        
        # Calculate scores (0-1 scale)
        if core_required:
            breakdown.core_skill_score = len(breakdown.matched_core) / len(core_required)
        else:
            breakdown.core_skill_score = 0.5  # Neutral if no core defined
        
        if core_preferred:
            breakdown.preferred_skill_score = len(breakdown.matched_preferred) / len(core_preferred)
        else:
            breakdown.preferred_skill_score = 0.5
        
        if secondary:
            breakdown.secondary_skill_score = len(breakdown.matched_secondary) / len(secondary)
        else:
            breakdown.secondary_skill_score = 0.5
        
        return breakdown
    
    def _calculate_semantic_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate TF-IDF semantic similarity."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.3  # Default on error
    
    def _calculate_final_score(
        self,
        breakdown: ScoringBreakdown,
        jd_analysis: JDAnalysisResult
    ) -> int:
        """
        Calculate final score using proper weights.
        Formula ensures high scores (80-95%) are achievable for strong matches.
        """
        # Get experience-level-specific weights
        level = jd_analysis.experience_level.value
        level_weights = EXPERIENCE_LEVEL_WEIGHTS.get(level, EXPERIENCE_LEVEL_WEIGHTS.get("mid", {}))
        
        core_weight = level_weights.get("core_skills_weight", 0.60)
        preferred_weight = level_weights.get("preferred_skills_weight", 0.25)
        secondary_weight = level_weights.get("optional_skills_weight", 0.15)
        
        # Calculate weighted skill score (main component - 75%)
        skill_score = (
            breakdown.core_skill_score * core_weight +
            breakdown.preferred_skill_score * preferred_weight +
            breakdown.secondary_skill_score * secondary_weight
        )
        
        # Semantic similarity component (15%)
        semantic_component = breakdown.semantic_similarity_score * 0.15
        
        # Learning potential bonus (up to 10% for interns)
        learning_bonus = breakdown.learning_potential_bonus
        
        # Raw score calculation
        raw_score = (skill_score * 0.75) + semantic_component + learning_bonus
        breakdown.raw_score = raw_score
        
        # RESCALE to allow high scores (80-95%)
        # If raw >= 0.7, scale to 80-95 range
        # If raw >= 0.5, scale to 60-80 range
        # If raw >= 0.3, scale to 40-60 range
        # Below 0.3, scale to 20-40 range
        
        if raw_score >= 0.75:
            normalized = 85 + (raw_score - 0.75) * 40  # 85-95
        elif raw_score >= 0.60:
            normalized = 70 + (raw_score - 0.60) * 100  # 70-85
        elif raw_score >= 0.45:
            normalized = 55 + (raw_score - 0.45) * 100  # 55-70
        elif raw_score >= 0.30:
            normalized = 40 + (raw_score - 0.30) * 100  # 40-55
        else:
            normalized = 20 + raw_score * 67  # 20-40
        
        breakdown.normalized_score = normalized
        return max(15, min(98, int(normalized)))
    
    def _get_recommendation_tier(
        self,
        score: int,
        breakdown: ScoringBreakdown
    ) -> RecommendationTier:
        """Get tiered recommendation instead of binary."""
        # Also consider core skill coverage
        core_coverage = breakdown.core_skill_score
        
        if score >= 80 and core_coverage >= 0.7:
            return RecommendationTier.STRONG_FIT
        elif score >= 65 and core_coverage >= 0.5:
            return RecommendationTier.GOOD_FIT
        elif score >= 50 and core_coverage >= 0.3:
            return RecommendationTier.PARTIAL_FIT
        elif score >= 35:
            return RecommendationTier.WEAK_FIT
        else:
            return RecommendationTier.NOT_RECOMMENDED
    
    def _score_to_fit_level(self, score: int) -> str:
        """Convert score to fit level string."""
        if score >= 75:
            return "High"
        elif score >= 55:
            return "Medium"
        elif score >= 35:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_explanation(
        self,
        breakdown: ScoringBreakdown,
        jd_analysis: JDAnalysisResult,
        score: int
    ) -> Tuple[str, List[str], List[str]]:
        """Generate human-readable explanation."""
        strengths = []
        improvements = []
        
        # Analyze core skill match
        core_pct = int(breakdown.core_skill_score * 100)
        if core_pct >= 70:
            strengths.append(f"Strong core skill match ({core_pct}% of required skills)")
        elif core_pct >= 40:
            improvements.append(f"Moderate core skill coverage ({core_pct}%)")
        else:
            improvements.append(f"Missing many core skills ({100-core_pct}% not found)")
        
        # Specific matched skills
        if breakdown.matched_core:
            strengths.append(f"Key skills found: {', '.join(breakdown.matched_core[:5])}")
        
        # Missing skills
        if breakdown.missing_core:
            improvements.append(f"Missing core skills: {', '.join(breakdown.missing_core[:3])}")
        
        # Preferred skills
        if breakdown.matched_preferred:
            strengths.append(f"Bonus skills: {', '.join(breakdown.matched_preferred[:3])}")
        
        # Learning potential
        if breakdown.learning_potential_bonus > 0:
            strengths.append("Shows learning initiative (projects, courses, certifications)")
        
        # Generate summary explanation
        role_title = jd_analysis.detected_role_title
        level = jd_analysis.experience_level.value
        
        if score >= 75:
            explanation = f"Strong candidate for {role_title} ({level} level). Has {core_pct}% of core skills with good semantic match."
        elif score >= 55:
            explanation = f"Suitable candidate for {role_title}. Has foundational skills but may need development in some areas."
        elif score >= 35:
            explanation = f"Partial match for {role_title}. Has some relevant skills but missing key requirements."
        else:
            explanation = f"Limited match for {role_title}. Would need significant upskilling to meet role requirements."
        
        return explanation, strengths, improvements


# Singleton instance
intelligent_scorer = IntelligentScoringEngine()


def evaluate_resume(resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
    """
    Evaluate a resume with the intelligent scoring engine.
    Returns a dictionary compatible with the existing API.
    """
    result = intelligent_scorer.evaluate_resume(resume_text, job_description, resume_name)
    
    return {
        "candidate_name": result.candidate_name,
        "current_title": result.extracted_current_title,
        "experience_years": result.extracted_experience_years,
        "match_score": result.match_score,
        "fit_level": result.fit_level,
        "recommendation": result.recommendation.value,
        "matched_skills": result.matched_skills[:10],
        "missing_skills": result.missing_core_skills[:5],
        "summary": result.explanation,
        "strengths": result.strengths,
        "improvement_areas": result.improvement_areas,
        "scoring_breakdown": {
            "core_skill_score": round(result.scoring_breakdown.core_skill_score * 100, 1),
            "preferred_skill_score": round(result.scoring_breakdown.preferred_skill_score * 100, 1),
            "semantic_similarity": round(result.scoring_breakdown.semantic_similarity_score * 100, 1),
            "learning_potential_bonus": round(result.scoring_breakdown.learning_potential_bonus * 100, 1),
        }
    }


def batch_evaluate_resumes(resumes: List[Dict], job_description: str) -> List[Dict]:
    """Batch evaluate multiple resumes."""
    results = []
    for resume in resumes:
        text = resume.get("text", "")
        name = resume.get("name", "Candidate")
        result = evaluate_resume(text, job_description, name)
        result["file_name"] = name
        results.append(result)
    
    # Sort by match score descending
    results.sort(key=lambda x: x["match_score"], reverse=True)
    
    return results
