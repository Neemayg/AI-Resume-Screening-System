
# backend/skill_taxonomy.py
"""
STRICT SKILL TAXONOMY - FIXED VERSION
Solves: Semantic Inflation problem with unidirectional matching
"""

from typing import Set, Tuple, Optional
from enum import Enum


class MatchType(Enum):
    """Types of skill matches with their credit values"""
    EXACT = "exact"           # 1.0 - Literal string match
    SEMANTIC = "semantic"     # 0.5 - Child skill matches parent requirement
    NONE = "none"             # 0.0 - No match


class SkillTaxonomy:
    """
    STRICT Skill Taxonomy with Unidirectional Matching.
    
    KEY PRINCIPLE:
    - If JD asks for GENERAL concept → Resume with SPECIFIC tool = PARTIAL MATCH
    - If JD asks for SPECIFIC tool → Resume with GENERAL concept = NO MATCH
    
    Example:
    - JD: "Machine Learning" + Resume: "scikit-learn" → ✓ Semantic Match (0.5)
    - JD: "scikit-learn" + Resume: "Machine Learning" → ✗ No Match (0.0)
    """
    
    # ═══════════════════════════════════════════════════════════════════
    # STRICT PARENT → CHILDREN MAPPING (Unidirectional)
    # Only children can satisfy parent requirements, NOT vice versa
    # ═══════════════════════════════════════════════════════════════════
    
    PARENT_TO_CHILDREN = {
        # Machine Learning - ONLY these specific tools/techniques satisfy "ML"
        "machine learning": {
            "scikit-learn", "sklearn", "xgboost", "lightgbm", "catboost",
            "random forest", "decision tree", "gradient boosting",
            "logistic regression", "linear regression", "svm", 
            "support vector machine", "naive bayes", "knn",
            "k-nearest neighbors", "ensemble methods"
        },
        
        # Deep Learning - ONLY these satisfy "Deep Learning"
        "deep learning": {
            "tensorflow", "pytorch", "keras", "neural network",
            "cnn", "rnn", "lstm", "transformer", "attention mechanism",
            "convolutional neural network", "recurrent neural network",
            "gan", "generative adversarial network", "autoencoder"
        },
        
        # NLP - ONLY these satisfy "NLP"
        "natural language processing": {
            "nltk", "spacy", "gensim", "huggingface", "transformers",
            "bert", "gpt", "word2vec", "fasttext", "text classification",
            "named entity recognition", "ner", "sentiment analysis",
            "tokenization", "lemmatization", "text mining"
        },
        "nlp": {  # Alias
            "nltk", "spacy", "gensim", "huggingface", "transformers",
            "bert", "gpt", "word2vec", "sentiment analysis"
        },
        
        # Computer Vision - ONLY these satisfy "Computer Vision"
        "computer vision": {
            "opencv", "yolo", "rcnn", "image classification",
            "object detection", "image segmentation", "face recognition",
            "ocr", "optical character recognition"
        },
        
        # Data Visualization
        "data visualization": {
            "matplotlib", "seaborn", "plotly", "bokeh", "altair",
            "tableau", "power bi", "d3.js", "grafana", "looker"
        },
        
        # Cloud Computing - General
        "cloud computing": {
            "aws", "azure", "gcp", "google cloud", "amazon web services",
            "microsoft azure"
        },
        "aws": {
            "ec2", "s3", "lambda", "sagemaker", "redshift", "dynamodb",
            "cloudformation", "ecs", "eks"
        },
        
        # Database - General
        "database": {
            "sql", "mysql", "postgresql", "mongodb", "redis",
            "cassandra", "dynamodb", "oracle", "sqlite"
        },
        "sql": {
            "mysql", "postgresql", "sqlite", "oracle", "sql server",
            "mssql", "tsql"
        },
        
        # Big Data
        "big data": {
            "hadoop", "spark", "pyspark", "hive", "kafka", "flink",
            "hdfs", "mapreduce", "databricks"
        },
        
        # Version Control
        "version control": {
            "git", "github", "gitlab", "bitbucket", "svn"
        },
        
        # Containerization
        "containerization": {
            "docker", "kubernetes", "k8s", "container", "podman"
        },
        
        # CI/CD
        "ci/cd": {
            "jenkins", "github actions", "gitlab ci", "circleci",
            "travis ci", "azure devops"
        }
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # EXACT SYNONYMS ONLY (Truly equivalent terms)
    # These are bidirectional - they mean the SAME thing
    # ═══════════════════════════════════════════════════════════════════
    
    EXACT_SYNONYMS = {
        # Abbreviations that are EXACTLY the same thing
        "ml": "machine learning",
        "dl": "deep learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "cv": "computer vision",
        
        # Library name variations
        "sklearn": "scikit-learn",
        "scikit learn": "scikit-learn",
        "tf": "tensorflow",
        "pytorch": "torch",
        
        # Database variations
        "postgres": "postgresql",
        "mongo": "mongodb",
        
        # Cloud abbreviations
        "gcp": "google cloud platform",
        "aws": "amazon web services",
        
        # Visualization
        "powerbi": "power bi",
        
        # Common typos/variations
        "javascript": "js",
        "typescript": "ts",
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # SKILLS THAT SHOULD NEVER CASCADE (Standalone skills)
    # These skills do NOT grant credit for anything else
    # ═══════════════════════════════════════════════════════════════════
    
    STANDALONE_SKILLS = {
        # Programming languages - knowing Python ≠ knowing ML
        "python", "java", "javascript", "c++", "c#", "ruby", "go",
        "rust", "scala", "kotlin", "swift", "php", "r", "matlab",
        
        # Basic tools
        "excel", "word", "powerpoint",
        
        # Generic terms
        "programming", "coding", "software development"
    }
    
    # ═══════════════════════════════════════════════════════════════════
    # MAIN MATCHING METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    @classmethod
    def skills_match(cls, jd_skill: str, resume_skill: str) -> Tuple[bool, MatchType, float]:
        """
        STRICT skill matching with proper credit assignment.
        
        Args:
            jd_skill: Skill required in job description
            resume_skill: Skill claimed in resume
            
        Returns:
            Tuple of (is_match, match_type, credit_value)
            - (True, EXACT, 1.0) - Exact match
            - (True, SEMANTIC, 0.5) - Semantic match (child→parent)
            - (False, NONE, 0.0) - No match
        """
        jd_lower = jd_skill.lower().strip()
        resume_lower = resume_skill.lower().strip()
        
        # ─────────────────────────────────────────────────────────────
        # CHECK 1: Exact String Match
        # ─────────────────────────────────────────────────────────────
        if jd_lower == resume_lower:
            return (True, MatchType.EXACT, 1.0)
        
        # ─────────────────────────────────────────────────────────────
        # CHECK 2: Exact Synonym Match (bidirectional)
        # ─────────────────────────────────────────────────────────────
        jd_normalized = cls._normalize_with_synonyms(jd_lower)
        resume_normalized = cls._normalize_with_synonyms(resume_lower)
        
        if jd_normalized == resume_normalized:
            return (True, MatchType.EXACT, 1.0)
        
        # ─────────────────────────────────────────────────────────────
        # CHECK 3: Standalone skill protection
        # If JD asks for a standalone skill, only exact match counts
        # ─────────────────────────────────────────────────────────────
        if jd_normalized in cls.STANDALONE_SKILLS:
            # Standalone skills require exact match only
            return (False, MatchType.NONE, 0.0)
        
        # ─────────────────────────────────────────────────────────────
        # CHECK 4: Semantic Match (UNIDIRECTIONAL)
        # Resume's SPECIFIC skill can match JD's GENERAL requirement
        # But NOT vice versa!
        # ─────────────────────────────────────────────────────────────
        
        # Get children of the JD skill (if it's a parent concept)
        jd_children = cls.PARENT_TO_CHILDREN.get(jd_normalized, set())
        
        # Check if resume skill is a valid child of JD requirement
        if resume_normalized in jd_children:
            return (True, MatchType.SEMANTIC, 0.5)
        
        # Also check normalized resume skill against children
        if jd_children:
            for child in jd_children:
                child_normalized = cls._normalize_with_synonyms(child)
                if resume_normalized == child_normalized:
                    return (True, MatchType.SEMANTIC, 0.5)
        
        # ─────────────────────────────────────────────────────────────
        # CHECK 5: NO reverse matching (parent → child)
        # If JD asks for "scikit-learn" and resume has "machine learning",
        # this is NOT a match
        # ─────────────────────────────────────────────────────────────
        
        # Explicitly prevent parent matching child requirement
        resume_children = cls.PARENT_TO_CHILDREN.get(resume_normalized, set())
        if jd_normalized in resume_children:
            # Resume has parent, JD wants child - NO MATCH
            return (False, MatchType.NONE, 0.0)
        
        # No match found
        return (False, MatchType.NONE, 0.0)
    
    @classmethod
    def _normalize_with_synonyms(cls, skill: str) -> str:
        """Normalize skill using exact synonyms."""
        skill = skill.lower().strip()
        
        # Check if it's a synonym key
        if skill in cls.EXACT_SYNONYMS:
            return cls.EXACT_SYNONYMS[skill]
        
        # Check if it's a synonym value (reverse lookup)
        for abbrev, full in cls.EXACT_SYNONYMS.items():
            if skill == full:
                return full  # Return canonical form
        
        return skill
    
    @classmethod
    def get_match_explanation(cls, jd_skill: str, resume_skill: str) -> str:
        """Get human-readable explanation of match logic."""
        is_match, match_type, credit = cls.skills_match(jd_skill, resume_skill)
        
        if match_type == MatchType.EXACT:
            return f"✓ EXACT: '{resume_skill}' = '{jd_skill}' (100% credit)"
        elif match_type == MatchType.SEMANTIC:
            return f"◐ SEMANTIC: '{resume_skill}' demonstrates '{jd_skill}' (50% credit)"
        else:
            return f"✗ NO MATCH: '{resume_skill}' ≠ '{jd_skill}'"
    
    @classmethod
    def validate_mapping(cls) -> None:
        """Debug utility to print the mapping logic."""
        print("=== STRICT SKILL TAXONOMY VALIDATION ===\n")
        
        test_cases = [
            # (JD Skill, Resume Skill, Expected Match)
            ("Machine Learning", "scikit-learn", True),   # Child → Parent ✓
            ("scikit-learn", "Machine Learning", False),  # Parent → Child ✗
            ("Python", "Machine Learning", False),        # Standalone ✗
            ("Machine Learning", "Python", False),        # Python not child of ML ✗
            ("NLP", "spacy", True),                       # Child → Parent ✓
            ("spacy", "NLP", False),                      # Parent → Child ✗
            ("tensorflow", "tf", True),                   # Exact synonym ✓
            ("Deep Learning", "pytorch", True),           # Child → Parent ✓
            ("pytorch", "Deep Learning", False),          # Parent → Child ✗
        ]
        
        for jd, resume, expected in test_cases:
            is_match, match_type, credit = cls.skills_match(jd, resume)
            status = "✓" if is_match == expected else "✗ WRONG"
            print(f"{status} JD:'{jd}' + Resume:'{resume}' → {match_type.value} ({credit})")


# ═══════════════════════════════════════════════════════════════════════
# STRICT SKILL CATEGORIZER (Updated)
# ═══════════════════════════════════════════════════════════════════════

class SkillCategorizer:
    """Categorizes JD skills into Core and Preferred buckets."""
    
    # Default CORE skills - these are almost always critical
    ALWAYS_CORE = {
        "python", "sql", "machine learning", "statistics",
        "data analysis", "communication"
    }
    
    # Keywords indicating CORE requirement
    CORE_INDICATORS = [
        "required", "must have", "must-have", "essential", "mandatory",
        "minimum", "need", "necessary", "critical", "key requirements",
        "qualifications:", "requirements:", "you have:", "you bring:"
    ]
    
    # Keywords indicating PREFERRED requirement
    PREFERRED_INDICATORS = [
        "preferred", "nice to have", "nice-to-have", "bonus", "plus",
        "advantage", "desired", "ideally", "optional", "additionally",
        "good to have", "beneficial", "asset", "extra credit"
    ]
    
    @classmethod
    def categorize_jd_skills(cls, jd_text: str, extracted_skills: list) -> dict:
        """
        Categorize extracted skills into CORE and PREFERRED.
        Limits core skills to prevent denominator inflation.
        """
        if not extracted_skills:
            return {"core": [], "preferred": []}
        
        jd_lower = jd_text.lower()
        core_skills = []
        preferred_skills = []
        
        for skill in extracted_skills:
            skill_lower = skill.lower()
            skill_pos = jd_lower.find(skill_lower)
            
            is_core = skill_lower in cls.ALWAYS_CORE
            
            if skill_pos != -1:
                # Check context around skill
                start = max(0, skill_pos - 150)
                context = jd_lower[start:skill_pos]
                
                has_core_indicator = any(ind in context for ind in cls.CORE_INDICATORS)
                has_pref_indicator = any(ind in context for ind in cls.PREFERRED_INDICATORS)
                
                if has_core_indicator and not has_pref_indicator:
                    is_core = True
                elif has_pref_indicator:
                    is_core = False
            
            if is_core:
                core_skills.append(skill)
            else:
                preferred_skills.append(skill)
        
        # LIMIT core skills to prevent inflation (max 8)
        if len(core_skills) > 8:
            # Keep most important, move rest to preferred
            overflow = core_skills[8:]
            core_skills = core_skills[:8]
            preferred_skills = overflow + preferred_skills
        
        # Ensure minimum core skills (at least 3)
        if len(core_skills) < 3 and preferred_skills:
            needed = min(3 - len(core_skills), len(preferred_skills))
            core_skills.extend(preferred_skills[:needed])
            preferred_skills = preferred_skills[needed:]
        
        return {
            "core": list(set(core_skills)),
            "preferred": list(set(preferred_skills))
        }
