
# backend/regex_patterns.py
"""
ENHANCED REGEX PATTERNS
Solves: Special character failures for C++, C#, .NET, Node.js, etc.
"""

import re
from typing import List, Set

class SkillExtractor:
    """
    Extracts skills from text with proper handling of special characters.
    """
    
    # ═══════════════════════════════════════════════════════════════
    # SPECIAL CHARACTER SKILLS (Need custom regex)
    # ═══════════════════════════════════════════════════════════════
    
    SPECIAL_SKILLS = {
        # C-family languages
        r"c\+\+": "C++",
        r"c#": "C#",
        r"c\s*sharp": "C#",
        r"\.net": ".NET",
        r"dotnet": ".NET",
        r"asp\.net": "ASP.NET",
        r"\.net\s*core": ".NET Core",
        
        # JavaScript ecosystem
        r"node\.?js": "Node.js",
        r"react\.?js": "React.js",
        r"vue\.?js": "Vue.js",
        r"next\.?js": "Next.js",
        r"express\.?js": "Express.js",
        r"d3\.?js": "D3.js",
        r"three\.?js": "Three.js",
        
        # Data Science
        r"scikit[\-\s]?learn": "scikit-learn",
        r"sci-kit\s*learn": "scikit-learn",
        r"tf[\-\s]?idf": "TF-IDF",
        r"tf/idf": "TF-IDF",
        
        # Databases
        r"pl/?sql": "PL/SQL",
        r"t[\-\s]?sql": "T-SQL",
        r"no[\-\s]?sql": "NoSQL",
        r"my[\-\s]?sql": "MySQL",
        r"postgre[\-\s]?sql": "PostgreSQL",
        r"mongo[\-\s]?db": "MongoDB",
        r"redis": "Redis",
        
        # Cloud & DevOps
        r"ci/?cd": "CI/CD",
        r"aws\s*lambda": "AWS Lambda",
        r"s3": "AWS S3",
        r"ec2": "AWS EC2",
        
        # Version Control
        r"git[\-\s]?hub": "GitHub",
        r"git[\-\s]?lab": "GitLab",
        
        # Other
        r"power\s*bi": "Power BI",
        r"objective[\-\s]?c": "Objective-C",
        r"r\s+programming": "R",
        r"\br\b(?=\s+language|\s+studio|\s+programming)": "R",
        r"f#": "F#",
    }
    
    # ═══════════════════════════════════════════════════════════════
    # STANDARD SKILLS (Can use word boundaries)
    # ═══════════════════════════════════════════════════════════════
    
    STANDARD_SKILLS = [
        # Programming Languages
        "python", "java", "javascript", "typescript", "ruby", "go", "golang",
        "rust", "scala", "kotlin", "swift", "php", "perl", "matlab", "julia",
        "haskell", "erlang", "clojure", "groovy", "lua", "dart", "cobol",
        
        # ML/AI Frameworks
        "tensorflow", "keras", "pytorch", "caffe", "mxnet", "theano",
        "paddle", "jax", "onnx", "mlflow", "kubeflow",
        
        # Data Science Libraries
        "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "bokeh", "altair", "statsmodels", "xgboost", "lightgbm", "catboost",
        
        # NLP Libraries
        "nltk", "spacy", "gensim", "huggingface", "transformers", "bert",
        "gpt", "word2vec", "fasttext",
        
        # Databases
        "oracle", "sqlite", "cassandra", "dynamodb", "couchbase",
        "elasticsearch", "neo4j", "graphql",
        
        # Big Data
        "hadoop", "spark", "pyspark", "hive", "kafka", "flink", "storm",
        "airflow", "luigi", "dbt", "presto", "impala", "databricks",
        
        # Cloud
        "aws", "azure", "gcp", "heroku", "digitalocean", "cloudflare",
        "sagemaker", "bigquery", "redshift", "snowflake",
        
        # DevOps
        "docker", "kubernetes", "jenkins", "ansible", "terraform",
        "prometheus", "grafana", "nginx", "apache",
        
        # Tools
        "git", "svn", "jira", "confluence", "slack", "notion",
        "jupyter", "colab", "vscode", "pycharm", "vim", "emacs",
        
        # Visualization
        "tableau", "looker", "metabase", "superset", "qlik", "domo",
        
        # Concepts (for matching)
        "machine learning", "deep learning", "artificial intelligence",
        "natural language processing", "computer vision", "data mining",
        "data analysis", "data science", "business intelligence",
        "statistical analysis", "predictive modeling", "neural network",
        "reinforcement learning", "supervised learning", "unsupervised learning",
        "feature engineering", "model deployment", "mlops", "automl",
        "time series", "forecasting", "regression", "classification",
        "clustering", "recommendation system", "anomaly detection",
        "sentiment analysis", "image recognition", "object detection",
        "speech recognition", "chatbot", "web scraping", "api development",
        "etl", "data pipeline", "data warehouse", "data lake"
    ]
    
    @classmethod
    def extract_skills(cls, text: str) -> Set[str]:
        """
        Extract all skills from text, handling special characters properly.
        
        Args:
            text: Resume or JD text
            
        Returns:
            Set of normalized skill names
        """
        if not text:
            return set()
        
        text_lower = text.lower()
        found_skills = set()
        
        # 1. Extract special character skills first
        for pattern, skill_name in cls.SPECIAL_SKILLS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_skills.add(skill_name)
        
        # 2. Extract standard skills with word boundaries
        for skill in cls.STANDARD_SKILLS:
            # Escape any regex special chars in skill name
            escaped_skill = re.escape(skill)
            
            # Create pattern with word boundaries
            pattern = r'\b' + escaped_skill + r'\b'
            
            if re.search(pattern, text_lower, re.IGNORECASE):
                found_skills.add(skill.title() if len(skill) > 2 else skill.upper())
        
        # 3. Special handling for single-letter languages (R, C)
        # Only match R when it's clearly the language
        r_patterns = [
            r'\br\s+programming',
            r'\br\s+language', 
            r'\br\s+studio',
            r'programming\s+in\s+r\b',
            r'\brstudio\b',
            r'\br\s+packages?',
            r'\br\s+script',
            r'tidyverse',
            r'ggplot2?',
            r'\bdplyr\b',
        ]
        for pattern in r_patterns:
            if re.search(pattern, text_lower):
                found_skills.add("R")
                break
        
        # Only match C when it's clearly the language (not C++ or C#)
        c_patterns = [
            r'\bc\s+programming',
            r'\bc\s+language',
            r'programming\s+in\s+c\b(?!\+|\#)',
            r'\bc/c\+\+',
            r'\bansi\s+c\b',
        ]
        for pattern in c_patterns:
            if re.search(pattern, text_lower):
                found_skills.add("C")
                break
        
        return found_skills
    
    @classmethod
    def extract_with_context(cls, text: str) -> dict:
        """
        Extract skills with their context (for debugging).
        
        Returns:
            {
                "Python": {"count": 5, "contexts": ["Python programming", ...]},
                ...
            }
        """
        if not text:
            return {}
        
        results = {}
        sentences = re.split(r'[.!?\n]', text)
        
        skills = cls.extract_skills(text)
        
        for skill in skills:
            results[skill] = {"count": 0, "contexts": []}
            
            skill_lower = skill.lower()
            for sentence in sentences:
                if skill_lower in sentence.lower():
                    results[skill]["count"] += 1
                    results[skill]["contexts"].append(sentence.strip()[:100])
        
        return results
