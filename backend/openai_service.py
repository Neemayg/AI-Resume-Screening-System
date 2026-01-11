"""
NLP-Based Resume Analysis Engine
Uses TF-IDF, cosine similarity, and smart extraction - no external AI needed
"""

import re
import logging
from typing import Dict, List, Tuple
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResumeAnalyzer:
    """Smart NLP-based resume analyzer."""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        
        # Load Tuning Configs
        self.context = {}
        self.skill_buckets = {}
        self.rec_thresholds = {}
        self.semantic_map = {}
        
        try:
            # Context
            if Path("tuning/context.json").exists():
                with open("tuning/context.json", 'r') as f: self.context = json.load(f)
            # Skill Buckets
            if Path("tuning/skill_buckets.json").exists():
                with open("tuning/skill_buckets.json", 'r') as f: self.skill_buckets = json.load(f)
            # Recommendation
            if Path("tuning/recommendation.json").exists():
                with open("tuning/recommendation.json", 'r') as f: self.rec_thresholds = json.load(f)
            # Semantic Map
            if Path("tuning/semantic_skill_map.json").exists():
                with open("tuning/semantic_skill_map.json", 'r') as f: self.semantic_map = json.load(f)
                
            logger.info(f"🔧 Tuning Loaded. Mode: {self.context.get('role_type', 'standard')}")
        except Exception as e:
            logger.warning(f"Tuning load error: {e}")
            
        # Fallback/Base Skills
        self.tech_skills = {
            # Programming Languages
            'python', 'javascript', 'typescript', 'java', 'c++', 'c#', 'ruby', 'go', 'rust', 'php', 'swift', 'kotlin', 'scala', 'r', 'dart', 'lua', 'perl', 'bash', 'shell',
            
            # Web Frontend
            'react', 'angular', 'vue', 'svelte', 'next.js', 'nuxt', 'html', 'css', 'sass', 'less', 'tailwind', 'bootstrap', 'jquery', 'redux', 'webpack', 'vite',
            
            # Web Backend
            'node.js', 'nodejs', 'express', 'django', 'flask', 'fastapi', 'spring', 'rails', 'laravel', 'asp.net', 'graphql', 'rest', 'api', 'microservices', 'serverless',
            
            # AI / ML / Data Science (CRITICAL for this use case)
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'opencv', 'nltk', 'spacy', 'scipy',
            'nlp', 'computer vision', 'deep learning', 'machine learning', 'transformers', 'huggingface', 'llm', 'langchain', 'openai', 'generative ai', 'rag',
            'data analysis', 'data science', 'big data', 'hadoop', 'spark', 'kafka', 'airflow', 'tableau', 'power bi',
            
            # Databases
            'postgresql', 'postgres', 'mysql', 'mongodb', 'redis', 'elasticsearch', 'cassandra', 'dynamodb', 'sqlite', 'mariadb', 'sql', 'nosql', 'oracle',
            
            # DevOps & Cloud
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'k8s', 'terraform', 'ansible', 'jenkins', 'circleci', 'github actions', 'gitlab ci',
            'linux', 'unix', 'ubuntu', 'nginx', 'apache', 'heroku', 'vercel', 'netlify',
            
            # Tools & Practices
            'git', 'github', 'gitlab', 'bitbucket', 'jira', 'confluence', 'agile', 'scrum', 'kanban', 'tdd', 'ci/cd', 'sdlc', 'devops'
        }
        
        # Add bucket skills to tech_skills
        if self.skill_buckets:
            for cat in self.skill_buckets.values():
                self.tech_skills.update(set(cat))
        
        # Soft Skills (New: Jobscan-style matching)
        self.soft_skills = {
            'communication', 'teamwork', 'leadership', 'problem solving', 'critical thinking', 'creativity', 'adaptability', 
            'time management', 'interpersonal', 'presentation', 'negotiation', 'mentoring', 'collaboration', 'emotional intelligence',
            'analytical', 'attention to detail', 'organizational', 'flexibility', 'initiative', 'dependability'
        }
    
    def extract_soft_skills(self, text: str) -> List[str]:
        """Extract soft skills from text."""
        text_lower = text.lower()
        found = []
        for skill in self.soft_skills:
            if skill in text_lower:
                found.append(skill)
        return found
    
    def extract_name(self, text: str) -> str:
        """Extract candidate name from resume."""
        lines = text.split('\n')
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) < 40:
                # Skip lines with common non-name content
                skip_words = ['email', 'phone', '@', 'http', 'github', 'linkedin', 'resume', 'cv', 'objective', 'summary']
                if not any(w in line.lower() for w in skip_words):
                    if re.match(r'^[A-Za-z\s\.\-]+$', line):
                        return line.title()
        return "Unknown Candidate"
    
    def extract_title(self, text: str) -> str:
        """Extract job title from resume."""
        title_patterns = [
            r'(?i)(software\s+(?:developer|engineer))',
            r'(?i)(full[\s\-]?stack\s+(?:developer|engineer))',
            r'(?i)(front[\s\-]?end\s+(?:developer|engineer))',
            r'(?i)(back[\s\-]?end\s+(?:developer|engineer))',
            r'(?i)(data\s+(?:scientist|analyst|engineer))',
            r'(?i)(devops\s+engineer)',
            r'(?i)(product\s+manager)',
            r'(?i)(senior\s+\w+\s*(?:developer|engineer)?)',
            r'(?i)(junior\s+\w+\s*(?:developer|engineer)?)',
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).title()
        return "Professional"
    
    def extract_experience_years(self, text: str) -> int:
        """
        Extract years of experience. Returns None if cannot be verified.
        IMPORTANT: Only extracts EXPLICITLY stated experience, not calculated from dates.
        This prevents hallucinating experience from education dates.
        """
        text_lower = text.lower()
        
        # Only match explicit experience statements like "5 years of experience"
        # or "5+ years experience" - not date ranges
        explicit_patterns = [
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:professional\s+)?experience',
            r'(?:professional\s+)?experience\s*(?:of\s+)?[:\s]+(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s+(?:of\s+)?(?:work|professional|industry)\s+experience',
            r'(?:over|more\s+than)\s+(\d+)\s*years?\s+(?:of\s+)?experience',
        ]
        
        for pattern in explicit_patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = int(match.group(1))
                # Sanity check - if someone claims 15+ years, verify they're not a student
                if years >= 5:
                    # Check if this is a student (education ongoing)
                    student_indicators = ['student', 'undergraduate', 'pursuing', 'currently studying', 
                                          'expected graduation', 'b.tech', 'btech', 'bachelor', 
                                          'freshman', 'sophomore', 'junior', 'senior year']
                    for indicator in student_indicators:
                        if indicator in text_lower:
                            # Student claiming 5+ years is suspicious, ignore
                            return None
                return years
        
        # If no explicit experience statement found, return None
        # We do NOT calculate from date ranges to avoid counting education dates
        return None
    
    def extract_education(self, text: str) -> str:
        """Extract education level."""
        text_lower = text.lower()
        if 'phd' in text_lower or 'doctor' in text_lower:
            return "PhD"
        elif any(x in text_lower for x in ["master's", 'masters', 'm.s.', 'mba', 'm.tech']):
            return "Master's Degree"
        elif any(x in text_lower for x in ["bachelor's", 'bachelors', 'b.s.', 'b.tech', 'b.e.']):
            return "Bachelor's Degree"
        return ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from text."""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.tech_skills:
            # Handle special characters (c++, c#, .net) which fail with standard \b
            if any(c in skill for c in ['+', '#', '.']):
                # Check not surrounded by alphanumeric characters
                pattern = r'(?<![a-z0-9])' + re.escape(skill) + r'(?![a-z0-9])'
            else:
                # For normal words, use standard word boundaries
                pattern = r'\b' + re.escape(skill) + r'\b'
            
            if re.search(pattern, text_lower):
                found_skills.append(skill)
        
        # Semantic Expansion: If "pandas" found, they know "EDA"
        for category, keywords in self.semantic_map.items():
            # Check if any keyword from this category is in the text
            if any(k.lower() in text_lower for k in keywords):
                # Add the category itself as a "Skill"
                found_skills.append(category.lower())
        
        return list(set(found_skills))
    
    def calculate_similarity(self, resume_text: str, jd_text: str) -> float:
        """Calculate TF-IDF cosine similarity between resume and JD."""
        try:
            tfidf_matrix = self.vectorizer.fit_transform([jd_text, resume_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def _classify_skills(self, found_skills: List[str]) -> Dict[str, List[str]]:
        """Sort found skills into buckets based on config."""
        classified = {
            "core": [],
            "project": [],
            "preferred": [],
            "other": []
        }
        
        if not self.skill_buckets:
            # Fallback if buckets not loaded
            classified["core"] = found_skills
            return classified
            
        found_set = set(found_skills)
        
        # Check against buckets
        for skill in found_set:
            if skill in self.skill_buckets.get('core_ai', []):
                classified["core"].append(skill)
            elif skill in self.skill_buckets.get('ml_projects', []):
                classified["project"].append(skill)
            elif skill in self.skill_buckets.get('preferred', []):
                classified["preferred"].append(skill)
            else:
                classified["other"].append(skill)
                
        return classified

    def analyze_resume(self, resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
        """
        Analyze a resume using NLP techniques and Bucketed Scoring.
        """
        # Extract candidate info
        name = self.extract_name(resume_text)
        if name == "Unknown Candidate":
            name = resume_name.replace('.pdf', '').replace('.docx', '').replace('resume_', '').replace('_', ' ').title()
        
        title = self.extract_title(resume_text)
        exp_years = self.extract_experience_years(resume_text)
        education = self.extract_education(resume_text)
        
        # Extract skills & Soft Skills
        raw_skills = self.extract_skills(resume_text) # Uses semantic map expansion implicitly
        
        # Classify Skills into Buckets
        classified = self._classify_skills(raw_skills)
        
        matched_soft = list(set(self.extract_soft_skills(resume_text)))
        
        # Calculate Score (Bucketed Formula)
        # Formula: Core*60 + Project*25 + Preferred*15
        
        # Core
        total_core = len(self.skill_buckets.get('core_ai', [])) or 1
        core_score_val = min(1.0, len(classified['core']) / total_core)
        
        # Project
        total_proj = len(self.skill_buckets.get('ml_projects', [])) or 1
        proj_score_val = min(1.0, len(classified['project']) / total_proj)
        
        # Preferred
        total_pref = len(self.skill_buckets.get('preferred', [])) or 1
        pref_score_val = min(1.0, len(classified['preferred']) / total_pref)
        
        # Weighted Sum
        raw_score = (core_score_val * 60) + (proj_score_val * 25) + (pref_score_val * 15)
        
        # Text Context Bonus (Small weight from TF-IDF)
        tfidf_score = self.calculate_similarity(resume_text, job_description) * 100
        final_score = raw_score + (tfidf_score * 0.1) # Small bump for context
        
        match_score = int(max(0, min(100, final_score)))
        
        # Normalization (Fix Compression - per user rule)
        # If core_score >= 0.5 and final_score < 55: final_score += 15
        if core_score_val >= 0.5 and match_score < 55:
            match_score += 15
            
        match_score = min(100, match_score)
        
        # Determine fit level
        # strong >= 75, good 60-74, partial 45-59, weak < 45
        if match_score >= 75:
            fit_level = "High"
        elif match_score >= 60:
            fit_level = "Medium"
        elif match_score >= 45:
            fit_level = "Partial"
        else:
            fit_level = "Low"
        
        # Identify Gaps
        missing_core = [s for s in self.skill_buckets.get('core_ai', []) if s not in classified['core']]
        missing_skills = missing_core + [s for s in self.skill_buckets.get('ml_projects', []) if s not in classified['project']]
        
        matched_all = classified['core'] + classified['project'] + classified['preferred'] + classified['other'] + matched_soft
        
        # Generate personalized summary
        summary = self._generate_summary(name, title, exp_years, education, matched_all, missing_skills, fit_level)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(fit_level, matched_all, missing_skills, exp_years)
        
        logger.info(f"✅ Analyzed {name}: Score {match_score}% ({fit_level}) | Core: {len(classified['core'])}/{total_core} | Proj: {len(classified['project'])}")
        
        return {
            "candidate_name": name,
            "current_title": title,
            "match_score": match_score,
            "fit_level": fit_level,
            "matched_skills": matched_all, # Full list for UI tags
            "matched_tech_skills": classified['core'], # Just for debug/backend structure
            "matched_soft_skills": matched_soft,
            "missing_skills": missing_skills[:5],
            "summary": summary,
            "recommendation": recommendation,
            "experience_years": exp_years
        }

    
    def _generate_summary(self, name: str, title: str, exp_years, education: str, 
                          matched_skills: List[str], missing_skills: List[str], fit_level: str) -> str:
        """Generate a personalized summary."""
        parts = []
        
        # Opening - only include years if verified
        if exp_years is not None and exp_years > 0:
            parts.append(f"{name} is a {title.lower()} with {exp_years}+ years of experience.")
        else:
            parts.append(f"{name} is a {title.lower()}.")
        
        # Skills highlight
        if matched_skills:
            top_skills = matched_skills[:4]
            if len(top_skills) >= 3:
                parts.append(f"Demonstrates proficiency in {', '.join(top_skills[:-1])} and {top_skills[-1]}.")
            else:
                parts.append(f"Has experience with {', '.join(top_skills)}.")
        
        # Education
        if education:
            parts.append(f"Holds a {education}.")
        
        # Gaps
        if fit_level != "High" and missing_skills:
            parts.append(f"Could strengthen skills in {', '.join(missing_skills[:2])}.")
        
        return ' '.join(parts)
    
    def _generate_recommendation(self, fit_level: str, matched_skills: List[str], 
                                  missing_skills: List[str], exp_years) -> str:
        """Generate hiring recommendation."""
        
        # Check Intent
        is_intern = self.context.get('role_type') == 'internship'
        
        if fit_level == "High":
            if is_intern:
                return "Top Intern Candidate. Strong skill alignment matches learning requirements. Recommended for immediate interview."
            
            if exp_years is not None and exp_years >= 5:
                return "Strong candidate for senior role. Recommend immediate interview. Demonstrated expertise aligns well with job requirements."
            else:
                return "Excellent match for the role. Proceed to technical interview. Shows strong potential and relevant skill set."
        
        elif fit_level == "Medium":
            if is_intern:
                return f"Promising Intern. Has foundation in {', '.join(matched_skills[:2]) if matched_skills else 'key areas'}. Good learning potential."
            
            if len(matched_skills) >= 4:
                return f"Decent skill match. Consider for screening call. May need training in {', '.join(missing_skills[:2])} to fully meet requirements."
            else:
                return "Good alignmnet with core skills. Worth a conversation to assess depth."
        
        elif fit_level == "Partial":
            if is_intern:
                return "Emerging Talent. Has some relevant exposure but may need structured mentorship. Keep in pool."
            return "Potential fit. possesses some key skills but has gaps. Consider if role allows for learning on the job."

        else:
            if is_intern:
                return "Skills do not currently match internship focus. Recommend checking if they have relevant coursework not listed."
                
            if exp_years is not None and exp_years > 0:
                return "Limited match with current requirements. May be better suited for a different role or team."
            else:
                return "Minimal skill overlap. Consider for entry-level positions or internship if applicable."


# Singleton instance
analyzer = ResumeAnalyzer()


def analyze_resume(resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
    """Analyze a single resume."""
    return analyzer.analyze_resume(resume_text, job_description, resume_name)


def batch_analyze_resumes(resumes: List[Dict], job_description: str) -> List[Dict]:
    """Analyze multiple resumes."""
    results = []
    
    for i, resume in enumerate(resumes):
        logger.info(f"📊 Analyzing resume {i+1}/{len(resumes)}: {resume.get('name', 'Unknown')}")
        
        result = analyze_resume(
            resume_text=resume.get('text', ''),
            job_description=job_description,
            resume_name=resume.get('name', 'Unknown')
        )
        result['file_name'] = resume.get('name', 'Unknown')
        results.append(result)
    
    # Sort by match score
    results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    # Add ranks
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results
