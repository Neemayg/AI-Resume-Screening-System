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
        
        # Load Tuning Intent (Freeze the role)
        self.intent = {}
        try:
            intent_path = Path("tuning/jd_intent.json")
            if intent_path.exists():
                with open(intent_path, 'r') as f:
                    self.intent = json.load(f)
                logger.info(f"❄️ Role Intent FROZEN: {self.intent}")
        except Exception as e:
            logger.warning(f"Failed to load tuning intent: {e}")
            
        # Load Semantic Map (New: Synonyms & Concept Mapping)
        self.semantic_map = {}
        try:
            map_path = Path("tuning/semantic_skill_map.json")
            if map_path.exists():
                with open(map_path, 'r') as f:
                    self.semantic_map = json.load(f)
                logger.info(f"🧠 Semantic Map LOADED: {list(self.semantic_map.keys())}")
        except Exception as e:
            logger.warning(f"Failed to load semantic map: {e}")
        
        # Common tech skills for matching
        # Comprehensive tech skills for matching
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
    
    def analyze_resume(self, resume_text: str, job_description: str, resume_name: str = "Candidate") -> Dict:
        """
        Analyze a resume using NLP techniques.
        """
        # Extract candidate info
        name = self.extract_name(resume_text)
        if name == "Unknown Candidate":
            name = resume_name.replace('.pdf', '').replace('.docx', '').replace('resume_', '').replace('_', ' ').title()
        
        title = self.extract_title(resume_text)
        exp_years = self.extract_experience_years(resume_text)
        education = self.extract_education(resume_text)
        
        # Extract skills
        resume_tech_skills = set(self.extract_skills(resume_text))
        jd_tech_skills = set(self.extract_skills(job_description))
        
        resume_soft_skills = set(self.extract_soft_skills(resume_text))
        jd_soft_skills = set(self.extract_soft_skills(job_description))
        
        # Combine for backward compatibility logic, but keep separate for reporting
        matched_tech = list(resume_tech_skills & jd_tech_skills)
        matched_soft = list(resume_soft_skills & jd_soft_skills)
        
        matched_skills = matched_tech + matched_soft
        missing_skills = list(jd_tech_skills - resume_tech_skills) # Focus on missing TECH skills for gaps
        
        # Calculate match score
        tfidf_score = self.calculate_similarity(resume_text, job_description)
        
        # New Scoring Formula (70% Core, 20% Preferred, 10% Project)
        # Note: in this simplified version, we treat ALL matched tech skills as Core since we don't have labeled JD sections yet.
        # Ideally, we'd split JD skills into 'Required' vs 'Nice to have'.
        
        # Core Tech Score (Base 70%)
        tech_coverage = len(matched_tech) / max(len(jd_tech_skills), 1) if jd_tech_skills else 0.5
        core_score = tech_coverage * 70
        
        # Text Similarity as Proxy for "Preferred/Context" (Base 20%)
        preferred_score = tfidf_score * 20
        
        # Project Bonus (Base 10%)
        project_bonus = 0
        resume_lower = resume_text.lower()
        if "machine learning" in resume_lower or "deep learning" in resume_lower or "scikit-learn" in resume_lower:
            project_bonus += 5
        if "nlp" in resume_lower or "natural language processing" in resume_lower or "spacy" in resume_lower:
            project_bonus += 5
            
        # Soft Skills Bonus (Extra 5%)
        soft_bonus = min(5, len(matched_soft) * 1)
        
        raw_score = core_score + preferred_score + project_bonus + soft_bonus
        
        match_score = int(max(0, min(100, raw_score)))  # Clamp between 0-100
        
        # Normalization (Boost candidates with good core skills but low overall score)
        if tech_coverage >= 0.5 and match_score < 60:
             # If they have >50% of skills but score is low (due to poor text match), boost them
             match_score += 15
        
        # Cap at 95 to allow room for "perfect" human judgement
        match_score = min(95, match_score)
        
        # Load Recommendation Logic
        self.rec_logic = {
            "strong_match": { "min": 75 },
            "good_match": { "min": 60 },
            "partial_match": { "min": 45 },
            "weak_match": { "max": 44 }
        }
        try:
            rec_path = Path("tuning/recommendation_logic.json")
            if rec_path.exists():
                with open(rec_path, 'r') as f:
                    self.rec_logic = json.load(f)
        except:
            pass
            
        # Determine fit level based on new logic
        if match_score >= self.rec_logic['strong_match']['min']:
            fit_level = "High"
        elif match_score >= self.rec_logic['good_match']['min']:
            fit_level = "Medium"
        elif match_score >= self.rec_logic['partial_match']['min']:
            fit_level = "Partial"
        else:
            fit_level = "Low"
        
        # Generate personalized summary
        summary = self._generate_summary(name, title, exp_years, education, matched_tech, missing_skills, fit_level)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(fit_level, matched_tech, missing_skills, exp_years)
        
        logger.info(f"✅ Analyzed: {name} - {match_score}% match ({fit_level}) [Tech: {len(matched_tech)}, Soft: {len(matched_soft)}]")
        
        return {
            "candidate_name": name,
            "current_title": title,
            "match_score": match_score,
            "fit_level": fit_level,

            "matched_skills": matched_skills, # Full list for UI tags
            "matched_tech_skills": matched_tech,
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
        is_intern = self.intent.get('role_type') == 'internship'
        
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
