
import sys
import os
import logging
from pathlib import Path

# Setup path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Mock logging
logging.basicConfig(level=logging.INFO)

from matching.text_normalizer import text_normalizer
from matching.skill_extractor import skill_extractor
from matching.skill_matcher import skill_matcher
from similarity_engine import similarity_engine
from nlp_engine import nlp_engine

# 1. Simulate Inputs
jd_text = """
We are looking for a Senior Full Stack Developer with 5+ years of experience.
Required skills: Python, React, Node.js, AWS, Docker, Kubernetes, PostgreSQL.
Experience with CI/CD and Agile methodologies is a plus.
"""

resume_text = """
Experienced Full Stack Developer with 6 years of experience.
Expertise in Python, Django, React.js, Node.js, and PostgreSQL.
Familiar with AWS (EC2, S3), Docker, and Jenkins.
Worked in Agile teams.
"""

print("\n--- 1. Extraction Debug ---")
jd_skills = nlp_engine.extract_skills(jd_text)
print(f"JD Skills extracted: {jd_skills['all']}")

resume_skills = nlp_engine.extract_skills(resume_text)
print(f"Resume Skills extracted: {resume_skills['all']}")

print("\n--- 2. Normalization Debug ---")
norm_jd = set(skill_extractor.normalize_skill_list(jd_skills['all']))
norm_res = set(skill_extractor.normalize_skill_list(resume_skills['all']))
print(f"Normalized JD: {norm_jd}")
print(f"Normalized Resume: {norm_res}")
print(f"Intersection: {norm_jd.intersection(norm_res)}")

print("\n--- 3. Skill Matcher Debug ---")
# Manually call match_skills with LISTS (imitating the fix)
match_result = skill_matcher.match_skills(jd_skills['all'], resume_skills['all'])
print(f"Skill Match Object: {match_result}")
print(f"Skill Match Score: {match_result.score}")

print("\n--- 4. Semantic Scorer Debug ---")
try:
    from matching.semantic_scorer import semantic_scorer
    sem_score = semantic_scorer.compute_semantic_similarity(resume_text, jd_text)
    print(f"Semantic Score: {sem_score}")
except ImportError:
    print("Semantic scorer not available")

print("\n--- 5. Enhanced Compute Similarity Debug ---")
# Preprocess
similarity_engine.jd_text = jd_text
similarity_engine.jd_skills = jd_skills['all']
similarity_engine.jd_preprocessed = text_normalizer.normalize_text(jd_text)
res_prep = text_normalizer.normalize_text(resume_text)

# Run full flow
result = similarity_engine.enhanced_compute_similarity(resume_text, res_prep)
print(f"Final Result: {result}")
