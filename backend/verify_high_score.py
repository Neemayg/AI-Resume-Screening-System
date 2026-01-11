
from openai_service import analyzer

# 1. Shruti's Resume Text (The one clearly focused on AI/ML)
resume_text = """
Shruti Kumari
Objective: Aspiring software developer with experience in AI/ML projects.
Skills: Python, TensorFlow, Pandas, NumPy, Scikit-Learn, Machine Learning, Deep Learning, Java, HTML.
Projects: 
- Diabetes Prediction: Machine Learning Application using Support Vector Machine (SVM) and Scikit-Learn.
- Grocery Delivery: MERN Stack application.
Education: B.Tech Computer Science 2022-2026.
"""

# 2. A "Correct" Job Description (Data Science Role)
matching_jd = """
We are looking for a Data Scientist Intern.
Requirements:
- Strong knowledge of Python and Machine Learning.
- Experience with libraries like Pandas, NumPy, and Scikit-Learn.
- Familiarity with TensorFlow or PyTorch for Deep Learning.
- Ability to build predictive models (e.g. SVM, Regression).
- Good understanding of AI concepts.
"""

# 3. Analyze
result = analyzer.analyze_resume(resume_text, matching_jd, "Shruti")

print(f"Candidate: {result['candidate_name']}")
print(f"Match Score: {result['match_score']}%")
print(f"Fit Level: {result['fit_level']}")
print(f"Matched Skills: {result['matched_skills']}")
print(f"Missing Skills: {result['missing_skills']}")
