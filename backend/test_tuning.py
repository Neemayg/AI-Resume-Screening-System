
from openai_service import analyzer
import json

# Define the Job Description (Data Science Intern)
jd_text = """
We are looking for a Data Science Intern to join our team.
Requirements:
- Strong knowledge of Python and Machine Learning.
- Experience with libraries like Pandas, NumPy, and Scikit-Learn.
- Familiarity with TensorFlow or PyTorch for Deep Learning is a plus.
- Understanding of NLP concepts (TF-IDF, Tokenization).
- Ability to build predictive models (e.g. SVM, Regression, Classification).
"""

# 1. Strong Candidate (Matches core + preferred + projects)
strong_resume = """
Name: Alice Strong
Objective: Aspiring Data Scientist with deep learning experience.
Skills: Python, TensorFlow, Keras, Pandas, NumPy, Scikit-Learn, Machine Learning, Deep Learning, SQL, Git.
Projects:
- Neural Network Classifier: Built a CNN using TensorFlow and Keras.
- NLP Sentiment Analysis: Used TF-IDF and NLTK for text classification.
Education: M.S. Data Science (GPA 3.8).
"""

# 2. Average Candidate (Matches core only)
average_resume = """
Name: Bob Average
Objective: Computer Science student looking for internship.
Skills: Python, Java, HTML, CSS, Pandas, NumPy.
Projects:
- Data Analysis of Sales: Used Pandas to analyze sales CSV files.
- Website Design: Built a personal website using HTML/CSS.
Education: B.S. Computer Science.
"""

# 3. Weak Candidate (Almost no match)
weak_resume = """
Name: Charlie Weak
Objective: Looking for a job in marketing or sales.
Skills: Microsoft Office, Excel, Powerpoint, Communication, Leadership.
Experience:
- Sales Associate: Handled customer queries.
- Marketing Intern: Managed social media accounts.
"""

print(f"{'Candidate':<20} | {'Score':<10} | {'Fit Level':<10} | {'Skills Found'}")
print("-" * 80)

for name, text in [("Strong", strong_resume), ("Average", average_resume), ("Weak", weak_resume)]:
    result = analyzer.analyze_resume(text, jd_text, name)
    print(f"{name:<20} | {result['match_score']}%{' ':<5} | {result['fit_level']:<10} | {len(result['matched_skills'])}")

