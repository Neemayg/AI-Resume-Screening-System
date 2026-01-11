
# backend/validation_test.py
"""
VALIDATION TESTS
Verifies the scoring fixes work correctly.
"""

from intern_scorer import InternScorer, InternRanker


def test_strong_intern_scores_high():
    """
    TEST: Strong AI/ML intern should score 70-85%.
    Previously scored 17-36% (BROKEN).
    """
    print("\n" + "="*70)
    print("TEST 1: STRONG INTERN SHOULD SCORE HIGH (70-85%)")
    print("="*70)
    
    scorer = InternScorer()
    
    jd_text = """
    Data Science Intern
    
    We are looking for a passionate Data Science Intern to join our team.
    
    Requirements:
    - Python programming
    - Machine Learning fundamentals
    - Data Analysis experience
    - Familiarity with Pandas, NumPy
    - Basic understanding of scikit-learn
    
    Nice to have:
    - SQL knowledge
    - Git/GitHub experience
    - TensorFlow or PyTorch
    """
    
    strong_resume = """
    John Smith
    Data Science Enthusiast
    
    Education:
    B.Tech in Computer Science, XYZ University (2022-2026)
    
    Skills:
    - Python, Pandas, NumPy, Matplotlib, Seaborn
    - Machine Learning: scikit-learn, Random Forest, XGBoost, SVM
    - Data Analysis, EDA, Data Visualization
    - SQL, MySQL
    - Git, GitHub
    - Jupyter Notebooks
    
    Projects:
    1. Customer Churn Prediction using Random Forest (Python, scikit-learn)
    2. Sentiment Analysis of Movie Reviews (NLP, Python)
    3. Housing Price Prediction (Linear Regression, Pandas)
    4. Exploratory Data Analysis on COVID-19 Dataset
    
    Coursework:
    - Machine Learning by Andrew Ng (Coursera)
    - Data Science Specialization
    """
    
    result = scorer.score_resume(strong_resume, jd_text)
    
    print(f"\nFinal Score: {result['final_score']}%")
    print(f"Category: {result['match_category']}")
    print(f"Guardrail Applied: {result['guardrail_applied']}")
    
    print("\nBreakdown:")
    for key, value in result['breakdown'].items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Validation
    if 70 <= result['final_score'] <= 85:
        print("\n✅ TEST PASSED: Strong intern scored in expected range (70-85%)")
    else:
        print(f"\n❌ TEST FAILED: Expected 70-85%, got {result['final_score']}%")


def test_weak_candidate_scores_low():
    """
    TEST: Weak candidate should score below 45%.
    """
    print("\n" + "="*70)
    print("TEST 2: WEAK CANDIDATE SHOULD SCORE LOW (<45%)")
    print("="*70)
    
    scorer = InternScorer()
    
    jd_text = """
    Data Science Intern - Python, Machine Learning, Data Analysis
    """
    
    weak_resume = """
    Jane Doe
    Marketing Student
    
    Education:
    Bachelor of Arts in Marketing, ABC University
    
    Skills:
    - Microsoft Office (Word, Excel, PowerPoint)
    - Social Media Marketing
    - Adobe Photoshop
    - Communication
    
    Experience:
    Marketing Intern at XYZ Company
    - Created social media content
    - Helped with marketing campaigns
    """
    
    result = scorer.score_resume(weak_resume, jd_text)
    
    print(f"\nFinal Score: {result['final_score']}%")
    print(f"Category: {result['match_category']}")
    
    if result['final_score'] < 45:
        print("\n✅ TEST PASSED: Weak candidate scored below 45%")
    else:
        print(f"\n❌ TEST FAILED: Expected <45%, got {result['final_score']}%")


def test_semantic_vs_explicit():
    """
    TEST: Explicit matches should score higher than semantic matches.
    """
    print("\n" + "="*70)
    print("TEST 3: EXPLICIT > SEMANTIC SCORING")
    print("="*70)
    
    scorer = InternScorer()
    
    jd_text = "Data Science Intern - Machine Learning, Python"
    
    # Resume with EXPLICIT "Machine Learning"
    explicit_resume = """
    Skills: Python, Machine Learning, Pandas, NumPy, scikit-learn
    Experience with Machine Learning algorithms
    """
    
    # Resume with only SEMANTIC indicators (no explicit "Machine Learning")
    semantic_resume = """
    Skills: Python, Random Forest, XGBoost, SVM, Pandas, NumPy
    Built classification and regression models
    """
    
    explicit_result = scorer.score_resume(explicit_resume, jd_text)
    semantic_result = scorer.score_resume(semantic_resume, jd_text)
    
    print(f"\nExplicit 'Machine Learning': {explicit_result['final_score']}%")
    print(f"Semantic (RF, XGBoost, SVM): {semantic_result['final_score']}%")
    
    print(f"\nExplicit ML matches: {explicit_result['breakdown']['core_skills']['explicit_matches']}")
    print(f"Semantic ML matches: {semantic_result['breakdown']['core_skills']['semantic_matches']}")
    
    if explicit_result['final_score'] > semantic_result['final_score']:
        print("\n✅ TEST PASSED: Explicit matches score higher than semantic")
    else:
        print("\n❌ TEST FAILED: Semantic should not equal or exceed explicit")


def test_no_experience_penalty():
    """
    TEST: Interns should NOT be penalized for lack of experience.
    """
    print("\n" + "="*70)
    print("TEST 4: NO EXPERIENCE PENALTY FOR INTERNS")
    print("="*70)
    
    scorer = InternScorer()
    
    jd_text = "Data Science Intern"
    
    # Student with no work experience but good skills
    student_resume = """
    Current Student
    B.Tech Computer Science (2022-2026)
    
    Skills: Python, Machine Learning, Pandas, NumPy, scikit-learn
    
    Projects:
    1. ML Project using Random Forest
    2. Data Analysis Project
    3. NLP Sentiment Analysis
    
    No work experience yet - first internship application
    """
    
    result = scorer.score_resume(student_resume, jd_text)
    
    print(f"\nStudent Score: {result['final_score']}%")
    print(f"Category: {result['match_category']}")
    
    # Should score well despite no experience
    if result['final_score'] >= 60:
        print("\n✅ TEST PASSED: Student not penalized for lack of experience")
    else:
        print(f"\n❌ TEST FAILED: Student penalized unfairly, got {result['final_score']}%")


def test_ranking_separation():
    """
    TEST: Rankings should have clear separation, not clumping.
    """
    print("\n" + "="*70)
    print("TEST 5: RANKING SEPARATION")
    print("="*70)
    
    ranker = InternRanker()
    
    jd_text = """
    Data Science Intern - Python, Machine Learning, 
    Data Analysis, Pandas, NumPy, scikit-learn
    """
    
    candidates = [
        {
            "id": "1",
            "name": "Alice (Strong)",
            "text": """
            Python, Machine Learning, Deep Learning, TensorFlow
            Pandas, NumPy, scikit-learn, Matplotlib
            5 ML projects, Kaggle competitions
            Data Analysis, Statistics, SQL, Git
            """
        },
        {
            "id": "2",
            "name": "Bob (Good)",
            "text": """
            Python, Pandas, NumPy
            Machine Learning basics, scikit-learn
            2 data analysis projects
            Some SQL experience
            """
        },
        {
            "id": "3",
            "name": "Charlie (Partial)",
            "text": """
            Python programming
            Basic data analysis with Excel
            Learning machine learning
            1 project
            """
        },
        {
            "id": "4",
            "name": "Dave (Weak)",
            "text": """
            Java developer
            Web development with React
            No ML or data science experience
            """
        }
    ]
    
    results = ranker.rank_candidates(candidates, jd_text)
    
    print("\nRANKED RESULTS:")
    print("-" * 50)
    for r in results:
        print(f"{r['rank_label']} {r['candidate_name']}: {r['final_score']}%")
    
    # Check separation
    scores = [r['final_score'] for r in results]
    min_gap = min(scores[i] - scores[i+1] for i in range(len(scores)-1))
    
    print(f"\nScore range: {max(scores)} - {min(scores)} = {max(scores) - min(scores)} pts")
    print(f"Minimum gap between adjacent ranks: {min_gap} pts")
    
    if max(scores) - min(scores) >= 30:
        print("\n✅ TEST PASSED: Good score distribution (30+ point spread)")
    else:
        print("\n❌ TEST FAILED: Scores too clumped together")


def run_all_tests():
    """Run all validation tests."""
    print("\n" + "="*70)
    print("RUNNING ALL VALIDATION TESTS")
    print("="*70)
    
    test_strong_intern_scores_high()
    test_weak_candidate_scores_low()
    test_semantic_vs_explicit()
    test_no_experience_penalty()
    test_ranking_separation()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    run_all_tests()
