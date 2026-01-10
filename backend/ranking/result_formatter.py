"""
Result Formatter
Formats ranking results for API responses
"""

from typing import List, Dict, Any
from dataclasses import asdict
from ranking.ranking_engine import RankedResume


class RankingResultFormatter:
    """
    Formats ranking results into different output formats
    """
    
    @staticmethod
    def to_summary(ranked_resumes: List[RankedResume]) -> Dict:
        """
        Create a summary of ranking results
        """
        if not ranked_resumes:
            return {'total': 0, 'ranked': []}
        
        total = len(ranked_resumes)
        categories = {}
        
        for resume in ranked_resumes:
            cat = resume.category
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_resumes': total,
            'category_breakdown': categories,
            'top_score': ranked_resumes[0].final_score,
            'lowest_score': ranked_resumes[-1].final_score,
            'average_score': sum(r.final_score for r in ranked_resumes) / total,
            'recommended_count': sum(
                1 for r in ranked_resumes 
                if 'Recommended' in r.recommendation
            )
        }
    
    @staticmethod
    def to_detailed_list(ranked_resumes: List[RankedResume]) -> List[Dict]:
        """
        Convert ranked resumes to detailed dictionary list
        """
        # Manual conversion to avoid deep copy issues with dataclasses if needed,
        # but asdict usually works fine.
        results = []
        for r in ranked_resumes:
             results.append({
                'rank': r.rank,
                'resume_id': r.resume_id,
                'candidate_name': r.candidate_name,
                'final_score': r.final_score,
                'category': r.category,
                'recommendation': r.recommendation,
                'score_breakdown': r.score_breakdown,
                'matched_skills': r.matched_skills,
                'missing_critical_skills': r.missing_critical_skills,
                'experience_match': r.experience_match,
                'strengths': r.strengths,
                'weaknesses': r.weaknesses,
                'processing_time_ms': r.processing_time_ms
            })
        return results
    
    @staticmethod
    def to_compact_list(ranked_resumes: List[RankedResume]) -> List[Dict]:
        """
        Convert to compact format for quick display
        """
        return [
            {
                'rank': r.rank,
                'name': r.candidate_name,
                'score': r.final_score,
                'category': r.category,
                'recommendation': r.recommendation
            }
            for r in ranked_resumes
        ]
    
    @staticmethod
    def to_comparison_table(ranked_resumes: List[RankedResume]) -> Dict:
        """
        Create a comparison table structure
        """
        if not ranked_resumes:
            return {'headers': [], 'rows': []}
        
        headers = [
            'Rank', 'Candidate', 'Score', 'Skills Match', 
            'Experience', 'Missing Skills', 'Recommendation'
        ]
        
        rows = []
        for r in ranked_resumes:
            skills_matched = len(r.matched_skills)
            skills_missing = len(r.missing_critical_skills)
            
            rows.append([
                r.rank,
                r.candidate_name,
                f"{r.final_score:.1f}%",
                f"{skills_matched} matched",
                r.experience_match.get('status', 'N/A'),
                f"{skills_missing} missing" if skills_missing > 0 else "None",
                r.recommendation
            ])
        
        return {
            'headers': headers,
            'rows': rows
        }
    
    @staticmethod
    def to_api_response(
        ranked_resumes: List[RankedResume],
        jd_analysis: Dict,
        include_details: bool = True
    ) -> Dict:
        """
        Create complete API response format
        """
        response = {
            'success': True,
            'job_analysis': {
                'detected_role': jd_analysis.get('detected_role', {}),
                'core_skills_required': jd_analysis.get('core_skills', []),
                'experience_required': jd_analysis.get('experience_requirements', {})
            },
            'summary': RankingResultFormatter.to_summary(ranked_resumes),
            'rankings': RankingResultFormatter.to_compact_list(ranked_resumes)
        }
        
        if include_details:
            response['detailed_rankings'] = RankingResultFormatter.to_detailed_list(ranked_resumes)
        
        return response
