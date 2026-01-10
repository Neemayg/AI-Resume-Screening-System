"""
Report Generator Module
Handles formatting and exporting of screening reports.
"""

import json
import logging
from typing import List, Dict
from datetime import datetime
from explainability.explanation_generator import explanation_generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates structured reports for screening sessions.
    """
    
    def generate_json_report(self, candidates: List[Dict], job_role: str) -> Dict:
        """
        Generate a full JSON report for all candidates in a session.
        """
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "job_role_detected": job_role,
            "total_candidates": len(candidates),
            "candidates": []
        }
        
        for candidate in candidates:
            # Generate explanation
            explanation = explanation_generator.generate_explanation(
                candidate['resume_name'],
                candidate['similarity_data'],
                job_role
            )
            
            candidate_report = {
                "id": candidate['resume_id'],
                "name": candidate['resume_name'],
                "rank": candidate.get('rank'),
                "score": candidate['score'],
                "category": candidate['similarity_data'].get('score_category'),
                "explanation": {
                    "summary": explanation.summary,
                    "strengths": explanation.strengths,
                    "gaps": explanation.gaps,
                    "tips": explanation.improvement_tips
                }
            }
            report_data['candidates'].append(candidate_report)
            
        return report_data
        
    def generate_markdown_summary(self, candidates: List[Dict], job_role: str) -> str:
        """
        Generate a markdown summary of the top candidates.
        """
        md = f"# Recruitment Screening Report\n"
        md += f"**Role:** {job_role}\n"
        md += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        
        md += "## Top Candidates\n\n"
        
        # Sort by score descending
        sorted_candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        top_candidates = sorted_candidates[:3]
        
        for i, c in enumerate(top_candidates, 1):
            name = c['resume_name']
            score = c['score']
            category = c['similarity_data'].get('score_category', 'N/A')
            
            explanation = explanation_generator.generate_explanation(
                name, c['similarity_data'], job_role
            )
            
            md += f"### {i}. {name} (Score: {score:.1f} - {category})\n"
            md += f"_{explanation.summary}_\n\n"
            md += "**Key Strengths:**\n"
            for s in explanation.strengths:
                md += f"- {s}\n"
            md += "\n"
            
        return md

# Global instance
report_generator = ReportGenerator()
