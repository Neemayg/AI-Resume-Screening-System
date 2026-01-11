
# backend/experience_parser.py
"""
CLEAN EXPERIENCE FORMATTING - FIXED VERSION
Solves: "0 months years" and robotic formatting issues
"""

import re
from datetime import datetime, date
from typing import Dict, Optional, List
from dateutil import parser as date_parser


class ExperienceParser:
    """Parse and format experience with clean output."""
    
    EDUCATION_KEYWORDS = [
        "university", "college", "school", "bachelor", "master", "phd",
        "degree", "student", "education", "graduation", "b.tech", "m.tech"
    ]
    
    WORK_KEYWORDS = [
        "experience", "work", "job", "position", "role", "company",
        "intern", "internship", "trainee", "associate", "engineer"
    ]
    
    @classmethod
    def parse_experience(cls, resume_text: str) -> Dict:
        """Parse experience with clean formatting."""
        result = {
            "total_years": 0,
            "total_months": 0,
            "formatted_experience": "Not specified",
            "is_student": False,
            "is_intern_candidate": False,
            "is_entry_level": True,
            "experience_level": "Entry Level",
            "work_entries": []
        }
        
        if not resume_text:
            return result
        
        text_lower = resume_text.lower()
        
        # Detect if student
        result["is_student"] = cls._detect_student_status(text_lower)
        result["is_intern_candidate"] = result["is_student"]
        
        # Parse date ranges
        work_dates = cls._extract_work_dates(resume_text)
        total_months = sum(d.get("duration_months", 0) for d in work_dates)
        
        result["total_months"] = total_months
        result["total_years"] = round(total_months / 12, 1)
        result["work_entries"] = work_dates
        
        # ═══════════════════════════════════════════════════════════
        # FIX 3: CLEAN FORMATTING
        # ═══════════════════════════════════════════════════════════
        result["formatted_experience"] = cls._format_experience_clean(
            total_months, 
            result["is_student"]
        )
        result["experience_level"] = cls._get_experience_level(
            total_months,
            result["is_student"]
        )
        result["is_entry_level"] = total_months < 24
        
        return result
    
    @classmethod
    def _format_experience_clean(cls, total_months: int, is_student: bool) -> str:
        """
        Format experience string CLEANLY.
        """
        if is_student and total_months == 0:
            return "Current Student"
        
        if total_months == 0:
            return "Entry Level"
        
        if total_months < 12:
            if total_months == 1:
                return "1 month"
            return f"{total_months} months"
        
        years = total_months // 12
        remaining_months = total_months % 12
        
        if remaining_months == 0:
            if years == 1:
                return "1 year"
            return f"{years} years"
        
        if years == 1:
            if remaining_months == 1:
                return "1 year, 1 month"
            return f"1 year, {remaining_months} months"
        
        # For 2+ years, just show years
        if remaining_months >= 6:
            return f"~{years + 1} years"
        return f"{years}+ years"
    
    @classmethod
    def _get_experience_level(cls, total_months: int, is_student: bool) -> str:
        """Get clean experience level label."""
        if is_student:
            if total_months == 0:
                return "Student"
            elif total_months < 6:
                return "Student with Internship Experience"
            else:
                return "Student with Work Experience"
        
        if total_months == 0:
            return "Entry Level"
        elif total_months < 6:
            return "Entry Level"
        elif total_months < 12:
            return "Early Career"
        elif total_months < 24:
            return "Junior"
        elif total_months < 48:
            return "Mid-Level"
        elif total_months < 84:
            return "Senior"
        elif total_months < 120:
            return "Lead/Principal"
        else:
            return "Executive/Expert"
    
    @classmethod
    def _detect_student_status(cls, text_lower: str) -> bool:
        """Detect if candidate is a current student."""
        student_indicators = [
            r"current(?:ly)?\s+(?:pursuing|studying|enrolled)",
            r"expected\s+graduation",
            r"graduating\s+in\s+\d{4}",
            r"student\s+at",
            r"\b(?:freshman|sophomore|junior|senior)\s+(?:year|student)",
            r"pursuing\s+(?:b\.?tech|m\.?tech|bachelor|master|degree)",
            r"20\d{2}\s*[-–]\s*(?:present|current|ongoing|expected)"
        ]
        
        for pattern in student_indicators:
            if re.search(pattern, text_lower):
                return True
        
        return False
    
    @classmethod
    def _extract_work_dates(cls, text: str) -> List[Dict]:
        """Extract work experience date ranges."""
        work_dates = []
        text_lower = text.lower()
        
        # Date patterns
        patterns = [
            r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4})\s*[-–—to]+\s*((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{4}|present|current|now)",
            r"(\d{1,2}/\d{4})\s*[-–—to]+\s*(\d{1,2}/\d{4}|present|current|now)",
            r"(\d{4})\s*[-–—to]+\s*(\d{4}|present|current|now)"
        ]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text_lower):
                start_str = match.group(1)
                end_str = match.group(2)
                
                # Get context
                context_start = max(0, match.start() - 100)
                context = text[context_start:match.start()].lower()
                
                # Skip if in education context
                if any(kw in context for kw in cls.EDUCATION_KEYWORDS):
                    if not any(kw in context for kw in cls.WORK_KEYWORDS):
                        continue
                
                # Parse dates
                start_date = cls._parse_date(start_str)
                end_date = cls._parse_date(end_str)
                
                if start_date and end_date:
                    months = (end_date.year - start_date.year) * 12 + \
                             (end_date.month - start_date.month)
                    
                    if 0 < months < 360:  # Sanity check
                        work_dates.append({
                            "start": start_str,
                            "end": end_str,
                            "duration_months": months
                        })
        
        return work_dates
    
    @classmethod
    def _parse_date(cls, date_str: str) -> Optional[date]:
        """Parse date string."""
        date_str = date_str.lower().strip()
        
        if date_str in ["present", "current", "now"]:
            return date.today()
        
        try:
            parsed = date_parser.parse(date_str, fuzzy=True)
            return parsed.date()
        except:
            pass
        
        # Year only
        year_match = re.match(r"(\d{4})", date_str)
        if year_match:
            return date(int(year_match.group(1)), 6, 1)
        
        return None


class ExperienceScorer:
    """Score experience fit for role level."""
    
    ROLE_LEVELS = {
        "intern": {"min": 0, "max": 12, "weight": 0.05},
        "entry": {"min": 0, "max": 24, "weight": 0.10},
        "junior": {"min": 12, "max": 36, "weight": 0.15},
        "mid": {"min": 36, "max": 60, "weight": 0.20},
        "senior": {"min": 60, "max": 120, "weight": 0.25},
    }
    
    @classmethod
    def detect_role_level(cls, jd_text: str) -> str:
        """Detect role level from JD."""
        jd_lower = jd_text.lower()
        
        if any(kw in jd_lower for kw in ["intern", "internship", "trainee"]):
            return "intern"
        if any(kw in jd_lower for kw in ["entry", "fresher", "graduate", "0-1 year"]):
            return "entry"
        if any(kw in jd_lower for kw in ["junior", "associate", "1-3 year"]):
            return "junior"
        if any(kw in jd_lower for kw in ["senior", "5+ year", "lead"]):
            return "senior"
        
        return "mid"
    
    @classmethod
    def calculate_experience_score(cls, exp_data: Dict, role_level: str) -> float:
        """Calculate experience fit score."""
        months = exp_data.get("total_months", 0)
        is_student = exp_data.get("is_student", False)
        
        config = cls.ROLE_LEVELS.get(role_level, cls.ROLE_LEVELS["mid"])
        
        # Intern: students are perfect
        if role_level == "intern":
            if is_student:
                return 1.0
            if months <= 12:
                return 0.9
            if months <= 24:
                return 0.7
            return 0.5  # Overqualified
        
        # Other roles
        min_months = config["min"]
        max_months = config["max"]
        
        if months < min_months:
            return max(0.3, months / max(min_months, 1))
        elif months <= max_months:
            return 1.0
        else:
            overage = (months - max_months) / 12
            return max(0.7, 1.0 - overage * 0.05)
