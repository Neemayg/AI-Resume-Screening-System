"""
Section parser for extracting structured information from JDs and Resumes.
Identifies and separates different sections of documents.
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import logging

from matching.text_normalizer import text_normalizer

logger = logging.getLogger(__name__)

@dataclass
class JDSections:
    """Structured representation of Job Description sections."""
    title: str = ""
    company: str = ""
    location: str = ""
    summary: str = ""
    responsibilities: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    qualifications: List[str] = field(default_factory=list)
    skills_required: List[str] = field(default_factory=list)
    skills_preferred: List[str] = field(default_factory=list)
    experience: str = ""
    education: List[str] = field(default_factory=list)
    benefits: List[str] = field(default_factory=list)
    raw_text: str = ""

@dataclass
class ResumeSections:
    """Structured representation of Resume sections."""
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    summary: str = ""
    experience: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    skills: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    projects: List[Dict] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    total_experience_years: float = 0.0
    raw_text: str = ""


class SectionParser:
    """Parse JDs and Resumes into structured sections."""
    
    # Section header patterns for JD
    JD_SECTION_PATTERNS = {
        'responsibilities': [
            r'responsibilities\s*:?',
            r'duties\s*:?',
            r'what you\'?ll do\s*:?',
            r'your role\s*:?',
            r'key responsibilities\s*:?',
            r'job duties\s*:?',
        ],
        'requirements': [
            r'requirements\s*:?',
            r'required\s*:?',
            r'must have\s*:?',
            r'what we\'?re looking for\s*:?',
            r'minimum requirements\s*:?',
            r'basic qualifications\s*:?',
        ],
        'qualifications': [
            r'qualifications\s*:?',
            r'preferred qualifications\s*:?',
            r'nice to have\s*:?',
            r'bonus points\s*:?',
            r'additional qualifications\s*:?',
        ],
        'skills': [
            r'skills\s*:?',
            r'technical skills\s*:?',
            r'required skills\s*:?',
            r'tech stack\s*:?',
            r'technologies\s*:?',
        ],
        'experience': [
            r'experience\s*:?',
            r'years of experience\s*:?',
            r'work experience\s*:?',
        ],
        'education': [
            r'education\s*:?',
            r'educational requirements\s*:?',
            r'degree\s*:?',
        ],
        'benefits': [
            r'benefits\s*:?',
            r'perks\s*:?',
            r'what we offer\s*:?',
            r'compensation\s*:?',
        ],
        'about': [
            r'about us\s*:?',
            r'about the company\s*:?',
            r'company overview\s*:?',
            r'who we are\s*:?',
        ]
    }
    
    # Section header patterns for Resume
    RESUME_SECTION_PATTERNS = {
        'summary': [
            r'summary\s*:?',
            r'professional summary\s*:?',
            r'objective\s*:?',
            r'career objective\s*:?',
            r'profile\s*:?',
            r'about me\s*:?',
        ],
        'experience': [
            r'experience\s*:?',
            r'work experience\s*:?',
            r'professional experience\s*:?',
            r'employment history\s*:?',
            r'work history\s*:?',
        ],
        'education': [
            r'education\s*:?',
            r'academic background\s*:?',
            r'educational qualifications\s*:?',
            r'academics\s*:?',
        ],
        'skills': [
            r'skills\s*:?',
            r'technical skills\s*:?',
            r'core competencies\s*:?',
            r'technologies\s*:?',
            r'expertise\s*:?',
        ],
        'certifications': [
            r'certifications?\s*:?',
            r'licenses?\s*:?',
            r'credentials?\s*:?',
            r'professional certifications?\s*:?',
        ],
        'projects': [
            r'projects?\s*:?',
            r'personal projects?\s*:?',
            r'key projects?\s*:?',
            r'portfolio\s*:?',
        ],
        'languages': [
            r'languages?\s*:?',
            r'spoken languages?\s*:?',
        ]
    }
    
    def parse_jd(self, jd_text: str) -> JDSections:
        """
        Parse a Job Description into structured sections.
        
        Args:
            jd_text: Raw JD text
            
        Returns:
            JDSections object with parsed content
        """
        sections = JDSections(raw_text=jd_text)
        
        # Extract title (usually first line or after "Title:")
        sections.title = self._extract_jd_title(jd_text)
        
        # Find section boundaries
        section_map = self._find_sections(jd_text, self.JD_SECTION_PATTERNS)
        
        # Extract each section
        for section_name, (start, end) in section_map.items():
            content = jd_text[start:end].strip()
            content = self._clean_section_content(content)
            
            if section_name == 'responsibilities':
                sections.responsibilities = self._extract_list_items(content)
            elif section_name == 'requirements':
                sections.requirements = self._extract_list_items(content)
            elif section_name == 'qualifications':
                sections.qualifications = self._extract_list_items(content)
            elif section_name == 'skills':
                all_skills = self._extract_list_items(content)
                # Separate required vs preferred if possible
                sections.skills_required = all_skills
            elif section_name == 'experience':
                sections.experience = content
            elif section_name == 'education':
                sections.education = self._extract_list_items(content)
            elif section_name == 'benefits':
                sections.benefits = self._extract_list_items(content)
        
        # If no sections found, treat entire text as summary
        if not any([sections.responsibilities, sections.requirements]):
            sections.summary = text_normalizer.normalize_text(jd_text)
        
        return sections
    
    def parse_resume(self, resume_text: str) -> ResumeSections:
        """
        Parse a Resume into structured sections.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            ResumeSections object with parsed content
        """
        sections = ResumeSections(raw_text=resume_text)
        
        # Extract contact info from beginning
        sections.name, sections.email, sections.phone = self._extract_contact_info(resume_text)
        
        # Find section boundaries
        section_map = self._find_sections(resume_text, self.RESUME_SECTION_PATTERNS)
        
        # Extract each section
        for section_name, (start, end) in section_map.items():
            content = resume_text[start:end].strip()
            content = self._clean_section_content(content)
            
            if section_name == 'summary':
                sections.summary = content
            elif section_name == 'experience':
                sections.experience = self._parse_experience_entries(content)
            elif section_name == 'education':
                sections.education = self._parse_education_entries(content)
            elif section_name == 'skills':
                sections.skills = self._extract_skills_from_section(content)
            elif section_name == 'certifications':
                sections.certifications = self._extract_list_items(content)
            elif section_name == 'projects':
                sections.projects = self._parse_project_entries(content)
            elif section_name == 'languages':
                sections.languages = self._extract_list_items(content)
        
        # Calculate total experience
        sections.total_experience_years = self._calculate_total_experience(sections.experience)
        
        return sections
    
    def _find_sections(
        self,
        text: str,
        patterns: Dict[str, List[str]]
    ) -> Dict[str, Tuple[int, int]]:
        """
        Find section boundaries in text.
        
        Returns:
            Dictionary mapping section name to (start, end) positions
        """
        section_positions = []
        
        for section_name, section_patterns in patterns.items():
            for pattern in section_patterns:
                regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
                matches = regex.finditer(text)
                
                for match in matches:
                    section_positions.append({
                        'name': section_name,
                        'start': match.end(),
                        'header_start': match.start()
                    })
                    break  # Take first match for each section
        
        # Sort by position
        section_positions.sort(key=lambda x: x['start'])
        
        # Calculate end positions
        result = {}
        for i, section in enumerate(section_positions):
            if i + 1 < len(section_positions):
                end = section_positions[i + 1]['header_start']
            else:
                end = len(text)
            
            result[section['name']] = (section['start'], end)
        
        return result
    
    def _extract_jd_title(self, text: str) -> str:
        """Extract job title from JD."""
        lines = text.strip().split('\n')
        
        # Check first few lines
        for line in lines[:5]:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Check for explicit title pattern
            title_match = re.match(r'(?:job\s*)?title\s*:\s*(.+)', line, re.IGNORECASE)
            if title_match:
                return title_match.group(1).strip()
            
            # Check for position pattern
            position_match = re.match(r'position\s*:\s*(.+)', line, re.IGNORECASE)
            if position_match:
                return position_match.group(1).strip()
            
            # First non-empty line might be title
            if len(line) < 100 and not line.endswith('.'):
                return line
        
        return ""
    
    def _extract_contact_info(self, text: str) -> Tuple[str, str, str]:
        """Extract name, email, phone from resume header."""
        lines = text.strip().split('\n')[:10]
        
        name = ""
        email = ""
        phone = ""
        
        # Email pattern
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        # Phone pattern
        phone_pattern = r'(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        
        for line in lines:
            if not email:
                email_match = re.search(email_pattern, line)
                if email_match:
                    email = email_match.group()
            
            if not phone:
                phone_match = re.search(phone_pattern, line)
                if phone_match:
                    phone = phone_match.group()
        
        # Name is typically first line
        first_line = lines[0].strip() if lines else ""
        if first_line and '@' not in first_line and not re.search(phone_pattern, first_line):
            name = first_line
        
        return name, email, phone
    
    def _clean_section_content(self, content: str) -> str:
        """Clean section content."""
        # Remove the section header if present
        lines = content.split('\n')
        if lines:
            # Check if first line is just the header
            first_line = lines[0].strip().lower()
            for patterns in list(self.JD_SECTION_PATTERNS.values()) + list(self.RESUME_SECTION_PATTERNS.values()):
                for pattern in patterns:
                    if re.match(pattern, first_line):
                        lines = lines[1:]
                        break
        
        return '\n'.join(lines).strip()
    
    def _extract_list_items(self, content: str) -> List[str]:
        """Extract list items from content."""
        items = []
        
        # Split by common list patterns
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove bullet points and numbering
            line = re.sub(r'^[\s][•●○◦▪▸►◆★☆✓✔→⇒➔➜\-\]\s*', '', line)
            line = re.sub(r'^[\s][\d]+[.)]\s', '', line)
            line = re.sub(r'^[\s][a-zA-Z][.)]\s', '', line)
            
            if line:
                items.append(line)
        
        return items
    
    def _extract_skills_from_section(self, content: str) -> List[str]:
        """Extract skills from a skills section."""
        skills = []
        
        # Split by common delimiters
        content = re.sub(r'[•●○◦▪▸►◆★☆✓✔→⇒➔➜]', ',', content)
        parts = re.split(r'[,;|\n]', content)
        
        for part in parts:
            skill = part.strip()
            # Clean up
            skill = re.sub(r'^[-\s]+', '', skill)
            skill = re.sub(r'[-\s]+$', '', skill)
            
            if skill and len(skill) > 1 and len(skill) < 50:
                skills.append(skill)
        
        return skills
    
    def _parse_experience_entries(self, content: str) -> List[Dict]:
        """Parse work experience entries."""
        entries = []
        
        # Pattern for date ranges
        date_pattern = r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]\.?\s\d{4}|' \
                       r'\d{1,2}/\d{4}|\d{4})\s*[-–—to]+\s*' \
                       r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]\.?\s\d{4}|' \
                       r'\d{1,2}/\d{4}|\d{4}|Present|Current|Now)'
        
        # Split content into potential entries
        lines = content.split('\n')
        current_entry = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check for date pattern (indicates new entry)
            date_match = re.search(date_pattern, line, re.IGNORECASE)
            
            if date_match:
                if current_entry:
                    entries.append(current_entry)
                
                current_entry = {
                    'title': '',
                    'company': '',
                    'start_date': date_match.group(1),
                    'end_date': date_match.group(2),
                    'description': '',
                    'bullets': []
                }
                
                # Extract title/company from the line
                before_date = line[:date_match.start()].strip()
                if before_date:
                    parts = re.split(r'\s+[-–—@|]\s+|\s+at\s+', before_date)
                    if len(parts) >= 2:
                        current_entry['title'] = parts[0].strip()
                        current_entry['company'] = parts[1].strip()
                    else:
                        current_entry['title'] = before_date
            
            elif current_entry:
                # Add to current entry
                if line.startswith(('-', '•', '*', '○')):
                    current_entry['bullets'].append(line.lstrip('-•*○ '))
                else:
                    current_entry['description'] += ' ' + line
        
        if current_entry:
            entries.append(current_entry)
        
        return entries
    
    def _parse_education_entries(self, content: str) -> List[Dict]:
        """Parse education entries."""
        entries = []
        
        # Common degree patterns
        degree_patterns = [
            r'(Ph\.?D\.?|Doctorate)',
            r'(M\.?S\.?|Master\'?s?|MBA|M\.?A\.?)',
            r'(B\.?S\.?|Bachelor\'?s?|B\.?A\.?|B\.?E\.?|B\.?Tech)',
            r'(Associate\'?s?|A\.?S\.?|A\.?A\.?)',
            r'(Diploma|Certificate)'
        ]
        
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            entry = {
                'degree': '',
                'field': '',
                'institution': '',
                'year': ''
            }
            
            # Check for degree
            for pattern in degree_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    entry['degree'] = match.group(1)
                    break
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', line)
            if year_match:
                entry['year'] = year_match.group()
            
            if entry['degree'] or entry['year']:
                entry['raw'] = line
                entries.append(entry)
        
        return entries
    
    def _parse_project_entries(self, content: str) -> List[Dict]:
        """Parse project entries."""
        entries = []
        
        lines = content.split('\n')
        current_project = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # New project (typically starts with project name)
            if not line.startswith(('-', '•', '*')) and len(line) < 100:
                if current_project:
                    entries.append(current_project)
                
                current_project = {
                    'name': line,
                    'description': '',
                    'technologies': []
                }
            elif current_project:
                current_project['description'] += ' ' + line.lstrip('-•* ')
        
        if current_project:
            entries.append(current_project)
        
        return entries
    
    def _calculate_total_experience(self, experience_entries: List[Dict]) -> float:
        """Calculate total years of experience from entries."""
        total_months = 0
        
        for entry in experience_entries:
            start = entry.get('start_date', '')
            end = entry.get('end_date', '')
            
            start_year = self._extract_year(start)
            end_year = self._extract_year(end)
            
            if start_year and end_year:
                # Simple year difference
                years = end_year - start_year
                if years >= 0:
                    total_months += max(12, years * 12)  # At least 1 year per entry
        
        return round(total_months / 12, 1)
    
    def _extract_year(self, date_str: str) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        
        # Handle "Present", "Current"
        if any(word in date_str.lower() for word in ['present', 'current', 'now']):
            from datetime import datetime
            return datetime.now().year
        
        # Extract 4-digit year
        match = re.search(r'(19|20)\d{2}', date_str)
        if match:
            return int(match.group())
        
        return None


# Singleton instance
section_parser = SectionParser()
