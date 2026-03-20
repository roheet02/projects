"""
JobPilot AI - Candidate Profile Models
"""

from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict
from datetime import datetime


class WorkExperience(BaseModel):
    company: str
    title: str
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    description: str = ""
    skills_used: List[str] = Field(default_factory=list)
    is_current: bool = False


class Education(BaseModel):
    institution: str
    degree: str
    field: str
    graduation_year: Optional[int] = None
    gpa: Optional[float] = None


class Project(BaseModel):
    name: str
    description: str
    technologies: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    impact: Optional[str] = None


class Certification(BaseModel):
    name: str
    issuer: str
    year: Optional[int] = None
    url: Optional[str] = None


class CandidateProfile(BaseModel):
    """
    Structured candidate profile extracted from resume.
    This is the central data model that drives all matching and outreach.
    """

    # Personal Info
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None

    # Professional Summary
    summary: str = ""
    headline: str = ""                    # e.g., "Senior Data Scientist | ML | LLMs"
    years_of_experience: Optional[float] = None
    current_role: Optional[str] = None
    current_company: Optional[str] = None

    # Skills
    technical_skills: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)  # e.g., NLP, CV, FinTech

    # Background
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    certifications: List[Certification] = Field(default_factory=list)
    publications: List[str] = Field(default_factory=list)

    # Preferences
    target_roles: List[str] = Field(default_factory=list)
    target_locations: List[str] = Field(default_factory=list)
    preferred_industries: List[str] = Field(default_factory=list)
    salary_expectation: Optional[str] = None
    notice_period: Optional[str] = None
    remote_preference: Optional[str] = None  # "remote", "hybrid", "onsite"

    # Raw
    raw_resume_text: str = ""
    resume_file_path: Optional[str] = None
    profile_embedding: Optional[List[float]] = None     # Cached embedding vector
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def all_skills(self) -> List[str]:
        """Combined skill list for matching."""
        return list(set(self.technical_skills + self.soft_skills + self.tools))

    @property
    def skill_string(self) -> str:
        """Space-joined skills for embedding."""
        return " ".join(self.all_skills)

    @property
    def full_profile_text(self) -> str:
        """
        Rich text representation of the profile for embedding.
        Used as the query vector for semantic job matching.
        """
        parts = [
            self.headline,
            self.summary,
            f"Skills: {', '.join(self.all_skills)}",
            f"Domains: {', '.join(self.domains)}",
        ]
        for exp in self.work_experience[:3]:  # Top 3 experiences
            parts.append(f"{exp.title} at {exp.company}: {exp.description[:200]}")
        for proj in self.projects[:3]:
            parts.append(f"Project: {proj.name} - {proj.description[:200]}")
        return "\n".join(filter(None, parts))
