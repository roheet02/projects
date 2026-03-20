"""
JobPilot AI - Job Data Models
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from datetime import datetime
from enum import Enum


class JobPortal(str, Enum):
    LINKEDIN = "linkedin"
    INDEED = "indeed"
    NAUKRI = "naukri"
    GLASSDOOR = "glassdoor"
    WELLFOUND = "wellfound"
    OTHER = "other"


class JobType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    REMOTE = "remote"
    HYBRID = "hybrid"
    INTERNSHIP = "internship"


class MatchStatus(str, Enum):
    NEW = "new"
    MATCHED = "matched"
    SHORTLISTED = "shortlisted"
    APPLIED = "applied"
    OUTREACH_SENT = "outreach_sent"
    RESPONDED = "responded"
    REJECTED = "rejected"
    INTERVIEW_SCHEDULED = "interview_scheduled"
    OFFER_RECEIVED = "offer_received"


class Job(BaseModel):
    """Represents a job posting discovered by the agent."""

    id: Optional[str] = None
    title: str
    company: str
    location: str
    job_type: Optional[JobType] = None
    portal: JobPortal = JobPortal.OTHER
    url: Optional[str] = None
    posted_date: Optional[datetime] = None
    description: str = ""
    required_skills: List[str] = Field(default_factory=list)
    preferred_skills: List[str] = Field(default_factory=list)
    experience_years: Optional[str] = None
    salary_range: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    match_score: Optional[float] = None       # Cosine similarity (0-1)
    match_reasons: List[str] = Field(default_factory=list)
    status: MatchStatus = MatchStatus.NEW
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    notes: Optional[str] = None


class JobSearchQuery(BaseModel):
    """Search parameters for job discovery."""

    roles: List[str] = Field(..., description="Target job titles")
    skills: List[str] = Field(default_factory=list, description="Key skills to match")
    locations: List[str] = Field(default_factory=list, description="Preferred locations")
    experience_years: Optional[int] = None
    job_type: Optional[JobType] = None
    portals: List[JobPortal] = Field(default_factory=lambda: [JobPortal.LINKEDIN])
    remote_ok: bool = True
    max_results: int = 50


class JobSearchResult(BaseModel):
    """Result of a job search session."""

    query: JobSearchQuery
    jobs: List[Job] = Field(default_factory=list)
    total_found: int = 0
    search_time_seconds: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
