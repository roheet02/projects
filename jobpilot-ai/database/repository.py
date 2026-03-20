"""
JobPilot AI - Database Repository
SQLite-based local storage using SQLAlchemy.
Persists jobs, candidate profiles, outreach messages, and analytics.
"""

import json
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger

from sqlalchemy import (
    create_engine, Column, String, Float, Integer,
    DateTime, Text, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from models.job import Job, MatchStatus
from models.candidate import CandidateProfile
from models.outreach import OutreachMessage, OutreachStatus
from config.settings import settings


Base = declarative_base()


# ------------------------------------------------------------------ #
# ORM Models
# ------------------------------------------------------------------ #

class JobRecord(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    location = Column(String)
    portal = Column(String)
    url = Column(String)
    description = Column(Text)
    required_skills = Column(JSON)
    preferred_skills = Column(JSON)
    experience_years = Column(String)
    salary_range = Column(String)
    industry = Column(String)
    match_score = Column(Float)
    match_reasons = Column(JSON)
    status = Column(String, default="new")
    discovered_at = Column(DateTime, default=datetime.utcnow)


class CandidateRecord(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String)
    headline = Column(String)
    years_of_experience = Column(Float)
    technical_skills = Column(JSON)
    domains = Column(JSON)
    target_roles = Column(JSON)
    profile_data = Column(JSON)   # Full profile JSON
    resume_path = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class OutreachRecord(Base):
    __tablename__ = "outreach"

    id = Column(String, primary_key=True)
    job_id = Column(String)
    candidate_name = Column(String)
    hr_name = Column(String)
    hr_company = Column(String)
    hr_email = Column(String)
    hr_linkedin_url = Column(String)
    channel = Column(String)
    subject = Column(String)
    body = Column(Text)
    status = Column(String, default="drafted")
    drafted_at = Column(DateTime, default=datetime.utcnow)
    sent_at = Column(DateTime)
    replied_at = Column(DateTime)
    human_edited = Column(Boolean, default=False)


class AnalyticsRecord(Base):
    __tablename__ = "analytics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_type = Column(String)   # job_found, outreach_sent, reply_received, etc.
    job_id = Column(String)
    company = Column(String)
    metadata = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow)


# ------------------------------------------------------------------ #
# Repository
# ------------------------------------------------------------------ #

class JobRepository:
    """
    Async database repository for all JobPilot AI data.
    Uses SQLite locally — no external database needed.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or settings.database_url
        self.engine = create_async_engine(self.db_url, echo=settings.debug)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        """Create all tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database initialized")

    async def save_job(self, job: Job) -> JobRecord:
        """Save or update a job record."""
        async with self.session_factory() as session:
            record = JobRecord(
                id=job.id,
                title=job.title,
                company=job.company,
                location=job.location,
                portal=job.portal.value,
                url=job.url,
                description=job.description,
                required_skills=job.required_skills,
                preferred_skills=job.preferred_skills,
                experience_years=job.experience_years,
                salary_range=job.salary_range,
                industry=job.industry,
                match_score=job.match_score,
                match_reasons=job.match_reasons,
                status=job.status.value,
            )
            await session.merge(record)
            await session.commit()
            return record

    async def save_profile(self, profile: CandidateProfile) -> CandidateRecord:
        """Save candidate profile."""
        import uuid
        async with self.session_factory() as session:
            record = CandidateRecord(
                id=str(uuid.uuid4()),
                name=profile.name,
                email=profile.email,
                headline=profile.headline,
                years_of_experience=profile.years_of_experience,
                technical_skills=profile.technical_skills,
                domains=profile.domains,
                target_roles=profile.target_roles,
                profile_data=profile.model_dump(exclude={"profile_embedding", "raw_resume_text"}),
                resume_path=profile.resume_file_path,
            )
            session.add(record)
            await session.commit()
            return record

    async def save_outreach(self, message: OutreachMessage) -> OutreachRecord:
        """Save outreach message."""
        async with self.session_factory() as session:
            record = OutreachRecord(
                id=message.id,
                job_id=message.job_id,
                candidate_name=message.candidate_name,
                hr_name=message.hr_contact.name,
                hr_company=message.hr_contact.company,
                hr_email=message.hr_contact.email,
                hr_linkedin_url=message.hr_contact.linkedin_url,
                channel=message.channel.value,
                subject=message.subject,
                body=message.body,
                status=message.status.value,
                human_edited=message.human_edited,
            )
            await session.merge(record)
            await session.commit()
            return record

    async def get_all_jobs(self, status: Optional[str] = None) -> List[JobRecord]:
        """Fetch all jobs, optionally filtered by status."""
        from sqlalchemy import select
        async with self.session_factory() as session:
            query = select(JobRecord)
            if status:
                query = query.where(JobRecord.status == status)
            result = await session.execute(query)
            return result.scalars().all()

    async def get_outreach_stats(self) -> Dict:
        """Get outreach performance statistics."""
        from sqlalchemy import func, select
        async with self.session_factory() as session:
            total = await session.execute(select(func.count(OutreachRecord.id)))
            sent = await session.execute(
                select(func.count(OutreachRecord.id))
                .where(OutreachRecord.status == "sent")
            )
            replied = await session.execute(
                select(func.count(OutreachRecord.id))
                .where(OutreachRecord.status == "replied")
            )
            return {
                "total_drafted": total.scalar(),
                "total_sent": sent.scalar(),
                "total_replied": replied.scalar(),
                "reply_rate": (
                    replied.scalar() / sent.scalar()
                    if sent.scalar() > 0 else 0
                ),
            }

    async def log_event(
        self,
        event_type: str,
        job_id: Optional[str] = None,
        company: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """Log an analytics event."""
        async with self.session_factory() as session:
            record = AnalyticsRecord(
                event_type=event_type,
                job_id=job_id,
                company=company,
                metadata=metadata or {},
            )
            session.add(record)
            await session.commit()
