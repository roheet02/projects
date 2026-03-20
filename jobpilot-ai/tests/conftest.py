"""
pytest configuration and shared fixtures for the JobPilot AI test suite.
"""

import sys
import os
import pytest

# Ensure the project root is on sys.path so all imports resolve correctly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Async event loop for all async tests ─────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy for all tests."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


# ── Shared lightweight fixtures ───────────────────────────────────────────────

@pytest.fixture
def minimal_profile():
    """Minimal CandidateProfile for quick tests that don't need full data."""
    from models.candidate import CandidateProfile
    return CandidateProfile(
        name="Test User",
        technical_skills=["Python", "SQL"],
        domains=["Machine Learning"],
    )


@pytest.fixture
def minimal_job():
    """Minimal Job object for quick tests."""
    from models.job import Job, JobPortal
    return Job(
        id="test-job-001",
        title="Data Scientist",
        company="TestCo",
        location="Bangalore",
        portal=JobPortal.LINKEDIN,
        required_skills=["Python", "SQL"],
        experience_years="2-4 years",
    )


@pytest.fixture
def minimal_hr_contact():
    """Minimal HRContact for outreach tests."""
    from models.outreach import HRContact
    return HRContact(
        name="Jane HR",
        title="Recruiter",
        company="TestCo",
        email="jane@testco.com",
        linkedin_url="https://linkedin.com/in/janehr",
    )
