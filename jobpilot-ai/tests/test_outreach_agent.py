"""
Tests for agents/outreach_agent.py

Covers:
  - Email generation structure (subject + body)
  - LinkedIn message character limits
  - OutreachMessage status enforcement
  - Batch generation
  - Human-in-the-loop approval check
  - Follow-up generation

Run:  pytest tests/test_outreach_agent.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from agents.outreach_agent import OutreachAgent
from models.job import Job, JobPortal
from models.candidate import CandidateProfile, WorkExperience
from models.outreach import (
    HRContact, OutreachMessage, OutreachChannel,
    OutreachStatus, OutreachBatch,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def profile():
    return CandidateProfile(
        name="Rohit Kumar",
        headline="Data Scientist | ML | NLP",
        years_of_experience=4.0,
        current_role="Data Scientist",
        current_company="Acme Corp",
        summary="Experienced DS with 4 years in NLP and ML.",
        technical_skills=["Python", "PyTorch", "SQL"],
        domains=["NLP", "FinTech"],
        work_experience=[
            WorkExperience(
                company="Acme Corp",
                title="Data Scientist",
                description="Built BERT-based NLP models, improved latency by 40%.",
                is_current=True,
            )
        ],
    )


@pytest.fixture
def job():
    return Job(
        id="job-001",
        title="Senior Data Scientist",
        company="Meesho",
        location="Bangalore",
        portal=JobPortal.LINKEDIN,
        description="Looking for DS with NLP and PyTorch expertise.",
        required_skills=["Python", "PyTorch", "NLP"],
        industry="E-commerce",
    )


@pytest.fixture
def hr_contact():
    return HRContact(
        id="hr-001",
        name="Priya Sharma",
        title="Technical Recruiter",
        company="Meesho",
        email="priya.sharma@meesho.com",
        linkedin_url="https://linkedin.com/in/priyasharma",
    )


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.aparse_json = AsyncMock(return_value={
        "subject": "Senior Data Scientist at Meesho — NLP Background",
        "body": (
            "Hi Priya,\n\nI noticed the Senior Data Scientist role at Meesho focused on NLP. "
            "I've built BERT-based pipelines at Acme Corp, improving latency by 40%.\n\n"
            "Would love 15 minutes to connect.\n\nBest,\nRohit"
        ),
    })
    return llm


@pytest.fixture
def agent(mock_llm):
    a = OutreachAgent(llm_client=mock_llm)
    # Stub browser agents so tests don't try to launch a browser
    a.linkedin_agent = MagicMock()
    a.gmail_agent = MagicMock()
    a._research_company_hook = AsyncMock(return_value="fast-growing e-commerce platform")
    return a


# Helper to run async tests
def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── Email Generation Tests ─────────────────────────────────────────────────────

class TestGenerateEmail:

    def test_returns_outreach_message(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert isinstance(msg, OutreachMessage)

    def test_subject_is_not_empty(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.subject and len(msg.subject) > 5

    def test_body_is_not_empty(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.body and len(msg.body) > 20

    def test_channel_is_email(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.channel == OutreachChannel.EMAIL

    def test_status_is_pending_review(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.status == OutreachStatus.PENDING_REVIEW

    def test_candidate_name_stored(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.candidate_name == profile.name

    def test_hr_contact_stored(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.hr_contact.name == hr_contact.name
        assert msg.hr_contact.company == hr_contact.company

    def test_message_has_id(self, agent, job, profile, hr_contact):
        msg = run(agent.generate_email(job, profile, hr_contact))
        assert msg.id is not None and len(msg.id) > 0


# ── LinkedIn Message Tests ─────────────────────────────────────────────────────

class TestGenerateLinkedInMessage:

    @pytest.fixture
    def agent_linkedin(self, mock_llm):
        mock_llm.aparse_json = AsyncMock(return_value={
            "connection_note": "Hi Priya, I'm a Data Scientist with NLP experience. Would love to connect!"
        })
        a = OutreachAgent(llm_client=mock_llm)
        a.linkedin_agent = MagicMock()
        a.gmail_agent = MagicMock()
        a._research_company_hook = AsyncMock(return_value=None)
        return a

    def test_connection_note_channel(self, agent_linkedin, job, profile, hr_contact):
        msg = run(agent_linkedin.generate_linkedin_message(job, profile, hr_contact, "connection"))
        assert msg.channel == OutreachChannel.LINKEDIN_CONNECTION

    def test_dm_channel(self, mock_llm, job, profile, hr_contact):
        mock_llm.aparse_json = AsyncMock(return_value={
            "message": "Hi Priya, I saw the DS role at Meesho..."
        })
        a = OutreachAgent(llm_client=mock_llm)
        a._research_company_hook = AsyncMock(return_value=None)
        msg = run(a.generate_linkedin_message(job, profile, hr_contact, "dm"))
        assert msg.channel == OutreachChannel.LINKEDIN_MESSAGE

    def test_connection_note_under_300_chars(self, agent_linkedin, job, profile, hr_contact):
        msg = run(agent_linkedin.generate_linkedin_message(job, profile, hr_contact, "connection"))
        assert len(msg.body) <= 300, f"Connection note too long: {len(msg.body)} chars"

    def test_status_pending_review(self, agent_linkedin, job, profile, hr_contact):
        msg = run(agent_linkedin.generate_linkedin_message(job, profile, hr_contact, "connection"))
        assert msg.status == OutreachStatus.PENDING_REVIEW


# ── Follow-up Generation Tests ─────────────────────────────────────────────────

class TestGenerateFollowup:

    @pytest.fixture
    def agent_followup(self, mock_llm):
        mock_llm.aparse_json = AsyncMock(return_value={
            "subject": "Following up — Senior Data Scientist",
            "body": "Hi Priya, just following up on my email from last week...",
        })
        a = OutreachAgent(llm_client=mock_llm)
        a._research_company_hook = AsyncMock(return_value=None)
        return a

    def test_followup_has_incremented_version(self, agent_followup, job, profile, hr_contact):
        original = OutreachMessage(
            id="orig-001",
            job_id=job.id,
            candidate_name=profile.name,
            hr_contact=hr_contact,
            channel=OutreachChannel.EMAIL,
            subject="Original Subject",
            body="Original body",
            version=1,
        )
        followup = run(agent_followup.generate_followup(original, days_since_sent=7))
        assert followup.version == 2

    def test_followup_status_is_pending(self, agent_followup, hr_contact, job, profile):
        original = OutreachMessage(
            id="orig-002", job_id=job.id, candidate_name=profile.name,
            hr_contact=hr_contact, channel=OutreachChannel.EMAIL, body="x",
        )
        followup = run(agent_followup.generate_followup(original, 7))
        assert followup.status == OutreachStatus.PENDING_REVIEW


# ── Batch Generation Tests ─────────────────────────────────────────────────────

class TestCreateBatch:

    def test_batch_has_messages(self, agent, profile, job, hr_contact):
        hr_map = {job.id: [hr_contact]}
        batch = run(agent.create_batch(
            jobs=[job],
            profile=profile,
            hr_contacts=hr_map,
            preferred_channel=OutreachChannel.EMAIL,
        ))
        assert isinstance(batch, OutreachBatch)
        assert batch.total >= 1

    def test_skips_jobs_without_contacts(self, agent, profile, job):
        batch = run(agent.create_batch(
            jobs=[job],
            profile=profile,
            hr_contacts={},           # no contacts
            preferred_channel=OutreachChannel.EMAIL,
        ))
        assert batch.total == 0

    def test_all_messages_pending_review(self, agent, profile, job, hr_contact):
        hr_map = {job.id: [hr_contact]}
        batch = run(agent.create_batch([job], profile, hr_map, OutreachChannel.EMAIL))
        for msg in batch.messages:
            assert msg.status == OutreachStatus.PENDING_REVIEW


# ── Send Enforcement Tests ─────────────────────────────────────────────────────

class TestSendApprovedMessage:

    def test_raises_if_not_approved(self, agent, job, profile, hr_contact):
        msg = OutreachMessage(
            id="m1", job_id=job.id, candidate_name=profile.name,
            hr_contact=hr_contact, channel=OutreachChannel.EMAIL,
            body="Test", status=OutreachStatus.PENDING_REVIEW,
        )
        with pytest.raises(ValueError, match="not approved"):
            run(agent.send_approved_message(msg))

    def test_calls_gmail_for_email_channel(self, agent, job, profile, hr_contact):
        agent.gmail_agent.compose_and_send = AsyncMock(return_value=True)
        msg = OutreachMessage(
            id="m2", job_id=job.id, candidate_name=profile.name,
            hr_contact=hr_contact, channel=OutreachChannel.EMAIL,
            subject="Test", body="Test body",
            status=OutreachStatus.APPROVED,
        )
        result = run(agent.send_approved_message(msg))
        agent.gmail_agent.compose_and_send.assert_called_once()
        assert result is True

    def test_calls_linkedin_for_connection_channel(self, agent, job, profile, hr_contact):
        agent.linkedin_agent.send_connection_request = AsyncMock(return_value=True)
        msg = OutreachMessage(
            id="m3", job_id=job.id, candidate_name=profile.name,
            hr_contact=hr_contact, channel=OutreachChannel.LINKEDIN_CONNECTION,
            body="Short note", status=OutreachStatus.APPROVED,
        )
        result = run(agent.send_approved_message(msg))
        agent.linkedin_agent.send_connection_request.assert_called_once()
        assert result is True


# ── to_display Formatting Test ────────────────────────────────────────────────

class TestOutreachMessageDisplay:

    def test_to_display_contains_hr_name(self, hr_contact, job, profile):
        msg = OutreachMessage(
            id="m1", job_id=job.id, candidate_name=profile.name,
            hr_contact=hr_contact, channel=OutreachChannel.EMAIL,
            subject="Test Subject", body="Test body.",
        )
        display = msg.to_display()
        assert "Priya Sharma" in display
        assert "Test Subject" in display
        assert "Test body." in display
