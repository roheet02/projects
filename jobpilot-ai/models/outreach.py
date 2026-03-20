"""
JobPilot AI - Outreach / HR Contact Models
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class OutreachChannel(str, Enum):
    EMAIL = "email"
    LINKEDIN_MESSAGE = "linkedin_message"
    LINKEDIN_CONNECTION = "linkedin_connection"


class OutreachStatus(str, Enum):
    DRAFTED = "drafted"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    SENT = "sent"
    OPENED = "opened"
    REPLIED = "replied"
    REJECTED = "rejected"


class HRContact(BaseModel):
    """HR or Recruiter contact for a company."""

    id: Optional[str] = None
    name: str
    title: str                          # e.g., "HR Manager", "Technical Recruiter"
    company: str
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    relevance_note: Optional[str] = None   # Why this contact was chosen
    discovered_at: datetime = Field(default_factory=datetime.utcnow)


class OutreachMessage(BaseModel):
    """
    A drafted outreach message — email or LinkedIn.
    Always goes through human review before sending.
    """

    id: Optional[str] = None
    job_id: str
    candidate_name: str
    hr_contact: HRContact
    channel: OutreachChannel

    # Email fields
    subject: Optional[str] = None       # For emails
    body: str = ""

    # Meta
    status: OutreachStatus = OutreachStatus.DRAFTED
    version: int = 1                    # Tracks iterations after edits
    tone: str = "professional"          # professional | friendly | concise
    personalization_notes: List[str] = Field(default_factory=list)

    # Timing
    drafted_at: datetime = Field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    replied_at: Optional[datetime] = None

    # Human edit
    human_edited: bool = False
    edit_notes: Optional[str] = None

    def to_display(self) -> str:
        """Pretty format for review in terminal/UI."""
        lines = [
            f"{'='*60}",
            f"TO: {self.hr_contact.name} ({self.hr_contact.title})",
            f"COMPANY: {self.hr_contact.company}",
            f"CHANNEL: {self.channel.value}",
        ]
        if self.subject:
            lines.append(f"SUBJECT: {self.subject}")
        lines += ["", "MESSAGE:", self.body, "="*60]
        return "\n".join(lines)


class OutreachBatch(BaseModel):
    """A batch of outreach messages for review."""

    batch_id: str
    messages: List[OutreachMessage] = Field(default_factory=list)
    total: int = 0
    pending_review: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)
