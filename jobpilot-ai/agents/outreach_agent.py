"""
JobPilot AI - Outreach Agent
Generates personalized cold emails and LinkedIn messages.
ALWAYS requires human review before sending (human-in-the-loop).

Legal note: This agent generates SUGGESTIONS. All messages require human
approval before being sent. This ensures compliance with CAN-SPAM, GDPR,
and LinkedIn's Terms of Service.
"""

import uuid
from typing import List, Optional
from string import Template
from loguru import logger

from models.job import Job
from models.candidate import CandidateProfile
from models.outreach import (
    HRContact, OutreachMessage, OutreachChannel,
    OutreachStatus, OutreachBatch
)
from llm.client import LLMClient
from llm.prompts import (
    EMAIL_GENERATION_SYSTEM, EMAIL_GENERATION_PROMPT,
    LINKEDIN_MESSAGE_SYSTEM, LINKEDIN_CONNECTION_PROMPT,
    LINKEDIN_DM_PROMPT, FOLLOWUP_EMAIL_PROMPT,
    COMPANY_RESEARCH_PROMPT,
)
from browser.linkedin_browser import LinkedInJobAgent
from browser.gmail_browser import GmailAgent
from config.settings import settings


class OutreachAgent:
    """
    AI agent that generates and (after approval) sends job outreach.

    Workflow:
    1. Generate personalized message (email or LinkedIn)
    2. Present to human for review/editing
    3. Human approves → agent sends via browser automation
    4. Track status and follow-ups
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.linkedin_agent = LinkedInJobAgent()
        self.gmail_agent = GmailAgent()

    async def generate_email(
        self,
        job: Job,
        profile: CandidateProfile,
        hr_contact: HRContact,
        tone: str = "professional",
    ) -> OutreachMessage:
        """
        Generate a personalized cold email for a job application.

        The email is:
        - Personalized to the specific HR contact and company
        - Highlights candidate's most relevant skills/achievements
        - Short and direct (150-200 words)
        - Has a clear call-to-action
        """
        logger.info(
            f"Generating email for {profile.name} → {hr_contact.name} "
            f"at {hr_contact.company} [{job.title}]"
        )

        # Get key achievement from work experience
        key_achievement = ""
        if profile.work_experience:
            exp = profile.work_experience[0]
            key_achievement = exp.description[:150] if exp.description else ""

        # Research company for personalization
        company_hook = await self._research_company_hook(job.company, job.industry)

        prompt = Template(EMAIL_GENERATION_PROMPT).substitute(
            candidate_name=profile.name,
            current_role=profile.current_role or profile.headline,
            key_skills=", ".join(profile.technical_skills[:5]),
            years_exp=profile.years_of_experience or "several",
            key_achievement=key_achievement or "N/A",
            summary=profile.summary[:300],
            hr_name=hr_contact.name.split()[0],   # First name
            hr_title=hr_contact.title,
            company=job.company + (f" ({company_hook})" if company_hook else ""),
            job_title=job.title,
            job_requirements=", ".join(job.required_skills[:6]),
        )

        result = await self.llm.aparse_json(
            user_prompt=prompt,
            system_prompt=EMAIL_GENERATION_SYSTEM,
        )

        return OutreachMessage(
            id=str(uuid.uuid4()),
            job_id=job.id or str(uuid.uuid4()),
            candidate_name=profile.name,
            hr_contact=hr_contact,
            channel=OutreachChannel.EMAIL,
            subject=result.get("subject", f"Interest in {job.title} at {job.company}"),
            body=result.get("body", ""),
            status=OutreachStatus.PENDING_REVIEW,
            tone=tone,
            personalization_notes=[company_hook] if company_hook else [],
        )

    async def generate_linkedin_message(
        self,
        job: Job,
        profile: CandidateProfile,
        hr_contact: HRContact,
        message_type: str = "connection",  # "connection" or "dm"
    ) -> OutreachMessage:
        """
        Generate a LinkedIn connection request note or direct message.
        """
        logger.info(
            f"Generating LinkedIn {message_type} for "
            f"{profile.name} → {hr_contact.name}"
        )

        if message_type == "connection":
            prompt = Template(LINKEDIN_CONNECTION_PROMPT).substitute(
                candidate_name=profile.name,
                current_role=profile.current_role or profile.headline,
                key_skill=profile.technical_skills[0] if profile.technical_skills else "data science",
                hr_name=hr_contact.name.split()[0],
                company=job.company,
                job_title=job.title,
            )
            result = await self.llm.aparse_json(
                user_prompt=prompt,
                system_prompt=LINKEDIN_MESSAGE_SYSTEM,
            )
            body = result.get("connection_note", "")
            channel = OutreachChannel.LINKEDIN_CONNECTION

        else:  # dm
            prompt = Template(LINKEDIN_DM_PROMPT).substitute(
                candidate_name=profile.name,
                headline=profile.headline,
                key_skills=", ".join(profile.technical_skills[:4]),
                hr_name=hr_contact.name.split()[0],
                hr_title=hr_contact.title,
                company=job.company,
                job_title=job.title,
            )
            result = await self.llm.aparse_json(
                user_prompt=prompt,
                system_prompt=LINKEDIN_MESSAGE_SYSTEM,
            )
            body = result.get("message", "")
            channel = OutreachChannel.LINKEDIN_MESSAGE

        return OutreachMessage(
            id=str(uuid.uuid4()),
            job_id=job.id or str(uuid.uuid4()),
            candidate_name=profile.name,
            hr_contact=hr_contact,
            channel=channel,
            body=body,
            status=OutreachStatus.PENDING_REVIEW,
        )

    async def generate_followup(
        self,
        original_message: OutreachMessage,
        days_since_sent: int,
    ) -> OutreachMessage:
        """Generate a polite follow-up message."""
        job_info = f"Job ID {original_message.job_id}"

        prompt = Template(FOLLOWUP_EMAIL_PROMPT).substitute(
            days_ago=days_since_sent,
            job_title=job_info,
            company=original_message.hr_contact.company,
            candidate_name=original_message.candidate_name,
        )

        result = await self.llm.aparse_json(user_prompt=prompt)

        return OutreachMessage(
            id=str(uuid.uuid4()),
            job_id=original_message.job_id,
            candidate_name=original_message.candidate_name,
            hr_contact=original_message.hr_contact,
            channel=original_message.channel,
            subject=result.get("subject", f"Following up - {original_message.subject}"),
            body=result.get("body", ""),
            status=OutreachStatus.PENDING_REVIEW,
            version=original_message.version + 1,
        )

    async def create_batch(
        self,
        jobs: List[Job],
        profile: CandidateProfile,
        hr_contacts: dict,  # job_id → List[HRContact]
        preferred_channel: OutreachChannel = OutreachChannel.EMAIL,
    ) -> OutreachBatch:
        """
        Generate outreach messages for a batch of jobs.
        All messages land in PENDING_REVIEW status until human approves.
        """
        batch_id = str(uuid.uuid4())[:8]
        messages = []

        logger.info(
            f"Generating outreach batch for {len(jobs)} jobs | "
            f"Channel: {preferred_channel.value}"
        )

        for job in jobs[:settings.max_daily_outreach]:
            contacts = hr_contacts.get(job.id, [])
            if not contacts:
                logger.warning(f"No HR contacts for job: {job.title} at {job.company}")
                continue

            # Use the most relevant contact
            contact = contacts[0]

            try:
                if preferred_channel == OutreachChannel.EMAIL:
                    msg = await self.generate_email(job, profile, contact)
                else:
                    msg = await self.generate_linkedin_message(
                        job, profile, contact,
                        message_type="connection" if preferred_channel == OutreachChannel.LINKEDIN_CONNECTION else "dm"
                    )
                messages.append(msg)
            except Exception as e:
                logger.error(f"Failed to generate outreach for {job.title}: {e}")

        batch = OutreachBatch(
            batch_id=batch_id,
            messages=messages,
            total=len(messages),
            pending_review=len(messages),
        )
        logger.success(f"Batch {batch_id}: {len(messages)} messages ready for review")
        return batch

    async def send_approved_message(
        self,
        message: OutreachMessage,
    ) -> bool:
        """
        Send a message that has been approved by the human.
        Uses browser automation to send via Gmail or LinkedIn.
        """
        if message.status != OutreachStatus.APPROVED:
            raise ValueError(
                f"Message {message.id} is not approved. "
                f"Current status: {message.status}"
            )

        logger.info(
            f"Sending approved {message.channel.value} to "
            f"{message.hr_contact.name} at {message.hr_contact.company}"
        )

        if message.channel == OutreachChannel.EMAIL:
            success = await self.gmail_agent.compose_and_send(
                message=message,
                dry_run=False,
            )
        elif message.channel in (
            OutreachChannel.LINKEDIN_CONNECTION,
            OutreachChannel.LINKEDIN_MESSAGE,
        ):
            if message.channel == OutreachChannel.LINKEDIN_CONNECTION:
                success = await self.linkedin_agent.send_connection_request(
                    linkedin_profile_url=message.hr_contact.linkedin_url or "",
                    connection_note=message.body,
                )
            else:
                success = await self.linkedin_agent.send_linkedin_message(
                    linkedin_profile_url=message.hr_contact.linkedin_url or "",
                    message=message.body,
                )
        else:
            logger.error(f"Unsupported channel: {message.channel}")
            return False

        return success

    async def _research_company_hook(
        self,
        company: str,
        industry: Optional[str],
    ) -> Optional[str]:
        """Get a personalization hook for a company."""
        try:
            prompt = Template(COMPANY_RESEARCH_PROMPT).substitute(
                company=company
            )
            result = await self.llm.aparse_json(user_prompt=prompt)
            return result.get("personalization_hook", "")
        except Exception:
            return None
