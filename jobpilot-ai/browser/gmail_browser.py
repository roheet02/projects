"""
JobPilot AI - Gmail Browser Agent
Uses AI browser automation to compose and send cold emails via Gmail.
The agent opens Gmail directly — no Gmail API credentials needed.

IMPORTANT: This agent only sends emails AFTER explicit human approval.
The outreach_review_required setting enforces this.
"""

from loguru import logger
from typing import Optional

from browser.browser_manager import AIBrowserAgent
from models.outreach import OutreachMessage
from config.settings import settings


class GmailAgent:
    """
    AI agent that opens Gmail in the browser and composes/sends emails.
    Uses browser automation — opens the actual Gmail website.
    """

    GMAIL_COMPOSE_URL = "https://mail.google.com/mail/u/0/#inbox"

    def __init__(self):
        self.ai_browser = AIBrowserAgent()

    async def compose_and_send(
        self,
        message: OutreachMessage,
        dry_run: bool = True,
    ) -> bool:
        """
        Open Gmail and send a cold email.

        Args:
            message: OutreachMessage with recipient, subject, and body
            dry_run: If True, compose but DO NOT click send (for preview)

        Returns:
            True if email was sent successfully
        """
        if settings.outreach_review_required and message.status.value != "approved":
            raise PermissionError(
                "Outreach review required. Message must be approved before sending. "
                "Set message.status = OutreachStatus.APPROVED after human review."
            )

        recipient = message.hr_contact.email
        if not recipient:
            raise ValueError(f"No email address for {message.hr_contact.name}")

        send_instruction = (
            "DO NOT click Send yet — just compose the email and leave it open for review."
            if dry_run
            else "Click the Send button to send the email."
        )

        task = f"""
        Go to Gmail (mail.google.com).

        Click the "Compose" button to open a new email.

        Fill in the email:
        - To: {recipient}
        - Subject: {message.subject}
        - Body: {message.body}

        {send_instruction}

        Report back whether the email was composed successfully.
        """

        action = "composing" if dry_run else "sending"
        logger.info(f"Gmail: {action} email to {recipient}")

        try:
            result = await self.ai_browser.run_task(task)
            success = "success" in result.lower() or "composed" in result.lower() or "sent" in result.lower()
            if success:
                logger.success(f"Email {'composed (dry run)' if dry_run else 'sent'} to {recipient}")
            return success
        except Exception as e:
            logger.error(f"Gmail action failed: {e}")
            return False

    async def check_inbox_for_replies(
        self,
        search_terms: list,
    ) -> list:
        """
        Check Gmail inbox for replies related to job applications.

        Args:
            search_terms: List of company names or email subjects to search for

        Returns:
            List of reply summaries
        """
        search_query = " OR ".join(f'"{term}"' for term in search_terms[:5])

        task = f"""
        Go to Gmail (mail.google.com).
        Search for emails with query: {search_query}

        For each email found in the last 30 days:
        1. Note the sender name and email
        2. Note the subject
        3. Note the date received
        4. Summarize the email content in 1-2 sentences
        5. Classify as: "positive_response", "rejection", "auto_reply", "interview_invite"

        Return as JSON array:
        [
          {{
            "sender": "...",
            "subject": "...",
            "date": "...",
            "summary": "...",
            "classification": "..."
          }}
        ]
        """

        logger.info(f"Checking Gmail for replies: {search_query}")
        try:
            result = await self.ai_browser.run_task(task)
            import json, re
            match = re.search(r'\[.*\]', result, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as e:
            logger.error(f"Gmail inbox check failed: {e}")
        return []
