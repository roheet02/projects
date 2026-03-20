"""
JobPilot AI - LinkedIn Browser Agent
Uses AI browser automation to:
  1. Search and extract job listings from LinkedIn Jobs
  2. Find HR/recruiter contacts at target companies
  3. Send LinkedIn connection requests and direct messages
"""

import json
from typing import List, Optional, Dict
from loguru import logger

from browser.browser_manager import AIBrowserAgent, BrowserManager
from models.job import Job, JobPortal
from models.outreach import HRContact, OutreachMessage, OutreachChannel, OutreachStatus
from config.settings import settings


class LinkedInJobAgent:
    """
    AI agent that directly opens LinkedIn and searches for jobs.
    No API key or scraping — opens the actual LinkedIn website.
    """

    LINKEDIN_JOBS_URL = "https://www.linkedin.com/jobs/search/"

    def __init__(self):
        self.ai_browser = AIBrowserAgent()
        self.browser_manager = BrowserManager()

    async def search_jobs(
        self,
        roles: List[str],
        locations: List[str],
        skills: List[str],
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Open LinkedIn Jobs and search for positions matching the criteria.

        Returns raw job data as list of dicts (further processed by parsers).
        """
        search_query = " OR ".join(roles[:3])  # LinkedIn OR search
        location = locations[0] if locations else "India"

        task = f"""
        Go to LinkedIn Jobs (linkedin.com/jobs/search).
        Search for jobs with query: "{search_query}"
        Location: "{location}"

        For each of the first {max_results} job listings visible:
        1. Click on the job to see details
        2. Extract: job title, company name, location, job type (remote/hybrid/onsite),
           full job description text, required skills mentioned, years of experience required,
           and the job URL

        Return the results as a JSON array with this structure:
        [
          {{
            "title": "...",
            "company": "...",
            "location": "...",
            "job_type": "remote/hybrid/onsite/full_time",
            "description": "...",
            "required_skills": ["skill1", "skill2"],
            "experience_years": "3-5 years",
            "url": "https://linkedin.com/jobs/..."
          }}
        ]

        Important:
        - If LinkedIn asks you to sign in, note that in the result
        - Focus on extracting real data, not placeholder text
        - Extract at least the first 5 jobs if possible
        """

        logger.info(f"LinkedIn job search: '{search_query}' in {location}")

        try:
            result = await self.ai_browser.run_task(task)

            # Try to parse JSON from result
            jobs_data = self._extract_json_from_result(result)
            logger.success(f"Found {len(jobs_data)} jobs on LinkedIn")
            return jobs_data

        except Exception as e:
            logger.error(f"LinkedIn job search failed: {e}")
            return []

    async def find_hr_contacts(
        self,
        company: str,
        target_role: str,
    ) -> List[Dict]:
        """
        Search LinkedIn for HR/recruiter contacts at a company.
        Returns list of contact dicts.
        """
        task = f"""
        Go to LinkedIn (linkedin.com) and search for HR or recruiter contacts at {company}.

        Search query: "recruiter OR HR manager OR talent acquisition {company}"

        For each person found (up to 5):
        1. Get their full name
        2. Get their job title
        3. Get their company name (confirm it's {company})
        4. Get their LinkedIn profile URL

        We are looking for: Technical Recruiter, HR Manager, Talent Acquisition,
        People Operations, HR Business Partner

        Return as JSON array:
        [
          {{
            "name": "...",
            "title": "...",
            "company": "{company}",
            "linkedin_url": "https://linkedin.com/in/...",
            "relevance": "why this person is relevant for {target_role} role"
          }}
        ]
        """

        logger.info(f"Finding HR contacts at {company}")
        try:
            result = await self.ai_browser.run_task(task)
            contacts = self._extract_json_from_result(result)
            logger.success(f"Found {len(contacts)} contacts at {company}")
            return contacts
        except Exception as e:
            logger.error(f"HR search failed: {e}")
            return []

    async def send_connection_request(
        self,
        linkedin_profile_url: str,
        connection_note: str,
    ) -> bool:
        """
        Send a LinkedIn connection request with a personalized note.
        ALWAYS requires human approval before calling this.
        """
        task = f"""
        Go to this LinkedIn profile: {linkedin_profile_url}

        Click the "Connect" button.
        If asked "How do you know this person?", select the most appropriate option.
        Click "Add a note" and type this message:
        "{connection_note}"
        Click "Send".

        Report back whether the connection request was sent successfully.
        """

        logger.info(f"Sending connection request to {linkedin_profile_url}")
        try:
            result = await self.ai_browser.run_task(task)
            success = "success" in result.lower() or "sent" in result.lower()
            return success
        except Exception as e:
            logger.error(f"Connection request failed: {e}")
            return False

    async def send_linkedin_message(
        self,
        linkedin_profile_url: str,
        message: str,
    ) -> bool:
        """
        Send a direct message to an existing LinkedIn connection.
        ALWAYS requires human approval before calling this.
        """
        task = f"""
        Go to this LinkedIn profile: {linkedin_profile_url}
        Click the "Message" button.
        Type this message in the message box:
        "{message}"
        Click "Send".

        Report back whether the message was sent successfully.
        """

        logger.info(f"Sending LinkedIn DM to {linkedin_profile_url}")
        try:
            result = await self.ai_browser.run_task(task)
            return "success" in result.lower() or "sent" in result.lower()
        except Exception as e:
            logger.error(f"LinkedIn message failed: {e}")
            return False

    def _extract_json_from_result(self, result: str) -> List[Dict]:
        """Extract JSON array from AI browser result string."""
        import re
        # Try to find JSON array in result
        match = re.search(r'\[.*\]', result, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        return []
