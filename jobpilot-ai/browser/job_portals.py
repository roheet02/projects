"""
JobPilot AI - Multi-Portal Job Browser Agent
Supports: LinkedIn, Indeed, Naukri, Glassdoor
The agent opens the actual website and navigates it like a human.
"""

import json
import re
from typing import List, Dict, Optional
from loguru import logger

from browser.browser_manager import AIBrowserAgent
from models.job import JobPortal


PORTAL_CONFIGS = {
    JobPortal.LINKEDIN: {
        "url": "https://www.linkedin.com/jobs/search/",
        "name": "LinkedIn",
    },
    JobPortal.INDEED: {
        "url": "https://www.indeed.com",
        "name": "Indeed",
    },
    JobPortal.NAUKRI: {
        "url": "https://www.naukri.com",
        "name": "Naukri",
    },
    JobPortal.GLASSDOOR: {
        "url": "https://www.glassdoor.com/Job/",
        "name": "Glassdoor",
    },
    JobPortal.WELLFOUND: {
        "url": "https://wellfound.com/jobs",
        "name": "Wellfound (AngelList)",
    },
}


class JobPortalAgent:
    """
    AI agent that browses job portals to discover relevant openings.
    Supports multiple portals with a unified interface.
    """

    def __init__(self):
        self.ai_browser = AIBrowserAgent()

    async def search_portal(
        self,
        portal: JobPortal,
        roles: List[str],
        location: str,
        skills: List[str],
        max_results: int = 20,
    ) -> List[Dict]:
        """
        Search a specific job portal for relevant openings.

        Args:
            portal: Which job portal to search
            roles: Target job titles
            location: Target location
            skills: Key skills for filtering
            max_results: Maximum jobs to extract

        Returns:
            List of raw job dicts
        """
        config = PORTAL_CONFIGS.get(portal)
        if not config:
            logger.warning(f"Unsupported portal: {portal}")
            return []

        portal_name = config["name"]
        url = config["url"]
        search_query = " ".join(roles[:2])
        skills_hint = ", ".join(skills[:5])

        task = f"""
        Go to {portal_name} ({url}).

        Search for jobs matching these criteria:
        - Job titles: {search_query}
        - Location: {location}
        - Key skills: {skills_hint}

        For each of the first {max_results} relevant job listings:
        1. Click on each job to see its full details
        2. Extract all available information:
           - Job title
           - Company name
           - Location (city, state, country)
           - Job type (remote/hybrid/onsite/full-time)
           - Salary range (if shown)
           - Required experience (years)
           - Key required skills (as a list)
           - A summary of the job description (2-3 sentences)
           - Direct URL of the job posting

        Return ONLY a valid JSON array (no other text):
        [
          {{
            "title": "...",
            "company": "...",
            "location": "...",
            "job_type": "...",
            "salary_range": "...",
            "experience_years": "...",
            "required_skills": ["skill1", "skill2"],
            "description": "...",
            "url": "..."
          }}
        ]

        Notes:
        - Skip promoted/sponsored jobs that are clearly not relevant
        - If you need to sign in, skip and try what's available without login
        - Extract real data only, no placeholders
        """

        logger.info(f"Searching {portal_name} for '{search_query}' in {location}")

        try:
            result = await self.ai_browser.run_task(task, url=url)
            jobs = self._parse_jobs(result)
            logger.success(f"{portal_name}: found {len(jobs)} jobs")
            return jobs
        except Exception as e:
            logger.error(f"{portal_name} search failed: {e}")
            return []

    async def search_all_portals(
        self,
        portals: List[JobPortal],
        roles: List[str],
        location: str,
        skills: List[str],
        max_per_portal: int = 15,
    ) -> List[Dict]:
        """
        Search multiple portals and aggregate results.
        Removes duplicate jobs (same title + company).
        """
        import asyncio

        logger.info(f"Searching {len(portals)} portals in parallel...")

        # Run all portal searches concurrently
        tasks = [
            self.search_portal(portal, roles, location, skills, max_per_portal)
            for portal in portals
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate and deduplicate
        all_jobs = []
        seen = set()

        for portal, portal_jobs in zip(portals, results):
            if isinstance(portal_jobs, Exception):
                logger.warning(f"Portal {portal} failed: {portal_jobs}")
                continue
            for job in portal_jobs:
                job["portal"] = portal.value
                key = f"{job.get('title','').lower()}_{job.get('company','').lower()}"
                if key not in seen:
                    seen.add(key)
                    all_jobs.append(job)

        logger.success(
            f"Aggregated {len(all_jobs)} unique jobs from {len(portals)} portals"
        )
        return all_jobs

    def _parse_jobs(self, result: str) -> List[Dict]:
        """Extract job dicts from AI browser result."""
        # Try direct JSON parse first
        try:
            data = json.loads(result.strip())
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON array in mixed text
        match = re.search(r'\[[\s\S]*\]', result)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        logger.warning("Could not parse job JSON from browser result")
        return []
