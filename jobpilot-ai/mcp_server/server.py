"""
JobPilot AI - MCP Server
Exposes JobPilot AI capabilities as MCP tools.
Compatible with Claude Desktop and any MCP-enabled client.

To use with Claude Desktop, add to claude_desktop_config.json:
{
  "mcpServers": {
    "jobpilot": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/jobpilot-ai"
    }
  }
}
"""

import asyncio
import json
from typing import Any, Optional
from loguru import logger

try:
    from fastmcp import FastMCP
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("fastmcp not installed. MCP server disabled.")


if MCP_AVAILABLE:
    mcp = FastMCP("JobPilot AI")

    from agents.orchestrator import JobPilotOrchestrator
    from models.job import JobPortal
    from models.outreach import OutreachChannel, OutreachStatus

    orchestrator = JobPilotOrchestrator()

    # ------------------------------------------------------------------ #
    # MCP TOOLS
    # ------------------------------------------------------------------ #

    @mcp.tool()
    async def parse_resume(resume_path: str) -> str:
        """
        Parse a resume file (PDF or DOCX) and extract a structured candidate profile.

        Args:
            resume_path: Absolute path to the resume file

        Returns:
            JSON string with the extracted candidate profile
        """
        profile = await orchestrator.load_profile(resume_path)
        return json.dumps({
            "name": profile.name,
            "headline": profile.headline,
            "years_of_experience": profile.years_of_experience,
            "technical_skills": profile.technical_skills,
            "domains": profile.domains,
            "target_roles": profile.target_roles,
            "current_role": profile.current_role,
            "current_company": profile.current_company,
            "summary": profile.summary[:300],
        }, indent=2)

    @mcp.tool()
    async def find_jobs(
        roles: str,
        locations: str,
        portals: str = "linkedin,indeed",
        max_per_portal: int = 15,
    ) -> str:
        """
        Search job portals and find relevant job openings.

        Args:
            roles: Comma-separated list of target job titles
                   e.g., "Data Scientist, ML Engineer, Applied Scientist"
            locations: Comma-separated list of locations
                       e.g., "Bangalore, Mumbai, Remote"
            portals: Comma-separated portal names: linkedin, indeed, naukri, glassdoor
            max_per_portal: Maximum jobs to find per portal

        Returns:
            JSON string with list of discovered jobs
        """
        role_list = [r.strip() for r in roles.split(",")]
        location_list = [l.strip() for l in locations.split(",")]
        portal_list = [
            JobPortal(p.strip().lower())
            for p in portals.split(",")
            if p.strip().lower() in [x.value for x in JobPortal]
        ]

        from browser.job_portals import JobPortalAgent
        agent = JobPortalAgent()

        raw_jobs = await agent.search_all_portals(
            portals=portal_list,
            roles=role_list,
            location=location_list[0],
            skills=[],
            max_per_portal=max_per_portal,
        )

        return json.dumps({
            "total": len(raw_jobs),
            "jobs": raw_jobs[:20],
        }, indent=2)

    @mcp.tool()
    async def match_jobs_to_profile(
        resume_path: str,
        top_k: int = 10,
    ) -> str:
        """
        Match discovered jobs against a candidate's resume profile using ML scoring.

        Args:
            resume_path: Path to resume file
            top_k: Number of top matches to return

        Returns:
            JSON with ranked job matches and scores
        """
        await orchestrator.load_profile(resume_path)
        if not orchestrator.found_jobs:
            return json.dumps({"error": "No jobs found. Run find_jobs first."})

        matched = await orchestrator.match_and_rank_jobs(top_k=top_k, explain=True)

        return json.dumps({
            "total_matched": len(matched),
            "threshold": orchestrator.job_matcher.threshold,
            "matches": [
                {
                    "rank": i+1,
                    "title": j.title,
                    "company": j.company,
                    "location": j.location,
                    "match_score": j.match_score,
                    "match_reasons": j.match_reasons,
                    "url": j.url,
                }
                for i, j in enumerate(matched)
            ]
        }, indent=2)

    @mcp.tool()
    async def find_hr_contacts(
        company: str,
        target_role: str,
    ) -> str:
        """
        Find HR and recruiter contacts at a specific company.

        Args:
            company: Company name to search
            target_role: Target job role (used to find relevant contacts)

        Returns:
            JSON with list of HR contacts (name, title, LinkedIn URL)
        """
        from browser.linkedin_browser import LinkedInJobAgent
        agent = LinkedInJobAgent()

        contacts = await agent.find_hr_contacts(
            company=company,
            target_role=target_role,
        )

        return json.dumps({
            "company": company,
            "contacts_found": len(contacts),
            "contacts": contacts,
        }, indent=2)

    @mcp.tool()
    async def draft_cold_email(
        resume_path: str,
        job_title: str,
        company: str,
        hr_name: str,
        hr_title: str,
        hr_linkedin_url: Optional[str] = None,
        job_requirements: str = "",
    ) -> str:
        """
        Draft a personalized cold email for a job application.

        Args:
            resume_path: Path to candidate's resume
            job_title: The job title being applied for
            company: Target company name
            hr_name: HR/recruiter's full name
            hr_title: HR/recruiter's job title
            hr_linkedin_url: Their LinkedIn profile URL (optional)
            job_requirements: Key job requirements (comma-separated)

        Returns:
            JSON with subject line and email body (ready for review)
        """
        from agents.outreach_agent import OutreachAgent
        from models.job import Job
        from models.outreach import HRContact

        # Load profile
        parser = __import__('core.resume_parser', fromlist=['ResumeParser']).ResumeParser()
        profile = await parser.aparse(resume_path)

        # Create job object
        job = Job(
            id="manual",
            title=job_title,
            company=company,
            location="",
            required_skills=[r.strip() for r in job_requirements.split(",") if r.strip()],
        )

        # Create HR contact
        contact = HRContact(
            name=hr_name,
            title=hr_title,
            company=company,
            linkedin_url=hr_linkedin_url,
        )

        agent = OutreachAgent()
        message = await agent.generate_email(job, profile, contact)

        return json.dumps({
            "status": "draft_ready",
            "subject": message.subject,
            "body": message.body,
            "recipient": f"{hr_name} ({hr_title}) at {company}",
            "note": "Review and edit before sending. Use send_outreach_message to send.",
            "message_id": message.id,
        }, indent=2)

    @mcp.tool()
    async def draft_linkedin_message(
        resume_path: str,
        job_title: str,
        company: str,
        hr_name: str,
        hr_title: str,
        hr_linkedin_url: str,
        message_type: str = "connection",
    ) -> str:
        """
        Draft a LinkedIn connection request or direct message.

        Args:
            resume_path: Path to candidate's resume
            job_title: Target job title
            company: Target company
            hr_name: HR/recruiter's name
            hr_title: Their title
            hr_linkedin_url: Their LinkedIn profile URL
            message_type: "connection" (300 char note) or "dm" (longer message)

        Returns:
            JSON with the drafted LinkedIn message
        """
        from agents.outreach_agent import OutreachAgent
        from models.job import Job
        from models.outreach import HRContact

        parser = __import__('core.resume_parser', fromlist=['ResumeParser']).ResumeParser()
        profile = await parser.aparse(resume_path)

        job = Job(id="manual", title=job_title, company=company, location="")
        contact = HRContact(
            name=hr_name, title=hr_title,
            company=company, linkedin_url=hr_linkedin_url
        )

        agent = OutreachAgent()
        message = await agent.generate_linkedin_message(
            job, profile, contact, message_type=message_type
        )

        return json.dumps({
            "status": "draft_ready",
            "type": message_type,
            "message": message.body,
            "char_count": len(message.body),
            "limit": 300 if message_type == "connection" else 1000,
            "recipient_url": hr_linkedin_url,
        }, indent=2)

    @mcp.tool()
    async def run_full_pipeline(
        resume_path: str,
        portals: str = "linkedin,indeed",
        channel: str = "email",
    ) -> str:
        """
        Run the complete JobPilot AI pipeline:
        parse resume → find jobs → match → research HR → generate outreach.

        Args:
            resume_path: Path to resume file
            portals: Comma-separated portal names (linkedin, indeed, naukri, glassdoor)
            channel: Outreach channel - "email", "linkedin_connection", or "linkedin_message"

        Returns:
            Summary of pipeline results
        """
        portal_list = [
            JobPortal(p.strip().lower())
            for p in portals.split(",")
            if p.strip().lower() in [x.value for x in JobPortal]
        ]

        channel_map = {
            "email": OutreachChannel.EMAIL,
            "linkedin_connection": OutreachChannel.LINKEDIN_CONNECTION,
            "linkedin_message": OutreachChannel.LINKEDIN_MESSAGE,
        }
        outreach_channel = channel_map.get(channel, OutreachChannel.EMAIL)

        await orchestrator.run_full_pipeline(
            resume_path=resume_path,
            portals=portal_list,
            channel=outreach_channel,
        )

        return json.dumps({
            "status": "pipeline_complete",
            "profile": orchestrator.profile.name if orchestrator.profile else None,
            "jobs_found": len(orchestrator.found_jobs),
            "jobs_matched": len(orchestrator.matched_jobs),
            "companies_researched": len(orchestrator.hr_contacts),
            "outreach_drafted": orchestrator.outreach_batch.total if orchestrator.outreach_batch else 0,
            "next_step": "Open Streamlit UI to review and approve outreach messages",
        }, indent=2)


def run_server():
    """Start the MCP server."""
    if not MCP_AVAILABLE:
        logger.error("fastmcp not installed. Run: pip install fastmcp")
        return

    logger.info("Starting JobPilot AI MCP Server...")
    mcp.run()


if __name__ == "__main__":
    run_server()
