"""
JobPilot AI - Main Orchestrator
The master agent that coordinates the entire job-hunting workflow.

Workflow:
  1. Parse resume → build candidate profile
  2. Generate semantic embedding for candidate
  3. Search job portals → raw job listings
  4. Score and rank jobs using ML matcher
  5. Research HR contacts for top-ranked jobs
  6. Generate personalized outreach (email + LinkedIn)
  7. Present all drafts for human review
  8. [After approval] Send via browser automation
  9. Track outcomes and compute analytics
"""

import asyncio
import uuid
from typing import List, Optional, Dict
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from core.resume_parser import ResumeParser
from core.job_matcher import JobMatcher
from core.embeddings import embedding_engine
from agents.outreach_agent import OutreachAgent
from browser.job_portals import JobPortalAgent
from browser.linkedin_browser import LinkedInJobAgent
from models.job import Job, JobPortal, JobSearchQuery, JobSearchResult, MatchStatus
from models.candidate import CandidateProfile
from models.outreach import (
    HRContact, OutreachBatch, OutreachChannel, OutreachMessage, OutreachStatus
)
from database.repository import JobRepository
from config.settings import settings


console = Console()


class JobPilotOrchestrator:
    """
    The brain of JobPilot AI.
    Coordinates all agents, manages state, and drives the workflow.
    """

    def __init__(self):
        self.resume_parser = ResumeParser()
        self.job_matcher = JobMatcher()
        self.outreach_agent = OutreachAgent()
        self.portal_agent = JobPortalAgent()
        self.linkedin_agent = LinkedInJobAgent()
        self.repository = JobRepository()

        # Session state
        self.profile: Optional[CandidateProfile] = None
        self.found_jobs: List[Job] = []
        self.matched_jobs: List[Job] = []
        self.hr_contacts: Dict[str, List[HRContact]] = {}
        self.outreach_batch: Optional[OutreachBatch] = None

    # ------------------------------------------------------------------ #
    # STEP 1: Resume Processing
    # ------------------------------------------------------------------ #

    async def load_profile(self, resume_path: str) -> CandidateProfile:
        """Parse resume and build candidate profile."""
        console.print(Panel("📄 [bold cyan]Step 1: Parsing Resume[/bold cyan]"))

        self.profile = await self.resume_parser.aparse(resume_path)

        # Generate and cache embedding vector
        profile_text = self.profile.full_profile_text
        self.profile.profile_embedding = embedding_engine.encode(profile_text)[0].tolist()

        # Display profile summary
        self._display_profile(self.profile)
        await self.repository.save_profile(self.profile)

        return self.profile

    def _display_profile(self, profile: CandidateProfile):
        table = Table(title="Candidate Profile", box=box.ROUNDED)
        table.add_column("Field", style="bold cyan", width=20)
        table.add_column("Value", style="white")

        table.add_row("Name", profile.name)
        table.add_row("Headline", profile.headline)
        table.add_row("Experience", f"{profile.years_of_experience} years")
        table.add_row("Current Role", f"{profile.current_role} at {profile.current_company}")
        table.add_row("Top Skills", ", ".join(profile.technical_skills[:8]))
        table.add_row("Domains", ", ".join(profile.domains))
        table.add_row("Target Roles", ", ".join(profile.target_roles))

        console.print(table)

    # ------------------------------------------------------------------ #
    # STEP 2: Job Discovery
    # ------------------------------------------------------------------ #

    async def discover_jobs(
        self,
        portals: Optional[List[JobPortal]] = None,
        custom_query: Optional[JobSearchQuery] = None,
    ) -> List[Job]:
        """Search job portals and extract relevant openings."""
        if not self.profile:
            raise RuntimeError("Load a profile first with load_profile()")

        console.print(Panel("🔍 [bold cyan]Step 2: Discovering Jobs[/bold cyan]"))

        # Build search parameters
        portals = portals or [JobPortal(p) for p in settings.job_portal_list]
        roles = custom_query.roles if custom_query else self.profile.target_roles or [self.profile.current_role]
        locations = custom_query.locations if custom_query else self.profile.target_locations or ["India"]
        skills = custom_query.skills if custom_query else self.profile.technical_skills

        console.print(
            f"Searching [bold]{', '.join(p.value for p in portals)}[/bold] "
            f"for [bold cyan]{', '.join(roles[:2])}[/bold cyan] "
            f"in [bold]{', '.join(locations[:2])}[/bold]..."
        )

        # Run portal searches
        raw_jobs = await self.portal_agent.search_all_portals(
            portals=portals,
            roles=roles,
            location=locations[0],
            skills=skills,
            max_per_portal=settings.max_jobs_per_search // len(portals),
        )

        # Convert to Job objects
        self.found_jobs = self._convert_raw_jobs(raw_jobs)
        console.print(f"[green]✓ Discovered {len(self.found_jobs)} unique jobs[/green]")

        # Save to DB
        for job in self.found_jobs:
            await self.repository.save_job(job)

        return self.found_jobs

    def _convert_raw_jobs(self, raw_jobs: List[Dict]) -> List[Job]:
        """Convert raw job dicts from browser to Job model objects."""
        jobs = []
        for i, raw in enumerate(raw_jobs):
            try:
                portal_str = raw.get("portal", "other")
                try:
                    portal = JobPortal(portal_str)
                except ValueError:
                    portal = JobPortal.OTHER

                job = Job(
                    id=str(uuid.uuid4()),
                    title=raw.get("title", "Unknown Role"),
                    company=raw.get("company", "Unknown Company"),
                    location=raw.get("location", ""),
                    portal=portal,
                    url=raw.get("url"),
                    description=raw.get("description", ""),
                    required_skills=raw.get("required_skills", []),
                    preferred_skills=raw.get("preferred_skills", []),
                    experience_years=raw.get("experience_years"),
                    salary_range=raw.get("salary_range"),
                    industry=raw.get("industry"),
                )
                jobs.append(job)
            except Exception as e:
                logger.warning(f"Failed to parse job #{i}: {e}")
        return jobs

    # ------------------------------------------------------------------ #
    # STEP 3: ML Matching
    # ------------------------------------------------------------------ #

    async def match_and_rank_jobs(
        self,
        top_k: int = 20,
        explain: bool = True,
    ) -> List[Job]:
        """Score and rank jobs using multi-signal ML matching."""
        if not self.profile or not self.found_jobs:
            raise RuntimeError("Run discover_jobs() first")

        console.print(Panel("🤖 [bold cyan]Step 3: ML Job Matching[/bold cyan]"))
        console.print(
            f"Scoring [bold]{len(self.found_jobs)}[/bold] jobs with "
            "semantic similarity + skill matching + experience analysis..."
        )

        self.matched_jobs = self.job_matcher.rank_jobs(
            jobs=self.found_jobs,
            profile=self.profile,
            top_k=top_k,
            explain=explain,
        )

        # Get market statistics
        stats = self.job_matcher.get_match_statistics(self.found_jobs, self.profile)

        self._display_matched_jobs(self.matched_jobs[:10], stats)
        return self.matched_jobs

    def _display_matched_jobs(self, jobs: List[Job], stats: Dict):
        # Stats panel
        console.print(f"\n[bold]Market Analysis:[/bold]")
        console.print(f"  Mean match score: [cyan]{stats.get('mean_match_score', 0):.0%}[/cyan]")
        console.print(f"  Top matches: [green]{stats.get('above_threshold', 0)}[/green] jobs")
        console.print(f"  Skill gaps to consider: {', '.join(stats.get('skill_gaps', [])[:5])}")

        # Job table
        table = Table(title="Top Matched Jobs", box=box.ROUNDED)
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", style="bold green", width=7)
        table.add_column("Title", style="bold white", width=30)
        table.add_column("Company", style="cyan", width=20)
        table.add_column("Location", width=15)
        table.add_column("Top Match Reason", style="dim", width=35)

        for i, job in enumerate(jobs, 1):
            score_color = "green" if job.match_score > 0.75 else "yellow" if job.match_score > 0.6 else "red"
            table.add_row(
                str(i),
                f"[{score_color}]{job.match_score:.0%}[/{score_color}]",
                job.title,
                job.company,
                job.location[:14],
                job.match_reasons[0][:34] if job.match_reasons else "N/A",
            )

        console.print(table)

    # ------------------------------------------------------------------ #
    # STEP 4: HR Research
    # ------------------------------------------------------------------ #

    async def research_hr_contacts(
        self,
        jobs: Optional[List[Job]] = None,
    ) -> Dict[str, List[HRContact]]:
        """Find HR/recruiter contacts for top matched jobs."""
        target_jobs = jobs or self.matched_jobs[:10]

        if not target_jobs:
            raise RuntimeError("No matched jobs to research")

        console.print(Panel("👥 [bold cyan]Step 4: HR Contact Research[/bold cyan]"))

        for job in target_jobs:
            console.print(f"  Researching contacts at [bold]{job.company}[/bold]...")
            try:
                raw_contacts = await self.linkedin_agent.find_hr_contacts(
                    company=job.company,
                    target_role=job.title,
                )
                contacts = [
                    HRContact(
                        id=str(uuid.uuid4()),
                        name=c.get("name", "Unknown"),
                        title=c.get("title", "HR"),
                        company=job.company,
                        linkedin_url=c.get("linkedin_url"),
                        relevance_note=c.get("relevance"),
                    )
                    for c in raw_contacts
                ]
                if contacts:
                    self.hr_contacts[job.id] = contacts
                    console.print(
                        f"    [green]✓ Found {len(contacts)} contacts[/green]"
                    )
                else:
                    console.print(f"    [yellow]⚠ No contacts found[/yellow]")
            except Exception as e:
                logger.warning(f"HR research failed for {job.company}: {e}")

        console.print(
            f"\n[green]✓ Found contacts for "
            f"{len(self.hr_contacts)}/{len(target_jobs)} companies[/green]"
        )
        return self.hr_contacts

    # ------------------------------------------------------------------ #
    # STEP 5: Outreach Generation
    # ------------------------------------------------------------------ #

    async def generate_outreach(
        self,
        channel: OutreachChannel = OutreachChannel.EMAIL,
    ) -> OutreachBatch:
        """Generate personalized outreach messages for all matched jobs."""
        if not self.profile or not self.matched_jobs:
            raise RuntimeError("Complete matching step first")

        console.print(Panel("✉️  [bold cyan]Step 5: Generating Outreach[/bold cyan]"))
        console.print(
            f"Generating [bold]{channel.value}[/bold] messages "
            f"for {min(len(self.matched_jobs), settings.max_daily_outreach)} jobs..."
        )

        self.outreach_batch = await self.outreach_agent.create_batch(
            jobs=self.matched_jobs,
            profile=self.profile,
            hr_contacts=self.hr_contacts,
            preferred_channel=channel,
        )

        console.print(
            f"\n[green]✓ Generated {self.outreach_batch.total} messages "
            f"— all pending your review[/green]"
        )
        console.print(
            "[yellow]Review each message in the UI before sending.[/yellow]"
        )
        return self.outreach_batch

    # ------------------------------------------------------------------ #
    # STEP 6: Human Review + Send
    # ------------------------------------------------------------------ #

    def review_messages(self) -> List[OutreachMessage]:
        """Display all messages pending review."""
        if not self.outreach_batch:
            return []

        pending = [
            m for m in self.outreach_batch.messages
            if m.status == OutreachStatus.PENDING_REVIEW
        ]

        console.print(
            Panel(
                f"[bold yellow]Review Queue: {len(pending)} messages pending approval[/bold yellow]\n"
                "Use the Streamlit UI or CLI to review, edit, and approve each message."
            )
        )
        for msg in pending:
            console.print(msg.to_display())

        return pending

    async def send_approved_messages(self) -> Dict[str, int]:
        """Send all approved messages via browser automation."""
        if not self.outreach_batch:
            return {}

        approved = [
            m for m in self.outreach_batch.messages
            if m.status == OutreachStatus.APPROVED
        ]

        if not approved:
            console.print("[yellow]No approved messages to send.[/yellow]")
            return {"sent": 0, "failed": 0}

        console.print(
            Panel(
                f"[bold green]Sending {len(approved)} approved messages...[/bold green]"
            )
        )

        results = {"sent": 0, "failed": 0}
        for msg in approved:
            success = await self.outreach_agent.send_approved_message(msg)
            if success:
                msg.status = OutreachStatus.SENT
                results["sent"] += 1
                console.print(f"  [green]✓ Sent to {msg.hr_contact.name}[/green]")
            else:
                results["failed"] += 1
                console.print(
                    f"  [red]✗ Failed: {msg.hr_contact.name}[/red]"
                )
            await asyncio.sleep(2)  # Small delay between sends

        console.print(
            f"\n[bold]Results:[/bold] "
            f"[green]{results['sent']} sent[/green] | "
            f"[red]{results['failed']} failed[/red]"
        )
        return results

    # ------------------------------------------------------------------ #
    # FULL PIPELINE RUN
    # ------------------------------------------------------------------ #

    async def run_full_pipeline(
        self,
        resume_path: str,
        portals: Optional[List[JobPortal]] = None,
        channel: OutreachChannel = OutreachChannel.EMAIL,
    ):
        """
        Run the complete JobPilot AI pipeline end-to-end.

        Steps:
        1. Parse resume
        2. Discover jobs
        3. ML matching
        4. HR research
        5. Generate outreach
        6. Present for review
        """
        console.print(
            Panel(
                "[bold cyan]🚀 JobPilot AI — Full Pipeline[/bold cyan]\n"
                "Autonomous job hunting agent starting...",
                border_style="cyan",
            )
        )

        start = datetime.utcnow()

        # Run pipeline steps
        await self.load_profile(resume_path)
        await self.discover_jobs(portals=portals)
        await self.match_and_rank_jobs(top_k=20, explain=True)
        await self.research_hr_contacts()
        await self.generate_outreach(channel=channel)

        elapsed = (datetime.utcnow() - start).seconds
        console.print(
            Panel(
                f"[bold green]✅ Pipeline complete in {elapsed}s[/bold green]\n\n"
                f"📊 Found: {len(self.found_jobs)} jobs\n"
                f"🎯 Matched: {len(self.matched_jobs)} above threshold\n"
                f"👥 HR Contacts: {len(self.hr_contacts)} companies\n"
                f"✉️  Outreach drafts: {self.outreach_batch.total if self.outreach_batch else 0}\n\n"
                "[yellow]Open the Streamlit UI to review and send messages.[/yellow]",
                border_style="green",
            )
        )
