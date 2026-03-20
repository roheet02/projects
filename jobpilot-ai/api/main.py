"""
JobPilot AI - FastAPI Application
REST API for the JobPilot AI agent system.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import tempfile
import os

from config.settings import settings
from agents.orchestrator import JobPilotOrchestrator
from models.job import JobPortal
from models.outreach import OutreachChannel, OutreachStatus


app = FastAPI(
    title="JobPilot AI API",
    description="Autonomous AI Job Hunting Agent",
    version=settings.app_version,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator instance
orchestrator = JobPilotOrchestrator()


# ------------------------------------------------------------------ #
# Request/Response Models
# ------------------------------------------------------------------ #

class JobSearchRequest(BaseModel):
    roles: List[str]
    locations: List[str]
    portals: List[str] = ["linkedin", "indeed"]
    max_per_portal: int = 15


class OutreachRequest(BaseModel):
    channel: str = "email"


class ApproveMessageRequest(BaseModel):
    message_id: str
    edited_body: Optional[str] = None
    edited_subject: Optional[str] = None


# ------------------------------------------------------------------ #
# Health Check
# ------------------------------------------------------------------ #

@app.get("/health")
async def health():
    return {"status": "healthy", "app": settings.app_name, "version": settings.app_version}


# ------------------------------------------------------------------ #
# Resume Endpoints
# ------------------------------------------------------------------ #

@app.post("/resume/parse")
async def parse_resume(file: UploadFile = File(...)):
    """Upload and parse a resume file (PDF or DOCX)."""
    if not file.filename.endswith((".pdf", ".docx")):
        raise HTTPException(400, "Only PDF and DOCX files are supported")

    suffix = ".pdf" if file.filename.endswith(".pdf") else ".docx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        profile = await orchestrator.load_profile(tmp_path)
        return {
            "name": profile.name,
            "headline": profile.headline,
            "years_of_experience": profile.years_of_experience,
            "technical_skills": profile.technical_skills,
            "domains": profile.domains,
            "target_roles": profile.target_roles,
            "summary": profile.summary[:500],
        }
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------ #
# Job Discovery Endpoints
# ------------------------------------------------------------------ #

@app.post("/jobs/search")
async def search_jobs(request: JobSearchRequest):
    """Search job portals for relevant openings."""
    if not orchestrator.profile:
        raise HTTPException(400, "Parse resume first")

    portal_list = []
    for p in request.portals:
        try:
            portal_list.append(JobPortal(p.lower()))
        except ValueError:
            pass

    jobs = await orchestrator.discover_jobs(portals=portal_list)
    return {
        "total": len(jobs),
        "jobs": [
            {
                "id": j.id,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "portal": j.portal.value,
                "url": j.url,
                "match_score": j.match_score,
            }
            for j in jobs
        ],
    }


@app.post("/jobs/match")
async def match_jobs(top_k: int = 20, explain: bool = True):
    """Run ML matching on discovered jobs."""
    if not orchestrator.found_jobs:
        raise HTTPException(400, "Search for jobs first")

    matched = await orchestrator.match_and_rank_jobs(top_k=top_k, explain=explain)
    return {
        "total_matched": len(matched),
        "matches": [
            {
                "rank": i + 1,
                "title": j.title,
                "company": j.company,
                "location": j.location,
                "match_score": j.match_score,
                "match_reasons": j.match_reasons,
                "required_skills": j.required_skills,
                "url": j.url,
            }
            for i, j in enumerate(matched)
        ],
    }


# ------------------------------------------------------------------ #
# HR Research Endpoints
# ------------------------------------------------------------------ #

@app.post("/hr/research")
async def research_hr():
    """Find HR contacts for matched jobs."""
    if not orchestrator.matched_jobs:
        raise HTTPException(400, "Run matching first")

    contacts = await orchestrator.research_hr_contacts()
    return {
        "companies_researched": len(contacts),
        "contacts": {
            job_id: [
                {
                    "name": c.name,
                    "title": c.title,
                    "company": c.company,
                    "linkedin_url": c.linkedin_url,
                }
                for c in contact_list
            ]
            for job_id, contact_list in contacts.items()
        },
    }


# ------------------------------------------------------------------ #
# Outreach Endpoints
# ------------------------------------------------------------------ #

@app.post("/outreach/generate")
async def generate_outreach(request: OutreachRequest):
    """Generate personalized outreach messages."""
    if not orchestrator.matched_jobs:
        raise HTTPException(400, "Complete matching first")

    channel_map = {
        "email": OutreachChannel.EMAIL,
        "linkedin_connection": OutreachChannel.LINKEDIN_CONNECTION,
        "linkedin_message": OutreachChannel.LINKEDIN_MESSAGE,
    }
    channel = channel_map.get(request.channel, OutreachChannel.EMAIL)

    batch = await orchestrator.generate_outreach(channel=channel)
    return {
        "batch_id": batch.batch_id,
        "total": batch.total,
        "pending_review": batch.pending_review,
        "messages": [
            {
                "id": m.id,
                "hr_name": m.hr_contact.name,
                "company": m.hr_contact.company,
                "channel": m.channel.value,
                "subject": m.subject,
                "body": m.body,
                "status": m.status.value,
            }
            for m in batch.messages
        ],
    }


@app.post("/outreach/approve")
async def approve_message(request: ApproveMessageRequest):
    """Approve a message for sending."""
    if not orchestrator.outreach_batch:
        raise HTTPException(400, "Generate outreach first")

    message = next(
        (m for m in orchestrator.outreach_batch.messages if m.id == request.message_id),
        None,
    )
    if not message:
        raise HTTPException(404, "Message not found")

    if request.edited_body:
        message.body = request.edited_body
        message.human_edited = True
    if request.edited_subject:
        message.subject = request.edited_subject
        message.human_edited = True

    message.status = OutreachStatus.APPROVED
    return {"status": "approved", "message_id": message.id}


@app.post("/outreach/send-approved")
async def send_approved():
    """Send all approved messages via browser automation."""
    results = await orchestrator.send_approved_messages()
    return results


# ------------------------------------------------------------------ #
# Analytics
# ------------------------------------------------------------------ #

@app.get("/analytics/overview")
async def analytics_overview():
    """Get analytics overview."""
    from analytics.dashboard import SearchAnalytics
    analytics = SearchAnalytics()

    jobs_data = [
        {"required_skills": j.required_skills, "location": j.location, "company": j.company}
        for j in orchestrator.found_jobs
    ]
    candidate_skills = orchestrator.profile.technical_skills if orchestrator.profile else []
    market = analytics.job_market_analysis(jobs_data, candidate_skills)

    matched_scores = [
        {"match_score": j.match_score}
        for j in orchestrator.matched_jobs
        if j.match_score
    ]
    match_stats = analytics.match_score_statistics(matched_scores)

    return {
        "market_analysis": market,
        "match_statistics": match_stats,
        "pipeline_status": {
            "profile_loaded": orchestrator.profile is not None,
            "jobs_found": len(orchestrator.found_jobs),
            "jobs_matched": len(orchestrator.matched_jobs),
            "hr_researched": len(orchestrator.hr_contacts),
            "outreach_drafted": orchestrator.outreach_batch.total if orchestrator.outreach_batch else 0,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=settings.debug)
