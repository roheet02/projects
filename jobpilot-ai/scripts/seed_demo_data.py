"""
JobPilot AI — Demo Data Seeder
Populates the local SQLite database with realistic sample data so you can
explore the Streamlit UI and analytics dashboard without running the full pipeline.

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --clear   # clear existing data first
"""

import asyncio
import sys
import os
import argparse
import uuid
from datetime import datetime, timedelta
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.progress import track

from database.repository import JobRepository
from models.job import Job, JobPortal, JobType, MatchStatus
from models.candidate import CandidateProfile, WorkExperience, Education, Project
from models.outreach import HRContact, OutreachMessage, OutreachChannel, OutreachStatus

console = Console()


# ── Sample Data ───────────────────────────────────────────────────────────────

SAMPLE_JOBS = [
    {
        "title": "Senior Data Scientist",
        "company": "Meesho",
        "location": "Bangalore, India",
        "portal": JobPortal.LINKEDIN,
        "job_type": JobType.HYBRID,
        "description": (
            "Looking for a Senior Data Scientist to build ML models for our "
            "recommendation engine. You'll work with Python, PyTorch, and large-scale "
            "data pipelines on AWS."
        ),
        "required_skills": ["Python", "PyTorch", "SQL", "Spark", "ML"],
        "preferred_skills": ["BERT", "NLP", "Airflow"],
        "experience_years": "3-6 years",
        "salary_range": "₹30-45 LPA",
        "industry": "E-commerce",
        "match_score": 0.89,
        "url": "https://linkedin.com/jobs/view/12345",
    },
    {
        "title": "ML Engineer",
        "company": "Razorpay",
        "location": "Bangalore, India",
        "portal": JobPortal.INDEED,
        "job_type": JobType.FULL_TIME,
        "description": (
            "Join our ML platform team building scalable ML infrastructure. "
            "Experience with MLOps, Docker, and model serving required."
        ),
        "required_skills": ["Python", "Docker", "Kubernetes", "scikit-learn", "MLOps"],
        "preferred_skills": ["Kubeflow", "MLflow", "PyTorch"],
        "experience_years": "2-5 years",
        "salary_range": "₹25-40 LPA",
        "industry": "FinTech",
        "match_score": 0.82,
        "url": "https://indeed.com/jobs/67890",
    },
    {
        "title": "Applied Scientist",
        "company": "Amazon",
        "location": "Hyderabad, India",
        "portal": JobPortal.LINKEDIN,
        "job_type": JobType.FULL_TIME,
        "description": (
            "Research and deploy NLP solutions for Amazon's customer experience team. "
            "Strong background in deep learning and NLP required."
        ),
        "required_skills": ["Python", "PyTorch", "NLP", "Deep Learning", "AWS"],
        "preferred_skills": ["Transformers", "RLHF", "Sagemaker"],
        "experience_years": "3-7 years",
        "salary_range": "₹40-70 LPA",
        "industry": "Technology",
        "match_score": 0.79,
        "url": "https://linkedin.com/jobs/view/22222",
    },
    {
        "title": "Data Scientist - NLP",
        "company": "Flipkart",
        "location": "Bangalore, India",
        "portal": JobPortal.NAUKRI,
        "job_type": JobType.HYBRID,
        "description": (
            "Work on NLP models for product search and cataloguing. "
            "Hands-on experience with BERT, transformers, and large datasets."
        ),
        "required_skills": ["Python", "NLP", "BERT", "TensorFlow", "SQL"],
        "preferred_skills": ["Spark", "Kafka", "GCP"],
        "experience_years": "2-5 years",
        "salary_range": "₹20-35 LPA",
        "industry": "E-commerce",
        "match_score": 0.76,
        "url": "https://naukri.com/jobs/33333",
    },
    {
        "title": "Data Scientist",
        "company": "CRED",
        "location": "Bangalore, India",
        "portal": JobPortal.LINKEDIN,
        "job_type": JobType.FULL_TIME,
        "description": (
            "Build credit risk models and fraud detection systems for our payments platform."
        ),
        "required_skills": ["Python", "SQL", "scikit-learn", "XGBoost", "Statistics"],
        "preferred_skills": ["Spark", "Airflow", "dbt"],
        "experience_years": "2-4 years",
        "salary_range": "₹18-30 LPA",
        "industry": "FinTech",
        "match_score": 0.72,
        "url": "https://linkedin.com/jobs/view/44444",
    },
    {
        "title": "ML Research Engineer",
        "company": "Google",
        "location": "Hyderabad, India",
        "portal": JobPortal.LINKEDIN,
        "job_type": JobType.FULL_TIME,
        "description": (
            "Research and productionise LLM-based features for Google products."
        ),
        "required_skills": ["Python", "TensorFlow", "JAX", "Deep Learning", "Research"],
        "preferred_skills": ["LLMs", "RLHF", "TPU"],
        "experience_years": "4-8 years",
        "salary_range": "₹50-90 LPA",
        "industry": "Technology",
        "match_score": 0.67,
        "url": "https://linkedin.com/jobs/view/55555",
    },
    {
        "title": "Business Analyst",
        "company": "McKinsey",
        "location": "Mumbai, India",
        "portal": JobPortal.INDEED,
        "job_type": JobType.FULL_TIME,
        "description": "Analyse business data and build dashboards for clients.",
        "required_skills": ["Excel", "PowerPoint", "SQL", "Tableau"],
        "preferred_skills": ["Python", "Power BI"],
        "experience_years": "1-3 years",
        "salary_range": "₹12-20 LPA",
        "industry": "Consulting",
        "match_score": 0.41,  # low match for a DS profile
        "url": "https://indeed.com/jobs/99999",
    },
]

SAMPLE_HR_CONTACTS = [
    {"name": "Priya Sharma",    "title": "Technical Recruiter",      "company": "Meesho",   "email": "priya.sharma@meesho.com",    "linkedin_url": "https://linkedin.com/in/priyasharma"},
    {"name": "Rahul Verma",     "title": "HR Manager",               "company": "Razorpay", "email": "rahul.verma@razorpay.com",   "linkedin_url": "https://linkedin.com/in/rahulverma"},
    {"name": "Anjali Singh",    "title": "Talent Acquisition Lead",  "company": "Amazon",   "email": "anjali.s@amazon.com",        "linkedin_url": "https://linkedin.com/in/anjalisingh"},
    {"name": "Deepak Nair",     "title": "Technical Recruiter",      "company": "Flipkart", "email": "deepak.nair@flipkart.com",   "linkedin_url": "https://linkedin.com/in/deepaknair"},
    {"name": "Sneha Patel",     "title": "HR Business Partner",      "company": "CRED",     "email": "sneha.patel@cred.club",      "linkedin_url": "https://linkedin.com/in/snehapatel"},
]

SAMPLE_OUTREACH = [
    {
        "subject": "Senior Data Scientist at Meesho — NLP & ML Background",
        "body": (
            "Hi Priya,\n\n"
            "I came across the Senior Data Scientist role at Meesho and was excited by "
            "the focus on recommendation systems. I've built similar PyTorch-based pipelines "
            "at my current company, improving model latency by 40%.\n\n"
            "I'm a Data Scientist with 4 years of experience in ML and NLP (Python, PyTorch, "
            "BERT, Spark). I'd love to learn more about the team's challenges.\n\n"
            "Would you be open to a 15-minute call this week?\n\nBest,\nRohit"
        ),
        "channel": OutreachChannel.EMAIL,
        "status": OutreachStatus.SENT,
        "days_ago": 5,
    },
    {
        "subject": "ML Engineer Role at Razorpay — MLOps Experience",
        "body": (
            "Hi Rahul,\n\nI saw the ML Engineer opening at Razorpay and it's a great fit — "
            "I've worked on ML platform tooling with Docker and Kubernetes. "
            "Happy to share more details if helpful.\n\nBest,\nRohit"
        ),
        "channel": OutreachChannel.EMAIL,
        "status": OutreachStatus.REPLIED,
        "days_ago": 8,
    },
    {
        "subject": None,
        "body": (
            "Hi Anjali, I'm a Data Scientist with strong NLP and deep learning experience "
            "— very interested in the Applied Scientist role at Amazon. Would love to connect!"
        ),
        "channel": OutreachChannel.LINKEDIN_CONNECTION,
        "status": OutreachStatus.SENT,
        "days_ago": 3,
    },
    {
        "subject": "Data Scientist NLP at Flipkart — BERT & Transformers",
        "body": (
            "Hi Deepak,\n\nI'd love to apply for the NLP Data Scientist role at Flipkart. "
            "My experience with BERT-based models for text classification aligns well with "
            "your product search use case.\n\nCould we connect?\n\nBest,\nRohit"
        ),
        "channel": OutreachChannel.EMAIL,
        "status": OutreachStatus.PENDING_REVIEW,
        "days_ago": 0,
    },
    {
        "subject": "Data Scientist at CRED — FinTech ML",
        "body": (
            "Hi Sneha, interested in the DS role at CRED — I have experience with "
            "credit risk modelling and fraud detection using Python and XGBoost. Happy to share my CV!"
        ),
        "channel": OutreachChannel.LINKEDIN_MESSAGE,
        "status": OutreachStatus.PENDING_REVIEW,
        "days_ago": 0,
    },
]


# ── Seeder ────────────────────────────────────────────────────────────────────

async def seed(clear: bool = False):
    repo = JobRepository()
    await repo.init_db()

    # Seed candidate profile
    console.print("\n[bold cyan]Seeding demo data...[/bold cyan]\n")

    profile = CandidateProfile(
        name="Rohit Kumar",
        email="rohit.kumar@example.com",
        phone="+91-9876543210",
        location="Bangalore, India",
        linkedin_url="https://linkedin.com/in/rohitkumar",
        github_url="https://github.com/rohitkumar",
        headline="Senior Data Scientist | ML | NLP | LLMs | Python",
        summary=(
            "Data Scientist with 4+ years of experience building production ML systems, "
            "NLP pipelines, and recommendation engines. Strong in Python, PyTorch, "
            "and cloud (AWS). Proven track record of reducing model latency by 40% "
            "and improving recommendation CTR by 18%."
        ),
        years_of_experience=4.5,
        current_role="Data Scientist",
        current_company="Acme Analytics",
        technical_skills=["Python", "PyTorch", "TensorFlow", "scikit-learn",
                          "SQL", "Spark", "BERT", "Transformers", "NLP"],
        tools=["Docker", "AWS", "Git", "Airflow", "Jupyter", "VS Code"],
        domains=["NLP", "Machine Learning", "Recommendation Systems", "FinTech"],
        work_experience=[
            WorkExperience(
                company="Acme Analytics",
                title="Data Scientist",
                start_date="2022-01",
                description=(
                    "Built NLP pipelines using BERT and PyTorch for customer intent classification. "
                    "Reduced model inference latency by 40%. Led a team of 2 junior data scientists."
                ),
                skills_used=["Python", "PyTorch", "BERT", "AWS", "Docker"],
                is_current=True,
            ),
            WorkExperience(
                company="DataFirst Startup",
                title="Junior Data Scientist",
                start_date="2020-06",
                end_date="2021-12",
                description=(
                    "Built recommendation systems using collaborative filtering and content-based methods. "
                    "Improved CTR by 18% on e-commerce platform."
                ),
                skills_used=["Python", "scikit-learn", "SQL", "Pandas"],
                is_current=False,
            ),
        ],
        education=[
            Education(
                institution="IIT Bombay",
                degree="B.Tech",
                field="Computer Science and Engineering",
                graduation_year=2020,
            )
        ],
        projects=[
            Project(
                name="LLM-Powered Job Hunting Agent",
                description="Multi-agent system for autonomous job search and personalized outreach.",
                technologies=["Python", "FastAPI", "LiteLLM", "Playwright", "sentence-transformers"],
                impact="Automates 90% of job hunting workflow",
            )
        ],
        target_roles=["Data Scientist", "ML Engineer", "Applied Scientist"],
        target_locations=["Bangalore", "Hyderabad", "Remote"],
        notice_period="30 days",
        remote_preference="hybrid",
    )

    await repo.save_profile(profile)
    console.print("  [green]✓[/green] Candidate profile seeded")

    # Seed jobs
    seeded_jobs = []
    for job_data in track(SAMPLE_JOBS, description="  Seeding jobs..."):
        job = Job(
            id=str(uuid.uuid4()),
            title=job_data["title"],
            company=job_data["company"],
            location=job_data["location"],
            portal=job_data["portal"],
            job_type=job_data.get("job_type"),
            description=job_data["description"],
            required_skills=job_data["required_skills"],
            preferred_skills=job_data.get("preferred_skills", []),
            experience_years=job_data.get("experience_years"),
            salary_range=job_data.get("salary_range"),
            industry=job_data.get("industry"),
            url=job_data.get("url"),
            match_score=job_data.get("match_score"),
            status=MatchStatus.MATCHED if job_data.get("match_score", 0) >= 0.65 else MatchStatus.NEW,
            discovered_at=datetime.utcnow() - timedelta(days=random.randint(0, 3)),
        )
        await repo.save_job(job)
        seeded_jobs.append(job)

    console.print(f"  [green]✓[/green] {len(seeded_jobs)} jobs seeded")

    # Seed HR contacts + outreach messages
    contact_objects = []
    for c in SAMPLE_HR_CONTACTS:
        contact_objects.append(HRContact(**c))

    for i, (outreach_data, contact) in enumerate(zip(SAMPLE_OUTREACH, contact_objects)):
        job = seeded_jobs[i]
        days_ago = outreach_data["days_ago"]
        drafted_at = datetime.utcnow() - timedelta(days=days_ago + 1)
        sent_at = datetime.utcnow() - timedelta(days=days_ago) if days_ago > 0 else None

        msg = OutreachMessage(
            id=str(uuid.uuid4()),
            job_id=job.id,
            candidate_name=profile.name,
            hr_contact=contact,
            channel=outreach_data["channel"],
            subject=outreach_data.get("subject"),
            body=outreach_data["body"],
            status=outreach_data["status"],
            drafted_at=drafted_at,
            sent_at=sent_at,
            replied_at=(
                datetime.utcnow() - timedelta(days=1)
                if outreach_data["status"] == OutreachStatus.REPLIED else None
            ),
        )
        await repo.save_outreach(msg)

    console.print(f"  [green]✓[/green] {len(SAMPLE_OUTREACH)} outreach messages seeded")

    # Log analytics events
    events = [
        ("job_found",       seeded_jobs[0].id,  "Meesho"),
        ("job_found",       seeded_jobs[1].id,  "Razorpay"),
        ("outreach_sent",   seeded_jobs[0].id,  "Meesho"),
        ("outreach_sent",   seeded_jobs[1].id,  "Razorpay"),
        ("outreach_sent",   seeded_jobs[2].id,  "Amazon"),
        ("reply_received",  seeded_jobs[1].id,  "Razorpay"),
    ]
    for event_type, job_id, company in events:
        await repo.log_event(event_type, job_id, company)

    console.print(f"  [green]✓[/green] {len(events)} analytics events seeded")

    console.print("\n[bold green]✅ Demo data seeded successfully![/bold green]")
    console.print("\nNow launch the UI to explore:")
    console.print("[cyan]  streamlit run ui/streamlit_app.py[/cyan]\n")


async def clear_data():
    """Drop and recreate all tables."""
    from sqlalchemy import text
    repo = JobRepository()
    async with repo.engine.begin() as conn:
        from database.repository import Base
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    console.print("[yellow]All existing data cleared.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Seed JobPilot AI with demo data")
    parser.add_argument("--clear", action="store_true", help="Clear existing data before seeding")
    args = parser.parse_args()

    async def run():
        if args.clear:
            await clear_data()
        await seed()

    asyncio.run(run())


if __name__ == "__main__":
    main()
