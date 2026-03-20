"""
JobPilot AI - Streamlit UI
Main user interface for the JobPilot AI agent system.

Run with: streamlit run ui/streamlit_app.py
"""

import asyncio
import json
from typing import Optional
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ------------------------------------------------------------------ #
# Page Configuration
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="JobPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------ #
# Async Runner Helper
# ------------------------------------------------------------------ #

def run_async(coro):
    """Run an async coroutine from Streamlit (sync context)."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ------------------------------------------------------------------ #
# Session State Initialization
# ------------------------------------------------------------------ #

def init_state():
    defaults = {
        "profile": None,
        "jobs": [],
        "matched_jobs": [],
        "hr_contacts": {},
        "outreach_batch": None,
        "pipeline_step": 0,
        "orchestrator": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #

def render_sidebar():
    with st.sidebar:
        st.image("https://via.placeholder.com/200x60?text=JobPilot+AI", width=200)
        st.markdown("---")

        st.markdown("### 🗺️ Navigation")
        page = st.radio(
            "Go to",
            ["🏠 Home", "📄 Resume", "🔍 Jobs", "🤖 Matching", "👥 HR Research",
             "✉️ Outreach", "📊 Analytics"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        # LLM Model
        st.selectbox(
            "LLM Model",
            ["gpt-4o-mini", "gpt-4o", "claude-3-haiku-20240307",
             "claude-3-5-sonnet-20241022", "groq/llama3-70b-8192"],
            key="llm_model_select",
        )

        # Match threshold
        st.slider(
            "Match Threshold",
            min_value=0.4, max_value=0.9, value=0.65, step=0.05,
            key="match_threshold",
            help="Minimum match score to shortlist a job",
        )

        # Outreach channel
        st.selectbox(
            "Preferred Outreach Channel",
            ["email", "linkedin_connection", "linkedin_message"],
            key="outreach_channel",
        )

        st.markdown("---")
        # Pipeline status
        steps = ["Resume", "Jobs", "Match", "HR", "Outreach"]
        current_step = st.session_state.pipeline_step
        st.markdown("### 📋 Pipeline Status")
        for i, step in enumerate(steps):
            if i < current_step:
                st.markdown(f"✅ {step}")
            elif i == current_step:
                st.markdown(f"🔄 **{step}** (current)")
            else:
                st.markdown(f"⬜ {step}")

    return page


# ------------------------------------------------------------------ #
# Pages
# ------------------------------------------------------------------ #

def page_home():
    st.title("🚀 JobPilot AI")
    st.markdown("""
    **Your autonomous AI job hunting agent.**

    JobPilot AI uses AI agents, semantic ML matching, and browser automation to:
    - 📄 Parse your resume and build your skill profile
    - 🔍 Search LinkedIn, Indeed, Naukri and other portals directly (no API needed)
    - 🤖 Score and rank jobs using semantic similarity + skill matching
    - 👥 Find HR contacts at target companies via LinkedIn
    - ✉️ Generate personalized cold emails and LinkedIn messages
    - 🔍 Let YOU review every message before sending (human-in-the-loop)
    - 📊 Track your application pipeline with analytics
    """)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Jobs Found", len(st.session_state.get("jobs", [])))
    with col2:
        st.metric("Jobs Matched", len(st.session_state.get("matched_jobs", [])))
    with col3:
        st.metric(
            "HR Contacts",
            sum(len(v) for v in st.session_state.get("hr_contacts", {}).values())
        )
    with col4:
        batch = st.session_state.get("outreach_batch")
        st.metric("Outreach Drafted", batch.total if batch else 0)

    st.markdown("---")
    st.info("👈 Start by uploading your resume in the **Resume** tab.")


def page_resume():
    st.title("📄 Resume Upload & Parsing")

    uploaded_file = st.file_uploader(
        "Upload your resume (PDF or DOCX)",
        type=["pdf", "docx"],
        help="Your resume is processed locally — never uploaded to external servers.",
    )

    if uploaded_file:
        # Save temp file
        import tempfile, os
        suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        if st.button("🔍 Parse Resume", type="primary"):
            with st.spinner("Parsing resume with AI..."):
                try:
                    from agents.orchestrator import JobPilotOrchestrator
                    if not st.session_state.orchestrator:
                        st.session_state.orchestrator = JobPilotOrchestrator()

                    profile = run_async(
                        st.session_state.orchestrator.load_profile(tmp_path)
                    )
                    st.session_state.profile = profile
                    st.session_state.pipeline_step = 1
                    st.success(f"✅ Resume parsed! Welcome, **{profile.name}**")
                except Exception as e:
                    st.error(f"Parsing failed: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.profile:
        profile = st.session_state.profile
        st.markdown("---")
        st.subheader("Your Profile")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {profile.name}")
            st.markdown(f"**Headline:** {profile.headline}")
            st.markdown(f"**Experience:** {profile.years_of_experience} years")
            st.markdown(f"**Current Role:** {profile.current_role} at {profile.current_company}")
            st.markdown(f"**Location:** {profile.location}")

        with col2:
            st.markdown("**Technical Skills:**")
            skills_html = " ".join(
                f'<span style="background:#0066cc;color:white;padding:2px 8px;border-radius:12px;margin:2px;display:inline-block">{s}</span>'
                for s in profile.technical_skills[:15]
            )
            st.markdown(skills_html, unsafe_allow_html=True)

        with st.expander("Summary"):
            st.write(profile.summary)

        with st.expander("Work Experience"):
            for exp in profile.work_experience:
                st.markdown(f"**{exp.title}** at {exp.company} ({exp.start_date} - {exp.end_date or 'Present'})")
                st.write(exp.description[:300])
                st.markdown("---")


def page_jobs():
    st.title("🔍 Job Discovery")

    if not st.session_state.profile:
        st.warning("Please parse your resume first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        portals = st.multiselect(
            "Job Portals",
            ["linkedin", "indeed", "naukri", "glassdoor", "wellfound"],
            default=["linkedin", "indeed"],
        )
    with col2:
        locations = st.text_input(
            "Locations (comma-separated)",
            value="Bangalore, Mumbai, Remote",
        )

    roles = st.text_input(
        "Target Roles (comma-separated)",
        value=", ".join(st.session_state.profile.target_roles[:3]),
    )

    if st.button("🔍 Search Jobs", type="primary"):
        with st.spinner(f"Searching {', '.join(portals)}... This may take 1-2 minutes as the agent browses the sites."):
            try:
                from models.job import JobPortal
                portal_list = [JobPortal(p) for p in portals]

                jobs = run_async(
                    st.session_state.orchestrator.discover_jobs(portals=portal_list)
                )
                st.session_state.jobs = jobs
                st.session_state.pipeline_step = 2
                st.success(f"✅ Found **{len(jobs)} jobs**")
            except Exception as e:
                st.error(f"Search failed: {e}")

    if st.session_state.jobs:
        st.markdown("---")
        st.subheader(f"Discovered Jobs ({len(st.session_state.jobs)})")

        jobs_df = pd.DataFrame([{
            "Title": j.title,
            "Company": j.company,
            "Location": j.location,
            "Portal": j.portal.value,
            "Score": f"{j.match_score:.0%}" if j.match_score else "—",
        } for j in st.session_state.jobs])

        st.dataframe(jobs_df, use_container_width=True, hide_index=True)


def page_matching():
    st.title("🤖 ML Job Matching")

    if not st.session_state.jobs:
        st.warning("Please discover jobs first.")
        return

    st.info(
        "Multi-signal matching uses: **Semantic similarity** (30%) + "
        "**Skill overlap** (40%) + **Experience match** (15%) + **Domain relevance** (15%)"
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Top K jobs to return", 5, 50, 20)
    with col2:
        explain = st.checkbox("Generate explanations (slower)", value=True)

    if st.button("🤖 Run Matching", type="primary"):
        with st.spinner("Scoring all jobs with ML..."):
            try:
                matched = run_async(
                    st.session_state.orchestrator.match_and_rank_jobs(
                        top_k=top_k, explain=explain
                    )
                )
                st.session_state.matched_jobs = matched
                st.session_state.pipeline_step = 3
                st.success(f"✅ Matched **{len(matched)} jobs** above threshold")
            except Exception as e:
                st.error(f"Matching failed: {e}")

    if st.session_state.matched_jobs:
        matched = st.session_state.matched_jobs

        # Score distribution chart
        scores = [j.match_score for j in matched if j.match_score]
        if scores:
            fig = px.histogram(
                x=scores, nbins=20,
                labels={"x": "Match Score", "y": "Number of Jobs"},
                title="Match Score Distribution",
                color_discrete_sequence=["#0066cc"],
            )
            fig.add_vline(x=0.65, line_dash="dash", line_color="orange",
                         annotation_text="Threshold")
            st.plotly_chart(fig, use_container_width=True)

        # Matched jobs table with color coding
        st.subheader("Top Matches")
        for job in matched[:15]:
            score = job.match_score or 0
            color = "green" if score > 0.75 else "orange" if score > 0.65 else "red"
            with st.expander(f"**{job.title}** at {job.company} — [{score:.0%}]"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Match Score", f"{score:.0%}")
                    st.write(f"**Location:** {job.location}")
                    st.write(f"**Experience:** {job.experience_years}")
                with col2:
                    if job.required_skills:
                        st.write("**Required Skills:**")
                        st.write(", ".join(job.required_skills[:8]))
                if job.match_reasons:
                    st.write("**Why it matches:**")
                    for reason in job.match_reasons:
                        st.write(f"• {reason}")
                if job.url:
                    st.link_button("View Job", job.url)


def page_outreach():
    st.title("✉️ Outreach Review & Send")

    if not st.session_state.outreach_batch:
        st.warning("Please complete the matching and HR research steps first.")
        return

    batch = st.session_state.outreach_batch
    messages = batch.messages

    st.info(
        f"**{len(messages)} messages drafted** — review each one before sending. "
        "All sends require your explicit approval."
    )

    for i, msg in enumerate(messages):
        status_color = {
            "pending_review": "🟡",
            "approved": "🟢",
            "sent": "✅",
            "rejected": "🔴",
        }.get(msg.status.value, "⬜")

        with st.expander(
            f"{status_color} Message {i+1}: {msg.hr_contact.name} at {msg.hr_contact.company}"
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                if msg.subject:
                    new_subject = st.text_input(
                        "Subject", value=msg.subject, key=f"subj_{msg.id}"
                    )
                    msg.subject = new_subject

                new_body = st.text_area(
                    "Message Body", value=msg.body, height=200, key=f"body_{msg.id}"
                )
                if new_body != msg.body:
                    msg.body = new_body
                    msg.human_edited = True

            with col2:
                st.markdown(f"**To:** {msg.hr_contact.name}")
                st.markdown(f"**Title:** {msg.hr_contact.title}")
                st.markdown(f"**Company:** {msg.hr_contact.company}")
                st.markdown(f"**Channel:** {msg.channel.value}")
                st.markdown(f"**Status:** {msg.status.value}")

                col_approve, col_reject = st.columns(2)
                with col_approve:
                    if st.button("✅ Approve", key=f"approve_{msg.id}"):
                        from models.outreach import OutreachStatus
                        msg.status = OutreachStatus.APPROVED
                        st.success("Approved!")
                        st.rerun()
                with col_reject:
                    if st.button("❌ Skip", key=f"reject_{msg.id}"):
                        from models.outreach import OutreachStatus
                        msg.status = OutreachStatus.REJECTED
                        st.rerun()

    # Send approved messages
    approved = [m for m in messages if m.status.value == "approved"]
    if approved:
        st.markdown("---")
        st.success(f"**{len(approved)} messages approved** and ready to send!")
        if st.button(f"🚀 Send All Approved Messages ({len(approved)})", type="primary"):
            with st.spinner("Sending via browser automation..."):
                try:
                    results = run_async(
                        st.session_state.orchestrator.send_approved_messages()
                    )
                    st.success(
                        f"✅ Sent {results['sent']} messages! "
                        f"{results['failed']} failed."
                    )
                except Exception as e:
                    st.error(f"Sending failed: {e}")


def page_analytics():
    st.title("📊 Analytics Dashboard")

    from analytics.dashboard import SearchAnalytics
    analytics = SearchAnalytics()

    jobs = st.session_state.jobs
    matched = st.session_state.matched_jobs
    profile = st.session_state.profile

    if not jobs:
        st.warning("Complete the job search to see analytics.")
        return

    # Market analysis
    market = analytics.job_market_analysis(
        [{"required_skills": j.required_skills, "location": j.location,
          "job_type": str(j.job_type), "company": j.company}
         for j in jobs],
        profile.technical_skills if profile else [],
    )

    st.subheader("📈 Job Market Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Jobs Found", len(jobs))
    col2.metric("Jobs Matched", len(matched))
    col3.metric("Match Rate", f"{len(matched)/max(len(jobs),1):.0%}")

    if market.get("top_demanded_skills"):
        st.subheader("Top Demanded Skills in Market")
        skills_df = pd.DataFrame(
            list(market["top_demanded_skills"].items()),
            columns=["Skill", "Count"]
        ).head(15)
        fig = px.bar(
            skills_df, x="Count", y="Skill",
            orientation="h",
            color="Count",
            color_continuous_scale="Blues",
            title="Skills in Highest Demand",
        )
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    if market.get("skill_gaps") and profile:
        st.subheader("🎯 Your Skill Gaps vs Market Demand")
        gaps_df = pd.DataFrame(
            list(market["skill_gaps"].items()),
            columns=["Skill", "Job Postings Requiring It"]
        ).head(10)
        fig2 = px.bar(
            gaps_df, x="Skill", y="Job Postings Requiring It",
            color="Job Postings Requiring It",
            color_continuous_scale="Oranges",
            title="Skills You're Missing vs Market Demand",
        )
        st.plotly_chart(fig2, use_container_width=True)

    if matched:
        st.subheader("Match Score Distribution")
        scores = [j.match_score for j in matched if j.match_score]
        match_stats = analytics.match_score_statistics(
            [{"match_score": s} for s in scores]
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Score", f"{match_stats.get('mean', 0):.0%}")
        col2.metric("Median Score", f"{match_stats.get('median', 0):.0%}")
        col3.metric(">75% Match", match_stats.get("above_75_pct", 0))
        col4.metric(">85% Match", match_stats.get("above_85_pct", 0))

        fig3 = px.histogram(
            x=scores, nbins=15,
            labels={"x": "Match Score"},
            title="Distribution of Match Scores",
            color_discrete_sequence=["#0066cc"],
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Recommendations
    st.subheader("💡 Recommendations")
    recs = analytics.generate_recommendations(market, {}, {})
    for rec in recs:
        st.info(f"💡 {rec}")


# ------------------------------------------------------------------ #
# Main App
# ------------------------------------------------------------------ #

def main():
    init_state()
    page = render_sidebar()

    page_map = {
        "🏠 Home": page_home,
        "📄 Resume": page_resume,
        "🔍 Jobs": page_jobs,
        "🤖 Matching": page_matching,
        "👥 HR Research": lambda: st.title("Coming in next update"),
        "✉️ Outreach": page_outreach,
        "📊 Analytics": page_analytics,
    }

    page_fn = page_map.get(page, page_home)
    page_fn()


if __name__ == "__main__":
    main()
