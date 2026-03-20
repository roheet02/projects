"""
JobPilot AI - Analytics Dashboard
Statistical analysis of job search progress and outreach effectiveness.
Generates charts and metrics for the Streamlit UI.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger


class SearchAnalytics:
    """
    Computes statistics and generates visualizations for:
    - Job market analysis (skill demand, salary distribution)
    - Match score distribution
    - Outreach performance (send rate, reply rate, conversion)
    - Skill gap analysis
    """

    def job_market_analysis(
        self,
        jobs: List[Dict],
        candidate_skills: List[str],
    ) -> Dict:
        """
        Analyze job market data to surface insights.

        Returns:
            Dict with charts-ready data for Streamlit
        """
        if not jobs:
            return {}

        df = pd.DataFrame(jobs)

        # 1. Skill demand analysis
        all_required_skills = []
        for skills_list in df.get("required_skills", pd.Series([])):
            if isinstance(skills_list, list):
                all_required_skills.extend(skills_list)

        skill_counts = pd.Series(all_required_skills).value_counts()
        top_skills = skill_counts.head(20).to_dict()

        # 2. Skill gap analysis (what market wants vs what you have)
        candidate_skill_set = set(s.lower() for s in candidate_skills)
        skill_gap = {
            skill: count
            for skill, count in top_skills.items()
            if skill.lower() not in candidate_skill_set
        }

        # 3. Location distribution
        location_dist = {}
        if "location" in df.columns:
            location_dist = df["location"].value_counts().head(10).to_dict()

        # 4. Job type distribution
        job_type_dist = {}
        if "job_type" in df.columns:
            job_type_dist = df["job_type"].value_counts().to_dict()

        # 5. Company distribution (which companies are hiring most)
        company_dist = {}
        if "company" in df.columns:
            company_dist = df["company"].value_counts().head(15).to_dict()

        # 6. Experience demand analysis
        experience_dist = {}
        if "experience_years" in df.columns:
            experience_dist = df["experience_years"].value_counts().to_dict()

        return {
            "total_jobs": len(df),
            "top_demanded_skills": top_skills,
            "skill_gaps": skill_gap,
            "location_distribution": location_dist,
            "job_type_distribution": job_type_dist,
            "top_hiring_companies": company_dist,
            "experience_demand": experience_dist,
        }

    def match_score_statistics(
        self,
        scored_jobs: List[Dict],
    ) -> Dict:
        """
        Statistical analysis of job match scores.
        Returns distribution stats and visualization data.
        """
        if not scored_jobs:
            return {}

        scores = [j.get("match_score", 0) for j in scored_jobs if j.get("match_score")]
        if not scores:
            return {}

        scores_arr = np.array(scores)

        # Histogram data
        bins = [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        hist, edges = np.histogram(scores_arr, bins=bins)
        histogram = {
            f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(hist[i])
            for i in range(len(hist))
        }

        return {
            "total_scored": len(scores),
            "mean": float(np.mean(scores_arr)),
            "median": float(np.median(scores_arr)),
            "std": float(np.std(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
            "percentile_25": float(np.percentile(scores_arr, 25)),
            "percentile_75": float(np.percentile(scores_arr, 75)),
            "above_65_pct": int((scores_arr >= 0.65).sum()),
            "above_75_pct": int((scores_arr >= 0.75).sum()),
            "above_85_pct": int((scores_arr >= 0.85).sum()),
            "histogram": histogram,
            "scores_list": scores,  # For plotly histogram
        }

    def outreach_funnel_analysis(
        self,
        outreach_records: List[Dict],
    ) -> Dict:
        """
        Analyze the outreach funnel:
        Drafted → Approved → Sent → Opened → Replied → Interview
        """
        if not outreach_records:
            return {}

        df = pd.DataFrame(outreach_records)
        statuses = df["status"].value_counts().to_dict() if "status" in df.columns else {}

        total = len(df)
        sent = statuses.get("sent", 0)
        replied = statuses.get("replied", 0)
        interviews = statuses.get("interview_scheduled", 0)

        funnel = {
            "drafted": total,
            "approved": statuses.get("approved", 0) + sent + replied + interviews,
            "sent": sent + replied + interviews,
            "replied": replied + interviews,
            "interviews": interviews,
        }

        return {
            "funnel": funnel,
            "conversion_rates": {
                "approval_rate": funnel["approved"] / total if total > 0 else 0,
                "send_rate": funnel["sent"] / max(funnel["approved"], 1),
                "reply_rate": funnel["replied"] / max(funnel["sent"], 1),
                "interview_rate": funnel["interviews"] / max(funnel["replied"], 1),
            },
            "by_channel": df.groupby("channel")["status"].value_counts().to_dict() if "channel" in df.columns else {},
            "by_company": df.groupby("hr_company")["status"].value_counts().to_dict() if "hr_company" in df.columns else {},
        }

    def timeline_analysis(
        self,
        events: List[Dict],
        days: int = 30,
    ) -> Dict:
        """Activity timeline showing job hunting progress over time."""
        if not events:
            return {}

        df = pd.DataFrame(events)
        if "timestamp" not in df.columns:
            return {}

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date

        # Daily activity counts
        daily = df.groupby(["date", "event_type"]).size().unstack(fill_value=0)
        return {
            "daily_activity": daily.to_dict(),
            "total_events": len(df),
            "event_breakdown": df["event_type"].value_counts().to_dict(),
        }

    def generate_recommendations(
        self,
        market_analysis: Dict,
        match_stats: Dict,
        outreach_stats: Dict,
    ) -> List[str]:
        """
        Generate actionable recommendations based on analytics data.
        """
        recs = []

        # Match score recommendations
        mean_score = match_stats.get("mean", 0)
        if mean_score < 0.6:
            skill_gaps = list(market_analysis.get("skill_gaps", {}).keys())[:3]
            if skill_gaps:
                recs.append(
                    f"Your match scores average {mean_score:.0%}. "
                    f"Consider adding these in-demand skills: {', '.join(skill_gaps)}"
                )

        # Outreach recommendations
        reply_rate = outreach_stats.get("conversion_rates", {}).get("reply_rate", 0)
        if reply_rate < 0.1:
            recs.append(
                "Reply rate below 10%. Try personalizing emails more — "
                "mention a specific project or recent news about the company."
            )
        elif reply_rate > 0.2:
            recs.append(
                f"Great reply rate of {reply_rate:.0%}! "
                "Your outreach style is working — keep using this approach."
            )

        # Channel recommendations
        channel_stats = outreach_stats.get("by_channel", {})
        if channel_stats:
            recs.append(
                "LinkedIn connection requests typically have higher response "
                "rates than cold emails for technical roles. Consider using both."
            )

        # Skill gap recommendations
        top_gaps = list(market_analysis.get("skill_gaps", {}).keys())[:3]
        if top_gaps:
            recs.append(
                f"Skill gaps vs. market demand: {', '.join(top_gaps)}. "
                "A quick online certification could significantly boost match scores."
            )

        return recs or ["Keep applying! Consistency is key to job search success."]
