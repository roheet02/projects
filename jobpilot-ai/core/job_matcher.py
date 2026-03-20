"""
JobPilot AI - Job Matcher
Multi-signal scoring engine that ranks jobs against a candidate profile.

Scoring signals:
  1. Semantic similarity (30%) — embedding-based meaning match
  2. Skill overlap (40%) — explicit skill matching with TF-IDF weighting
  3. Experience match (15%) — years of experience alignment
  4. Domain relevance (15%) — industry/domain overlap

Final score is a weighted combination of all signals (0–1 scale).
"""

import re
from typing import List, Tuple, Dict, Optional
import numpy as np
from loguru import logger
from string import Template

from models.job import Job
from models.candidate import CandidateProfile
from core.embeddings import embedding_engine
from llm.client import LLMClient
from llm.prompts import MATCH_EXPLANATION_PROMPT
from config.settings import settings


class MatchScore:
    """Detailed breakdown of a job-candidate match score."""

    def __init__(
        self,
        semantic_score: float,
        skill_score: float,
        experience_score: float,
        domain_score: float,
        matched_skills: List[str],
        missing_skills: List[str],
    ):
        self.semantic_score = semantic_score
        self.skill_score = skill_score
        self.experience_score = experience_score
        self.domain_score = domain_score
        self.matched_skills = matched_skills
        self.missing_skills = missing_skills

        # Weighted final score
        self.final_score = (
            0.30 * semantic_score +
            0.40 * skill_score +
            0.15 * experience_score +
            0.15 * domain_score
        )

    def to_dict(self) -> Dict:
        return {
            "final_score": round(self.final_score, 3),
            "semantic": round(self.semantic_score, 3),
            "skill_overlap": round(self.skill_score, 3),
            "experience_match": round(self.experience_score, 3),
            "domain_relevance": round(self.domain_score, 3),
            "matched_skills": self.matched_skills,
            "missing_skills": self.missing_skills,
        }


class JobMatcher:
    """
    Ranks and filters jobs based on multi-signal scoring.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.threshold = settings.job_match_threshold

    def score_job(self, job: Job, profile: CandidateProfile) -> MatchScore:
        """
        Compute multi-signal match score for a job against a candidate profile.
        """

        # 1. Semantic similarity (embedding cosine similarity)
        candidate_text = profile.full_profile_text
        job_text = f"{job.title} {job.description} {' '.join(job.required_skills)}"

        semantic_score = embedding_engine.semantic_match_score(
            candidate_text, job_text
        )
        semantic_score = max(0.0, min(1.0, semantic_score))  # Clamp 0-1

        # 2. Skill overlap score
        skill_score, matched, missing = self._compute_skill_score(job, profile)

        # 3. Experience match
        experience_score = self._compute_experience_score(job, profile)

        # 4. Domain relevance
        domain_score = self._compute_domain_score(job, profile)

        return MatchScore(
            semantic_score=semantic_score,
            skill_score=skill_score,
            experience_score=experience_score,
            domain_score=domain_score,
            matched_skills=matched,
            missing_skills=missing,
        )

    def _compute_skill_score(
        self, job: Job, profile: CandidateProfile
    ) -> Tuple[float, List[str], List[str]]:
        """
        Compute skill overlap score.
        Uses normalized Jaccard similarity with partial string matching.
        """
        candidate_skills = set(s.lower() for s in profile.all_skills)
        required_skills = set(s.lower() for s in job.required_skills)
        preferred_skills = set(s.lower() for s in job.preferred_skills)

        if not required_skills:
            return 0.5, [], []  # No skills listed — neutral score

        # Direct matches
        matched = []
        missing = []

        for skill in job.required_skills:
            skill_lower = skill.lower()
            # Check for partial match (e.g., "PyTorch" matches "pytorch")
            found = any(
                skill_lower in cand or cand in skill_lower
                for cand in candidate_skills
            )
            if found:
                matched.append(skill)
            else:
                missing.append(skill)

        # Score: % of required skills matched
        required_score = len(matched) / len(required_skills) if required_skills else 0.5

        # Bonus for preferred skills
        preferred_matched = sum(
            1 for s in preferred_skills
            if any(s in c or c in s for c in candidate_skills)
        )
        preferred_bonus = 0.1 * (preferred_matched / len(preferred_skills)) if preferred_skills else 0

        final_skill_score = min(1.0, required_score + preferred_bonus)
        return final_skill_score, matched, missing

    def _compute_experience_score(
        self, job: Job, profile: CandidateProfile
    ) -> float:
        """
        Score based on years of experience alignment.
        Returns 1.0 for perfect match, penalizes under/over qualification.
        """
        if not job.experience_years or profile.years_of_experience is None:
            return 0.7  # No data — give benefit of doubt

        # Parse "3-5 years" or "5+" or "2 years"
        exp_str = job.experience_years.lower()
        candidate_exp = profile.years_of_experience

        # Extract numbers
        numbers = re.findall(r'\d+', exp_str)
        if not numbers:
            return 0.7

        if len(numbers) == 1:
            required_min = float(numbers[0])
            required_max = required_min + 2  # Add buffer
        else:
            required_min = float(numbers[0])
            required_max = float(numbers[1])

        # Scoring: perfect if within range
        if required_min <= candidate_exp <= required_max:
            return 1.0
        elif candidate_exp < required_min:
            gap = required_min - candidate_exp
            return max(0.0, 1.0 - (gap * 0.2))  # -20% per year under
        else:
            # Overqualified — slight penalty
            gap = candidate_exp - required_max
            return max(0.6, 1.0 - (gap * 0.05))  # Small penalty for over-qual

    def _compute_domain_score(
        self, job: Job, profile: CandidateProfile
    ) -> float:
        """Score based on industry/domain overlap."""
        if not job.industry or not profile.domains:
            return 0.6  # Neutral

        job_domain = job.industry.lower()
        candidate_domains = [d.lower() for d in profile.domains]

        # Check if any candidate domain appears in job domain description
        for domain in candidate_domains:
            if domain in job_domain or job_domain in domain:
                return 1.0

        # Semantic similarity fallback
        return embedding_engine.semantic_match_score(
            " ".join(profile.domains), job.industry
        )

    def rank_jobs(
        self,
        jobs: List[Job],
        profile: CandidateProfile,
        top_k: Optional[int] = None,
        explain: bool = False,
    ) -> List[Job]:
        """
        Score and rank all jobs against the candidate profile.

        Args:
            jobs: List of Job objects to score
            profile: CandidateProfile of the candidate
            top_k: Return only top K jobs (None = all above threshold)
            explain: Use LLM to generate match explanations (slower)

        Returns:
            Sorted list of jobs with match_score and match_reasons populated
        """
        logger.info(f"Scoring {len(jobs)} jobs for {profile.name}...")

        scored_jobs = []
        for job in jobs:
            try:
                score = self.score_job(job, profile)
                job.match_score = round(score.final_score, 3)

                if explain and score.final_score >= self.threshold:
                    job.match_reasons = self._generate_explanation(job, profile, score)
                else:
                    # Quick explanation from score breakdown
                    job.match_reasons = [
                        f"Skill match: {', '.join(score.matched_skills[:3])}",
                        f"Semantic relevance: {score.semantic_score:.0%}",
                        f"Experience alignment: {score.experience_score:.0%}",
                    ]

                if score.final_score >= self.threshold:
                    scored_jobs.append(job)

            except Exception as e:
                logger.warning(f"Failed to score job {job.title}: {e}")
                continue

        # Sort by score descending
        scored_jobs.sort(key=lambda j: j.match_score or 0, reverse=True)

        if top_k:
            scored_jobs = scored_jobs[:top_k]

        logger.success(
            f"Scoring complete: {len(scored_jobs)}/{len(jobs)} jobs "
            f"above {self.threshold:.0%} threshold"
        )
        return scored_jobs

    def _generate_explanation(
        self,
        job: Job,
        profile: CandidateProfile,
        score: MatchScore,
    ) -> List[str]:
        """Use LLM to generate human-readable match explanations."""
        try:
            prompt = Template(MATCH_EXPLANATION_PROMPT).substitute(
                candidate_name=profile.name,
                candidate_skills=", ".join(profile.technical_skills[:15]),
                candidate_experience=f"{profile.years_of_experience} years as {profile.current_role}",
                candidate_domains=", ".join(profile.domains),
                job_title=job.title,
                company=job.company,
                required_skills=", ".join(job.required_skills[:10]),
                job_description=job.description[:500],
            )
            result = self.llm.parse_json(prompt)
            return result.get("reasons", score.matched_skills[:3])
        except Exception:
            return score.matched_skills[:3]

    def get_match_statistics(
        self,
        jobs: List[Job],
        profile: CandidateProfile,
    ) -> Dict:
        """
        Generate statistical summary of the job market for this candidate.
        Useful for the analytics dashboard.
        """
        all_scores = []
        skill_demand = {}

        for job in jobs:
            score = self.score_job(job, profile)
            all_scores.append(score.final_score)
            for skill in job.required_skills:
                skill_demand[skill] = skill_demand.get(skill, 0) + 1

        if not all_scores:
            return {}

        scores_array = np.array(all_scores)
        top_demanded_skills = sorted(
            skill_demand.items(), key=lambda x: x[1], reverse=True
        )[:20]

        return {
            "total_jobs_analyzed": len(jobs),
            "mean_match_score": float(np.mean(scores_array)),
            "median_match_score": float(np.median(scores_array)),
            "std_match_score": float(np.std(scores_array)),
            "above_threshold": int((scores_array >= self.threshold).sum()),
            "score_distribution": {
                "excellent (>0.8)": int((scores_array > 0.8).sum()),
                "good (0.6-0.8)": int(((scores_array >= 0.6) & (scores_array <= 0.8)).sum()),
                "fair (0.4-0.6)": int(((scores_array >= 0.4) & (scores_array < 0.6)).sum()),
                "poor (<0.4)": int((scores_array < 0.4).sum()),
            },
            "top_demanded_skills": dict(top_demanded_skills),
            "your_matched_skills": [
                s for s in profile.technical_skills
                if s in skill_demand
            ],
            "skill_gaps": [
                s for s, count in top_demanded_skills[:10]
                if s.lower() not in [x.lower() for x in profile.all_skills]
            ],
        }
