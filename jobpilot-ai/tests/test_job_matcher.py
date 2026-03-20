"""
Tests for core/job_matcher.py

Covers:
  - Skill overlap scoring (exact, partial, empty)
  - Experience matching (under/over/exact)
  - Domain scoring
  - Full rank_jobs pipeline
  - Edge cases (no skills, missing data)

Run:  pytest tests/test_job_matcher.py -v
"""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from models.job import Job, JobPortal
from models.candidate import CandidateProfile, WorkExperience
from core.job_matcher import JobMatcher, MatchScore


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_profile():
    return CandidateProfile(
        name="Rohit Kumar",
        headline="Senior Data Scientist | ML | NLP | LLMs",
        years_of_experience=4.5,
        current_role="Data Scientist",
        current_company="Acme Corp",
        technical_skills=["Python", "PyTorch", "scikit-learn", "SQL", "Spark", "BERT"],
        soft_skills=["Communication", "Leadership"],
        tools=["Docker", "AWS", "Git"],
        domains=["NLP", "Machine Learning", "FinTech"],
        summary=(
            "Data scientist with 4.5 years of experience building ML pipelines, "
            "NLP models, and recommendation systems."
        ),
        work_experience=[
            WorkExperience(
                company="Acme Corp",
                title="Data Scientist",
                start_date="2021-06",
                description="Built NLP models using BERT and PyTorch. Reduced latency by 40%.",
                skills_used=["Python", "PyTorch", "BERT"],
                is_current=True,
            )
        ],
    )


@pytest.fixture
def matching_job():
    """A job that closely matches the sample profile."""
    return Job(
        id="job-001",
        title="Senior Data Scientist",
        company="TechCorp",
        location="Bangalore",
        portal=JobPortal.LINKEDIN,
        description=(
            "We are looking for an experienced data scientist to build NLP pipelines "
            "and recommendation systems using Python and PyTorch."
        ),
        required_skills=["Python", "PyTorch", "scikit-learn", "SQL"],
        preferred_skills=["Spark", "BERT"],
        experience_years="3-5 years",
        industry="FinTech",
    )


@pytest.fixture
def mismatched_job():
    """A job that doesn't match the profile well."""
    return Job(
        id="job-002",
        title="iOS Developer",
        company="MobileInc",
        location="Mumbai",
        portal=JobPortal.INDEED,
        description="Build iOS apps using Swift and Objective-C.",
        required_skills=["Swift", "Objective-C", "Xcode", "UIKit"],
        preferred_skills=["CoreML"],
        experience_years="2-4 years",
        industry="Mobile Development",
    )


@pytest.fixture
def matcher():
    return JobMatcher()


# ── Skill Overlap Tests ────────────────────────────────────────────────────────

class TestSkillOverlap:

    def test_exact_skill_match(self, matcher, sample_profile, matching_job):
        """All required skills match → score close to 1.0"""
        score, matched, missing = matcher._compute_skill_score(matching_job, sample_profile)
        assert score >= 0.9
        assert "Python" in matched
        assert "PyTorch" in matched

    def test_partial_string_match(self, matcher, sample_profile):
        """'pytorch' (lowercase) should match 'PyTorch' in candidate skills."""
        job = Job(
            id="j", title="ML Engineer", company="X", location="Y",
            required_skills=["pytorch", "sklearn"],
        )
        score, matched, missing = matcher._compute_skill_score(job, sample_profile)
        assert len(matched) >= 1

    def test_no_skills_listed(self, matcher, sample_profile):
        """Job with no required skills → neutral score 0.5"""
        job = Job(
            id="j", title="Data Analyst", company="X", location="Y",
            required_skills=[],
        )
        score, matched, missing = matcher._compute_skill_score(job, sample_profile)
        assert score == 0.5
        assert matched == []

    def test_zero_overlap(self, matcher, sample_profile, mismatched_job):
        """No skill overlap → low score"""
        score, matched, missing = matcher._compute_skill_score(mismatched_job, sample_profile)
        assert score < 0.2
        assert len(missing) > 0

    def test_preferred_skills_bonus(self, matcher, sample_profile):
        """Preferred skills add a small bonus on top of required score."""
        job_with_preferred = Job(
            id="j", title="ML Engineer", company="X", location="Y",
            required_skills=["Python"],
            preferred_skills=["PyTorch", "Spark"],  # both in candidate profile
        )
        job_without_preferred = Job(
            id="j2", title="ML Engineer", company="X", location="Y",
            required_skills=["Python"],
            preferred_skills=[],
        )
        score_with, _, _ = matcher._compute_skill_score(job_with_preferred, sample_profile)
        score_without, _, _ = matcher._compute_skill_score(job_without_preferred, sample_profile)
        assert score_with > score_without

    def test_score_capped_at_one(self, matcher, sample_profile):
        """Score must never exceed 1.0"""
        job = Job(
            id="j", title="X", company="Y", location="Z",
            required_skills=["Python"],
            preferred_skills=["PyTorch", "Spark", "SQL", "Docker", "AWS"],
        )
        score, _, _ = matcher._compute_skill_score(job, sample_profile)
        assert score <= 1.0


# ── Experience Matching Tests ──────────────────────────────────────────────────

class TestExperienceMatch:

    def test_within_range(self, matcher, sample_profile):
        """4.5 years matches '3-5 years' perfectly → 1.0"""
        job = Job(id="j", title="X", company="Y", location="Z", experience_years="3-5 years")
        score = matcher._compute_experience_score(job, sample_profile)
        assert score == 1.0

    def test_underqualified(self, matcher):
        """Candidate with 1 year vs '3-5 years' required → penalised"""
        profile = CandidateProfile(
            name="Junior Dev", years_of_experience=1.0,
            technical_skills=[], domains=[],
        )
        job = Job(id="j", title="X", company="Y", location="Z", experience_years="3-5 years")
        score = matcher._compute_experience_score(job, profile)
        assert score < 0.7

    def test_overqualified(self, matcher, sample_profile):
        """10 years vs '2-3 years' → small penalty but not disqualifying"""
        profile = CandidateProfile(
            name="Senior Dev", years_of_experience=10.0,
            technical_skills=[], domains=[],
        )
        job = Job(id="j", title="X", company="Y", location="Z", experience_years="2-3 years")
        score = matcher._compute_experience_score(job, profile)
        assert 0.5 <= score <= 1.0

    def test_missing_experience_in_job(self, matcher, sample_profile):
        """No experience requirement → neutral score 0.7"""
        job = Job(id="j", title="X", company="Y", location="Z", experience_years=None)
        score = matcher._compute_experience_score(job, sample_profile)
        assert score == 0.7

    def test_missing_experience_in_profile(self, matcher):
        """Candidate with unknown experience → neutral score 0.7"""
        profile = CandidateProfile(
            name="Unknown", years_of_experience=None,
            technical_skills=[], domains=[],
        )
        job = Job(id="j", title="X", company="Y", location="Z", experience_years="2-4 years")
        score = matcher._compute_experience_score(job, profile)
        assert score == 0.7

    def test_single_number_experience(self, matcher, sample_profile):
        """'5 years' (single number) should parse and score correctly."""
        job = Job(id="j", title="X", company="Y", location="Z", experience_years="5 years")
        score = matcher._compute_experience_score(job, sample_profile)
        assert score >= 0.7


# ── Domain Relevance Tests ─────────────────────────────────────────────────────

class TestDomainRelevance:

    def test_exact_domain_match(self, matcher, sample_profile):
        """'FinTech' in both profile and job → score 1.0"""
        job = Job(id="j", title="X", company="Y", location="Z", industry="FinTech")
        score = matcher._compute_domain_score(job, sample_profile)
        assert score == 1.0

    def test_partial_domain_match(self, matcher, sample_profile):
        """'NLP Engineer' contains 'NLP' which is in candidate domains."""
        job = Job(id="j", title="X", company="Y", location="Z", industry="NLP Platform")
        score = matcher._compute_domain_score(job, sample_profile)
        assert score == 1.0

    def test_no_industry_in_job(self, matcher, sample_profile):
        """Job has no industry → neutral 0.6"""
        job = Job(id="j", title="X", company="Y", location="Z", industry=None)
        score = matcher._compute_domain_score(job, sample_profile)
        assert score == 0.6

    def test_no_domains_in_profile(self, matcher):
        """Profile has no domains → neutral 0.6"""
        profile = CandidateProfile(
            name="X", technical_skills=[], domains=[],
        )
        job = Job(id="j", title="X", company="Y", location="Z", industry="FinTech")
        score = matcher._compute_domain_score(job, profile)
        assert score == 0.6


# ── Full Scoring Tests ─────────────────────────────────────────────────────────

class TestFullScoring:

    @patch("core.job_matcher.embedding_engine")
    def test_score_job_returns_match_score(self, mock_emb, matcher, sample_profile, matching_job):
        """score_job should return a MatchScore with a valid final_score."""
        # Mock the embedding engine to return a fixed semantic score
        mock_emb.semantic_match_score.return_value = 0.82
        mock_emb.encode.return_value = np.array([[0.1] * 384])

        score = matcher.score_job(matching_job, sample_profile)

        assert isinstance(score, MatchScore)
        assert 0.0 <= score.final_score <= 1.0
        assert len(score.matched_skills) > 0

    @patch("core.job_matcher.embedding_engine")
    def test_matching_job_scores_higher_than_mismatched(
        self, mock_emb, matcher, sample_profile, matching_job, mismatched_job
    ):
        """A relevant job must score higher than an irrelevant one."""
        mock_emb.semantic_match_score.side_effect = [0.80, 0.15]
        mock_emb.encode.return_value = np.array([[0.1] * 384])

        good_score  = matcher.score_job(matching_job,   sample_profile)
        bad_score   = matcher.score_job(mismatched_job, sample_profile)

        assert good_score.final_score > bad_score.final_score

    @patch("core.job_matcher.embedding_engine")
    def test_rank_jobs_filters_below_threshold(
        self, mock_emb, matcher, sample_profile, matching_job, mismatched_job
    ):
        """rank_jobs should only return jobs above the threshold."""
        mock_emb.semantic_match_score.side_effect = [0.80, 0.10]
        mock_emb.encode.return_value = np.array([[0.1] * 384])
        matcher.threshold = 0.65

        ranked = matcher.rank_jobs([matching_job, mismatched_job], sample_profile)

        # Only the matching job should survive
        assert len(ranked) == 1
        assert ranked[0].title == "Senior Data Scientist"

    @patch("core.job_matcher.embedding_engine")
    def test_rank_jobs_sorted_by_score(self, mock_emb, matcher, sample_profile):
        """rank_jobs output must be sorted highest-first."""
        mock_emb.semantic_match_score.side_effect = [0.70, 0.85, 0.75]
        mock_emb.encode.return_value = np.array([[0.1] * 384])
        matcher.threshold = 0.0  # include all

        jobs = [
            Job(id="j1", title="Job A", company="X", location="Y",
                required_skills=["Python", "SQL"], experience_years="3-5 years"),
            Job(id="j2", title="Job B", company="X", location="Y",
                required_skills=["Python", "PyTorch", "scikit-learn"], experience_years="4-6 years"),
            Job(id="j3", title="Job C", company="X", location="Y",
                required_skills=["Python", "Spark"], experience_years="2-4 years"),
        ]

        ranked = matcher.rank_jobs(jobs, sample_profile)
        scores = [j.match_score for j in ranked]
        assert scores == sorted(scores, reverse=True)

    @patch("core.job_matcher.embedding_engine")
    def test_rank_jobs_empty_list(self, mock_emb, matcher, sample_profile):
        """rank_jobs with no jobs should return empty list without error."""
        mock_emb.semantic_match_score.return_value = 0.8
        result = matcher.rank_jobs([], sample_profile)
        assert result == []

    @patch("core.job_matcher.embedding_engine")
    def test_get_match_statistics(self, mock_emb, matcher, sample_profile):
        """get_match_statistics should return expected keys and valid values."""
        mock_emb.semantic_match_score.return_value = 0.75
        mock_emb.encode.return_value = np.array([[0.1] * 384])

        jobs = [
            Job(id=f"j{i}", title="DS", company="Co", location="BLR",
                required_skills=["Python", "SQL"],
                experience_years="2-4 years", industry="FinTech")
            for i in range(5)
        ]

        stats = matcher.get_match_statistics(jobs, sample_profile)

        assert "total_jobs_analyzed" in stats
        assert "mean_match_score" in stats
        assert "score_distribution" in stats
        assert "skill_gaps" in stats
        assert stats["total_jobs_analyzed"] == 5
        assert 0.0 <= stats["mean_match_score"] <= 1.0


# ── MatchScore Unit Tests ──────────────────────────────────────────────────────

class TestMatchScore:

    def test_weighted_formula(self):
        """MatchScore.final_score must follow the 30/40/15/15 formula."""
        ms = MatchScore(
            semantic_score=1.0,
            skill_score=1.0,
            experience_score=1.0,
            domain_score=1.0,
            matched_skills=["Python"],
            missing_skills=[],
        )
        assert ms.final_score == pytest.approx(1.0)

    def test_zero_scores(self):
        ms = MatchScore(
            semantic_score=0.0,
            skill_score=0.0,
            experience_score=0.0,
            domain_score=0.0,
            matched_skills=[],
            missing_skills=["Python"],
        )
        assert ms.final_score == pytest.approx(0.0)

    def test_to_dict_keys(self):
        ms = MatchScore(0.7, 0.8, 0.9, 0.6, ["Python"], ["Scala"])
        d = ms.to_dict()
        for key in ["final_score", "semantic", "skill_overlap",
                    "experience_match", "domain_relevance",
                    "matched_skills", "missing_skills"]:
            assert key in d
