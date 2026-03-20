"""
Tests for core/resume_parser.py

Covers:
  - Text extraction helpers (clean_resume_text)
  - Profile building from parsed LLM output
  - Handling of missing / partial data
  - File extension routing (pdf vs docx)
  - LLM error handling

Run:  pytest tests/test_resume_parser.py -v
"""

import pytest
import tempfile
import os
from unittest.mock import MagicMock, patch, mock_open

from core.resume_parser import (
    clean_resume_text,
    extract_resume_text,
    ResumeParser,
)
from models.candidate import CandidateProfile


# ── Sample Data ────────────────────────────────────────────────────────────────

SAMPLE_RESUME_TEXT = """
Rohit Kumar
rohit@example.com | +91-9876543210 | Bangalore, India
linkedin.com/in/rohitkumar | github.com/rohitkumar

SUMMARY
Data Scientist with 4 years of experience in ML, NLP, and recommendation systems.

SKILLS
Python, PyTorch, TensorFlow, SQL, Spark, scikit-learn, Docker, AWS

EXPERIENCE
Senior Data Scientist — Acme Corp (2022 – Present)
  Built NLP pipelines using BERT. Improved model accuracy by 15%.

Data Analyst — StartupXYZ (2020 – 2022)
  Analysed user behaviour data using Python and SQL.

EDUCATION
B.Tech Computer Science — IIT Bombay (2020)

CERTIFICATIONS
AWS Certified Machine Learning Specialist (2023)
"""

SAMPLE_LLM_OUTPUT = {
    "name": "Rohit Kumar",
    "email": "rohit@example.com",
    "phone": "+91-9876543210",
    "location": "Bangalore, India",
    "linkedin_url": "https://linkedin.com/in/rohitkumar",
    "github_url": "https://github.com/rohitkumar",
    "portfolio_url": None,
    "summary": "Data Scientist with 4 years of experience in ML, NLP.",
    "headline": "Senior Data Scientist | ML | NLP",
    "years_of_experience": 4.0,
    "current_role": "Senior Data Scientist",
    "current_company": "Acme Corp",
    "technical_skills": ["Python", "PyTorch", "TensorFlow", "SQL", "Spark"],
    "soft_skills": ["Communication"],
    "tools": ["Docker", "AWS", "Git"],
    "domains": ["NLP", "Machine Learning"],
    "work_experience": [
        {
            "company": "Acme Corp",
            "title": "Senior Data Scientist",
            "start_date": "2022-01",
            "end_date": None,
            "description": "Built NLP pipelines using BERT.",
            "skills_used": ["Python", "PyTorch", "BERT"],
            "is_current": True,
        },
        {
            "company": "StartupXYZ",
            "title": "Data Analyst",
            "start_date": "2020-01",
            "end_date": "2022-01",
            "description": "Analysed user behaviour data.",
            "skills_used": ["Python", "SQL"],
            "is_current": False,
        },
    ],
    "education": [
        {
            "institution": "IIT Bombay",
            "degree": "B.Tech",
            "field": "Computer Science",
            "graduation_year": 2020,
            "gpa": None,
        }
    ],
    "projects": [],
    "certifications": [
        {
            "name": "AWS Certified Machine Learning Specialist",
            "issuer": "Amazon",
            "year": 2023,
            "url": None,
        }
    ],
    "publications": [],
    "target_roles": ["Data Scientist", "ML Engineer"],
    "target_locations": ["Bangalore"],
    "preferred_industries": ["FinTech"],
    "salary_expectation": None,
    "notice_period": "30 days",
    "remote_preference": "hybrid",
}


# ── Text Cleaning Tests ────────────────────────────────────────────────────────

class TestCleanResumeText:

    def test_removes_excess_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_resume_text(text)
        assert "\n\n\n" not in result

    def test_removes_excess_spaces(self):
        text = "Hello     World"
        result = clean_resume_text(text)
        assert "  " not in result

    def test_strips_whitespace(self):
        text = "   Hello World   "
        result = clean_resume_text(text)
        assert result == result.strip()

    def test_preserves_content(self):
        text = "Name: Rohit Kumar\nSkills: Python, SQL"
        result = clean_resume_text(text)
        assert "Rohit Kumar" in result
        assert "Python" in result

    def test_empty_string(self):
        result = clean_resume_text("")
        assert result == ""

    def test_only_whitespace(self):
        result = clean_resume_text("   \n\n\t\t   ")
        assert result == ""


# ── File Extension Routing Tests ───────────────────────────────────────────────

class TestExtractResumeText:

    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            extract_resume_text("/fake/path/resume.txt.xyz")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            extract_resume_text("/nonexistent/path/resume.pdf")

    def test_txt_file_reads_content(self, tmp_path):
        txt_file = tmp_path / "resume.txt"
        txt_file.write_text("Rohit Kumar\nData Scientist")
        result = extract_resume_text(str(txt_file))
        assert "Rohit Kumar" in result

    @patch("core.resume_parser.extract_text_from_pdf")
    def test_pdf_calls_pdf_extractor(self, mock_pdf, tmp_path):
        mock_pdf.return_value = "PDF content"
        pdf_file = tmp_path / "resume.pdf"
        pdf_file.write_bytes(b"%PDF fake")
        result = extract_resume_text(str(pdf_file))
        mock_pdf.assert_called_once()
        assert result == "PDF content"

    @patch("core.resume_parser.extract_text_from_docx")
    def test_docx_calls_docx_extractor(self, mock_docx, tmp_path):
        mock_docx.return_value = "DOCX content"
        docx_file = tmp_path / "resume.docx"
        docx_file.write_bytes(b"PK fake")
        result = extract_resume_text(str(docx_file))
        mock_docx.assert_called_once()
        assert result == "DOCX content"


# ── Profile Building Tests ─────────────────────────────────────────────────────

class TestResumeParserBuildProfile:

    @pytest.fixture
    def parser(self):
        mock_llm = MagicMock()
        mock_llm.parse_json.return_value = SAMPLE_LLM_OUTPUT
        mock_llm.aparse_json = MagicMock()
        return ResumeParser(llm_client=mock_llm)

    def test_builds_correct_name(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert profile.name == "Rohit Kumar"

    def test_builds_skills(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert "Python" in profile.technical_skills
        assert "PyTorch" in profile.technical_skills

    def test_builds_work_experience(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert len(profile.work_experience) == 2
        assert profile.work_experience[0].company == "Acme Corp"
        assert profile.work_experience[0].is_current is True

    def test_builds_education(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert len(profile.education) == 1
        assert profile.education[0].institution == "IIT Bombay"

    def test_builds_certifications(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert len(profile.certifications) == 1
        assert profile.certifications[0].name == "AWS Certified Machine Learning Specialist"

    def test_stores_raw_resume_text(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert profile.raw_resume_text == SAMPLE_RESUME_TEXT

    def test_stores_file_path(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/my/resume.pdf")
        assert profile.resume_file_path == "/my/resume.pdf"

    def test_handles_missing_optional_fields(self, parser):
        """Profile with minimal data should not raise."""
        minimal = {
            "name": "Test User",
            "technical_skills": [],
            "soft_skills": [],
            "tools": [],
            "domains": [],
            "work_experience": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "publications": [],
            "target_roles": [],
        }
        profile = parser._build_profile(minimal, "", "/fake.pdf")
        assert profile.name == "Test User"
        assert profile.email is None
        assert profile.years_of_experience is None

    def test_all_skills_property(self, parser):
        """all_skills should combine technical, soft, and tools."""
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        all_s = profile.all_skills
        assert "Python" in all_s
        assert "Docker" in all_s
        assert "Communication" in all_s

    def test_full_profile_text_not_empty(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        text = profile.full_profile_text
        assert len(text) > 50
        assert "Python" in text

    def test_returns_candidate_profile_instance(self, parser):
        profile = parser._build_profile(SAMPLE_LLM_OUTPUT, SAMPLE_RESUME_TEXT, "/fake.pdf")
        assert isinstance(profile, CandidateProfile)


# ── Full Parse (mocked LLM + file) ────────────────────────────────────────────

class TestResumeParserParse:

    @patch("core.resume_parser.extract_resume_text")
    def test_parse_calls_llm_and_returns_profile(self, mock_extract, tmp_path):
        mock_extract.return_value = SAMPLE_RESUME_TEXT

        mock_llm = MagicMock()
        mock_llm.parse_json.return_value = SAMPLE_LLM_OUTPUT

        parser = ResumeParser(llm_client=mock_llm)
        fake_path = str(tmp_path / "resume.pdf")

        # Create a dummy file so Path.exists() passes
        open(fake_path, "w").close()

        profile = parser.parse(fake_path)

        assert isinstance(profile, CandidateProfile)
        assert profile.name == "Rohit Kumar"
        mock_llm.parse_json.assert_called_once()

    @patch("core.resume_parser.extract_resume_text")
    def test_parse_truncates_long_resume(self, mock_extract, tmp_path):
        """Resumes > 8000 chars should be truncated before LLM call."""
        long_text = "A" * 10000
        mock_extract.return_value = long_text

        mock_llm = MagicMock()
        mock_llm.parse_json.return_value = SAMPLE_LLM_OUTPUT

        parser = ResumeParser(llm_client=mock_llm)
        fake_path = str(tmp_path / "resume.pdf")
        open(fake_path, "w").close()

        parser.parse(fake_path)

        # The LLM was called with a truncated prompt
        call_args = mock_llm.parse_json.call_args
        user_prompt = call_args[1].get("user_prompt", call_args[0][0] if call_args[0] else "")
        assert len(user_prompt) < len(long_text) + 1000  # prompt adds template overhead
