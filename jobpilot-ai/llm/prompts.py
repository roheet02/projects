"""
JobPilot AI - LLM Prompt Templates
All prompts used across the system.
"""

from string import Template


# ============================================================
# RESUME PARSING
# ============================================================

RESUME_EXTRACTION_SYSTEM = """
You are an expert resume parser. Extract structured information from the provided
resume text and return it as valid JSON matching the schema provided.
Be thorough — capture all skills, tools, technologies, and domain knowledge.
For missing fields, use null. Do NOT invent information not present in the resume.
"""

RESUME_EXTRACTION_PROMPT = """
Extract all information from this resume text and return a JSON object with this exact schema:

{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "+1-xxx",
  "location": "City, Country",
  "linkedin_url": "https://...",
  "github_url": "https://...",
  "portfolio_url": "https://...",
  "summary": "Professional summary paragraph",
  "headline": "Short professional headline (e.g. Senior Data Scientist | ML | NLP)",
  "years_of_experience": 5.5,
  "current_role": "Current Job Title",
  "current_company": "Company Name",
  "technical_skills": ["Python", "PyTorch", "SQL"],
  "soft_skills": ["Leadership", "Communication"],
  "tools": ["VS Code", "Docker", "AWS"],
  "domains": ["NLP", "Computer Vision", "FinTech"],
  "work_experience": [
    {
      "company": "Company",
      "title": "Job Title",
      "start_date": "2021-06",
      "end_date": "2024-01",
      "description": "What you did",
      "skills_used": ["Python", "Spark"],
      "is_current": false
    }
  ],
  "education": [
    {
      "institution": "University",
      "degree": "B.Tech",
      "field": "Computer Science",
      "graduation_year": 2019,
      "gpa": null
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "description": "What it does",
      "technologies": ["Python", "TensorFlow"],
      "url": null,
      "impact": "Reduced latency by 40%"
    }
  ],
  "certifications": [
    {
      "name": "AWS Certified ML Specialist",
      "issuer": "Amazon",
      "year": 2023,
      "url": null
    }
  ],
  "publications": [],
  "target_roles": ["Data Scientist", "ML Engineer"],
  "notice_period": "30 days",
  "remote_preference": "hybrid"
}

RESUME TEXT:
$resume_text
"""


# ============================================================
# JOB DESCRIPTION PARSING
# ============================================================

JOB_PARSING_SYSTEM = """
You are an expert at analyzing job descriptions. Extract structured information
from job postings and return valid JSON. Be precise about required skills vs. nice-to-have.
"""

JOB_PARSING_PROMPT = """
Parse this job description and return a JSON object:

{
  "title": "Job Title",
  "company": "Company Name",
  "location": "City, Country / Remote",
  "job_type": "full_time | part_time | contract | remote | hybrid",
  "experience_years": "3-5 years",
  "salary_range": "$80k-$120k or null",
  "required_skills": ["Python", "Machine Learning"],
  "preferred_skills": ["Kubernetes", "MLOps"],
  "description_summary": "2-3 sentence summary of the role",
  "company_size": "startup | mid-size | enterprise | null",
  "industry": "FinTech | HealthTech | E-commerce | etc."
}

JOB DESCRIPTION:
$job_text
"""


# ============================================================
# JOB MATCHING
# ============================================================

MATCH_EXPLANATION_PROMPT = """
Explain why this candidate is a good match for this job in 3 concise bullet points.
Focus on skill overlap, domain relevance, and experience alignment.
Be specific — mention actual skills and technologies.

CANDIDATE PROFILE:
Name: $candidate_name
Skills: $candidate_skills
Experience: $candidate_experience
Domains: $candidate_domains

JOB:
Title: $job_title at $company
Required Skills: $required_skills
Description: $job_description

Return exactly 3 bullet points as a JSON array of strings:
{"reasons": ["reason 1", "reason 2", "reason 3"]}
"""


# ============================================================
# HR RESEARCH
# ============================================================

HR_SEARCH_PROMPT = """
I need to find the right HR or recruiter contact at $company for a $role position.
Based on the information below, suggest:
1. The best job title to search for on LinkedIn (e.g., "Technical Recruiter", "HR Manager", "Talent Acquisition")
2. A LinkedIn search query to find this person
3. Any publicly known company careers page or email pattern if known

Company: $company
Role: $role
Industry: $industry

Return JSON:
{
  "target_titles": ["Technical Recruiter", "HR Manager"],
  "linkedin_search_query": "Technical Recruiter at Company Name",
  "careers_page_hint": "careers.company.com or null",
  "email_pattern_hint": "firstname.lastname@company.com or null"
}
"""


# ============================================================
# OUTREACH GENERATION
# ============================================================

EMAIL_GENERATION_SYSTEM = """
You are an expert at writing personalized, professional cold outreach emails for job seekers.
Your emails are:
- Short (150-200 words max)
- Personalized and specific (not generic templates)
- Professional but warm in tone
- Focused on value — what the candidate brings, not just what they want
- Clear call-to-action
- Compliant with anti-spam best practices (no misleading subject lines)
"""

EMAIL_GENERATION_PROMPT = """
Write a personalized cold email from the candidate to the HR contact about a specific job opening.

CANDIDATE:
Name: $candidate_name
Current Role: $current_role
Key Skills: $key_skills
Years of Experience: $years_exp
Key Achievement: $key_achievement
Summary: $summary

HR CONTACT:
Name: $hr_name
Title: $hr_title
Company: $company

JOB:
Title: $job_title
Key Requirements: $job_requirements

Write:
1. A compelling subject line (max 8 words)
2. The email body (150-200 words)

Format as JSON:
{
  "subject": "Subject line here",
  "body": "Full email body here"
}

Rules:
- Address HR by first name
- Mention 1-2 specific things about the company/role (not generic)
- Highlight 2-3 relevant skills/experiences
- End with a clear, non-pushy CTA
- Do NOT use clichés like "I hope this email finds you well"
- Do NOT attach or mention a resume in the cold email
"""


LINKEDIN_MESSAGE_SYSTEM = """
You are an expert at writing LinkedIn connection requests and direct messages for job seekers.
LinkedIn messages must be SHORT (300 character limit for connection requests, 1000 for DMs).
Be direct, specific, and human. Avoid corporate speak.
"""

LINKEDIN_CONNECTION_PROMPT = """
Write a LinkedIn connection request note (max 300 characters) from the candidate to the HR.

CANDIDATE: $candidate_name, $current_role, $key_skill
HR: $hr_name at $company
CONTEXT: Interested in $job_title role

Return JSON:
{
  "connection_note": "Short note here (max 300 chars)"
}
"""

LINKEDIN_DM_PROMPT = """
Write a LinkedIn direct message from the candidate expressing interest in the job.

CANDIDATE:
Name: $candidate_name
Headline: $headline
Key Skills: $key_skills

HR CONTACT: $hr_name, $hr_title at $company
JOB: $job_title

Write a natural, brief DM (max 150 words). Return JSON:
{
  "message": "DM text here"
}
"""


# ============================================================
# FOLLOW-UP GENERATION
# ============================================================

FOLLOWUP_EMAIL_PROMPT = """
Write a brief, polite follow-up email for a cold outreach sent $days_ago days ago with no response.
Keep it to 3-4 sentences. Acknowledge this is a follow-up. Add one new value point.

Original message was about: $job_title at $company
Candidate: $candidate_name

Return JSON: {"subject": "...", "body": "..."}
"""


# ============================================================
# COMPANY RESEARCH
# ============================================================

COMPANY_RESEARCH_PROMPT = """
Based on publicly available information, provide a brief research summary for $company.
Focus on: tech stack, company culture, recent news, growth stage, and team size.
This will be used to personalize a job application outreach.

Return JSON:
{
  "tech_stack": ["Python", "AWS"],
  "culture_keywords": ["fast-paced", "data-driven"],
  "recent_news": "Brief highlight if known",
  "growth_stage": "Series B | IPO | Enterprise",
  "team_size_estimate": "50-200",
  "personalization_hook": "One specific thing to mention in outreach"
}
"""
