# 🚀 JobPilot AI

**Autonomous AI Job Hunting Agent** — Find relevant jobs, research HR contacts, and send personalized cold outreach — all running on your personal laptop.

---

## What is JobPilot AI?

JobPilot AI is a multi-agent system that automates the most tedious parts of job searching:

| Manual Task | What JobPilot AI Does |
|---|---|
| Search LinkedIn for jobs | AI agent opens LinkedIn/Indeed/Naukri directly and finds relevant openings |
| Read each JD and decide if it fits | ML scoring engine (semantic + skill matching) ranks jobs by fit score |
| Find the right HR to contact | AI agent searches LinkedIn for HR/Talent contacts at target companies |
| Write cold emails | LLM generates personalized emails referencing your specific skills + their role |
| Send emails one by one | Agent opens Gmail/LinkedIn and sends after YOUR approval |
| Track who you contacted | SQLite database + analytics dashboard tracks everything |

**Human-in-the-loop**: Every message is shown to you for review before sending. The agent NEVER sends anything without your explicit approval.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JobPilot AI System                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │   Streamlit │  │   FastAPI    │  │      MCP Server        │ │
│  │     UI      │  │   REST API   │  │  (Claude Desktop)      │ │
│  └──────┬──────┘  └──────┬───────┘  └──────────┬─────────────┘ │
│         └────────────────┼──────────────────────┘               │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                  Orchestrator Agent                        │  │
│  │  (coordinates all agents, manages state, drives workflow)  │  │
│  └───┬──────────────┬────────────────┬──────────────┬────────┘  │
│      ▼              ▼                ▼              ▼           │
│  ┌───────┐  ┌─────────────┐  ┌──────────┐  ┌────────────────┐ │
│  │Resume │  │Job Discovery│  │HR Search │  │Outreach Agent  │ │
│  │Parser │  │   Agent     │  │  Agent   │  │(email/LinkedIn)│ │
│  └───┬───┘  └──────┬──────┘  └────┬─────┘  └───────┬────────┘ │
│      │              │              │                │           │
│  ┌───▼───────────────▼──────────────▼────────────────▼───────┐ │
│  │               Browser Automation Layer                     │ │
│  │      (Playwright + browser-use AI browser agent)           │ │
│  │   LinkedIn │ Indeed │ Naukri │ Gmail │ Glassdoor           │ │
│  └─────────────────────────────────────────────────────────── ┘ │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │                    Core ML Layer                           │  │
│  │  Resume Parser │ Embedding Engine │ Job Matcher (scoring) │  │
│  │  sentence-transformers │ scikit-learn │ ChromaDB           │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│  ┌───────────────────────▼───────────────────────────────────┐  │
│  │              Storage & Analytics Layer                     │  │
│  │          SQLite (local) │ ChromaDB │ Analytics             │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ML Matching Algorithm

JobPilot AI uses a **4-signal weighted scoring system**:

| Signal | Weight | Method |
|---|---|---|
| Semantic Similarity | 30% | Cosine similarity of sentence-transformer embeddings |
| Skill Overlap | 40% | Normalized Jaccard with partial string matching |
| Experience Match | 15% | Penalized linear score for over/under qualification |
| Domain Relevance | 15% | Semantic similarity of domain keywords |

**Final Score = 0.30×Semantic + 0.40×Skill + 0.15×Experience + 0.15×Domain**

---

## Project Structure

```
jobpilot-ai/
├── config/         Settings (Pydantic, .env)
├── core/           Resume parser, ML matcher, embeddings
├── agents/         Outreach agent, orchestrator
├── browser/        Playwright browser agents (LinkedIn, Gmail, portals)
├── llm/            LiteLLM client, prompt templates
├── mcp_server/     FastMCP server for Claude Desktop integration
├── api/            FastAPI REST API
├── database/       SQLAlchemy ORM, SQLite repository
├── analytics/      Statistics and dashboard data
├── ui/             Streamlit web interface
└── tests/          Unit and integration tests
```

---

## Quick Start

### 1. Install dependencies
```bash
cd jobpilot-ai
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your LLM API key (OpenAI/Claude/Groq/etc.)
```

### 3. Run the Streamlit UI
```bash
streamlit run ui/streamlit_app.py
```

### 4. Or run the FastAPI server
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Or use from Python
```python
import asyncio
from agents.orchestrator import JobPilotOrchestrator

async def main():
    agent = JobPilotOrchestrator()
    await agent.run_full_pipeline(
        resume_path="./my_resume.pdf",
        portals=["linkedin", "indeed"],
        channel="email",
    )

asyncio.run(main())
```

---

## MCP Server (Claude Desktop)

Add to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "jobpilot": {
      "command": "python",
      "args": ["-m", "mcp_server.server"],
      "cwd": "/path/to/jobpilot-ai"
    }
  }
}
```

Then from Claude Desktop you can say:
> "Parse my resume at ~/resume.pdf, find ML Engineer jobs in Bangalore, and draft cold emails for the top 5 matches"

---

## Supported LLM Providers

| Provider | Model Example | Set in .env |
|---|---|---|
| OpenAI | `gpt-4o-mini` | `OPENAI_API_KEY=sk-...` |
| Anthropic | `claude-3-haiku-20240307` | `ANTHROPIC_API_KEY=sk-ant-...` |
| Groq (free) | `groq/llama3-70b-8192` | `GROQ_API_KEY=gsk_...` |
| Ollama (local) | `ollama/llama3.1` | `OLLAMA_BASE_URL=http://localhost:11434` |

---

## Legal & Compliance Notes

- **Human-in-the-loop**: All outreach requires your explicit approval before sending
- **Daily limits**: Default max 20 outreach messages/day (configurable)
- **CAN-SPAM / GDPR**: Cold emails include genuine contact info and a way to opt out
- **LinkedIn ToS**: Sending connection requests at a reasonable pace (2-3/day) is within normal use
- **No bulk/spam**: This tool is designed for quality, personalized outreach — not mass blasting

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| LLM | LiteLLM (provider-agnostic) |
| Browser Automation | Playwright + browser-use |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| ML Matching | scikit-learn + numpy |
| Vector Store | ChromaDB (local) |
| Database | SQLite + SQLAlchemy |
| UI | Streamlit |
| MCP Server | FastMCP |

---

## Author

Built by **Rohit** — Data Scientist | AI Engineer

---

*JobPilot AI is a personal project for automating job search workflows. Use responsibly.*
