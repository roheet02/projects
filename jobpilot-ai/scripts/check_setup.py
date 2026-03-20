"""
JobPilot AI — Setup Checker
Verifies that your environment is correctly configured before running the pipeline.
Checks: Python version, dependencies, .env file, LLM connectivity, browser install.

Usage:
    python scripts/check_setup.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

PASS = "[bold green]✅ PASS[/bold green]"
FAIL = "[bold red]❌ FAIL[/bold red]"
WARN = "[bold yellow]⚠️  WARN[/bold yellow]"


def check(label: str, fn) -> tuple:
    """Run a check function, return (status_str, detail_str)."""
    try:
        result = fn()
        return (PASS, result or "")
    except Exception as e:
        return (FAIL, str(e))


# ── Individual Checks ──────────────────────────────────────────────────────────

def check_python_version():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        raise RuntimeError(f"Python 3.10+ required, got {v.major}.{v.minor}")
    return f"Python {v.major}.{v.minor}.{v.micro}"


def check_env_file():
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        raise FileNotFoundError(
            ".env file not found. Run: cp .env.example .env  then add your API key."
        )
    return ".env file found"


def check_llm_key():
    from config.settings import settings
    if settings.openai_api_key:
        return f"OpenAI key set (model: {settings.llm_model})"
    if settings.anthropic_api_key:
        return f"Anthropic key set (model: {settings.llm_model})"
    if settings.groq_api_key:
        return f"Groq key set (model: {settings.llm_model})"
    if settings.ollama_base_url:
        return f"Ollama configured ({settings.ollama_base_url})"
    raise ValueError(
        "No LLM API key found in .env. Set one of: "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, or OLLAMA_BASE_URL"
    )


def check_fastapi():
    import fastapi
    return f"fastapi {fastapi.__version__}"


def check_pydantic():
    import pydantic
    return f"pydantic {pydantic.__version__}"


def check_litellm():
    import litellm
    return f"litellm {litellm.__version__}"


def check_sentence_transformers():
    import sentence_transformers
    return f"sentence-transformers {sentence_transformers.__version__}"


def check_chromadb():
    import chromadb
    return f"chromadb {chromadb.__version__}"


def check_sqlalchemy():
    import sqlalchemy
    return f"sqlalchemy {sqlalchemy.__version__}"


def check_streamlit():
    import streamlit
    return f"streamlit {streamlit.__version__}"


def check_playwright():
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        return "Playwright + Chromium OK"
    except Exception as e:
        raise RuntimeError(
            f"Playwright/Chromium not ready: {e}\n"
            "Fix: playwright install chromium"
        )


def check_browser_use():
    try:
        import browser_use
        return f"browser-use {browser_use.__version__}"
    except ImportError:
        raise ImportError(
            "browser-use not installed. Run: pip install browser-use\n"
            "(Optional — AI browser agent capabilities require this)"
        )


def check_pymupdf():
    try:
        import fitz
        return f"PyMuPDF (fitz) {fitz.version[0]}"
    except ImportError:
        raise ImportError(
            "PyMuPDF not installed. Run: pip install PyMuPDF\n"
            "(Required for PDF resume parsing)"
        )


def check_python_docx():
    import docx
    return "python-docx OK"


def check_fastmcp():
    try:
        import fastmcp
        return f"fastmcp OK"
    except ImportError:
        raise ImportError(
            "fastmcp not installed. Run: pip install fastmcp\n"
            "(Optional — required only for MCP Server / Claude Desktop integration)"
        )


def check_llm_connectivity():
    """Quick live LLM call to verify the API key works."""
    from config.settings import settings
    import litellm
    response = litellm.completion(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Say 'OK' in one word."}],
        max_tokens=5,
    )
    reply = response.choices[0].message.content.strip()
    return f"LLM responded: '{reply}'"


# ── Runner ────────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print("[bold cyan]JobPilot AI — Setup Checker[/bold cyan]")
    console.print("[dim]Checking your environment...[/dim]\n")

    checks = [
        # Category, label, check_fn, required?
        ("Environment",    "Python Version",         check_python_version,         True),
        ("Environment",    ".env File",              check_env_file,               True),
        ("Environment",    "LLM API Key",            check_llm_key,                True),

        ("Dependencies",   "FastAPI",                check_fastapi,                True),
        ("Dependencies",   "Pydantic",               check_pydantic,               True),
        ("Dependencies",   "LiteLLM",                check_litellm,                True),
        ("Dependencies",   "sentence-transformers",  check_sentence_transformers,  True),
        ("Dependencies",   "ChromaDB",               check_chromadb,               True),
        ("Dependencies",   "SQLAlchemy",             check_sqlalchemy,             True),
        ("Dependencies",   "Streamlit",              check_streamlit,              True),
        ("Dependencies",   "PyMuPDF",                check_pymupdf,                True),
        ("Dependencies",   "python-docx",            check_python_docx,            True),

        ("Optional",       "browser-use",            check_browser_use,            False),
        ("Optional",       "FastMCP (Claude Desktop)", check_fastmcp,              False),

        ("Connectivity",   "Playwright + Chromium",  check_playwright,             True),
        ("Connectivity",   "LLM Live Call",          check_llm_connectivity,       True),
    ]

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold cyan")
    table.add_column("Category",   style="dim",  width=14)
    table.add_column("Check",      style="bold", width=28)
    table.add_column("Status",                   width=12)
    table.add_column("Detail",     style="dim",  width=42)

    failures = []
    warnings = []

    for category, label, fn, required in checks:
        with console.status(f"  Checking {label}..."):
            status, detail = check(label, fn)

        if "FAIL" in status:
            if required:
                failures.append(label)
            else:
                status = WARN
                warnings.append(label)

        table.add_row(category, label, status, detail[:60])

    console.print(table)
    console.print()

    if failures:
        console.print(f"[bold red]❌ {len(failures)} required check(s) failed:[/bold red]")
        for f in failures:
            console.print(f"   • {f}")
        console.print("\n[yellow]Fix the issues above before running the pipeline.[/yellow]\n")
        sys.exit(1)
    elif warnings:
        console.print(
            f"[bold yellow]⚠️  Setup mostly complete "
            f"({len(warnings)} optional item(s) missing).[/bold yellow]"
        )
        for w in warnings:
            console.print(f"   • {w} (optional)")
        console.print("\n[green]✅ Core setup is ready. You can run the pipeline.[/green]\n")
    else:
        console.print("[bold green]✅ All checks passed! JobPilot AI is ready to run.[/bold green]\n")
        console.print("Next step:")
        console.print("[cyan]  streamlit run ui/streamlit_app.py[/cyan]")
        console.print("[cyan]  python scripts/run_pipeline.py --resume your_resume.pdf[/cyan]\n")


if __name__ == "__main__":
    main()
