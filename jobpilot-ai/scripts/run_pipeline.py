"""
JobPilot AI — CLI Pipeline Runner
Run the full job hunting pipeline from your terminal.

Usage:
    python scripts/run_pipeline.py --resume path/to/resume.pdf
    python scripts/run_pipeline.py --resume resume.pdf --portals linkedin indeed --channel email
    python scripts/run_pipeline.py --resume resume.pdf --portals linkedin --channel linkedin_connection
    python scripts/run_pipeline.py --resume resume.pdf --step match   # run only matching step
"""

import asyncio
import sys
import os
import argparse

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.panel import Panel
from loguru import logger

from agents.orchestrator import JobPilotOrchestrator
from models.job import JobPortal
from models.outreach import OutreachChannel

console = Console()

PORTAL_MAP = {
    "linkedin": JobPortal.LINKEDIN,
    "indeed": JobPortal.INDEED,
    "naukri": JobPortal.NAUKRI,
    "glassdoor": JobPortal.GLASSDOOR,
    "wellfound": JobPortal.WELLFOUND,
}

CHANNEL_MAP = {
    "email": OutreachChannel.EMAIL,
    "linkedin_connection": OutreachChannel.LINKEDIN_CONNECTION,
    "linkedin_message": OutreachChannel.LINKEDIN_MESSAGE,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="JobPilot AI — Autonomous Job Hunting Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline (all steps)
  python scripts/run_pipeline.py --resume resume.pdf

  # Custom portals and outreach channel
  python scripts/run_pipeline.py --resume resume.pdf --portals linkedin naukri --channel linkedin_connection

  # Just parse resume and match jobs (no outreach)
  python scripts/run_pipeline.py --resume resume.pdf --step match

  # Run only outreach generation (after jobs are cached)
  python scripts/run_pipeline.py --resume resume.pdf --step outreach
        """,
    )
    parser.add_argument(
        "--resume", required=True,
        help="Path to resume file (PDF or DOCX)",
    )
    parser.add_argument(
        "--portals", nargs="+",
        default=["linkedin", "indeed"],
        choices=list(PORTAL_MAP.keys()),
        help="Job portals to search (default: linkedin indeed)",
    )
    parser.add_argument(
        "--channel",
        default="email",
        choices=list(CHANNEL_MAP.keys()),
        help="Outreach channel (default: email)",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=["all", "parse", "discover", "match", "hr", "outreach"],
        help="Run only a specific pipeline step (default: all)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Number of top matched jobs to process (default: 20)",
    )
    parser.add_argument(
        "--no-explain", action="store_true",
        help="Skip LLM match explanations (faster)",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    # Validate resume file
    if not os.path.exists(args.resume):
        console.print(f"[red]Error: Resume file not found: {args.resume}[/red]")
        sys.exit(1)

    portals = [PORTAL_MAP[p] for p in args.portals]
    channel = CHANNEL_MAP[args.channel]

    console.print(Panel(
        f"[bold cyan]🚀 JobPilot AI — CLI Runner[/bold cyan]\n"
        f"Resume: [yellow]{args.resume}[/yellow]\n"
        f"Portals: [yellow]{', '.join(args.portals)}[/yellow]\n"
        f"Channel: [yellow]{args.channel}[/yellow]\n"
        f"Step: [yellow]{args.step}[/yellow]",
        border_style="cyan",
    ))

    orchestrator = JobPilotOrchestrator()

    try:
        if args.step == "all":
            await orchestrator.run_full_pipeline(
                resume_path=args.resume,
                portals=portals,
                channel=channel,
            )

        elif args.step == "parse":
            await orchestrator.load_profile(args.resume)

        elif args.step == "discover":
            await orchestrator.load_profile(args.resume)
            await orchestrator.discover_jobs(portals=portals)

        elif args.step == "match":
            await orchestrator.load_profile(args.resume)
            await orchestrator.discover_jobs(portals=portals)
            await orchestrator.match_and_rank_jobs(
                top_k=args.top_k,
                explain=not args.no_explain,
            )

        elif args.step == "hr":
            await orchestrator.load_profile(args.resume)
            await orchestrator.discover_jobs(portals=portals)
            await orchestrator.match_and_rank_jobs(top_k=args.top_k)
            await orchestrator.research_hr_contacts()

        elif args.step == "outreach":
            await orchestrator.load_profile(args.resume)
            await orchestrator.discover_jobs(portals=portals)
            await orchestrator.match_and_rank_jobs(top_k=args.top_k)
            await orchestrator.research_hr_contacts()
            await orchestrator.generate_outreach(channel=channel)
            orchestrator.review_messages()

        console.print("\n[bold green]✅ Done![/bold green]")
        console.print("Open the Streamlit UI to review and send messages:")
        console.print("[cyan]  streamlit run ui/streamlit_app.py[/cyan]\n")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Pipeline failed: {e}[/red]")
        logger.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
