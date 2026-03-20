"""
JobPilot AI - Browser Manager
Manages Playwright browser instances for the job hunting agents.
The agent opens actual websites/apps directly — no scraping APIs needed.
"""

import asyncio
from typing import Optional, Any
from contextlib import asynccontextmanager
from loguru import logger

from config.settings import settings


class BrowserManager:
    """
    Manages async Playwright browser sessions.
    Provides context-managed browser/page access for agents.
    """

    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None

    async def start(self, headless: Optional[bool] = None):
        """Launch the browser."""
        from playwright.async_api import async_playwright

        if headless is None:
            headless = settings.browser_headless

        logger.info(f"Starting browser (headless={headless})")
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=headless,
            slow_mo=settings.browser_slow_mo,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self._context = await self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
        )
        logger.success("Browser ready")

    async def stop(self):
        """Close the browser."""
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._context = None
        logger.info("Browser closed")

    async def new_page(self):
        """Open a new browser page."""
        if not self._context:
            await self.start()
        page = await self._context.new_page()
        page.set_default_timeout(settings.playwright_timeout)
        return page

    @asynccontextmanager
    async def page(self):
        """Context manager for a browser page."""
        pg = await self.new_page()
        try:
            yield pg
        finally:
            await pg.close()

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, *args):
        await self.stop()


class AIBrowserAgent:
    """
    AI-powered browser agent using browser-use library.
    Combines Playwright with LLM to perform browser tasks via natural language.

    This is the KEY component: instead of hard-coded scraping logic,
    the agent receives a natural language task and figures out how to execute it.
    """

    def __init__(self, llm_model: Optional[str] = None):
        self.model = llm_model or settings.llm_model

    async def run_task(
        self,
        task: str,
        url: Optional[str] = None,
        max_steps: int = 20,
    ) -> str:
        """
        Run a browser task described in natural language.

        Args:
            task: Natural language description of what to do
                  e.g., "Search for Python data scientist jobs in Bangalore on LinkedIn
                         and extract the first 10 job titles and companies"
            url: Optional starting URL
            max_steps: Max browser actions before giving up

        Returns:
            Task result as a string
        """
        try:
            from browser_use import Agent
            from browser_use.browser.browser import Browser, BrowserConfig

            # Configure browser
            browser = Browser(
                config=BrowserConfig(
                    headless=settings.browser_headless,
                    disable_security=True,
                )
            )

            # Build LLM from LiteLLM (provider-agnostic)
            llm = self._get_langchain_llm()

            full_task = task
            if url:
                full_task = f"Go to {url}. Then: {task}"

            agent = Agent(
                task=full_task,
                llm=llm,
                browser=browser,
                max_actions_per_step=max_steps,
            )

            result = await agent.run()
            return str(result)

        except ImportError:
            logger.warning("browser-use not installed, falling back to Playwright")
            return await self._playwright_fallback(task, url)

    def _get_langchain_llm(self):
        """Get a LangChain-compatible LLM from settings."""
        model = settings.llm_model

        if model.startswith("gpt") or "openai" in model:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                api_key=settings.openai_api_key,
                temperature=0,
            )
        elif "claude" in model or "anthropic" in model:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=model.replace("anthropic/", ""),
                api_key=settings.anthropic_api_key,
                temperature=0,
            )
        elif model.startswith("groq/"):
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=model.replace("groq/", ""),
                api_key=settings.groq_api_key,
                temperature=0,
            )
        else:
            # Default fallback
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    async def _playwright_fallback(self, task: str, url: Optional[str]) -> str:
        """Simple Playwright fallback when browser-use is not available."""
        manager = BrowserManager()
        async with manager.page() as page:
            if url:
                await page.goto(url, wait_until="networkidle")
            return f"[Playwright fallback] Navigated to {url}. Task: {task}"


# Global instances
browser_manager = BrowserManager()
ai_browser = AIBrowserAgent()
