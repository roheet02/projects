"""
JobPilot AI - LLM Client
Provider-agnostic wrapper using LiteLLM.
Supports: OpenAI, Claude, Groq, Ollama, Mistral, and more.
"""

import json
from typing import Optional, List, Dict, Any
import litellm
from litellm import completion, acompletion
from loguru import logger

from config.settings import settings


# Configure LiteLLM
litellm.set_verbose = settings.debug


class LLMClient:
    """
    Provider-agnostic LLM client.
    Change LLM_MODEL in .env to switch providers — no code changes needed.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ):
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> List[Dict]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def chat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Synchronous LLM call."""
        messages = self._build_messages(user_prompt, system_prompt)
        try:
            extra = {}
            if json_mode:
                extra["response_format"] = {"type": "json_object"}

            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra,
                **kwargs,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    async def achat(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False,
        **kwargs,
    ) -> str:
        """Async LLM call."""
        messages = self._build_messages(user_prompt, system_prompt)
        try:
            extra = {}
            if json_mode:
                extra["response_format"] = {"type": "json_object"}

            response = await acompletion(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **extra,
                **kwargs,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            raise

    def parse_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Call LLM and parse JSON response."""
        response = self.chat(user_prompt, system_prompt, json_mode=True)
        try:
            # Strip markdown code fences if present
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            logger.warning("JSON parse failed, attempting extraction...")
            # Try to extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from: {response[:200]}")

    async def aparse_json(
        self,
        user_prompt: str,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async call and parse JSON response."""
        response = await self.achat(user_prompt, system_prompt, json_mode=True)
        try:
            if response.startswith("```"):
                response = response.split("```")[1]
                if response.startswith("json"):
                    response = response[4:]
            return json.loads(response.strip())
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise ValueError(f"Could not parse JSON from: {response[:200]}")


# Default singleton instance
llm = LLMClient()
