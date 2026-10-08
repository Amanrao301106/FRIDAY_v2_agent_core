"""LLM-backed task planner for FRIDAY."""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from config import FridayConfig
from intelligence.response_parser import ActionRequest, ParsedResponse, ParseError, parse_response
from utils.logger import get_logger

if TYPE_CHECKING:
    from intelligence.llm_client import OllamaClient

log = get_logger("agent.planner")


_PLANNER_PROMPT = """
You are FRIDAY's task planner for a Windows desktop agent.

Turn the user's goal into a SHORT, ordered list of executable actions.
Return ONLY valid JSON using this schema:
{
  "actions": [
    {
      "action": "open_app|close_app|open_website|type_text|press_key|hotkey|click|scroll|send_whatsapp|speak",
      "value": "",
      "keys": [],
      "name": "",
      "message": "",
      "delay_after": 0,
      "fail_strategy": "retry|ask_user|fail_fast|skip",
      "description": "human readable step"
    }
  ],
  "spoken_response": "short optional acknowledgement",
  "requires_confirmation": false,
  "meta": {
    "intent": "task",
    "confidence": 0.0,
    "plan_complexity": "low|medium|high",
    "optimized": true
  }
}

Rules:
- Plan only actions FRIDAY can execute with the listed action types.
- Never invent shell commands or arbitrary executable names.
- Prefer simple deterministic steps.
- Keep a plan to at most {max_steps} actions.
- A communication action such as send_whatsapp must set requires_confirmation=true.
- If the goal cannot be completed with available actions, return a speak action asking for the missing capability.
- Do not claim an action happened; only produce the plan.
"""


class TaskPlanner:
    """Create or repair an action plan using the configured LLM."""

    def __init__(self, cfg: FridayConfig, llm: OllamaClient) -> None:
        self._cfg = cfg
        self._llm = llm

    def plan(self, goal: str, context: Optional[str] = None) -> ParsedResponse:
        prompt = _PLANNER_PROMPT.replace("{max_steps}", str(self._cfg.security.max_chain_length))
        user_message = goal
        if context:
            user_message += f"\n\nCURRENT TASK CONTEXT:\n{context}"

        raw = self._llm.query(prompt, user_message, use_cache=False)
        parsed = parse_response(raw)
        if len(parsed.actions) > self._cfg.security.max_chain_length:
            raise ParseError("Planner produced too many actions")
        return parsed
