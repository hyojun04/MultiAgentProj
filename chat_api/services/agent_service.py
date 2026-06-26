import logging
import json
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import Message, MessageSendParams, SendMessageRequest, TextPart

from chat_api.config import get_settings


logger = logging.getLogger(__name__)


class AgentService:
    def __init__(self, agent_client: Any | None = None):
        self.agent_client = agent_client
        self.httpx_client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if self.agent_client:
            return

        settings = get_settings()
        self.httpx_client = httpx.AsyncClient(timeout=600.0)
        card_resolver = A2ACardResolver(
            httpx_client=self.httpx_client,
            base_url=settings.orchestrator_agent_url,
        )
        agent_card = await card_resolver.get_agent_card()
        self.agent_client = A2AClient(
            httpx_client=self.httpx_client,
            agent_card=agent_card,
        )
        logger.info("[CHAT API] Connected to orchestrator agent: %s", agent_card.name)

    async def close(self) -> None:
        if self.httpx_client:
            await self.httpx_client.aclose()
            self.httpx_client = None
        self.agent_client = None

    async def send_message(
        self,
        content: str,
        conversation_id: int,
        history: list[dict[str, Any]] | None = None,
        memory_context: str = "",
    ) -> str:
        if not self.agent_client:
            await self.initialize()

        payload = self._build_message_payload(
            content=content,
            conversation_id=conversation_id,
            history=history,
            memory_context=memory_context,
        )
        message = Message(
            kind="message",
            role="user",
            parts=[TextPart(kind="text", text=json.dumps(payload, ensure_ascii=False))],
            message_id=uuid4().hex,
        )
        request = SendMessageRequest(
            id=uuid4().hex,
            params=MessageSendParams(message=message),
        )

        response = await self.agent_client.send_message(request)
        response_text = self._extract_response_text(response)
        if not response_text:
            raise RuntimeError("Agent returned an empty response")

        return response_text

    @staticmethod
    def _build_message_payload(
        content: str,
        conversation_id: int,
        history: list[dict[str, Any]] | None = None,
        memory_context: str = "",
    ) -> dict[str, Any]:
        conversation_history = []
        for message in history or []:
            message_content = message.get("content")
            if not message_content:
                continue
            conversation_history.append(
                {
                    "role": str(message.get("role", "")),
                    "content": str(message_content),
                }
            )

        return {
            "current_query": content,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history,
            "memory_context": memory_context or "",
        }

    @staticmethod
    def _extract_response_text(response: Any) -> str | None:
        result = response.root.result if hasattr(response, "root") else response.result

        if hasattr(result, "artifacts") and result.artifacts:
            for artifact in result.artifacts:
                for part in artifact.parts or []:
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        return part.root.text

        if hasattr(result, "status") and result.status:
            status_message = getattr(result.status, "message", None)
            if status_message:
                for part in status_message.parts or []:
                    if hasattr(part, "root") and hasattr(part.root, "text"):
                        return part.root.text

        if hasattr(result, "history") and result.history:
            for msg in reversed(result.history):
                if hasattr(msg, "role") and "agent" in str(msg.role):
                    for part in msg.parts or []:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            return part.root.text

        return None
