import logging
from typing import Any, AsyncIterator
from uuid import uuid4

import httpx
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import Message, MessageSendParams, SendMessageRequest, TextPart

try:
    from a2a.types import SendStreamingMessageRequest
except ImportError:
    SendStreamingMessageRequest = None

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

    async def send_message(self, content: str, conversation_id: int) -> str:
        """
        기존 비스트리밍 방식.
        기존 /messages API 호환용으로 유지한다.
        """
        if not self.agent_client:
            await self.initialize()

        message = self._build_message(content)
        request = SendMessageRequest(
            id=uuid4().hex,
            params=MessageSendParams(message=message),
        )

        response = await self.agent_client.send_message(request)

        response_text = self._extract_response_text(response)

        if not response_text:
            raise RuntimeError("Agent returned an empty response")

        return response_text

    async def stream_message(
        self,
        content: str,
        conversation_id: int,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Orchestrator Agent의 스트리밍 응답을 받아서
        ChatService가 처리할 수 있는 표준 이벤트로 변환한다.

        yield 형식:
        {
            "type": "status" | "final_delta" | "done",
            "content": "...",
            "stage": "...",
            "agent": "calendar_agent" | ...
        }
        """

        if not self.agent_client:
            await self.initialize()

        yield {
            "type": "status",
            "stage": "agent_connecting",
            "agent": "orchestrator",
            "content": "오케스트레이터에 요청을 전달하고 있습니다...",
        }

        streaming_method = (
            getattr(self.agent_client, "send_message_streaming", None)
            or getattr(self.agent_client, "send_message_stream", None)
        )

        # A2A SDK 또는 현재 AgentCard가 streaming을 지원하지 않는 경우 fallback
        if not streaming_method or SendStreamingMessageRequest is None:
            logger.warning(
                "[CHAT API] A2A streaming is not available. Falling back to normal send_message."
            )

            assistant_content = await self.send_message(content, conversation_id)

            yield {
                "type": "final_delta",
                "stage": "final_response",
                "agent": None,
                "content": assistant_content,
            }

            yield {
                "type": "done",
                "stage": "complete",
                "agent": None,
                "content": assistant_content,
            }

            return

        message = self._build_message(content)

        request = SendStreamingMessageRequest(
            id=uuid4().hex,
            params=MessageSendParams(message=message),
        )

        assistant_parts: list[str] = []
        final_content = ""

        try:
            async for response in streaming_method(request):
                result = self._get_result(response)

                # Orchestrator가 DataPart로 구조화된 이벤트를 보낸 경우
                structured_event = self._extract_structured_event(result)

                if structured_event:
                    event_type = structured_event.get("type", "status")
                    event_content = structured_event.get("content", "")

                    if event_type == "status":
                        yield {
                            "type": "status",
                            "stage": structured_event.get("stage"),
                            "agent": structured_event.get("agent"),
                            "content": event_content,
                        }

                    elif event_type == "final_delta":
                        assistant_parts.append(event_content)

                        yield {
                            "type": "final_delta",
                            "stage": structured_event.get("stage", "final_response"),
                            "agent": structured_event.get("agent"),
                            "content": event_content,
                        }

                    elif event_type == "done":
                        final_content = event_content or "".join(assistant_parts)

                    continue

                text = self._extract_text_from_result(result)

                if not text:
                    continue

                is_final = self._is_final_result(result)

                # Orchestrator의 중간 상태 메시지는 이모지 prefix로 구분
                if not is_final and self._is_status_text(text):
                    yield {
                        "type": "status",
                        "stage": self._guess_stage_from_status_text(text),
                        "agent": self._guess_agent_from_status_text(text),
                        "content": text,
                    }
                    continue

                # 최종 완료 이벤트인데 이미 delta를 받은 경우,
                # 같은 최종 답변이 중복으로 출력되지 않도록 done만 처리
                if is_final:
                    if assistant_parts:
                        final_content = "".join(assistant_parts)
                    else:
                        final_content = text
                        assistant_parts.append(text)

                        yield {
                            "type": "final_delta",
                            "stage": "final_response",
                            "agent": None,
                            "content": text,
                        }

                    continue

                # 상태 메시지가 아니라면 최종 응답 조각으로 간주
                assistant_parts.append(text)

                yield {
                    "type": "final_delta",
                    "stage": "final_response",
                    "agent": None,
                    "content": text,
                }

            if not final_content:
                final_content = "".join(assistant_parts)

            yield {
                "type": "done",
                "stage": "complete",
                "agent": None,
                "content": final_content,
            }

        except Exception as exc:
            logger.exception("[CHAT API] Agent streaming failed")

            yield {
                "type": "status",
                "stage": "error",
                "agent": None,
                "content": f"에이전트 스트리밍 중 오류가 발생했습니다: {str(exc)}",
            }

            raise

    def _build_message(self, content: str) -> Message:
        return Message(
            kind="message",
            role="user",
            parts=[TextPart(kind="text", text=content)],
            message_id=uuid4().hex,
        )

    @staticmethod
    def _get_result(response: Any) -> Any:
        if hasattr(response, "root") and hasattr(response.root, "result"):
            return response.root.result

        if hasattr(response, "result"):
            return response.result

        return response

    @classmethod
    def _extract_response_text(cls, response: Any) -> str | None:
        result = cls._get_result(response)
        return cls._extract_text_from_result(result)

    @classmethod
    def _extract_text_from_result(cls, result: Any) -> str | None:
        """
        A2A 응답 result에서 text를 최대한 유연하게 추출한다.
        """

        if not result:
            return None

        # 1. artifacts 안의 TextPart
        if hasattr(result, "artifacts") and result.artifacts:
            for artifact in result.artifacts:
                for part in getattr(artifact, "parts", []) or []:
                    text = cls._extract_text_from_part(part)
                    if text:
                        return text

        # 2. artifact 단일 객체
        if hasattr(result, "artifact") and result.artifact:
            artifact = result.artifact
            for part in getattr(artifact, "parts", []) or []:
                text = cls._extract_text_from_part(part)
                if text:
                    return text

        # 3. status.message 안의 TextPart
        if hasattr(result, "status") and result.status:
            status_message = getattr(result.status, "message", None)

            if status_message:
                for part in getattr(status_message, "parts", []) or []:
                    text = cls._extract_text_from_part(part)
                    if text:
                        return text

        # 4. message 안의 TextPart
        if hasattr(result, "message") and result.message:
            for part in getattr(result.message, "parts", []) or []:
                text = cls._extract_text_from_part(part)
                if text:
                    return text

        # 5. result 자체가 parts를 가진 경우
        if hasattr(result, "parts") and result.parts:
            for part in result.parts:
                text = cls._extract_text_from_part(part)
                if text:
                    return text

        # 6. history에서 마지막 agent 메시지
        if hasattr(result, "history") and result.history:
            for msg in reversed(result.history):
                if hasattr(msg, "role") and "agent" in str(msg.role):
                    for part in getattr(msg, "parts", []) or []:
                        text = cls._extract_text_from_part(part)
                        if text:
                            return text

        return None

    @staticmethod
    def _extract_text_from_part(part: Any) -> str | None:
        if hasattr(part, "root"):
            part_root = part.root

            if hasattr(part_root, "text"):
                return part_root.text

            if hasattr(part_root, "data"):
                data = part_root.data

                if isinstance(data, dict):
                    content = data.get("content")
                    if isinstance(content, str):
                        return content

        if hasattr(part, "text"):
            return part.text

        return None

    @classmethod
    def _extract_structured_event(cls, result: Any) -> dict[str, Any] | None:
        """
        Orchestrator가 DataPart로 아래 형식의 이벤트를 보낸 경우 추출한다.

        {
            "type": "status" | "final_delta" | "done",
            "content": "...",
            "stage": "...",
            "agent": "..."
        }
        """

        if not result:
            return None

        data = cls._extract_data_from_result(result)

        if isinstance(data, dict) and "type" in data:
            return data

        text = cls._extract_text_from_result(result)

        if not text:
            return None

        # 혹시 JSON string으로 들어온 경우까지 처리
        text = text.strip()

        if not text.startswith("{"):
            return None

        try:
            parsed = json.loads(text)
        except Exception:
            return None

        if isinstance(parsed, dict) and "type" in parsed:
            return parsed

        return None

    @classmethod
    def _extract_data_from_result(cls, result: Any) -> Any:
        if hasattr(result, "artifacts") and result.artifacts:
            for artifact in result.artifacts:
                for part in getattr(artifact, "parts", []) or []:
                    data = cls._extract_data_from_part(part)
                    if data is not None:
                        return data

        if hasattr(result, "artifact") and result.artifact:
            artifact = result.artifact

            for part in getattr(artifact, "parts", []) or []:
                data = cls._extract_data_from_part(part)
                if data is not None:
                    return data

        if hasattr(result, "status") and result.status:
            status_message = getattr(result.status, "message", None)

            if status_message:
                for part in getattr(status_message, "parts", []) or []:
                    data = cls._extract_data_from_part(part)
                    if data is not None:
                        return data

        if hasattr(result, "message") and result.message:
            for part in getattr(result.message, "parts", []) or []:
                data = cls._extract_data_from_part(part)
                if data is not None:
                    return data

        if hasattr(result, "parts") and result.parts:
            for part in result.parts:
                data = cls._extract_data_from_part(part)
                if data is not None:
                    return data

        return None

    @staticmethod
    def _extract_data_from_part(part: Any) -> Any:
        if hasattr(part, "root"):
            part_root = part.root

            if hasattr(part_root, "data"):
                return part_root.data

        if hasattr(part, "data"):
            return part.data

        return None

    @staticmethod
    def _is_final_result(result: Any) -> bool:
        """
        A2A streaming chunk가 최종 완료 이벤트인지 추정한다.
        SDK 버전에 따라 state 표현이 다를 수 있어 문자열 기반으로 넓게 처리한다.
        """

        if not result:
            return False

        if getattr(result, "final", False) is True:
            return True

        if getattr(result, "is_task_complete", False) is True:
            return True

        status = getattr(result, "status", None)

        if status:
            state = getattr(status, "state", None)

            if state and "completed" in str(state).lower():
                return True

        kind = getattr(result, "kind", None)

        if kind and "completed" in str(kind).lower():
            return True

        return False

    @staticmethod
    def _is_status_text(text: str) -> bool:
        """
        OrchestratorAgent.stream()에서 보내는 상태 메시지인지 구분한다.
        """

        stripped = text.strip()

        status_prefixes = (
            "🤔",
            "📋",
            "🔄",
            "✅",
            "⚠️",
            "📝",
            "오케스트레이터",
        )

        return stripped.startswith(status_prefixes)

    @staticmethod
    def _guess_stage_from_status_text(text: str) -> str | None:
        if "분석" in text:
            return "analyzing"

        if "계획" in text or "요청" in text:
            return "plan_created"

        if "호출" in text or "처리" in text or "작업" in text:
            return "agent_running"

        if "완료" in text:
            return "agent_completed"

        if "정리" in text:
            return "finalizing"

        if "오류" in text or "실패" in text or "문제" in text:
            return "error"

        return None

    @staticmethod
    def _guess_agent_from_status_text(text: str) -> str | None:
        lower_text = text.lower()

        if "calendar" in lower_text or "캘린더" in text:
            return "calendar_agent"

        if "web" in lower_text or "웹" in text:
            return "web_research"

        if "file" in lower_text or "drive" in lower_text or "파일" in text:
            return "file_management"

        if "rag" in lower_text or "문서" in text:
            return "internal_rag"

        return None