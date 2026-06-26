from typing import Any, AsyncIterator

from chat_api.errors import api_error
from chat_api.repositories.chat_repository import ChatRepository
from chat_api.services.agent_service import AgentService
from chat_api.services.memory_service import MemoryService


class ChatService:
    def __init__(
        self,
        repository: ChatRepository,
        agent_service: AgentService,
        memory_service: MemoryService,
    ):
        self.repository = repository
        self.agent_service = agent_service
        self.memory_service = memory_service

    def list_conversations(self, user_id: int) -> list[dict[str, Any]]:
        return self.repository.list_conversations(user_id)

    def create_conversation(self, user_id: int, title: str, chat_type: str) -> dict[str, Any]:
        return self.repository.create_conversation(user_id, title, chat_type)

    def delete_conversation(self, conversation_id: int, user_id: int) -> None:
        self._require_conversation(conversation_id, user_id)
        self.repository.delete_conversation(conversation_id, user_id)

    def update_conversation_title(
        self,
        conversation_id: int,
        user_id: int,
        title: str,
    ) -> dict[str, Any]:
        self._require_conversation(conversation_id, user_id)
        return self.repository.update_conversation_title(conversation_id, user_id, title)

    def list_messages(self, conversation_id: int, user_id: int) -> list[dict[str, Any]]:
        self._require_conversation(conversation_id, user_id)
        return self.repository.list_messages(conversation_id)

    async def send_message(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
    ) -> dict[str, Any]:
        self._require_conversation(conversation_id, user_id)

        # 현재 사용자 메시지를 저장하기 전에 최근 대화 기록을 가져온다.
        # 그래야 현재 질문이 history에 중복으로 들어가지 않는다.
        recent_messages = self.repository.list_recent_messages(conversation_id, limit=20)

        user_message = self.repository.create_message(
            conversation_id,
            "user",
            content,
        )

        # 장기메모리는 현재 요청과 분리해서 전달한다.
        # 메모리 내용이 실제 실행 명령으로 섞이지 않게 하기 위함.
        memory_context = self.memory_service.retrieve_context(user_id, content)

        try:
            assistant_content = await self.agent_service.send_message(
                content,
                conversation_id,
                history=recent_messages,
                memory_context=memory_context,
            )
        except Exception as exc:
            raise api_error(502, "AGENT_CALL_FAILED", str(exc)) from exc

        assistant_message = self.repository.create_message(
            conversation_id,
            "assistant",
            assistant_content,
        )

        self.repository.touch_conversation(conversation_id, user_id)

        # 기억할 가치가 있으면 추출/저장
        self.memory_service.extract_and_store(
            user_id=user_id,
            user_query=content,
            assistant_answer=assistant_content,
            conversation_id=conversation_id,
        )

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
        }

    async def send_message_stream(
        self,
        conversation_id: int,
        user_id: int,
        content: str,
    ) -> AsyncIterator[dict[str, Any]]:
        self._require_conversation(conversation_id, user_id)

        # 현재 사용자 메시지를 저장하기 전에 최근 대화 기록을 가져온다.
        recent_messages = self.repository.list_recent_messages(conversation_id, limit=20)

        # 1. 사용자 메시지는 먼저 DB에 저장
        user_message = self.repository.create_message(
            conversation_id,
            "user",
            content,
        )

        yield {
            "event": "user_message",
            "data": {
                "message": user_message,
            },
        }

        # 2. 장기 메모리는 content에 붙이지 않고 별도 필드로 전달
        memory_context = self.memory_service.retrieve_context(user_id, content)

        assistant_parts: list[str] = []
        final_content = ""

        try:
            # 3. AgentService에서 Orchestrator 스트림 이벤트 받기
            async for event in self.agent_service.stream_message(
                content,
                conversation_id,
                history=recent_messages,
                memory_context=memory_context,
            ):
                event_type = event.get("type", "status")
                event_content = event.get("content", "") or ""

                if event_type == "status":
                    yield {
                        "event": "status",
                        "data": {
                            "content": event_content,
                            "stage": event.get("stage"),
                            "agent": event.get("agent"),
                        },
                    }

                elif event_type == "final_delta":
                    delta = self._merge_delta_safely(assistant_parts, event_content)

                    # 이미 출력한 내용과 같은 전체 답변이 다시 들어온 경우 무시
                    if not delta:
                        continue

                    assistant_parts.append(delta)

                    yield {
                        "event": "delta",
                        "data": {
                            "content": delta,
                        },
                    }

                elif event_type == "done":
                    current_content = self._deduplicate_repeated_content(
                        "".join(assistant_parts)
                    )
                    done_content = self._deduplicate_repeated_content(event_content)

                    # 이미 delta로 받은 내용이 있으면 done의 content는 저장용으로도 신뢰하지 않음
                    # A2A 마지막 완료 응답이 전체 답변을 한 번 더 들고 오는 경우가 있기 때문
                    if current_content:
                        final_content = current_content
                    else:
                        final_content = done_content

        except Exception as exc:
            yield {
                "event": "error",
                "data": {
                    "code": "AGENT_CALL_FAILED",
                    "message": str(exc),
                },
            }
            return

        if not final_content:
            final_content = "".join(assistant_parts)

        final_content = self._deduplicate_repeated_content(final_content)

        # 4. assistant 최종 응답 DB 저장
        assistant_message = self.repository.create_message(
            conversation_id,
            "assistant",
            final_content,
        )

        self.repository.touch_conversation(conversation_id, user_id)

        # 5. 장기 메모리 저장
        self.memory_service.extract_and_store(
            user_id=user_id,
            user_query=content,
            assistant_answer=final_content,
            conversation_id=conversation_id,
        )

        yield {
            "event": "done",
            "data": {
                "assistant_message": assistant_message,
            },
        }

    def _require_conversation(self, conversation_id: int, user_id: int) -> dict[str, Any]:
        conversation = self.repository.get_conversation(conversation_id, user_id)

        if not conversation or conversation.get("is_archived"):
            raise api_error(404, "CONVERSATION_NOT_FOUND", "Conversation not found")

        return conversation

    @classmethod
    def _merge_delta_safely(
        cls,
        assistant_parts: list[str],
        incoming_content: str,
    ) -> str:
        """
        이미 출력된 전체 답변이 final_delta로 한 번 더 들어오는 경우 중복 방지.
        """

        incoming_content = cls._deduplicate_repeated_content(incoming_content)

        if not incoming_content:
            return ""

        current_content = "".join(assistant_parts)

        if not current_content:
            return incoming_content

        # 같은 전체 답변이 다시 들어온 경우
        if incoming_content.strip() == current_content.strip():
            return ""

        # incoming_content가 전체 완성본이고, 앞부분이 이미 출력된 경우
        if incoming_content.startswith(current_content):
            rest = incoming_content[len(current_content):]
            return rest

        # 긴 문장이 동일하게 다시 들어온 경우 방지
        if len(incoming_content.strip()) >= 20 and current_content.endswith(incoming_content):
            return ""

        return incoming_content

    @staticmethod
    def _deduplicate_repeated_content(content: str) -> str:
        """
        '답변답변'처럼 같은 최종 답변이 정확히 2번 붙은 경우 제거.
        """

        if not content:
            return ""

        stripped = content.strip()

        if len(stripped) < 20:
            return content

        center = len(stripped) // 2

        # 공백/개행 차이를 고려해서 가운데 기준 ±5 범위 확인
        for offset in range(-5, 6):
            split_index = center + offset

            if split_index <= 0 or split_index >= len(stripped):
                continue

            first = stripped[:split_index].strip()
            second = stripped[split_index:].strip()

            if first and first == second:
                return first

        return content