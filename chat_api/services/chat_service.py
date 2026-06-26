from typing import Any

from chat_api.errors import api_error
from chat_api.repositories.chat_repository import ChatRepository
from chat_api.services.agent_service import AgentService
from chat_api.services.memory_service import MemoryService

class ChatService:
    def __init__(self, repository: ChatRepository, agent_service: AgentService, memory_service : MemoryService):
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

        recent_messages = self.repository.list_recent_messages(conversation_id, limit=20)
        user_message = self.repository.create_message(conversation_id, "user", content)

        # 장기메모리는 현재 요청과 분리해서 전달해 과거/참고 정보가 실행 대상으로 섞이지 않게 한다.
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

        # 기억할 가치가 있으면 추출/저장 (원본 content 기준, 실패해도 응답엔 영향 없음)
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

    def _require_conversation(self, conversation_id: int, user_id: int) -> dict[str, Any]:
        conversation = self.repository.get_conversation(conversation_id, user_id)
        if not conversation or conversation.get("is_archived"):
            raise api_error(404, "CONVERSATION_NOT_FOUND", "Conversation not found")
        return conversation
