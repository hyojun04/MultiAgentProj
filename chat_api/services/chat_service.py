from typing import Any

from chat_api.errors import api_error
from chat_api.repositories.chat_repository import ChatRepository
from chat_api.services.agent_service import AgentService


class ChatService:
    def __init__(self, repository: ChatRepository, agent_service: AgentService):
        self.repository = repository
        self.agent_service = agent_service

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

        user_message = self.repository.create_message(conversation_id, "user", content)
        try:
            assistant_content = await self.agent_service.send_message(content, conversation_id)
        except Exception as exc:
            raise api_error(502, "AGENT_CALL_FAILED", str(exc)) from exc

        assistant_message = self.repository.create_message(
            conversation_id,
            "assistant",
            assistant_content,
        )
        self.repository.touch_conversation(conversation_id, user_id)

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
        }

    def _require_conversation(self, conversation_id: int, user_id: int) -> dict[str, Any]:
        conversation = self.repository.get_conversation(conversation_id, user_id)
        if not conversation or conversation.get("is_archived"):
            raise api_error(404, "CONVERSATION_NOT_FOUND", "Conversation not found")
        return conversation
