import asyncio
import unittest
from datetime import datetime, timezone

from chat_api.services.chat_service import ChatService


NOW = datetime.now(timezone.utc).isoformat()


class FakeRepository:
    def __init__(self):
        self.conversation = {
            "conversation_id": 1,
            "user_id": 7,
            "title": "새 채팅",
            "chat_type": "general",
            "is_archived": False,
            "created_at": NOW,
            "updated_at": NOW,
            "last_message_at": None,
        }
        self.messages = []
        self.recent_messages = []
        self.touched = False

    def get_conversation(self, conversation_id, user_id):
        if conversation_id == 1 and user_id == 7:
            return self.conversation
        return None

    def create_message(self, conversation_id, role, content):
        message = {
            "message_id": len(self.messages) + 1,
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": NOW,
        }
        self.messages.append(message)
        return message

    def list_recent_messages(self, conversation_id, limit=20):
        return self.recent_messages[-limit:]

    def touch_conversation(self, conversation_id, user_id):
        self.touched = True


class FakeAgentService:
    def __init__(self):
        self.calls = []

    async def send_message(
        self,
        content,
        conversation_id,
        history=None,
        memory_context="",
    ):
        self.calls.append(
            {
                "content": content,
                "conversation_id": conversation_id,
                "history": history,
                "memory_context": memory_context,
            }
        )
        return f"answer to {content}"


class FakeMemoryService:
    def __init__(self, context=""):
        self.context = context
        self.stored = []

    def retrieve_context(self, user_id, query):
        return self.context

    def extract_and_store(self, **kwargs):
        self.stored.append(kwargs)


class ChatServiceTest(unittest.TestCase):
    def test_send_message_persists_user_and_assistant_messages(self):
        repository = FakeRepository()
        service = ChatService(repository, FakeAgentService(), FakeMemoryService())

        result = asyncio.run(service.send_message(1, 7, "hello"))

        self.assertEqual(result["user_message"]["role"], "user")
        self.assertEqual(result["assistant_message"]["role"], "assistant")
        self.assertEqual(result["assistant_message"]["content"], "answer to hello")
        self.assertTrue(repository.touched)

    def test_send_message_passes_previous_messages_and_memory_separately(self):
        repository = FakeRepository()
        repository.recent_messages = [
            {
                "message_id": 10,
                "conversation_id": 1,
                "role": "user",
                "content": "이전 요청은 실행하지 마",
                "created_at": NOW,
            },
            {
                "message_id": 11,
                "conversation_id": 1,
                "role": "assistant",
                "content": "알겠습니다.",
                "created_at": NOW,
            },
        ]
        agent_service = FakeAgentService()
        memory_service = FakeMemoryService("[사용자에 대해 기억하고 있는 정보]\n- Python 선호")
        service = ChatService(repository, agent_service, memory_service)

        asyncio.run(service.send_message(1, 7, "현재 요청만 처리해줘"))

        self.assertEqual(agent_service.calls[0]["content"], "현재 요청만 처리해줘")
        self.assertEqual(agent_service.calls[0]["history"], repository.recent_messages)
        self.assertEqual(
            agent_service.calls[0]["memory_context"],
            "[사용자에 대해 기억하고 있는 정보]\n- Python 선호",
        )
        self.assertNotIn(
            repository.messages[0],
            agent_service.calls[0]["history"],
            "current user message should not be duplicated in history",
        )


if __name__ == "__main__":
    unittest.main()
