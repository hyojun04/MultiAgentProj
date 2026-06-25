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

    def touch_conversation(self, conversation_id, user_id):
        self.touched = True


class FakeAgentService:
    async def send_message(self, content, conversation_id):
        return f"answer to {content}"


class ChatServiceTest(unittest.TestCase):
    def test_send_message_persists_user_and_assistant_messages(self):
        repository = FakeRepository()
        service = ChatService(repository, FakeAgentService())

        result = asyncio.run(service.send_message(1, 7, "hello"))

        self.assertEqual(result["user_message"]["role"], "user")
        self.assertEqual(result["assistant_message"]["role"], "assistant")
        self.assertEqual(result["assistant_message"]["content"], "answer to hello")
        self.assertTrue(repository.touched)


if __name__ == "__main__":
    unittest.main()
