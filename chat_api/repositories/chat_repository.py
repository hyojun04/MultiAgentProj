from datetime import datetime, timezone
from typing import Any

from supabase import Client, create_client

from chat_api.config import get_settings


class ChatRepository:
    def __init__(self, client: Client | None = None):
        settings = get_settings()
        self.client = client or create_client(settings.supabase_url, settings.supabase_key)

    def list_conversations(self, user_id: int) -> list[dict[str, Any]]:
        response = (
            self.client.table("chat_conversations")
            .select("*")
            .eq("user_id", user_id)
            .eq("is_archived", False)
            .order("updated_at", desc=True)
            .execute()
        )
        return response.data or []

    def create_conversation(self, user_id: int, title: str, chat_type: str) -> dict[str, Any]:
        response = (
            self.client.table("chat_conversations")
            .insert({"user_id": user_id, "title": title, "chat_type": chat_type})
            .execute()
        )
        return self._one(response.data)

    def get_conversation(self, conversation_id: int, user_id: int) -> dict[str, Any] | None:
        response = (
            self.client.table("chat_conversations")
            .select("*")
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else None

    def delete_conversation(self, conversation_id: int, user_id: int) -> None:
        (
            self.client.table("chat_conversations")
            .delete()
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )

    def update_conversation_title(
        self,
        conversation_id: int,
        user_id: int,
        title: str,
    ) -> dict[str, Any]:
        response = (
            self.client.table("chat_conversations")
            .update({"title": title, "updated_at": self._now_iso()})
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )
        return self._one(response.data)

    def list_messages(self, conversation_id: int) -> list[dict[str, Any]]:
        response = (
            self.client.table("chat_messages")
            .select("*")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
            .execute()
        )
        return response.data or []

    def create_message(self, conversation_id: int, role: str, content: str) -> dict[str, Any]:
        response = (
            self.client.table("chat_messages")
            .insert({"conversation_id": conversation_id, "role": role, "content": content})
            .execute()
        )
        return self._one(response.data)

    def touch_conversation(self, conversation_id: int, user_id: int) -> None:
        now = self._now_iso()
        (
            self.client.table("chat_conversations")
            .update({"updated_at": now, "last_message_at": now})
            .eq("conversation_id", conversation_id)
            .eq("user_id", user_id)
            .execute()
        )

    @staticmethod
    def _one(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
        if not rows:
            raise RuntimeError("Supabase returned no rows")
        return rows[0]

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
