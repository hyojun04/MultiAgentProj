from typing import Any

from supabase import Client, create_client

from chat_api.config import get_settings


class MemoryRepository:
    """user_memories 테이블 접근 (ChatRepository와 동일한 패턴)."""

    def __init__(self, client: Client | None = None):
        settings = get_settings()
        self.client = client or create_client(settings.supabase_url, settings.supabase_key)

    def search_memories(
        self,
        user_id: int,
        query_embedding: list[float],
        match_count: int = 5,
    ) -> list[dict[str, Any]]:
        """질문 임베딩과 의미적으로 가까운 기억을 match_user_memories RPC로 검색."""
        response = self.client.rpc(
            "match_user_memories",
            {
                "query_embedding": query_embedding,
                "target_user_id": user_id,
                "match_count": match_count,
            },
        ).execute()
        return response.data or []

    def insert_memory(
        self,
        user_id: int,
        content: str,
        embedding: list[float],
        memory_type: str = "fact",
        source_conversation_id: int | None = None,
    ) -> dict[str, Any]:
        """추출된 기억 1건을 user_memories에 저장."""
        response = (
            self.client.table("user_memories")
            .insert(
                {
                    "user_id": user_id,
                    "content": content,
                    "embedding": embedding,
                    "memory_type": memory_type,
                    "source_conversation_id": source_conversation_id,
                }
            )
            .execute()
        )
        rows = response.data or []
        return rows[0] if rows else {}