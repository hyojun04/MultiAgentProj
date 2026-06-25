from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ConversationCreateRequest(BaseModel):
    title: str = Field(default="새 채팅", min_length=1, max_length=255)
    chat_type: str = Field(default="general", min_length=1, max_length=30)


class ConversationUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationResponse(BaseModel):
    conversation_id: int
    user_id: int
    title: str
    chat_type: str
    is_archived: bool
    created_at: datetime
    updated_at: datetime
    last_message_at: Optional[datetime] = None
