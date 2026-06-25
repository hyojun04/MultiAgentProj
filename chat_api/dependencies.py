from fastapi import Header

from chat_api.errors import api_error
from chat_api.repositories.chat_repository import ChatRepository
from chat_api.services.agent_service import AgentService
from chat_api.services.chat_service import ChatService


agent_service = AgentService()


def get_user_id(x_user_id: int | None = Header(default=None, alias="X-User-Id")) -> int:
    if x_user_id is None:
        raise api_error(401, "USER_ID_REQUIRED", "X-User-Id header is required")
    if x_user_id <= 0:
        raise api_error(422, "INVALID_USER_ID", "X-User-Id must be a positive integer")
    return x_user_id


def get_chat_service() -> ChatService:
    return ChatService(ChatRepository(), agent_service)
