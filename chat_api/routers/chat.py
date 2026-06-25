from fastapi import APIRouter, Depends, Response, status

from chat_api.dependencies import get_chat_service, get_user_id
from chat_api.schemas.conversations import (
    ConversationCreateRequest,
    ConversationResponse,
    ConversationUpdateRequest,
)
from chat_api.schemas.messages import MessageCreateRequest, MessageResponse, SendMessageResponse
from chat_api.services.chat_service import ChatService


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.get("/conversations", response_model=list[ConversationResponse])
def list_conversations(
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    return service.list_conversations(user_id)


@router.post(
    "/conversations",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: ConversationCreateRequest,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    return service.create_conversation(user_id, request.title, request.chat_type)


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(
    conversation_id: int,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    service.delete_conversation(conversation_id, user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.patch("/conversations/{conversation_id}", response_model=ConversationResponse)
def update_conversation_title(
    conversation_id: int,
    request: ConversationUpdateRequest,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    return service.update_conversation_title(conversation_id, user_id, request.title)


@router.get(
    "/conversations/{conversation_id}/messages",
    response_model=list[MessageResponse],
)
def list_messages(
    conversation_id: int,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    return service.list_messages(conversation_id, user_id)


@router.post(
    "/conversations/{conversation_id}/messages",
    response_model=SendMessageResponse,
    status_code=status.HTTP_201_CREATED,
)
async def send_message(
    conversation_id: int,
    request: MessageCreateRequest,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    return await service.send_message(conversation_id, user_id, request.content)
