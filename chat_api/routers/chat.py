import json
from typing import Any

from fastapi import APIRouter, Depends, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse

from chat_api.dependencies import get_chat_service, get_user_id
from chat_api.schemas.conversations import (
    ConversationCreateRequest,
    ConversationResponse,
    ConversationUpdateRequest,
)
from chat_api.schemas.messages import (
    MessageCreateRequest,
    MessageResponse,
    SendMessageResponse,
)
from chat_api.services.chat_service import ChatService


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


def sse_event(event: str, data: dict[str, Any]) -> str:
    """
    Server-Sent Events 형식으로 변환

    event: status
    data: {"content": "..."}
    """
    encoded_data = jsonable_encoder(data)
    json_data = json.dumps(encoded_data, ensure_ascii=False)

    return f"event: {event}\ndata: {json_data}\n\n"


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


@router.post("/conversations/{conversation_id}/messages/stream")
async def send_message_stream(
    conversation_id: int,
    request: MessageCreateRequest,
    user_id: int = Depends(get_user_id),
    service: ChatService = Depends(get_chat_service),
):
    """
    메시지 전송 스트리밍 API

    SSE 이벤트 종류:
    - user_message: 사용자 메시지 저장 완료
    - status: 현재 처리 상태
    - delta: assistant 최종 응답 일부
    - done: assistant 메시지 저장 완료
    - error: 오류 발생
    """

    async def event_generator():
        try:
            async for item in service.send_message_stream(
                conversation_id=conversation_id,
                user_id=user_id,
                content=request.content,
            ):
                event_name = item.get("event", "message")
                data = item.get("data", {})

                yield sse_event(event_name, data)

        except Exception as exc:
            yield sse_event(
                "error",
                {
                    "code": "STREAM_ERROR",
                    "message": str(exc),
                },
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )