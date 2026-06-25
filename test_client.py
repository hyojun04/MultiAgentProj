import asyncio
import httpx


BASE_URL = "http://127.0.0.1:8000"
USER_ID = 1


async def create_conversation(client: httpx.AsyncClient) -> dict:
    response = await client.post(
        "/api/v1/chat/conversations",
        json={
            "title": "멀티턴 테스트 채팅",
            "chat_type": "general",
        },
    )
    response.raise_for_status()
    return response.json()


async def list_conversations(client: httpx.AsyncClient) -> list:
    response = await client.get("/api/v1/chat/conversations")
    response.raise_for_status()
    return response.json()


async def list_messages(client: httpx.AsyncClient, conversation_id: int) -> list:
    response = await client.get(
        f"/api/v1/chat/conversations/{conversation_id}/messages"
    )
    response.raise_for_status()
    return response.json()


async def send_message(
    client: httpx.AsyncClient,
    conversation_id: int,
    content: str,
) -> dict:
    response = await client.post(
        f"/api/v1/chat/conversations/{conversation_id}/messages",
        json={
            "content": content,
        },
    )
    response.raise_for_status()
    return response.json()


def print_message(role: str, content: str):
    if role == "user":
        print(f"\n👤 사용자:\n{content}")
    elif role == "assistant":
        print(f"\n🤖 Assistant:\n{content}")
    else:
        print(f"\n[{role}]\n{content}")


def print_conversation_info(conversation: dict):
    print("\n✅ 채팅방 생성 완료")
    print(f"   conversation_id: {conversation['conversation_id']}")
    print(f"   user_id: {conversation['user_id']}")
    print(f"   title: {conversation['title']}")
    print(f"   chat_type: {conversation['chat_type']}")


async def main():
    print("=" * 60)
    print("Backend Chat API 멀티턴 테스트")
    print("=" * 60)

    headers = {
        "X-User-Id": str(USER_ID),
    }

    async with httpx.AsyncClient(
        base_url=BASE_URL,
        headers=headers,
        timeout=600.0,
    ) as client:
        try:
            # 서버 상태 확인
            health_response = await client.get("/health")
            health_response.raise_for_status()
            print("✅ Backend 서버 연결 확인")

            # 멀티턴 테스트용 채팅방 생성
            conversation = await create_conversation(client)
            conversation_id = conversation["conversation_id"]

            print_conversation_info(conversation)

        except Exception as e:
            print(f"\n❌ 초기화 실패: {e}")
            return

        test_queries = [
            "2026년 AI 에이전트 트렌드를 5줄로 요약해서 'AI_에이전트_트렌드_2026_test.txt' 이름으로 드라이브에 저장해줘",
            "FileManagementAgent 폴더에 있는 파일들 중 하나만 DB에 인덱싱해줘",
            '인덱싱된 문서에서 "에이전틱 AI"에 대해 설명해줘. 출처 파일명도 알려줘.',
            "2026-06-27 17:00부터 18:00까지 'AI 에이전트 시연 준비 회의' 일정 등록해줘",
            "이번주 일정 알려줘",
            "2026-06-27 AI 에이전트 시연 준비 회의 일정 찾아줘",
            "2026-06-27 AI 에이전트 시연 준비 회의를 18:00부터 19:00까지로 변경해줘",
            "2026-06-27 17:30부터 18:30까지 '충돌 테스트 회의' 일정 등록해줘",
            "2026-06-27 오후에 비어있는 시간 알려줘",
            "2026-06-27 AI 에이전트 시연 준비 회의 일정 삭제해줘",
        ]

        print("\n" + "-" * 60)
        print("쿼리 예시")
        print("-" * 60)

        for query in test_queries:
            print(f"Q. {query}")

        print("\n명령어")
        print("- exit, quit, q : 종료")
        print("- history : 현재 채팅방 메시지 목록 조회")
        print("- new : 새 채팅방 생성")
        print("- list : 채팅방 목록 조회")

        while True:
            print("\n" + "=" * 60)
            print(f"현재 conversation_id: {conversation_id}")
            query = input("질문을 입력하세요: ").strip()

            if not query:
                print("질문이 비어 있습니다. 다시 입력해주세요.")
                continue

            if query.lower() in ["exit", "quit", "q"]:
                print("테스트를 종료합니다.")
                break

            if query.lower() == "new":
                try:
                    conversation = await create_conversation(client)
                    conversation_id = conversation["conversation_id"]
                    print_conversation_info(conversation)
                except Exception as e:
                    print(f"\n❌ 새 채팅방 생성 실패: {e}")
                continue

            if query.lower() == "list":
                try:
                    conversations = await list_conversations(client)
                    print("\n📋 채팅방 목록")
                    for conv in conversations:
                        print(
                            f"- id={conv['conversation_id']} | "
                            f"title={conv['title']} | "
                            f"updated_at={conv['updated_at']}"
                        )
                except Exception as e:
                    print(f"\n❌ 채팅방 목록 조회 실패: {e}")
                continue

            if query.lower() == "history":
                try:
                    messages = await list_messages(client, conversation_id)
                    print("\n📜 메시지 히스토리")
                    if not messages:
                        print("메시지가 없습니다.")
                    for msg in messages:
                        print_message(msg["role"], msg["content"])
                except Exception as e:
                    print(f"\n❌ 메시지 목록 조회 실패: {e}")
                continue

            try:
                print(f"\n📤 요청:\n{query}")

                result = await send_message(
                    client=client,
                    conversation_id=conversation_id,
                    content=query,
                )

                user_message = result.get("user_message")
                assistant_message = result.get("assistant_message")

                if user_message:
                    print_message(
                        user_message["role"],
                        user_message["content"],
                    )

                if assistant_message:
                    print_message(
                        assistant_message["role"],
                        assistant_message["content"],
                    )
                else:
                    print("\n📥 assistant 응답 없음")

            except httpx.HTTPStatusError as e:
                print("\n❌ HTTP 오류")
                print(f"status_code: {e.response.status_code}")
                print(f"response: {e.response.text}")

            except Exception as e:
                print(f"\n❌ 오류: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n테스트를 종료합니다.")
