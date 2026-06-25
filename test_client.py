import asyncio
import httpx
from uuid import uuid4

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import (
    Message,
    MessageSendParams,
    SendMessageRequest,
    TextPart,
)


async def test_agent(agent_url: str, query: str):
    """
    에이전트 테스트

    Args:
        agent_url: 에이전트 서버 URL
        query: 테스트 쿼리
    """
    async with httpx.AsyncClient(timeout=600.0) as client:  # 10분 타임아웃
        try:
            # 에이전트 카드 가져오기
            card_resolver = A2ACardResolver(
                httpx_client=client,
                base_url=agent_url,
            )
            agent_card = await card_resolver.get_agent_card()

            print(f"\n✅ 에이전트: {agent_card.name}")
            print(f"   설명: {agent_card.description}")

            # A2A 클라이언트 생성
            a2a_client = A2AClient(
                httpx_client=client,
                agent_card=agent_card,
            )

            # 메시지 생성
            message = Message(
                kind="message",
                role="user",
                parts=[TextPart(kind="text", text=query)],
                message_id=uuid4().hex,
            )

            request = SendMessageRequest(
                id=uuid4().hex,
                params=MessageSendParams(message=message),
            )

            print(f"\n📤 요청: {query}")

            # 메시지 전송
            response = await a2a_client.send_message(request)

            result = response.root.result

            # 응답 텍스트 추출
            response_text = None

            # 1. artifacts에서 응답 찾기
            if hasattr(result, "artifacts") and result.artifacts:
                for artifact in result.artifacts:
                    if artifact.parts:
                        for part in artifact.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                response_text = part.root.text
                                break
                    if response_text:
                        break

            # 2. status.message에서 응답 찾기
            if not response_text and hasattr(result, "status") and result.status:
                status = result.status
                if hasattr(status, "message") and status.message:
                    for part in status.message.parts:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            response_text = part.root.text
                            break

            # 3. history에서 마지막 agent 메시지 찾기
            if not response_text and hasattr(result, "history") and result.history:
                for msg in reversed(result.history):
                    if hasattr(msg, "role") and str(msg.role) == "Role.agent":
                        for part in msg.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                response_text = part.root.text
                                break
                    if response_text:
                        break

            if response_text:
                print(f"\n📥 응답:\n{response_text}")
            else:
                print("\n📥 응답 없음")

        except Exception as e:
            print(f"\n❌ 오류: {str(e)}")


async def main():
    print("=" * 60)
    print("A2A 오케스트레이터 통합 테스트")
    print("=" * 60)

    orchestrator_url = "http://localhost:10010"

    test_queries = [
        "2026년 AI 에이전트 트렌드를 5줄로 요약해서 'AI_에이전트_트렌드_2026_test.txt' 이름으로 드라이브에 저장해줘",
        "FileManagementAgent 폴더에 있는 파일들 중 하나만 DB에 인덱싱해줘",
        '인덱싱된 문서에서 "에이전틱 AI"에 대해 설명해줘. 출처 파일명도 알려줘.',
        "2026-06-26 17:00부터 18:00까지 'AI 에이전트 시연 준비 회의' 일정 등록해줘",
        "이번주 일정 알려줘",
        "2026-06-27 AI 에이전트 시연 준비 회의 일정 찾아줘",
        "2026-06-27 AI 에이전트 시연 준비 회의를 18:00부터 19:00까지로 변경해줘",
        "2026-06-27 오후에 비어있는 시간 알려줘",
        "2026-06-27 AI 에이전트 시연 준비 회의 일정 삭제해줘",
    ]

    print("\n" + "-" * 60)
    print("쿼리 예시")
    print("-" * 60)

    for query in test_queries:
        print(f"Q. {query}")

    print("\n종료하려면 exit, quit, q를 입력하거나 Ctrl + C를 누르세요.")

    while True:
        print("\n" + "=" * 60)
        query = input("질문을 입력하세요: ").strip()

        if not query:
            print("질문이 비어 있습니다. 다시 입력해주세요.")
            continue

        if query.lower() in ["exit", "quit", "q"]:
            print("테스트를 종료합니다.")
            break

        await test_agent(orchestrator_url, query)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n테스트를 종료합니다.")
