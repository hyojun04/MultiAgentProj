import logging
import os
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent_executor import CalendarAgentExecutor


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


HOST = os.getenv("CALENDAR_AGENT_HOST", "0.0.0.0")
PORT = int(os.getenv("CALENDAR_AGENT_PORT", "10014"))
PUBLIC_URL = os.getenv("CALENDAR_AGENT_PUBLIC_URL", f"http://localhost:{PORT}")


def create_agent_card() -> AgentCard:
    list_skill = AgentSkill(
        id="calendar_list_events",
        name="Google Calendar 일정 조회",
        description="오늘 일정 또는 이번 주 일정을 조회합니다.",
        tags=["calendar", "schedule", "list"],
        examples=[
            "오늘 일정 알려줘",
            "이번 주 일정 조회해줘",
            "오늘 캘린더에 뭐가 있어?"
        ],
    )

    create_skill = AgentSkill(
        id="calendar_create_event",
        name="Google Calendar 일정 등록",
        description="제목, 날짜, 시작 시간, 종료 시간을 바탕으로 Google Calendar에 일정을 등록합니다.",
        tags=["calendar", "schedule", "create"],
        examples=[
            "내일 오후 2시에 AI 에이전트 프로젝트 회의 일정 등록해줘",
            "2026-06-26 14:00부터 15:00까지 시연 준비 회의 등록해줘",
        ],
    )

    capabilities = AgentCapabilities(
        streaming=True,
        input_modes=["text"],
        output_modes=["text"],
    )

    return AgentCard(
        name="Calendar Agent",
        description="Google Calendar 일정을 조회하고 등록하는 업무 지원 에이전트",
        url=PUBLIC_URL,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=capabilities,
        skills=[list_skill, create_skill],
    )


def main():
    agent_card = create_agent_card()
    agent_executor = CalendarAgentExecutor()
    task_store = InMemoryTaskStore()

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )

    server_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )

    logger.info("=" * 60)
    logger.info("[CALENDAR AGENT] 서버 시작")
    logger.info(f"[CALENDAR AGENT] 서버 주소: {PUBLIC_URL}")
    logger.info(f"[CALENDAR AGENT] 에이전트 카드: {PUBLIC_URL}/.well-known/agent-card.json")
    logger.info("=" * 60)

    uvicorn.run(
        server_app.build(),
        host=HOST,
        port=PORT,
    )


if __name__ == "__main__":
    main()