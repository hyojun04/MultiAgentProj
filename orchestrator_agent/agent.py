import os
import json
import logging
import httpx
from typing import AsyncIterator, Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from uuid import uuid4

from a2a.client import A2AClient, A2ACardResolver
from a2a.types import MessageSendParams, SendMessageRequest, TextPart, Message

logger = logging.getLogger(__name__)

load_dotenv()


AGENT_URLS = {
    "internal_rag": os.getenv("RAG_AGENT_URL", "http://localhost:10012"),
    "web_research": os.getenv("WEB_AGENT_URL", "http://localhost:10011"),
    "file_management": os.getenv("FILE_AGENT_URL", "http://localhost:10013"),
    "calendar_agent": os.getenv("CALENDAR_AGENT_URL", "http://localhost:10014"),
}

AGENT_DISPLAY_NAMES = {
    "internal_rag": "Internal RAG Agent",
    "web_research": "Web Research Agent",
    "file_management": "File Management Agent",
    "calendar_agent": "Calendar Agent",
}


class OrchestratorAgent:
    """
    오케스트레이터 에이전트

    기능:
    1. 사용자 요청 Intent 분석
    2. 하위 에이전트 실행 Plan 생성
    3. A2A Client로 Remote Agent 호출
    4. 여러 결과 통합 후 최종 답변 생성
    5. 처리 상태와 최종 응답 스트리밍
    """

    SYSTEM_PROMPT = """
당신은 사용자 요청을 분석하여 적절한 하위 에이전트 실행 계획을 만드는 Orchestrator입니다.
직접 작업하지 말고, 어떤 에이전트를 어떤 순서로 호출할지만 결정하세요.

사용 가능한 agent 이름은 반드시 아래 4개 중 하나만 사용합니다.
- internal_rag
- web_research
- file_management
- calendar_agent

응답은 반드시 JSON object만 반환합니다.
마크다운, 설명문, 코드블록은 절대 포함하지 마세요.

응답 형식:
{
  "intent": "INTERNAL_SEARCH|WEB_SEARCH|FILE_OPERATION|CALENDAR_OPERATION|HYBRID|DIRECT",
  "plan": [
    {
      "agent": "internal_rag|web_research|file_management|calendar_agent",
      "query": "하위 에이전트에게 전달할 자연어 요청"
    }
  ],
  "direct_answer": "DIRECT일 때만 작성, 아니면 빈 문자열"
}

공통 규칙:
1. 사용자의 원래 의도를 보존합니다.
2. 날짜, 시간, 파일명, 폴더명, 저장할 파일명은 절대 생략하지 않습니다.
3. 사용자가 YYYY-MM-DD 형식 날짜를 입력하면 하위 에이전트 query에도 반드시 그대로 포함합니다.
4. "하나만", "전부", "3개", "최신순" 같은 수량/조건을 유지합니다.
5. plan은 실제 실행 순서대로 작성합니다.
6. 앞 단계 결과가 필요한 경우 [이전 결과] 또는 [파일 목록] placeholder를 사용합니다.
7. 사용자의 단순 오타나 띄어쓰기는 자연스럽게 보정합니다.
8. 이전 대화는 현재 요청의 대명사, 생략된 대상, 맥락 해석에만 사용합니다.
9. plan은 반드시 현재 사용자 요청만 대상으로 생성하고, 이전 대화의 과거 요청을 다시 실행하지 않습니다.
10. 하위 에이전트 query에는 필요한 맥락만 요약해서 포함하고, 전체 이전 대화를 그대로 전달하지 않습니다.

대화 기록 처리 규칙:
1. 사용자 입력에 [이전 대화]와 [현재 사용자 질문]이 함께 들어오면 [현재 사용자 질문]을 실제 요청으로 봅니다.
2. "총 몇 개야?", "그 일정은?", "방금 말한 거", "위 내용"처럼 이전 대화를 가리키는 질문은 [이전 대화]를 참고합니다.
3. 이전 대화만으로 답할 수 있으면 DIRECT로 답변합니다.
4. 이전 대화만으로 부족하고 실제 작업이 필요할 때만 하위 에이전트를 호출합니다.

에이전트 역할:
1. internal_rag
- DB에 인덱싱된 문서 검색
- 내부 문서/RAG 검색
- Google Drive 파일을 storage_ref 기반으로 DB에 인덱싱
- "인덱싱된 문서", "DB에 저장된 문서", "내부 문서", "RAG", "문서에서 찾아줘" 요청 담당

2. web_research
- 외부 웹 검색
- 최신 뉴스, 최신 트렌드, 현재 정보 조사
- "최신", "오늘", "~년 트렌드", "웹에서 찾아줘", "조사해줘" 요청 담당

3. file_management
- Google Drive 파일/폴더 목록 조회
- 파일 검색, 업로드, 저장, 다운로드, 삭제, 수정
- storage_ref 확인 요청 담당

4. calendar_agent
- Google Calendar 일정 조회
- Google Calendar 일정 검색
- Google Calendar 일정 등록
- Google Calendar 일정 수정
- Google Calendar 일정 삭제
- 빈 시간 확인
- 오늘/이번 주 일정 요약
- "오늘 일정", "이번 주 일정", "캘린더", "일정 검색", "일정 수정", "일정 삭제", "빈 시간", "회의 가능한 시간", "일정 요약", "회의 잡아줘", "스케줄 알려줘" 요청 담당

라우팅 규칙:

A. 인덱싱된 문서 검색
사용자가 인덱싱된 문서, DB 저장 문서, 내부 문서, RAG에서 찾으라고 하면 internal_rag만 호출합니다.

B. 웹 검색
최신 정보, 뉴스, 트렌드, 외부 조사가 목적이면 web_research를 호출합니다.

C. 웹 검색 후 Drive 저장
웹 조사 결과를 파일로 저장하라는 요청은 web_research 후 file_management를 호출합니다.
두 번째 query에는 반드시 [이전 결과]를 포함합니다.

D. Drive 파일 DB 인덱싱
Google Drive 파일을 DB에 인덱싱하라는 요청은 file_management 후 internal_rag를 호출합니다.
internal_rag query에는 반드시 [파일 목록]을 포함합니다.

E. Drive 파일 관리
파일 목록 조회, 폴더 조회, storage_ref 확인, 파일 검색은 file_management만 호출합니다.

F. Calendar 일정 조회/검색/등록/수정/삭제/요약
캘린더, 일정, 스케줄, 회의 시간, 빈 시간 관련 요청은 calendar_agent를 호출합니다.
날짜가 포함된 경우 YYYY-MM-DD 날짜를 반드시 query에 그대로 포함합니다.
일정 수정/삭제/검색에서 날짜가 포함되어 있으면 날짜를 절대 생략하지 않습니다.

예:
사용자: 이번주 일정 알려줘
응답:
{
  "intent": "CALENDAR_OPERATION",
  "plan": [
    {
      "agent": "calendar_agent",
      "query": "이번 주 일정을 조회해줘."
    }
  ],
  "direct_answer": ""
}

예:
사용자: 2026-06-27 17:00부터 18:00까지 'AI 에이전트 시연 준비 회의' 일정 등록해줘
응답:
{
  "intent": "CALENDAR_OPERATION",
  "plan": [
    {
      "agent": "calendar_agent",
      "query": "2026-06-27 17:00부터 18:00까지 'AI 에이전트 시연 준비 회의' 일정을 등록해줘."
    }
  ],
  "direct_answer": ""
}

예:
사용자: 2026-06-27 AI 에이전트 시연 준비 회의 일정 삭제해줘
응답:
{
  "intent": "CALENDAR_OPERATION",
  "plan": [
    {
      "agent": "calendar_agent",
      "query": "2026-06-27 날짜의 'AI 에이전트 시연 준비 회의' 일정을 삭제해줘."
    }
  ],
  "direct_answer": ""
}

예:
사용자: 2026-06-27 AI 에이전트 시연 준비 회의를 18:00부터 19:00까지로 변경해줘
응답:
{
  "intent": "CALENDAR_OPERATION",
  "plan": [
    {
      "agent": "calendar_agent",
      "query": "2026-06-27 날짜의 'AI 에이전트 시연 준비 회의' 일정을 18:00부터 19:00까지로 변경해줘."
    }
  ],
  "direct_answer": ""
}

G. 단순 대화
인사, 기능 설명, 도움말처럼 하위 에이전트가 필요 없으면 DIRECT로 답합니다.

예:
사용자: 안녕
응답:
{
  "intent": "DIRECT",
  "plan": [],
  "direct_answer": "안녕하세요. 웹 검색, Google Drive 파일 관리, 문서 인덱싱, 인덱싱된 문서 검색, Google Calendar 일정 관리를 도와드릴 수 있습니다."
}

예:
사용자:
[이전 대화]
사용자: 이번주 일정 봐줘
assistant: 이번 주 일정은 다음과 같습니다:
1. AI 에이전트 시연 준비 회의
2. Zoom 회의
3. 멀티 에이전트 구조 점검 회의

[현재 사용자 질문]
총 몇 개야?

응답:
{
  "intent": "DIRECT",
  "plan": [],
  "direct_answer": "방금 조회한 이번 주 일정은 총 3개입니다."
}

금지 사항:
- agent 이름을 새로 만들지 마세요.
- 사용자가 요청하지 않은 에이전트를 호출하지 마세요.
- 파일 인덱싱 요청에서 file_management 단계를 생략하지 마세요.
- 웹 검색 후 저장 요청에서 file_management 단계를 생략하지 마세요.
- 인덱싱된 문서 검색 요청을 web_research로 보내지 마세요.
- 캘린더/일정/스케줄 요청을 DIRECT로 처리하지 마세요.
- Calendar 요청에서 사용자가 입력한 날짜와 시간을 생략하지 마세요.
"""

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 필요합니다")

        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.httpx_client = None
        self.remote_agents: Dict[str, A2AClient] = {}
        self.initialized = False

    async def initialize(self) -> None:
        """Remote Agent 연결 초기화"""
        if self.initialized:
            return

        self.httpx_client = httpx.AsyncClient(timeout=300.0)

        for name, url in AGENT_URLS.items():
            try:
                card_resolver = A2ACardResolver(
                    httpx_client=self.httpx_client,
                    base_url=url,
                )
                agent_card = await card_resolver.get_agent_card()

                client = A2AClient(
                    httpx_client=self.httpx_client,
                    agent_card=agent_card,
                )

                self.remote_agents[name] = client
                logger.info(f"[ORCHESTRATOR] [INIT] {name} 연결 완료: {agent_card.name}")

            except Exception as e:
                logger.error(f"[ORCHESTRATOR] [ERROR] {name} 연결 실패 ({url}): {e}")

        self.initialized = True
        logger.info(
            f"[ORCHESTRATOR] [INIT] 초기화 완료 "
            f"(연결된 에이전트: {len(self.remote_agents)}개)"
        )

    async def close(self) -> None:
        """리소스 정리"""
        if self.httpx_client:
            await self.httpx_client.aclose()

    async def analyze_intent(self, state: Dict[str, Any] | str) -> Dict[str, Any]:
        """사용자 질문 분석 및 플랜 생성"""
        input_state = self._parse_input_state(state)
        intent_prompt = self._build_intent_prompt(input_state)

        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": intent_prompt},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    @staticmethod
    def _parse_input_state(raw_input: Dict[str, Any] | str) -> Dict[str, Any]:
        if isinstance(raw_input, dict):
            data = raw_input
        else:
            try:
                parsed = json.loads(raw_input)
            except (TypeError, json.JSONDecodeError):
                parsed = None

            data = parsed if isinstance(parsed, dict) else {"current_query": raw_input}

        current_query = str(data.get("current_query") or data.get("query") or "")
        history = data.get("conversation_history") or []

        if not isinstance(history, list):
            history = []

        normalized_history = []

        for message in history:
            if not isinstance(message, dict):
                continue

            content = message.get("content")

            if not content:
                continue

            normalized_history.append(
                {
                    "role": str(message.get("role", "")),
                    "content": str(content),
                }
            )

        return {
            "current_query": current_query,
            "conversation_id": data.get("conversation_id"),
            "conversation_history": normalized_history,
            "memory_context": str(data.get("memory_context") or ""),
        }

    @staticmethod
    def _build_intent_prompt(state: Dict[str, Any]) -> str:
        sections = []
        history = state.get("conversation_history") or []

        if history:
            history_lines = []

            for message in history:
                role = message.get("role", "unknown")
                content = message.get("content", "")
                history_lines.append(f"{role}: {content[:2000]}")

            sections.append(
                "[이전 대화 - 참고용, 실행 금지]\n"
                + "\n".join(history_lines)
            )

        memory_context = state.get("memory_context")

        if memory_context:
            sections.append(f"[사용자 장기기억 - 참고용]\n{memory_context[:4000]}")

        sections.append(
            "[현재 사용자 요청 - 이것만 실행 대상]\n"
            f"{state.get('current_query', '')}"
        )

        return "\n\n".join(sections)

    @staticmethod
    def _build_file_management_query(agent_query: str, state: Dict[str, Any]) -> str:
        history = state.get("conversation_history") or []

        if not history:
            return agent_query

        history_lines = []

        for message in history:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            history_lines.append(f"{role}: {content[:4000]}")

        return (
            f"{agent_query}\n\n"
            "[현재 사용자 요청 - 실행 대상]\n"
            f"{state.get('current_query', '')}\n\n"
            "[이전 대화 - 참조 전용, 실행 금지]\n"
            f"{chr(10).join(history_lines)}\n\n"
            "[파일 작업 규칙]\n"
            "- 실행해야 하는 명령은 [현재 사용자 요청]과 위 file_management 요청뿐입니다.\n"
            "- [이전 대화]의 사용자 요청을 다시 실행하지 마세요.\n"
            "- 사용자가 '조사한 내용', '위 내용', '1번 자료', '2번 자료'처럼 이전 대화를 가리키면 [이전 대화]에서 해당 assistant 답변을 찾아 참조하세요.\n"
            "- 이전 assistant 답변을 파일로 저장하거나 업데이트할 때는 원문을 요약, 재작성, 번역, 보정하지 말고 그대로 사용하세요."
        )

    async def call_remote_agent(self, agent_name: str, query: str) -> Dict[str, Any]:
        """Remote Agent 호출"""
        if agent_name not in self.remote_agents:
            return {
                "success": False,
                "content": f"에이전트 '{agent_name}'를 찾을 수 없습니다.",
                "artifacts": [],
            }

        client = self.remote_agents[agent_name]

        try:
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

            response = await client.send_message(request)

            content = ""
            artifacts = []

            result = response.root.result if hasattr(response, "root") else response.result
            logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} result 타입: {type(result)}")

            if result:
                if hasattr(result, "artifacts") and result.artifacts:
                    logger.info(
                        f"[ORCHESTRATOR] [REMOTE] {agent_name} artifacts 수: "
                        f"{len(result.artifacts)}"
                    )

                    for artifact in result.artifacts:
                        artifact_data = {
                            "name": getattr(artifact, "name", None),
                        }

                        parts = getattr(artifact, "parts", []) or []
                        logger.info(
                            f"[ORCHESTRATOR] [REMOTE] artifact.name: "
                            f"{getattr(artifact, 'name', None)}, parts: {len(parts)}"
                        )

                        for part in parts:
                            if hasattr(part, "root"):
                                part_root = part.root

                                if hasattr(part_root, "text"):
                                    content = part_root.text
                                    artifact_data["text"] = content
                                    logger.info(
                                        f"[ORCHESTRATOR] [REMOTE] TextPart 추출: "
                                        f"{content[:100] if content else 'None'}..."
                                    )

                                elif hasattr(part_root, "data"):
                                    data = part_root.data
                                    artifact_data["data"] = data

                                    if isinstance(data, dict):
                                        logger.info(
                                            f"[ORCHESTRATOR] [REMOTE] DataPart keys: "
                                            f"{list(data.keys())}"
                                        )

                        artifacts.append(artifact_data)

                else:
                    logger.info(f"[ORCHESTRATOR] [REMOTE] {agent_name} artifacts 없음")

                if not content and hasattr(result, "history") and result.history:
                    for msg in reversed(result.history):
                        if hasattr(msg, "role") and "agent" in str(msg.role):
                            for part in msg.parts:
                                if hasattr(part, "root") and hasattr(part.root, "text"):
                                    content = part.root.text
                                    break

                        if content:
                            break

            logger.info(
                f"[ORCHESTRATOR] [REMOTE] {agent_name} 최종: "
                f"content={content[:100] if content else 'None'}..., "
                f"artifacts={len(artifacts)}"
            )

            return {
                "success": True if content else False,
                "content": content or "응답을 받지 못했습니다.",
                "artifacts": artifacts,
            }

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] [REMOTE] {agent_name} 호출 실패: {e}")

            return {
                "success": False,
                "content": f"에이전트 호출 실패: {str(e)}",
                "artifacts": [],
            }

    async def generate_final_response(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        여러 에이전트 결과를 통합하여 최종 답변 생성
        기존 비스트리밍 호환용 함수
        """
        results_text = json.dumps(results, ensure_ascii=False, indent=2)

        response = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": self._final_response_system_prompt(),
                },
                {
                    "role": "user",
                    "content": f"""원본 질문: {query}

에이전트 결과:
{results_text}

위 정보를 바탕으로 사용자 질문에 답변해주세요.""",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content

    async def generate_final_response_stream(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> AsyncIterator[str]:
        """
        여러 에이전트 결과를 통합하여 최종 답변을 token 단위로 스트리밍
        """
        results_text = json.dumps(results, ensure_ascii=False, indent=2)

        stream = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": self._final_response_system_prompt(),
                },
                {
                    "role": "user",
                    "content": f"""원본 질문: {query}

에이전트 결과:
{results_text}

위 정보를 바탕으로 사용자 질문에 답변해주세요.""",
                },
            ],
            temperature=0,
            stream=True,
        )

        async for chunk in stream:
            if not chunk.choices:
                continue

            delta = chunk.choices[0].delta.content

            if delta:
                yield delta

    def _final_response_system_prompt(self) -> str:
        return """
당신은 여러 에이전트의 실행 결과를 사용자에게 정리하는 응답 생성기입니다.

규칙:
1. 에이전트 결과에 있는 내용만 바탕으로 답변하세요.
2. 실패한 단계가 있으면 실패 사실과 원인을 명확히 말하세요.
3. 성공한 단계는 무엇이 성공했는지 구체적으로 말하세요.
4. 파일명, storage_ref, Google Drive 링크, chunk 수, 검색 출처, event_id, Calendar 링크가 있으면 포함하세요.
5. 에이전트 결과에 없는 내용을 추측하지 마세요.
6. 사용자가 요청한 형식이나 수량 조건을 유지하세요.
7. 여러 단계 작업이면 단계별로 간단히 정리하세요.
""".strip()

    async def stream(
        self,
        query: str | Dict[str, Any],
        session_id: str = "default",
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        스트리밍 처리

        yield type:
        - status: 현재 처리 상태
        - final_delta: 최종 응답의 일부 토큰
        - done: 최종 완료 이벤트
        """

        if not self.initialized:
            await self.initialize()

        state = self._parse_input_state(query)
        current_query = state["current_query"]

        yield {
            "type": "status",
            "stage": "analyzing",
            "agent": None,
            "is_task_complete": False,
            "require_user_input": False,
            "content": "🤔 질문을 분석하고 있습니다...",
        }

        try:
            analysis = await self.analyze_intent(state)
            logger.info(
                f"[ORCHESTRATOR] Intent 분석 결과: "
                f"{json.dumps(analysis, ensure_ascii=False, indent=2)}"
            )
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Intent 분석 실패: {e}")

            yield {
                "type": "done",
                "stage": "error",
                "agent": None,
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"분석 중 오류: {str(e)}",
            }

            return

        intent = analysis.get("intent", "DIRECT")
        plan = analysis.get("plan", [])
        direct_answer = analysis.get("direct_answer", "")

        logger.info(
            f"[ORCHESTRATOR] intent={intent}, "
            f"plan개수={len(plan)}, "
            f"remote_agents={list(self.remote_agents.keys())}"
        )

        for i, step in enumerate(plan):
            logger.info(
                f"[ORCHESTRATOR] Plan[{i}]: "
                f"agent={step.get('agent')}, "
                f"query={step.get('query', '')[:100]}..."
            )

        if not plan and intent != "DIRECT":
            logger.warning(f"[ORCHESTRATOR] plan이 비어있음! intent={intent}")

        if intent == "DIRECT" and direct_answer:
            yield {
                "type": "final_delta",
                "stage": "direct_answer",
                "agent": None,
                "is_task_complete": False,
                "require_user_input": False,
                "content": direct_answer,
            }

            yield {
                "type": "done",
                "stage": "complete",
                "agent": None,
                "is_task_complete": True,
                "require_user_input": False,
                "content": direct_answer,
                "artifacts": [],
            }

            return

        if not plan:
            yield {
                "type": "done",
                "stage": "complete",
                "agent": None,
                "is_task_complete": True,
                "require_user_input": False,
                "content": "처리할 작업이 없습니다.",
                "artifacts": [],
            }

            return

        yield {
            "type": "status",
            "stage": "plan_created",
            "agent": None,
            "is_task_complete": False,
            "require_user_input": False,
            "content": f"📋 {len(plan)}개 에이전트에 요청할 계획을 세웠습니다.",
            "plan": plan,
        }

        results = []
        all_artifacts = []
        previous_result_content = ""
        previous_step_failed = False
        file_list_from_artifacts = []

        for i, step in enumerate(plan):
            agent_name = step.get("agent")
            agent_query = step.get("query", current_query)
            display_name = AGENT_DISPLAY_NAMES.get(agent_name, agent_name)

            if previous_step_failed:
                logger.warning(f"[ORCHESTRATOR] 이전 스텝 실패로 {agent_name} 스킵")

                results.append({
                    "agent": agent_name,
                    "success": False,
                    "content": "이전 단계 실패로 스킵됨",
                })

                yield {
                    "type": "status",
                    "stage": "agent_skipped",
                    "agent": agent_name,
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"⚠️ 이전 단계 실패로 {display_name} 실행을 건너뜁니다.",
                }

                continue

            if agent_name == "internal_rag" and file_list_from_artifacts:
                files_text = "\n".join([
                    f"- {f['filename']} ({f['storage_ref']})"
                    for f in file_list_from_artifacts
                ])

                agent_query = f"{agent_query}\n\n사용 가능한 파일 목록:\n{files_text}"

                logger.info(
                    f"[ORCHESTRATOR] RAG에 "
                    f"{len(file_list_from_artifacts)}개 파일 정보 전달"
                )

            elif previous_result_content and i > 0:
                if "[이전 결과]" in agent_query or "[검색 결과]" in agent_query:
                    agent_query = agent_query.replace("[이전 결과]", previous_result_content)
                    agent_query = agent_query.replace("[검색 결과]", previous_result_content)
                else:
                    agent_query = (
                        f"{agent_query}\n\n"
                        f"[이전 에이전트 결과]:\n"
                        f"{previous_result_content[:2000]}"
                    )

            if agent_name == "file_management":
                agent_query = self._build_file_management_query(agent_query, state)

            yield {
                "type": "status",
                "stage": "agent_running",
                "agent": agent_name,
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"🔄 {display_name}가 작업을 처리하고 있습니다...",
                "agent_query": agent_query,
            }

            logger.info(
                f"[ORCHESTRATOR] [CALL] {agent_name} 호출: "
                f"{agent_query[:200]}..."
            )

            result = await self.call_remote_agent(agent_name, agent_query)

            logger.info(
                f"[ORCHESTRATOR] [CALL] {agent_name} 결과: "
                f"success={result['success']}, "
                f"content={result['content'][:100] if result['content'] else 'None'}"
            )

            actual_success = result["success"]
            previous_step_failed = not actual_success

            if actual_success:
                yield {
                    "type": "status",
                    "stage": "agent_completed",
                    "agent": agent_name,
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"✅ {display_name} 작업이 완료되었습니다.",
                }
            else:
                yield {
                    "type": "status",
                    "stage": "agent_failed",
                    "agent": agent_name,
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": f"⚠️ {display_name} 작업 중 문제가 발생했습니다.",
                }

            if agent_name == "file_management" and actual_success:
                for artifact in result.get("artifacts", []):
                    artifact_name = artifact.get("name")
                    artifact_data = artifact.get("data")

                    if artifact_name == "file_list" and isinstance(artifact_data, dict):
                        files = artifact_data.get("files", [])

                        if files:
                            file_list_from_artifacts.extend(files)

                            logger.info(
                                f"[ORCHESTRATOR] DataPart에서 "
                                f"{len(files)}개 파일 추출"
                            )

            results.append({
                "agent": agent_name,
                "success": actual_success,
                "content": result["content"],
            })

            all_artifacts.extend(result.get("artifacts", []))

            if actual_success:
                previous_result_content = result["content"]

        yield {
            "type": "status",
            "stage": "finalizing",
            "agent": None,
            "is_task_complete": False,
            "require_user_input": False,
            "content": "📝 결과를 정리하고 있습니다...",
        }

        final_parts = []

        try:
            async for delta in self.generate_final_response_stream(current_query, results):
                final_parts.append(delta)

                yield {
                    "type": "final_delta",
                    "stage": "final_response",
                    "agent": None,
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": delta,
                }

            final_response = "".join(final_parts).strip()

            if not final_response:
                final_response = await self.generate_final_response(current_query, results)

        except Exception as e:
            logger.error(f"[ORCHESTRATOR] 최종 응답 스트리밍 실패: {e}")
            final_response = await self.generate_final_response(current_query, results)

        yield {
            "type": "done",
            "stage": "complete",
            "agent": None,
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_response,
            "artifacts": all_artifacts,
        }