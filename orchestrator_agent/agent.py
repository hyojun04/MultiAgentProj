import os
import json
import logging
from datetime import datetime
import httpx
from typing import AsyncIterator, Dict, Any, List
from dotenv import load_dotenv
from openai import AsyncOpenAI
from uuid import uuid4
 
try:
    from prompts import ORCHESTRATOR_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
except ImportError:  # 패키지로 import 될 때
    from .prompts import ORCHESTRATOR_SYSTEM_PROMPT, FINAL_RESPONSE_SYSTEM_PROMPT
 
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
 
    SYSTEM_PROMPT = ORCHESTRATOR_SYSTEM_PROMPT
 
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
 
        try:
            from zoneinfo import ZoneInfo
            now = datetime.now(ZoneInfo("Asia/Seoul"))
        except Exception:
            now = datetime.now()
        weekday_kr = ["월", "화", "수", "목", "금", "토", "일"][now.weekday()]
        today_str = f"{now.strftime('%Y-%m-%d')} ({weekday_kr}요일)"
        system_prompt = self.SYSTEM_PROMPT.replace("<<TODAY>>", today_str)
 
        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": intent_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
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
            model="gpt-5-mini",
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
            model="gpt-5-mini",
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
        return FINAL_RESPONSE_SYSTEM_PROMPT
 
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