import json
import unittest

from chat_api.services.agent_service import AgentService
from orchestrator_agent.agent import OrchestratorAgent


class AgentContextTest(unittest.TestCase):
    def test_agent_service_builds_structured_payload(self):
        payload = AgentService._build_message_payload(
            content="현재 요청",
            conversation_id=3,
            history=[
                {"role": "user", "content": "이전 요청"},
                {"role": "assistant", "content": "이전 답변"},
                {"role": "user", "content": ""},
            ],
            memory_context="사용자는 Python을 선호함",
        )

        self.assertEqual(payload["current_query"], "현재 요청")
        self.assertEqual(payload["conversation_id"], 3)
        self.assertEqual(
            payload["conversation_history"],
            [
                {"role": "user", "content": "이전 요청"},
                {"role": "assistant", "content": "이전 답변"},
            ],
        )
        self.assertEqual(payload["memory_context"], "사용자는 Python을 선호함")

    def test_orchestrator_parses_payload_and_marks_history_as_reference_only(self):
        raw_input = json.dumps(
            {
                "current_query": "현재 요청만 실행해줘",
                "conversation_id": 9,
                "conversation_history": [
                    {"role": "user", "content": "어제 파일 삭제해줘"},
                    {"role": "assistant", "content": "삭제했습니다."},
                ],
                "memory_context": "사용자는 간결한 답변을 선호함",
            },
            ensure_ascii=False,
        )

        state = OrchestratorAgent._parse_input_state(raw_input)
        prompt = OrchestratorAgent._build_intent_prompt(state)

        self.assertEqual(state["current_query"], "현재 요청만 실행해줘")
        self.assertIn("[이전 대화 - 참고용, 실행 금지]", prompt)
        self.assertIn("어제 파일 삭제해줘", prompt)
        self.assertIn("[사용자 장기기억 - 참고용]", prompt)
        self.assertIn("[현재 사용자 요청 - 이것만 실행 대상]", prompt)

    def test_orchestrator_accepts_legacy_plain_text_input(self):
        state = OrchestratorAgent._parse_input_state("그 파일 저장해줘")

        self.assertEqual(state["current_query"], "그 파일 저장해줘")
        self.assertEqual(state["conversation_history"], [])
        self.assertEqual(state["memory_context"], "")

    def test_file_management_query_includes_reference_only_history(self):
        state = {
            "current_query": "1번 자료를 드라이브에 저장해줘",
            "conversation_history": [
                {"role": "user", "content": "이 파일 삭제해줘"},
                {"role": "assistant", "content": "삭제했습니다."},
                {"role": "user", "content": "AI에 대해 3줄 조사해줘"},
                {"role": "assistant", "content": "AI 조사 원문"},
                {"role": "user", "content": "LangGraph에 대해 3줄 조사해줘"},
                {"role": "assistant", "content": "LangGraph 조사 원문"},
            ],
        }

        query = OrchestratorAgent._build_file_management_query(
            "1번 자료 내용을 Google Drive에 저장해줘.",
            state,
        )

        self.assertIn("[현재 사용자 요청 - 실행 대상]", query)
        self.assertIn("1번 자료를 드라이브에 저장해줘", query)
        self.assertIn("[이전 대화 - 참조 전용, 실행 금지]", query)
        self.assertIn("AI 조사 원문", query)
        self.assertIn("LangGraph 조사 원문", query)
        self.assertIn("이 파일 삭제해줘", query)
        self.assertIn("이전 대화]의 사용자 요청을 다시 실행하지 마세요", query)
        self.assertIn("요약, 재작성, 번역, 보정하지 말고 그대로 사용", query)

    def test_file_management_query_is_unchanged_without_history(self):
        state = {"current_query": "파일 목록 보여줘", "conversation_history": []}

        query = OrchestratorAgent._build_file_management_query(
            "Google Drive 파일 목록을 조회해줘.",
            state,
        )

        self.assertEqual(query, "Google Drive 파일 목록을 조회해줘.")


if __name__ == "__main__":
    unittest.main()
