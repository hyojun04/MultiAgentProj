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


if __name__ == "__main__":
    unittest.main()
