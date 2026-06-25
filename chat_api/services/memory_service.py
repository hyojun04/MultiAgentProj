import logging
from typing import Any
 
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field
 
from chat_api.repositories.memory_repository import MemoryRepository
 
logger = logging.getLogger(__name__)
 
 
class ExtractedMemory(BaseModel):
    should_remember: bool = Field(description="장기적으로 재사용할 가치가 있는 정보인지")
    content: str | None = Field(default=None, description="기억할 내용 한 문장 요약")
    memory_type: str = Field(default="fact", description="fact 또는 preference")
 
 
class MemoryService:
    """장기메모리 검색/추출/저장. chat_api 레이어에서만 사용하며 에이전트는 건드리지 않음."""
 
    def __init__(
        self,
        repository: MemoryRepository | None = None,
        embedding_model: str = "text-embedding-3-small",
        extract_model: str = "gpt-4o-mini",
        match_count: int = 5,
    ):
        self.repository = repository or MemoryRepository()
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.extract_llm = ChatOpenAI(model=extract_model, temperature=0)
        self.match_count = match_count
 
    def retrieve_context(self, user_id: int, query: str) -> str:
        """질문과 관련된 기억을 검색해 프롬프트용 텍스트로 반환. 실패/없으면 빈 문자열."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            memories = self.repository.search_memories(
                user_id, query_embedding, self.match_count
            )
        except Exception as exc:  # 메모리 실패가 채팅 자체를 막지 않도록
            logger.error("[MEMORY] 검색 실패: %s", exc)
            return ""
 
        if not memories:
            return ""
 
        lines = ["[사용자에 대해 기억하고 있는 정보]"]
        for m in memories:
            lines.append(f"- ({m.get('memory_type', 'fact')}) {m.get('content', '')}")
        return "\n".join(lines)
 
    def extract_and_store(
        self,
        user_id: int,
        user_query: str,
        assistant_answer: str,
        conversation_id: int | None = None,
    ) -> None:
        """대화에서 기억할 가치가 있는 정보만 추출해 저장. 실패해도 조용히 넘어감."""
        try:
            structured = self.extract_llm.with_structured_output(ExtractedMemory)
            result: ExtractedMemory = structured.invoke(
                [
                    {
                        "role": "system",
                        "content": (
                            "대화에서 향후 이 사용자와의 상호작용에 재사용할 사실/선호가 있는지 판단하세요.\n"
                            "기억할 가치 있음: 사용자의 소속·역할, 반복 참조할 선호 설정, "
                            "사용자가 명시적으로 '기억해줘'라고 한 정보.\n"
                            "기억할 가치 없음: 단발성 질문/답변, 일반 지식.\n"
                            "가치가 없으면 should_remember를 false로 두세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"사용자 질문: {user_query}\n\n에이전트 답변: {assistant_answer}"
                        ),
                    },
                ]
            )
            if not result.should_remember or not result.content:
                return
 
            embedding = self.embeddings.embed_query(result.content)
            self.repository.insert_memory(
                user_id=user_id,
                content=result.content,
                embedding=embedding,
                memory_type=result.memory_type or "fact",
                source_conversation_id=conversation_id,
            )
            logger.info("[MEMORY] 저장: user_id=%s, content=%s", user_id, result.content[:50])
        except Exception as exc:
            logger.error("[MEMORY] 저장 실패: %s", exc)