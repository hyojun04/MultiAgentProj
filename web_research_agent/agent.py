import os
import sys
import json
import logging
import asyncio
from datetime import datetime
from typing import AsyncIterator, Dict, Any, List

import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

# Windows 이벤트 루프 정책 설정
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


SYSTEM_PROMPT = """
당신은 웹 조사 전문가입니다.

역할:
- 사용자의 요청에 맞는 최신 웹 정보를 조사합니다.
- 검색 결과를 바탕으로 핵심 내용을 정리합니다.
- 출처가 있는 정보와 추론을 구분합니다.
- 파일로 저장될 수 있는 형태의 완성된 조사 결과를 작성합니다.

답변 규칙:
1. 검색 결과에 없는 내용은 확정적으로 말하지 마세요.
2. 최신 정보가 중요한 질문은 검색 결과 기준으로 정리하세요.
3. 결과는 Markdown 형식으로 작성하세요.
4. 마지막에는 반드시 "참고 링크" 섹션을 포함하세요.
5. 참고 링크에는 제목과 URL을 함께 표시하세요.
6. 오류가 발생했거나 검색 결과가 부족하면 성공한 것처럼 답하지 마세요.
"""


def _safe_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _needs_recent_search(query: str) -> bool:
    normalized = (query or "").lower()

    recent_keywords = [
        "최신",
        "최근",
        "현재",
        "요즘",
        "오늘",
        "올해",
        "뉴스",
        "동향",
        "트렌드",
        "latest",
        "recent",
        "current",
        "news",
        "trend",
    ]

    return any(keyword in normalized for keyword in recent_keywords)


def _build_search_query(query: str) -> str:
    clean_query = (query or "").strip()

    if not clean_query:
        return clean_query

    if _needs_recent_search(clean_query):
        return f"{clean_query} 최신 정보"

    return clean_query


def _extract_sources(results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    sources = []

    for item in results:
        title = item.get("title") or "제목 없음"
        url = item.get("url") or ""
        content = item.get("content") or ""
        published_date = item.get("published_date") or ""

        if not url:
            continue

        sources.append({
            "title": title,
            "url": url,
            "published_date": published_date,
            "snippet": content,
        })

    return sources


def _format_sources_for_prompt(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "검색 결과 없음"

    lines = []

    for idx, item in enumerate(results, start=1):
        title = item.get("title") or "제목 없음"
        url = item.get("url") or ""
        content = item.get("content") or ""
        published_date = item.get("published_date") or ""

        lines.append(f"[출처 {idx}]")
        lines.append(f"제목: {title}")

        if published_date:
            lines.append(f"게시일: {published_date}")

        if url:
            lines.append(f"URL: {url}")

        if content:
            lines.append(f"내용: {content}")

        lines.append("")

    return "\n".join(lines).strip()


def _format_reference_links(sources: List[Dict[str, str]]) -> str:
    if not sources:
        return "참고 링크를 찾지 못했습니다."

    lines = []

    for idx, source in enumerate(sources, start=1):
        title = source.get("title") or "출처"
        url = source.get("url") or ""

        if url:
            lines.append(f"{idx}. [{title}]({url})")
        else:
            lines.append(f"{idx}. {title}")

    return "\n".join(lines)


def _append_reference_links(content: str, sources: List[Dict[str, str]]) -> str:
    """
    모델이 참고 링크 섹션을 빠뜨릴 경우를 대비해서 코드에서 한 번 더 보장합니다.
    """
    content = (content or "").strip()
    reference_links = _format_reference_links(sources)

    if "## 참고 링크" in content or "# 참고 링크" in content or "참고 출처" in content:
        return content

    return f"{content}\n\n## 참고 링크\n{reference_links}".strip()


class WebResearchAgent:
    """
    Tavily API 기반 웹 검색 에이전트

    MCP tool calling에 맡기면 모델이 topic='news'처럼 잘못된 값을 넣을 수 있으므로,
    코드에서 Tavily API를 직접 호출하고 topic은 항상 general로 고정합니다.
    """

    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o")
        self.initialized = False

    async def initialize(self) -> None:
        if self.initialized:
            return

        if not TAVILY_API_KEY:
            raise ValueError("TAVILY_API_KEY 환경 변수가 필요합니다")

        self.initialized = True
        logger.info("[WEB AGENT] [INIT] 초기화 완료")

    async def _search_web(self, query: str) -> Dict[str, Any]:
        search_query = _build_search_query(query)

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": search_query,
            "topic": "general",
            "search_depth": "advanced",
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
            "max_results": 6,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(TAVILY_SEARCH_URL, json=payload)
            response.raise_for_status()
            return response.json()

    async def _summarize_results(
        self,
        original_query: str,
        search_data: Dict[str, Any],
        sources: List[Dict[str, str]],
    ) -> str:
        results = search_data.get("results", []) or []
        tavily_answer = search_data.get("answer") or ""
        sources_text = _format_sources_for_prompt(results)
        reference_links = _format_reference_links(sources)

        today = datetime.now().strftime("%Y-%m-%d")

        user_prompt = f"""
오늘 날짜: {today}

사용자 요청:
{original_query}

Tavily 요약 답변:
{tavily_answer}

웹 검색 결과:
{sources_text}

위 검색 결과를 바탕으로 사용자 요청에 대한 조사 결과를 작성하세요.

작성 형식:
# 조사 결과

## 핵심 요약
- 핵심 내용을 3~5개 bullet로 정리

## 상세 내용
- 중요한 내용을 주제별로 정리
- 날짜, 일정, 개최지, 주요 이슈처럼 사용자가 궁금해할 만한 정보를 포함
- 검색 결과에 없는 내용은 추측하지 말 것

## 참고 링크
{reference_links}
""".strip()

        response = await self.model.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        content = response.content

        if isinstance(content, list):
            return "\n".join(
                item.get("text", "")
                if isinstance(item, dict)
                else str(item)
                for item in content
            ).strip()

        return str(content).strip()

    async def stream(
        self,
        query: str,
    ) -> AsyncIterator[Dict[str, Any]]:
        if not self.initialized:
            await self.initialize()

        try:
            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "🔍 웹 검색을 준비하고 있습니다.",
            }

            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": "🔍 웹에서 관련 정보를 검색하고 있습니다.",
            }

            search_data = await self._search_web(query)
            results = search_data.get("results", []) or []
            sources = _extract_sources(results)

            if not results:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "success": False,
                    "content": "웹 검색 결과를 찾지 못했습니다. 검색어를 조금 더 구체적으로 입력해 주세요.",
                    "data": {
                        "success": False,
                        "error": "NO_SEARCH_RESULTS",
                        "query": query,
                        "sources": [],
                    },
                }
                return

            yield {
                "is_task_complete": False,
                "require_user_input": False,
                "content": f"🧾 검색 결과 {len(results)}개와 참고 링크 {len(sources)}개를 정리하고 있습니다.",
            }

            final_message = await self._summarize_results(
                original_query=query,
                search_data=search_data,
                sources=sources,
            )

            final_message = _append_reference_links(final_message, sources)

            if not final_message:
                yield {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "success": False,
                    "content": "검색은 완료했지만 조사 결과를 정리하지 못했습니다.",
                    "data": {
                        "success": False,
                        "error": "EMPTY_SUMMARY",
                        "query": query,
                        "search_results": results,
                        "sources": sources,
                    },
                }
                return

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "success": True,
                "content": final_message,
                "data": {
                    "success": True,
                    "query": query,
                    "answer": search_data.get("answer"),
                    "search_results": results,
                    "sources": sources,
                    "source_count": len(sources),
                },
            }

        except Exception as e:
            logger.exception("[WEB AGENT] 처리 중 오류 발생")

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "success": False,
                "content": f"웹 조사에 실패했습니다.\n\n오류: {str(e)}",
                "data": {
                    "success": False,
                    "error": str(e),
                    "query": query,
                    "sources": [],
                },
            }


if __name__ == "__main__":
    async def test():
        print("=" * 60)
        print("Web Research Agent 테스트")
        print("=" * 60)

        agent = WebResearchAgent()
        query = "최신 월드컵에 대한 내용을 조사해줘"

        print(f"\n👤 Query: {query}")
        print("-" * 60)

        async for chunk in agent.stream(query):
            content = chunk.get("content", "")
            is_complete = chunk.get("is_task_complete", False)
            success = chunk.get("success")
            data = chunk.get("data") or {}

            if not is_complete:
                print(f"{content}")
            else:
                print(f"\n✅ success: {success}")
                print(f"\n💬 {content}")
                print(f"\n🔗 sources: {len(data.get('sources', []))}개")

    asyncio.run(test())