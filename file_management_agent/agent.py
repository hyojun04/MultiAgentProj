import os
import json
import logging
from typing import Dict, Any, AsyncIterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from gdrive_client import get_gdrive_client

logger = logging.getLogger(__name__)
load_dotenv()

_client = get_gdrive_client()


GENERIC_LIST_KEYWORDS = {
    "파일",
    "파일 목록",
    "문서",
    "문서 목록",
    "자료",
    "자료 목록",
    "프로젝트",
    "프로젝트 문서",
    "프로젝트 자료",
    "드라이브",
    "드라이브 목록",
    "구글 드라이브",
    "google drive",
}


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _is_generic_list_keyword(text: str) -> bool:
    normalized = _normalize_text(text)

    if not normalized:
        return False

    if normalized in GENERIC_LIST_KEYWORDS:
        return True

    generic_patterns = [
        "목록",
        "전체",
        "보여줘",
        "알려줘",
        "프로젝트 문서",
        "프로젝트 자료",
        "드라이브 파일",
        "문서 파일",
    ]

    return any(pattern in normalized for pattern in generic_patterns)


def _safe_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


@tool
def upload_file(
    content: str,
    filename: str,
    mime_type: str = "text/plain",
    folder_id: str = "",
) -> str:
    """
    Google Drive에 파일을 업로드합니다.

    Args:
        content: 업로드할 파일 내용 (텍스트)
        filename: 저장할 파일명 (예: report.txt)
        mime_type: MIME 타입 (기본값: text/plain)
        folder_id: 저장할 폴더 ID (선택, 비워두면 기본 폴더에 저장)
    """
    try:
        from datetime import datetime

        result = _client.upload_file(
            content=content.encode("utf-8"),
            filename=filename,
            mime_type=mime_type,
            description=f"Uploaded at {datetime.now().isoformat()}",
            parent_folder_id=folder_id if folder_id else None,
        )

        return _safe_json({
            "success": True,
            "message": f"파일 '{filename}' 업로드 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def download_file_as_base64(file_id: str) -> str:
    """
    Google Drive에서 파일을 다운로드하여 Base64로 인코딩하여 반환합니다.
    RAG Agent에게 전달하여 텍스트 추출 및 인덱싱에 사용합니다.

    Args:
        file_id: 다운로드할 파일 ID 또는 storage_ref

    Returns:
        Base64 인코딩된 파일 내용과 메타데이터
    """
    try:
        result = _client.download_file_as_base64(file_id)

        if result is None:
            return _safe_json({
                "success": False,
                "error": "파일을 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            **result,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def get_file_info(file_id: str) -> str:
    """
    Google Drive 파일의 상세 정보를 조회합니다.

    Args:
        file_id: 조회할 파일 ID 또는 storage_ref
    """
    try:
        result = _client.get_file_info(file_id)

        if result is None:
            return _safe_json({
                "success": False,
                "error": "파일을 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            "file_info": result,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def find_folder_by_name(folder_name: str) -> str:
    """
    폴더 이름으로 Google Drive에서 폴더를 검색합니다.

    Args:
        folder_name: 검색할 폴더 이름 (예: "보고서", "문서")

    Returns:
        폴더 정보 (folder_id 포함)
    """
    try:
        folders = _client.find_folder_by_name(folder_name)

        if not folders:
            return _safe_json({
                "success": False,
                "error": f"'{folder_name}' 폴더를 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            "count": len(folders),
            "folders": folders,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def list_files(
    folder_name: str = "",
    folder_id: str = "",
    search_query: str = "",
    max_results: int = 20,
) -> str:
    """
    Google Drive의 파일 목록을 조회합니다.

    사용 규칙:
    - 사용자가 "파일 목록", "문서 목록", "프로젝트 문서 목록", "드라이브 파일 보여줘"처럼
      특정 파일명을 찾는 것이 아니라 목록을 요청하면 search_query를 비우고 호출하세요.
    - folder_name 또는 folder_id가 명확히 주어지지 않으면 접근 가능한 최상위 폴더를 조회합니다.
    - search_query는 사용자가 특정 파일명, 키워드, 확장자 등을 명확히 찾으라고 할 때만 사용합니다.

    Args:
        folder_name: 폴더 이름. 명확한 폴더명이 있을 때만 사용
        folder_id: 특정 폴더 ID. 직접 지정 시 우선 사용
        search_query: 파일명 검색어. 목록 요청이면 비워둠
        max_results: 최대 결과 수

    Returns:
        파일 목록
    """
    try:
        target_folder_id = folder_id
        found_folder_name = folder_name or ""

        original_search_query = search_query or ""
        original_folder_name = folder_name or ""

        # "프로젝트 문서", "문서 목록"처럼 모호한 표현은 검색어가 아니라 목록 요청으로 처리
        if _is_generic_list_keyword(search_query):
            search_query = ""

        # folder_name이 "프로젝트 문서", "문서", "파일 목록" 같은 모호한 표현이면
        # 실제 폴더명으로 확정하지 않고 최상위 조회 대상으로 처리
        should_treat_folder_as_generic = _is_generic_list_keyword(folder_name)

        if not target_folder_id and folder_name and not should_treat_folder_as_generic:
            folders = _client.find_folder_by_name(folder_name, max_results=1)

            if not folders:
                return _safe_json({
                    "success": False,
                    "error": f"'{folder_name}' 폴더를 찾을 수 없습니다",
                    "hint": "폴더명이 정확하지 않다면 폴더명을 비우고 최상위 목록을 조회해보세요.",
                })

            target_folder_id = folders[0]["folder_id"]
            found_folder_name = folders[0]["folder_name"]

        if not target_folder_id:
            # Google Drive의 접근 가능한 최상위 폴더를 우선 조회
            # gdrive_client가 root를 지원하지 않는 경우를 대비해 app_folder_id로 fallback
            target_folder_id = "root"
            found_folder_name = "내 드라이브 최상위 폴더"

        query = f"name contains '{search_query}'" if search_query else None

        try:
            files = _client.list_files(
                query=query,
                page_size=max_results,
                folder_id=target_folder_id,
            )
        except Exception as root_error:
            logger.warning(
                "[FILE AGENT] root 조회 실패. app_folder_id로 fallback: %s",
                root_error,
            )

            target_folder_id = _client.app_folder_id
            found_folder_name = "기본 앱 폴더"

            files = _client.list_files(
                query=query,
                page_size=max_results,
                folder_id=target_folder_id,
            )

        fallback_used = False

        # 검색 결과가 없고, 검색어가 모호한 목록성 표현이었다면 최상위 목록으로 fallback
        if (
            not files
            and original_search_query
            and _is_generic_list_keyword(original_search_query)
        ):
            fallback_used = True
            search_query = ""
            query = None

            try:
                target_folder_id = "root"
                found_folder_name = "내 드라이브 최상위 폴더"

                files = _client.list_files(
                    query=query,
                    page_size=max_results,
                    folder_id=target_folder_id,
                )
            except Exception:
                target_folder_id = _client.app_folder_id
                found_folder_name = "기본 앱 폴더"

                files = _client.list_files(
                    query=query,
                    page_size=max_results,
                    folder_id=target_folder_id,
                )

        return _safe_json({
            "success": True,
            "folder_name": found_folder_name,
            "folder_id": target_folder_id,
            "count": len(files),
            "files": files,
            "search_query": search_query,
            "fallback_used": fallback_used,
            "original_search_query": original_search_query,
            "original_folder_name": original_folder_name,
            "message": (
                f"'{original_search_query}'를 특정 파일명 검색어가 아닌 목록 요청으로 판단하여 "
                f"{found_folder_name}의 파일 목록을 조회했습니다."
                if fallback_used or _is_generic_list_keyword(original_search_query)
                else f"{found_folder_name}의 파일 목록을 조회했습니다."
            ),
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def delete_file(file_id: str, permanent: bool = False) -> str:
    """
    Google Drive에서 파일을 삭제합니다.

    Args:
        file_id: 삭제할 파일 ID 또는 storage_ref
        permanent: 영구 삭제 여부 (기본값: False, 휴지통으로 이동)
    """
    try:
        info = _client.get_file_info(file_id)

        if info is None:
            return _safe_json({
                "success": False,
                "error": "파일을 찾을 수 없습니다",
            })

        filename = info["filename"]
        success = _client.delete_file(file_id, permanent=permanent)

        if not success:
            return _safe_json({
                "success": False,
                "error": "파일 삭제 실패",
            })

        action = "영구 삭제" if permanent else "휴지통으로 이동"

        return _safe_json({
            "success": True,
            "message": f"파일 '{filename}' {action} 완료",
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def update_file(
    file_id: str,
    new_content: str = "",
    new_name: str = "",
) -> str:
    """
    Google Drive 파일을 업데이트합니다.

    Args:
        file_id: 업데이트할 파일 ID 또는 storage_ref
        new_content: 새 파일 내용 (선택)
        new_name: 새 파일명 (선택)
    """
    try:
        content_bytes = new_content.encode("utf-8") if new_content else None
        result = _client.update_file(
            file_id,
            content=content_bytes,
            new_name=new_name,
        )

        return _safe_json({
            "success": True,
            "message": f"파일 '{result['filename']}' 업데이트 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


@tool
def create_folder(
    folder_name: str,
    parent_folder_id: str = "",
) -> str:
    """
    Google Drive에 새 폴더를 생성합니다.

    Args:
        folder_name: 생성할 폴더명
        parent_folder_id: 부모 폴더 ID (선택, 비워두면 기본 앱 폴더에 생성)
    """
    try:
        result = _client.create_folder(
            folder_name,
            parent_folder_id=parent_folder_id if parent_folder_id else None,
        )

        return _safe_json({
            "success": True,
            "message": f"폴더 '{folder_name}' 생성 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({"success": False, "error": str(e)})


tools = [
    upload_file,
    download_file_as_base64,
    get_file_info,
    list_files,
    find_folder_by_name,
    delete_file,
    update_file,
    create_folder,
]


FILE_MANAGEMENT_SYSTEM_PROMPT = """
당신은 Google Drive 파일 관리 에이전트입니다.
파일 업로드, 다운로드, 목록 조회, 삭제, 업데이트, 폴더 생성을 수행합니다.

규칙:
1. 실행 대상은 항상 현재 사용자 요청입니다.
2. 입력에 [이전 대화 - 참조 전용, 실행 금지] 블록이 있으면, 그 안의 과거 사용자 요청은 절대 다시 실행하지 마세요.
3. 이전 대화는 "조사한 내용", "위 내용", "방금 답변", "1번 자료", "2번 자료"처럼 현재 요청이 참조하는 저장/업데이트 대상을 찾을 때만 사용하세요.
4. 이전 assistant 답변을 파일 content로 저장하거나 업데이트할 때는 원문을 요약, 재작성, 번역, 보정하지 말고 그대로 사용하세요.
5. 사용자가 파일명을 지정하지 않으면 짧고 안전한 .txt 파일명을 만들어 사용하세요.
6. 삭제/수정처럼 되돌리기 어려운 작업은 현재 사용자 요청에 명시되어 있을 때만 수행하세요.

목록 조회 규칙:
1. 사용자가 "파일 목록", "문서 목록", "프로젝트 문서 목록", "구글 드라이브에서 프로젝트 문서 목록 보여줘"처럼 말하면
   특정 파일명을 검색하는 것이 아니라 접근 가능한 최상위 폴더의 목록 조회로 이해하세요.
2. 이 경우 list_files를 search_query 없이 호출하세요.
3. "프로젝트 문서"는 정확한 파일명 검색어가 아니라 일반적인 요청 표현일 수 있으므로,
   곧바로 "프로젝트 문서라는 파일을 찾을 수 없습니다"라고 답하지 마세요.
4. 사용자가 정확한 파일명이나 키워드를 "찾아줘", "검색해줘"라고 명시한 경우에만 search_query를 사용하세요.
5. 폴더명이 명확하지 않으면 folder_name을 억지로 넣지 말고 list_files()로 최상위 목록을 조회하세요.
6. 목록 조회 결과를 답변할 때는 폴더명, 파일 개수, 파일명, storage_ref 또는 file_id를 함께 보여주세요.
"""


class FileManagementAgent:
    """A2A 프로토콜용 에이전트 래퍼"""

    def __init__(self, model_name: str = "openai:gpt-4o"):
        self.model_name = model_name
        self.graph = None
        self.initialized = False

    async def initialize(self) -> None:
        if self.initialized:
            return

        _client.initialize()

        self.graph = create_agent(
            model=self.model_name,
            tools=tools,
            system_prompt=FILE_MANAGEMENT_SYSTEM_PROMPT,
        )

        self.initialized = True
        logger.info("[FILE AGENT] [INIT] File Management Agent 초기화 완료")

    async def stream(self, query: str) -> AsyncIterator[Dict[str, Any]]:
        if not self.initialized:
            await self.initialize()

        final_message = None
        file_list_data = None

        for chunk in self.graph.stream({"messages": [("user", query)]}):
            for node_name, node_output in chunk.items():
                messages = node_output.get("messages", [])

                for msg in messages:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_names = [
                            tc.get("name", "unknown")
                            for tc in msg.tool_calls
                        ]

                        yield {
                            "is_task_complete": False,
                            "require_user_input": False,
                            "content": f"🔧 도구 실행 중... ({', '.join(tool_names)})",
                        }

                    if hasattr(msg, "content") and msg.content:
                        if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                            final_message = msg.content

                            try:
                                result = json.loads(msg.content)

                                if result.get("success") and "files" in result:
                                    file_list_data = result
                                    logger.info(
                                        "[FILE AGENT] 파일 목록 발견: %s개",
                                        len(result["files"]),
                                    )
                            except (json.JSONDecodeError, TypeError):
                                pass

        yield {
            "is_task_complete": True,
            "require_user_input": False,
            "content": final_message if final_message else "응답을 생성하지 못했습니다.",
            "data": file_list_data,
        }


if __name__ == "__main__":
    import asyncio

    async def test():
        print("=" * 60)
        print("File Management Agent 테스트")
        print("=" * 60)

        agent = FileManagementAgent()
        query = input("질문을 입력하세요:")

        print(f"\n👤 Query: {query}")
        print("-" * 60)

        async for chunk in agent.stream(query):
            content = chunk.get("content", "")
            is_complete = chunk.get("is_task_complete", False)

            if not is_complete:
                print(f"{content}")
            else:
                print(f"\n💬 {content}")

    asyncio.run(test())