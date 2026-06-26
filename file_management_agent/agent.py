import re
import json
import logging
from typing import Dict, Any, AsyncIterator, List

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

from gdrive_client import get_gdrive_client

logger = logging.getLogger(__name__)
load_dotenv()

_client = get_gdrive_client()

APP_FOLDER_NAME = "FileManagementAgent"
MAX_FILE_GUARD_RESULTS = 1000


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


def _safe_json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False)


def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()


def _ensure_client_initialized() -> None:
    if not getattr(_client, "initialized", False):
        _client.initialize()


def _get_app_folder_id() -> str:
    _ensure_client_initialized()

    if not getattr(_client, "app_folder_id", None):
        raise RuntimeError(f"{APP_FOLDER_NAME} 폴더 ID를 확인할 수 없습니다.")

    return _client.app_folder_id


def _escape_drive_query_value(value: str) -> str:
    """
    Google Drive query에 들어갈 문자열을 안전하게 escape합니다.
    """
    return (value or "").replace("\\", "\\\\").replace("'", "\\'")


def _parse_storage_ref(file_id: str) -> str:
    """
    gdrive://file/{FILE_ID} 형태와 순수 file_id를 모두 처리합니다.
    """
    if file_id and file_id.startswith("gdrive://file/"):
        return file_id.replace("gdrive://file/", "")
    return file_id or ""


def _coerce_max_results(max_results: int) -> int:
    """
    모델이 max_results를 이상하게 넣어도 Google API 호출이 터지지 않도록 보정합니다.
    """
    try:
        value = int(max_results)
    except (TypeError, ValueError):
        value = 20

    return max(1, min(value, 100))


def _is_generic_list_keyword(text: str) -> bool:
    normalized = _normalize_text(text)

    if not normalized:
        return False

    if normalized in GENERIC_LIST_KEYWORDS:
        return True

    generic_patterns = [
        "목록",
        "전체 목록",
        "전체 파일",
        "보여줘",
        "알려줘",
        "프로젝트 문서",
        "프로젝트 자료",
        "드라이브 파일",
        "문서 파일",
    ]

    return any(pattern in normalized for pattern in generic_patterns)


def _clean_search_keyword(keyword: str) -> str:
    """
    사용자가 자연어로 넘긴 검색어에서 실제 파일명 검색에 필요한 키워드만 남깁니다.

    예:
    - "'AI' 또는 '인공지능' 관련 키워드가 포함된 모든 파일" -> AI, 인공지능
    - "드라이브에서 AI 관련 파일 찾아줘" -> AI
    - "최신 축구 정보 파일 찾아줘" -> 최신 축구 정보
    """
    keyword = (keyword or "").strip()
    keyword = keyword.strip("'\"“”‘’`")

    # 앞쪽의 장소/대상 표현 제거
    keyword = re.sub(
        r"^(구글\s*드라이브|google\s*drive|drive|드라이브|내\s*드라이브)(에서|의|내)?\s*",
        "",
        keyword,
        flags=re.IGNORECASE,
    ).strip()

    # 뒤쪽의 요청/설명 표현 제거
    keyword = re.sub(
        r"\s*(관련|키워드|포함|포함된|모든|전체|파일|문서|자료|찾아줘|찾아|검색|보여줘|알려줘|목록).*$",
        "",
        keyword,
        flags=re.IGNORECASE,
    ).strip()

    return keyword


def _extract_search_keywords(search_query: str) -> List[str]:
    """
    단일 검색어뿐 아니라 'AI 또는 인공지능' 같은 다중 키워드를 지원합니다.
    """
    text = (search_query or "").strip()

    if not text:
        return []

    # 따옴표로 감싼 키워드 우선 추출: 'AI', "인공지능"
    quoted_keywords = re.findall(r"[\"'“”‘’]([^\"'“”‘’]+)[\"'“”‘’]", text)
    if quoted_keywords:
        keywords = [_clean_search_keyword(k) for k in quoted_keywords]
        return [k for k in keywords if k and not _is_generic_list_keyword(k)]

    # 또는/or/,/| 기준 분리
    parts = re.split(r"\s+또는\s+|\s+혹은\s+|\s+or\s+|\s+OR\s+|,|\||/", text)
    keywords = [_clean_search_keyword(part) for part in parts]

    result = []
    for keyword in keywords:
        if not keyword:
            continue
        if _is_generic_list_keyword(keyword):
            continue
        if keyword not in result:
            result.append(keyword)

    return result


def _build_name_query(search_query: str) -> str | None:
    """
    Google Drive 파일명 검색 query 생성.
    여러 키워드가 있으면 OR 조건으로 검색합니다.
    """
    keywords = _extract_search_keywords(search_query)

    if not keywords:
        return None

    name_conditions = [
        f"name contains '{_escape_drive_query_value(keyword)}'"
        for keyword in keywords
    ]

    return " or ".join(name_conditions)


def _is_directly_in_app_folder(file_id: str) -> bool:
    """
    FileManagementAgent 폴더 바로 아래에 있는 파일/폴더인지 확인합니다.
    외부 파일 ID로 조회/수정/삭제/다운로드하지 않기 위한 안전장치입니다.
    """
    app_folder_id = _get_app_folder_id()
    parsed_file_id = _parse_storage_ref(file_id)

    if not parsed_file_id:
        return False

    files = _client.list_files(
        query=None,
        page_size=MAX_FILE_GUARD_RESULTS,
        folder_id=app_folder_id,
    )

    return any(item.get("file_id") == parsed_file_id for item in files)


def _assert_file_in_app_folder(file_id: str) -> str:
    parsed_file_id = _parse_storage_ref(file_id)

    if not _is_directly_in_app_folder(parsed_file_id):
        raise PermissionError(f"{APP_FOLDER_NAME} 폴더 내부의 파일만 접근할 수 있습니다.")

    return parsed_file_id


def _safe_parent_folder_id(parent_folder_id: str = "") -> str | None:
    """
    업로드/폴더 생성 시 외부 parent_folder_id가 들어와도 무시하고 앱 폴더에 저장합니다.
    """
    app_folder_id = _get_app_folder_id()
    parsed_parent_id = _parse_storage_ref(parent_folder_id)

    if not parsed_parent_id:
        return None

    if parsed_parent_id == app_folder_id:
        return app_folder_id

    logger.warning(
        "[FILE AGENT] 외부 parent_folder_id는 무시하고 %s 폴더에 저장합니다.",
        APP_FOLDER_NAME,
    )
    return None


@tool
def upload_file(
    content: str,
    filename: str,
    mime_type: str = "text/plain",
    folder_id: str = "",
) -> str:
    """
    FileManagementAgent 폴더 내부에 파일을 업로드합니다.

    Args:
        content: 업로드할 파일 내용
        filename: 저장할 파일명
        mime_type: MIME 타입
        folder_id: 저장할 폴더 ID. 외부 ID는 무시되고 FileManagementAgent 폴더에 저장
    """
    try:
        from datetime import datetime

        _ensure_client_initialized()

        safe_folder_id = _safe_parent_folder_id(folder_id)

        result = _client.upload_file(
            content=content.encode("utf-8"),
            filename=filename,
            mime_type=mime_type,
            description=f"Uploaded at {datetime.now().isoformat()}",
            parent_folder_id=safe_folder_id,
        )

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "message": f"파일 '{filename}' 업로드 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def download_file_as_base64(file_id: str) -> str:
    """
    FileManagementAgent 폴더 내부 파일을 다운로드하여 Base64로 인코딩하여 반환합니다.

    Args:
        file_id: 다운로드할 파일 ID 또는 storage_ref
    """
    try:
        _ensure_client_initialized()

        safe_file_id = _assert_file_in_app_folder(file_id)
        result = _client.download_file_as_base64(safe_file_id)

        if result is None:
            return _safe_json({
                "success": False,
                "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                "error": "파일을 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            **result,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def get_file_info(file_id: str) -> str:
    """
    FileManagementAgent 폴더 내부 파일의 상세 정보를 조회합니다.

    Args:
        file_id: 조회할 파일 ID 또는 storage_ref
    """
    try:
        _ensure_client_initialized()

        safe_file_id = _assert_file_in_app_folder(file_id)
        result = _client.get_file_info(safe_file_id)

        if result is None:
            return _safe_json({
                "success": False,
                "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                "error": "파일을 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "file_info": result,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def find_folder_by_name(folder_name: str) -> str:
    """
    FileManagementAgent 폴더 바로 아래에서만 폴더 이름을 검색합니다.
    Google Drive 전체를 검색하지 않습니다.

    Args:
        folder_name: 검색할 하위 폴더 이름

    Returns:
        폴더 정보
    """
    try:
        app_folder_id = _get_app_folder_id()

        query = (
            f"name contains '{_escape_drive_query_value(folder_name)}' "
            "and mimeType='application/vnd.google-apps.folder'"
        )

        folders = _client.list_files(
            query=query,
            page_size=10,
            folder_id=app_folder_id,
        )

        folder_results = [
            {
                "folder_id": folder.get("file_id"),
                "storage_ref": folder.get("storage_ref"),
                "folder_name": folder.get("filename"),
                "mime_type": folder.get("mime_type"),
                "created_at": folder.get("created_at"),
                "updated_at": folder.get("updated_at"),
                "web_view_link": folder.get("web_view_link"),
            }
            for folder in folders
            if folder.get("mime_type") == "application/vnd.google-apps.folder"
        ]

        if not folder_results:
            return _safe_json({
                "success": False,
                "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                "error": f"{APP_FOLDER_NAME} 폴더 내부에서 '{folder_name}' 폴더를 찾을 수 없습니다",
            })

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "count": len(folder_results),
            "folders": folder_results,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def list_files(
    folder_name: str = "",
    folder_id: str = "",
    search_query: str = "",
    max_results: int = 20,
) -> str:
    """
    FileManagementAgent 폴더 내부의 파일 목록을 조회합니다.

    중요:
    - Google Drive 전체 또는 최상위 폴더는 절대 조회하지 않습니다.
    - 직접 전달된 folder_id는 신뢰하지 않고 기본적으로 무시합니다.
    - folder_name이 있으면 FileManagementAgent 폴더 바로 아래의 하위 폴더만 검색합니다.
    - search_query는 FileManagementAgent 폴더 내부 파일명 검색에만 사용합니다.
    - "AI 또는 인공지능"처럼 여러 키워드도 지원합니다.

    Args:
        folder_name: 하위 폴더 이름. 꼭 필요한 경우에만 사용
        folder_id: 직접 전달된 폴더 ID. 외부 접근 방지를 위해 기본적으로 무시
        search_query: 파일명 검색어
        max_results: 최대 결과 수
    """
    try:
        app_folder_id = _get_app_folder_id()
        safe_max_results = _coerce_max_results(max_results)

        original_search_query = search_query or ""
        original_folder_name = folder_name or ""
        original_folder_id = folder_id or ""

        target_folder_id = app_folder_id
        found_folder_name = APP_FOLDER_NAME

        if original_folder_id and _parse_storage_ref(original_folder_id) != app_folder_id:
            logger.warning(
                "[FILE AGENT] 직접 전달된 folder_id는 무시하고 %s 폴더 내부만 조회합니다.",
                APP_FOLDER_NAME,
            )

        # folder_name이 있으면 FileManagementAgent 폴더 내부의 하위 폴더만 검색
        if folder_name and not _is_generic_list_keyword(folder_name):
            folder_query = (
                f"name contains '{_escape_drive_query_value(folder_name)}' "
                "and mimeType='application/vnd.google-apps.folder'"
            )

            folders = _client.list_files(
                query=folder_query,
                page_size=1,
                folder_id=app_folder_id,
            )

            if not folders:
                return _safe_json({
                    "success": False,
                    "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                    "error": f"{APP_FOLDER_NAME} 폴더 내부에서 '{folder_name}' 폴더를 찾을 수 없습니다",
                    "original_folder_name": original_folder_name,
                })

            target_folder_id = folders[0]["file_id"]
            found_folder_name = folders[0]["filename"]

        extracted_keywords = _extract_search_keywords(search_query)

        # 목록성 표현이면 검색어 제거
        if _is_generic_list_keyword(search_query) and not extracted_keywords:
            search_query = ""

        query = _build_name_query(search_query)

        files = _client.list_files(
            query=query,
            page_size=safe_max_results,
            folder_id=target_folder_id,
        )

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "folder_name": found_folder_name,
            "folder_id": target_folder_id,
            "count": len(files),
            "files": files,
            "search_query": search_query,
            "extracted_keywords": extracted_keywords,
            "original_search_query": original_search_query,
            "original_folder_name": original_folder_name,
            "original_folder_id": original_folder_id,
            "message": (
                f"{APP_FOLDER_NAME} 폴더 내부에서 "
                f"{', '.join(extracted_keywords)} 키워드가 포함된 파일을 조회했습니다."
                if extracted_keywords
                else f"{APP_FOLDER_NAME} 폴더 내부 파일 목록을 조회했습니다."
            ),
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def delete_file(file_id: str, permanent: bool = False) -> str:
    """
    FileManagementAgent 폴더 내부 파일만 삭제합니다.

    Args:
        file_id: 삭제할 파일 ID 또는 storage_ref
        permanent: 영구 삭제 여부
    """
    try:
        _ensure_client_initialized()

        safe_file_id = _assert_file_in_app_folder(file_id)
        info = _client.get_file_info(safe_file_id)

        if info is None:
            return _safe_json({
                "success": False,
                "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                "error": "파일을 찾을 수 없습니다",
            })

        filename = info["filename"]
        success = _client.delete_file(safe_file_id, permanent=permanent)

        if not success:
            return _safe_json({
                "success": False,
                "scope": f"{APP_FOLDER_NAME} 폴더 내부",
                "error": "파일 삭제 실패",
            })

        action = "영구 삭제" if permanent else "휴지통으로 이동"

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "message": f"파일 '{filename}' {action} 완료",
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def update_file(
    file_id: str,
    new_content: str = "",
    new_name: str = "",
) -> str:
    """
    FileManagementAgent 폴더 내부 파일만 업데이트합니다.

    Args:
        file_id: 업데이트할 파일 ID 또는 storage_ref
        new_content: 새 파일 내용
        new_name: 새 파일명
    """
    try:
        _ensure_client_initialized()

        safe_file_id = _assert_file_in_app_folder(file_id)
        content_bytes = new_content.encode("utf-8") if new_content else None

        result = _client.update_file(
            safe_file_id,
            content=content_bytes,
            new_name=new_name,
        )

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "message": f"파일 '{result['filename']}' 업데이트 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


@tool
def create_folder(
    folder_name: str,
    parent_folder_id: str = "",
) -> str:
    """
    FileManagementAgent 폴더 내부에 새 폴더를 생성합니다.

    Args:
        folder_name: 생성할 폴더명
        parent_folder_id: 부모 폴더 ID. 외부 ID는 무시되고 FileManagementAgent 폴더에 생성
    """
    try:
        _ensure_client_initialized()

        safe_parent_id = _safe_parent_folder_id(parent_folder_id)

        result = _client.create_folder(
            folder_name,
            parent_folder_id=safe_parent_id,
        )

        return _safe_json({
            "success": True,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "message": f"폴더 '{folder_name}' 생성 완료",
            **result,
        })
    except Exception as e:
        return _safe_json({
            "success": False,
            "scope": f"{APP_FOLDER_NAME} 폴더 내부",
            "error": str(e),
        })


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

가장 중요한 원칙:
- 모든 작업 범위는 Google Drive의 "FileManagementAgent" 폴더 내부로만 제한합니다.
- 사용자가 "구글 드라이브", "드라이브", "내 드라이브", "Drive에서", "Google Drive에서"라고 말해도 FileManagementAgent 폴더 내부를 의미하는 것으로 해석하세요.
- 절대 Google Drive 전체나 내 드라이브 최상위 폴더를 탐색하지 마세요.
- 항상 FileManagementAgent 폴더를 기준 폴더로 생각하세요.
- 사용자가 별도의 폴더명을 말하더라도, 그 폴더는 FileManagementAgent 폴더 안에 있는 하위 폴더로만 해석하세요.
- FileManagementAgent 폴더 밖의 파일이나 폴더는 존재하지 않는 것처럼 취급하세요.

도구 사용 규칙:
1. 파일 목록을 보여달라는 요청은 기본적으로 list_files()를 folder_name, folder_id, search_query 없이 호출하세요.
   예:
   - "구글 드라이브 파일 목록 보여줘"
   - "프로젝트 문서 목록 보여줘"
   - "문서 목록 보여줘"
   - "파일 뭐 있어?"
   위 요청들은 모두 FileManagementAgent 폴더 내부 목록 조회입니다.

2. 특정 키워드로 파일을 찾으라는 요청은 list_files(search_query=...)를 사용하세요.
   예:
   - "월드컵 파일 찾아줘" → list_files(search_query="월드컵")
   - "최신 축구 정보 파일 찾아줘" → list_files(search_query="최신 축구 정보")
   - "AI 또는 인공지능 관련 파일 찾아줘" → list_files(search_query="AI 또는 인공지능")

3. "프로젝트 문서", "문서", "파일", "자료", "드라이브" 같은 일반 표현은 search_query로 사용하지 마세요.
   이런 표현은 특정 파일명이 아니라 목록 요청으로 판단하세요.

4. find_folder_by_name 도구를 사용하더라도 FileManagementAgent 폴더 내부의 하위 폴더만 검색됩니다.
   Google Drive 전체 폴더 검색으로 생각하지 마세요.

5. folder_id를 직접 사용할 때는 반드시 이전 list_files 또는 find_folder_by_name 결과에서 나온 folder_id만 사용하세요.
   그래도 도구 내부에서 직접 전달된 외부 folder_id는 무시하고 FileManagementAgent 폴더 기준으로 처리합니다.

6. upload_file에서 folder_id가 명확하지 않으면 항상 비워두세요.
   folder_id를 비워두면 FileManagementAgent 폴더에 저장됩니다.

7. 파일을 저장할 때는 사용자가 지정한 파일명을 그대로 사용하세요.
   사용자가 파일명을 지정하지 않으면 짧고 안전한 .txt 파일명을 만들어 사용하세요.

8. 삭제/수정처럼 되돌리기 어려운 작업은 현재 사용자 요청에 명시되어 있을 때만 수행하세요.

이전 대화 처리 규칙:
1. 실행 대상은 항상 현재 사용자 요청입니다.
2. 입력에 [이전 대화 - 참조 전용, 실행 금지] 블록이 있으면, 그 안의 과거 사용자 요청은 절대 다시 실행하지 마세요.
3. 이전 대화는 "조사한 내용", "위 내용", "방금 답변", "1번 자료", "2번 자료"처럼 현재 요청이 참조하는 저장/업데이트 대상을 찾을 때만 사용하세요.
4. 이전 assistant 답변을 파일 content로 저장하거나 업데이트할 때는 원문을 요약, 재작성, 번역, 보정하지 말고 그대로 사용하세요.

답변 규칙:
1. 목록 조회 결과를 답변할 때는 "FileManagementAgent 폴더 내부 기준"이라고 명시하세요.
2. 파일 목록에는 파일명, 파일 타입, storage_ref 또는 file_id, 링크가 있으면 링크를 함께 보여주세요.
3. 검색 결과가 없으면 "FileManagementAgent 폴더 내부에서 찾을 수 없습니다."라고 답하세요.
   절대 "Google Drive 전체에서 찾을 수 없습니다"라고 말하지 마세요.
"""


class FileManagementAgent:
    """A2A 프로토콜용 에이전트 래퍼"""

    def __init__(self, model_name: str = "openai:gpt-4.1"):
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