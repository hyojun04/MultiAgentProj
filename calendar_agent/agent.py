import os
import logging
import asyncio
from datetime import datetime, timedelta
from typing import AsyncIterator, Dict, Any, Optional, Literal
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

logger = logging.getLogger(__name__)

load_dotenv()

# 일정 조회 + 일정 등록을 위해 calendar.events 사용
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "Asia/Seoul")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 기존 Drive Agent의 credentials.json을 재사용
_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH",
    os.path.join(_PROJECT_ROOT, "file_management_agent", "credentials.json")
)

# Calendar 전용 token.json 사용
# Drive token과 scope가 다르기 때문에 token은 분리하는 것이 안전함
_TOKEN_PATH = os.getenv(
    "CALENDAR_TOKEN_PATH",
    os.path.join(_PROJECT_ROOT, "calendar_agent", "token.json")
)


class CalendarRequest(BaseModel):
    action: Literal["list_events", "create_event", "unsupported"] = Field(
        description="list_events: 일정 조회, create_event: 일정 등록, unsupported: 지원하지 않는 요청"
    )
    range_type: Optional[Literal["today", "week"]] = Field(
        default=None,
        description="list_events일 때 today 또는 week"
    )
    title: Optional[str] = Field(
        default=None,
        description="create_event일 때 일정 제목"
    )
    date: Optional[str] = Field(
        default=None,
        description="create_event일 때 날짜. YYYY-MM-DD 형식"
    )
    start_time: Optional[str] = Field(
        default=None,
        description="create_event일 때 시작 시간. HH:MM 24시간 형식"
    )
    end_time: Optional[str] = Field(
        default=None,
        description="create_event일 때 종료 시간. HH:MM 24시간 형식. 없으면 1시간 뒤로 처리"
    )
    reason: Optional[str] = Field(
        default=None,
        description="판단 이유 또는 부족한 정보"
    )


class GoogleCalendarClient:
    def __init__(self):
        self.service = None
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return

        creds = None

        if os.path.exists(_TOKEN_PATH):
            creds = Credentials.from_authorized_user_file(_TOKEN_PATH, SCOPES)

            # 기존 token에 필요한 scope가 없으면 재인증
            if not creds.has_scopes(SCOPES):
                logger.warning("[CALENDAR AGENT] 기존 token scope 부족. 재인증 필요")
                creds = None

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(_CREDENTIALS_PATH):
                    raise FileNotFoundError(
                        f"credentials.json을 찾을 수 없습니다: {_CREDENTIALS_PATH}"
                    )

                flow = InstalledAppFlow.from_client_secrets_file(
                    _CREDENTIALS_PATH,
                    SCOPES
                )
                creds = flow.run_local_server(port=0)

            with open(_TOKEN_PATH, "w", encoding="utf-8") as token:
                token.write(creds.to_json())

        self.service = build("calendar", "v3", credentials=creds)
        self._initialized = True
        logger.info("[CALENDAR AGENT] Google Calendar 클라이언트 초기화 완료")

    def list_events(self, range_type: str = "today") -> str:
        self._ensure_initialized()

        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)

        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if range_type == "week":
            # 이번 주 월요일 00:00 ~ 다음 주 월요일 00:00
            start_dt = today_start - timedelta(days=today_start.weekday())
            end_dt = start_dt + timedelta(days=7)
            title = "이번 주"
        else:
            start_dt = today_start
            end_dt = today_start + timedelta(days=1)
            title = "오늘"

        logger.info(
            f"[CALENDAR AGENT] 일정 조회: {title}, "
            f"{start_dt.isoformat()} ~ {end_dt.isoformat()}"
        )

        events_result = self.service.events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
            timeZone=TIMEZONE,
        ).execute()

        events = events_result.get("items", [])

        if not events:
            return f"📅 {title} 등록된 일정이 없습니다."

        lines = [f"📅 **{title} 일정**\n"]

        for i, event in enumerate(events, 1):
            summary = event.get("summary", "제목 없음")
            start = event.get("start", {})
            end = event.get("end", {})

            start_value = start.get("dateTime") or start.get("date")
            end_value = end.get("dateTime") or end.get("date")

            start_text = self._format_datetime(start_value)
            end_text = self._format_datetime(end_value)

            lines.append(f"{i}. **{summary}**")
            lines.append(f"   - 시간: {start_text} ~ {end_text}")

            html_link = event.get("htmlLink")
            if html_link:
                lines.append(f"   - 링크: {html_link}")

            lines.append("")

        return "\n".join(lines)

    def create_event(
        self,
        title: str,
        date: str,
        start_time: str,
        end_time: Optional[str] = None,
    ) -> str:
        self._ensure_initialized()

        tz = ZoneInfo(TIMEZONE)

        start_dt = datetime.strptime(
            f"{date} {start_time}",
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=tz)

        if end_time:
            end_dt = datetime.strptime(
                f"{date} {end_time}",
                "%Y-%m-%d %H:%M"
            ).replace(tzinfo=tz)
        else:
            end_dt = start_dt + timedelta(hours=1)

        if end_dt <= start_dt:
            return "일정 종료 시간이 시작 시간보다 늦어야 합니다."

        event_body = {
            "summary": title,
            "start": {
                "dateTime": start_dt.isoformat(),
                "timeZone": TIMEZONE,
            },
            "end": {
                "dateTime": end_dt.isoformat(),
                "timeZone": TIMEZONE,
            },
        }

        created_event = self.service.events().insert(
            calendarId="primary",
            body=event_body
        ).execute()

        link = created_event.get("htmlLink", "")

        return (
            "✅ Google Calendar에 일정을 등록했습니다.\n\n"
            f"- 제목: {title}\n"
            f"- 날짜: {date}\n"
            f"- 시간: {start_time} ~ {end_time or end_dt.strftime('%H:%M')}\n"
            f"- 링크: {link}"
        )

    def _format_datetime(self, value: Optional[str]) -> str:
        if not value:
            return "시간 정보 없음"

        try:
            if "T" in value:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.astimezone(ZoneInfo(TIMEZONE)).strftime("%Y-%m-%d %H:%M")
            return value
        except Exception:
            return value


class CalendarAgent:
    """
    Google Calendar Agent

    지원 기능:
    1. list_events
       - 오늘 일정 조회
       - 이번 주 일정 조회

    2. create_event
       - 제목, 날짜, 시작 시간, 종료 시간으로 일정 등록
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.calendar_client = GoogleCalendarClient()

    def parse_request(self, query: str) -> CalendarRequest:
        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)

        messages = [
            {
                "role": "system",
                "content": f"""
당신은 Google Calendar 요청 분석기입니다.
사용자 요청을 분석하여 list_events, create_event, unsupported 중 하나로 분류하세요.

현재 날짜와 시간:
- 오늘 날짜: {now.strftime("%Y-%m-%d")}
- 현재 시간: {now.strftime("%H:%M")}
- 시간대: {TIMEZONE}

지원 기능:
1. list_events
- 오늘 일정 조회
- 이번 주 일정 조회

2. create_event
- 제목, 날짜, 시작 시간, 종료 시간으로 일정 등록
- 날짜는 YYYY-MM-DD 형식
- 시간은 HH:MM 24시간 형식
- "내일", "다음 주 월요일" 같은 상대 날짜는 현재 날짜 기준으로 계산
- "오후 2시"는 14:00으로 변환
- 종료 시간이 없으면 end_time은 null로 둔다

분류 규칙:
- "오늘 일정", "오늘 캘린더", "오늘 스케줄" → list_events, range_type=today
- "이번 주 일정", "이번주 일정", "이번 주 캘린더" → list_events, range_type=week
- "등록해줘", "추가해줘", "일정 잡아줘", "캘린더에 넣어줘" → create_event
- 지원하지 않는 요청 → unsupported

응답은 반드시 구조화된 필드에 맞게 작성하세요.
"""
            },
            {
                "role": "user",
                "content": query
            }
        ]

        structured_llm = self.llm.with_structured_output(CalendarRequest)
        return structured_llm.invoke(messages)

    def run(self, query: str) -> str:
        logger.info(f"[CALENDAR AGENT] 요청 수신: {query}")

        parsed = self.parse_request(query)
        logger.info(f"[CALENDAR AGENT] 분석 결과: {parsed.model_dump()}")

        if parsed.action == "list_events":
            range_type = parsed.range_type or "today"
            return self.calendar_client.list_events(range_type=range_type)

        if parsed.action == "create_event":
            missing = []

            if not parsed.title:
                missing.append("제목")
            if not parsed.date:
                missing.append("날짜")
            if not parsed.start_time:
                missing.append("시작 시간")

            if missing:
                return (
                    "일정 등록에 필요한 정보가 부족합니다.\n"
                    f"부족한 정보: {', '.join(missing)}\n\n"
                    "예시: 2026-06-26 14:00~15:00에 'AI 에이전트 회의' 일정 등록해줘."
                )

            return self.calendar_client.create_event(
                title=parsed.title,
                date=parsed.date,
                start_time=parsed.start_time,
                end_time=parsed.end_time,
            )

        return (
            "현재 Calendar Agent는 다음 기능만 지원합니다.\n\n"
            "1. 오늘 일정 조회\n"
            "2. 이번 주 일정 조회\n"
            "3. 제목, 날짜, 시작 시간, 종료 시간으로 일정 등록"
        )

    async def stream(
        self,
        query: str
    ) -> AsyncIterator[Dict[str, Any]]:
        yield {
            "is_task_complete": False,
            "require_user_input": False,
            "content": "📅 캘린더 요청을 분석하고 있습니다..."
        }

        try:
            answer = await asyncio.to_thread(self.run, query)

            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": answer
            }

        except Exception as e:
            logger.error(f"[CALENDAR AGENT] [ERROR] 처리 실패: {e}")
            yield {
                "is_task_complete": True,
                "require_user_input": False,
                "content": f"Calendar Agent 처리 중 오류가 발생했습니다: {str(e)}"
            }