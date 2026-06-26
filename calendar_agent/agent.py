import os
import re
import logging
import asyncio
from datetime import datetime, timedelta, time
from difflib import SequenceMatcher
from typing import AsyncIterator, Dict, Any, Optional, Literal, List, Tuple
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

# 일정 조회/등록/수정/삭제를 위해 calendar.events 사용
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

TIMEZONE = os.getenv("CALENDAR_TIMEZONE", "Asia/Seoul")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 기존 Drive Agent의 credentials.json 재사용
_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_CREDENTIALS_PATH",
    os.path.join(_PROJECT_ROOT, "file_management_agent", "credentials.json")
)

# Calendar 전용 token.json 사용
_TOKEN_PATH = os.getenv(
    "CALENDAR_TOKEN_PATH",
    os.path.join(_PROJECT_ROOT, "calendar_agent", "token.json")
)


class CalendarRequest(BaseModel):
    action: Literal[
        "list_events",
        "search_events",
        "create_event",
        "update_event",
        "delete_event",
        "find_free_time",
        "summarize_events",
        "unsupported",
    ] = Field(
        description=(
            "list_events: 일정 조회, search_events: 일정 검색, create_event: 일정 등록, "
            "update_event: 일정 수정, delete_event: 일정 삭제, find_free_time: 빈 시간 확인, "
            "summarize_events: 일정 요약, unsupported: 지원하지 않는 요청"
        )
    )

    range_type: Optional[Literal["today", "week", "custom"]] = Field(
        default=None,
        description="조회/검색/요약/빈 시간 확인 범위. today, week, custom 중 하나"
    )

    keyword: Optional[str] = Field(
        default=None,
        description="검색/수정/삭제 대상 일정 키워드. 명령어를 제외한 일정 이름 중심으로 추출"
    )

    title: Optional[str] = Field(
        default=None,
        description="create_event에서 일정 제목. search/update/delete에서는 대상 제목으로도 사용 가능"
    )

    new_title: Optional[str] = Field(
        default=None,
        description="update_event에서 변경할 새 제목"
    )

    date: Optional[str] = Field(
        default=None,
        description="일정 날짜 또는 검색/수정/삭제 범위 날짜. YYYY-MM-DD 형식"
    )

    start_date: Optional[str] = Field(
        default=None,
        description="custom 범위 시작 날짜. YYYY-MM-DD 형식"
    )

    end_date: Optional[str] = Field(
        default=None,
        description="custom 범위 종료 날짜. YYYY-MM-DD 형식"
    )

    start_time: Optional[str] = Field(
        default=None,
        description="create_event 시작 시간 또는 update_event 새 시작 시간. HH:MM 24시간 형식"
    )

    end_time: Optional[str] = Field(
        default=None,
        description="create_event 종료 시간 또는 update_event 새 종료 시간. HH:MM 24시간 형식"
    )

    new_date: Optional[str] = Field(
        default=None,
        description="update_event에서 변경할 새 날짜. YYYY-MM-DD 형식"
    )

    new_start_time: Optional[str] = Field(
        default=None,
        description="update_event에서 변경할 새 시작 시간. HH:MM 형식"
    )

    new_end_time: Optional[str] = Field(
        default=None,
        description="update_event에서 변경할 새 종료 시간. HH:MM 형식"
    )

    description: Optional[str] = Field(
        default=None,
        description="일정 설명 또는 수정할 설명"
    )

    location: Optional[str] = Field(
        default=None,
        description="일정 장소 또는 수정할 장소"
    )

    reminder_minutes: Optional[int] = Field(
        default=None,
        description="일정 시작 몇 분 전에 알림을 받을지"
    )

    duration_minutes: Optional[int] = Field(
        default=None,
        description="빈 시간 확인 시 필요한 회의 길이. 기본 60분"
    )

    time_period: Optional[
        Literal["morning", "afternoon", "evening", "business_hours", "all_day"]
    ] = Field(
        default=None,
        description="빈 시간 확인 시간대. morning, afternoon, evening, business_hours, all_day"
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

    def _tz(self):
        return ZoneInfo(TIMEZONE)

    def _parse_date_time(self, date: str, clock: str) -> datetime:
        return datetime.strptime(
            f"{date} {clock}",
            "%Y-%m-%d %H:%M"
        ).replace(tzinfo=self._tz())

    def _date_range(
        self,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        default_future_days: int = 90,
    ) -> Tuple[datetime, datetime, str]:
        tz = self._tz()
        now = datetime.now(tz)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if date:
            start_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = start_dt + timedelta(days=1)
            return start_dt, end_dt, date

        if range_type == "week":
            start_dt = today_start - timedelta(days=today_start.weekday())
            end_dt = start_dt + timedelta(days=7)
            return start_dt, end_dt, "이번 주"

        if range_type == "today":
            return today_start, today_start + timedelta(days=1), "오늘"

        if start_date and end_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=tz)
            end_dt = datetime.strptime(end_date, "%Y-%m-%d").replace(tzinfo=tz) + timedelta(days=1)
            return start_dt, end_dt, f"{start_date} ~ {end_date}"

        start_dt = today_start - timedelta(days=7)
        end_dt = today_start + timedelta(days=default_future_days)
        return start_dt, end_dt, f"최근 7일 ~ 향후 {default_future_days}일"

    def _format_datetime(self, value: Optional[str]) -> str:
        if not value:
            return "시간 정보 없음"

        try:
            if "T" in value:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.astimezone(self._tz()).strftime("%Y-%m-%d %H:%M")
            return value
        except Exception:
            return value

    def _event_start_dt(self, event: Dict[str, Any]) -> Optional[datetime]:
        start = event.get("start", {})
        value = start.get("dateTime") or start.get("date")

        if not value:
            return None

        try:
            if "T" in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(self._tz())
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=self._tz())
        except Exception:
            return None

    def _event_end_dt(self, event: Dict[str, Any]) -> Optional[datetime]:
        end = event.get("end", {})
        value = end.get("dateTime") or end.get("date")

        if not value:
            return None

        try:
            if "T" in value:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(self._tz())
            return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=self._tz())
        except Exception:
            return None

    def _normalize_for_match(self, text: Optional[str]) -> str:
        if not text:
            return ""

        text = text.lower()
        text = re.sub(r"[^0-9a-zA-Z가-힣\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _token_f1_score(self, query: str, target: str) -> float:
        query_tokens = set(self._normalize_for_match(query).split())
        target_tokens = set(self._normalize_for_match(target).split())

        if not query_tokens or not target_tokens:
            return 0.0

        overlap = len(query_tokens & target_tokens)

        if overlap == 0:
            return 0.0

        precision = overlap / len(query_tokens)
        recall = overlap / len(target_tokens)

        return 2 * precision * recall / (precision + recall)

    def _similarity_score(self, query: str, event: Dict[str, Any]) -> float:
        summary = event.get("summary", "")
        description = event.get("description", "")
        location = event.get("location", "")

        query_norm = self._normalize_for_match(query)
        summary_norm = self._normalize_for_match(summary)
        combined_norm = self._normalize_for_match(
            f"{summary} {description} {location}"
        )

        if not query_norm or not combined_norm:
            return 0.0

        contains_score = 0.0

        if summary_norm and summary_norm in query_norm:
            contains_score = 0.98
        elif query_norm in combined_norm:
            contains_score = 0.95

        seq_summary_score = SequenceMatcher(None, query_norm, summary_norm).ratio()
        seq_combined_score = SequenceMatcher(None, query_norm, combined_norm).ratio()
        token_score = self._token_f1_score(query_norm, combined_norm)

        return max(
            contains_score,
            seq_summary_score,
            seq_combined_score * 0.9,
            token_score,
        )

    def _rank_events_by_similarity(
        self,
        events: List[Dict[str, Any]],
        query: str,
        threshold: float = 0.35,
    ) -> List[Dict[str, Any]]:
        ranked = []

        for event in events:
            score = self._similarity_score(query, event)

            if score >= threshold:
                event["_match_score"] = score
                ranked.append(event)

        ranked.sort(
            key=lambda event: (
                event.get("_match_score", 0),
                self._event_start_dt(event) or datetime.min.replace(tzinfo=self._tz())
            ),
            reverse=True
        )

        return ranked

    def _render_event_lines(self, events: List[Dict[str, Any]], title: str) -> str:
        if not events:
            return f"📅 {title} 등록된 일정이 없습니다."

        lines = [f"📅 **{title}**\n"]

        for i, event in enumerate(events, 1):
            summary = event.get("summary", "제목 없음")
            event_id = event.get("id", "")

            start = event.get("start", {})
            end = event.get("end", {})

            start_value = start.get("dateTime") or start.get("date")
            end_value = end.get("dateTime") or end.get("date")

            start_text = self._format_datetime(start_value)
            end_text = self._format_datetime(end_value)

            lines.append(f"{i}. **{summary}**")
            lines.append(f"   - 시간: {start_text} ~ {end_text}")
            lines.append(f"   - event_id: {event_id}")

            match_score = event.get("_match_score")
            if match_score is not None:
                lines.append(f"   - 유사도: {match_score:.2f}")

            location = event.get("location")
            if location:
                lines.append(f"   - 장소: {location}")

            description = event.get("description")
            if description:
                short_description = description.replace("\n", " ")
                if len(short_description) > 80:
                    short_description = short_description[:80] + "..."
                lines.append(f"   - 설명: {short_description}")

            html_link = event.get("htmlLink")
            if html_link:
                lines.append(f"   - 링크: {html_link}")

            lines.append("")

        return "\n".join(lines)

    def _fetch_events(
        self,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        keyword: Optional[str] = None,
        max_results: int = 100,
    ) -> Tuple[List[Dict[str, Any]], str]:
        self._ensure_initialized()

        start_dt, end_dt, range_title = self._date_range(
            range_type=range_type,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(
            f"[CALENDAR AGENT] 일정 조회: {range_title}, "
            f"{start_dt.isoformat()} ~ {end_dt.isoformat()}, keyword={keyword}"
        )

        # q 검색에 의존하지 않고, 기간 안의 일정을 넓게 가져온 뒤 Python에서 유사도 계산
        result = self.service.events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=max_results,
            timeZone=TIMEZONE,
        ).execute()

        events = result.get("items", [])

        if keyword:
            events = self._rank_events_by_similarity(
                events=events,
                query=keyword,
                threshold=0.35,
            )

        return events, range_title

    def _find_one_event(
        self,
        keyword: Optional[str],
        range_type: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        if not keyword:
            return None, "수정/삭제할 일정의 제목 또는 내용을 알려주세요."

        events, range_title = self._fetch_events(
            range_type=range_type,
            date=date,
            start_date=start_date,
            end_date=end_date,
            keyword=keyword,
            max_results=100,
        )

        if not events:
            return None, f"'{keyword}'와 비슷한 일정을 찾지 못했습니다. 검색 범위: {range_title}"

        top_event = events[0]
        top_score = top_event.get("_match_score", 0)

        if len(events) == 1 and top_score >= 0.40:
            return top_event, None

        if len(events) > 1:
            second_score = events[1].get("_match_score", 0)

            if top_score >= 0.75 and (top_score - second_score) >= 0.20:
                return top_event, None

        rendered = self._render_event_lines(
            events[:5],
            f"'{keyword}'와 비슷한 일정 후보"
        )

        return None, (
            "비슷한 일정이 여러 개라서 바로 수정/삭제하지 않았습니다.\n"
            "날짜나 시간을 더 구체적으로 말해 주세요.\n\n"
            f"{rendered}"
        )

    def _find_conflicting_events(
        self,
        start_dt: datetime,
        end_dt: datetime,
    ) -> List[Dict[str, Any]]:
        """
        요청한 시간대와 겹치는 기존 일정을 찾는다.
        겹침 조건: 기존 시작 < 새 종료 and 기존 종료 > 새 시작
        """
        self._ensure_initialized()

        result = self.service.events().list(
            calendarId="primary",
            timeMin=start_dt.isoformat(),
            timeMax=end_dt.isoformat(),
            singleEvents=True,
            orderBy="startTime",
            maxResults=20,
            timeZone=TIMEZONE,
        ).execute()

        events = result.get("items", [])
        conflicts = []

        for event in events:
            event_start = self._event_start_dt(event)
            event_end = self._event_end_dt(event)

            if not event_start or not event_end:
                continue

            if event_start < end_dt and event_end > start_dt:
                conflicts.append(event)

        return conflicts

    def _suggest_available_slots(
        self,
        date: str,
        start_dt: datetime,
        end_dt: datetime,
        limit: int = 3,
    ) -> List[Tuple[datetime, datetime]]:
        """
        충돌이 발생했을 때 같은 날짜에서 가능한 대체 시간대를 추천한다.
        요청한 시작 시간 이후부터 21:00까지 탐색한다.
        """
        duration = end_dt - start_dt
        tz = self._tz()

        day_start = datetime.combine(
            start_dt.date(),
            time(9, 0),
            tzinfo=tz,
        )
        day_end = datetime.combine(
            start_dt.date(),
            time(21, 0),
            tzinfo=tz,
        )

        cursor = max(start_dt, day_start)

        events, _ = self._fetch_events(
            date=date,
            max_results=100,
        )

        busy_slots = []

        for event in events:
            busy_start = self._event_start_dt(event)
            busy_end = self._event_end_dt(event)

            if not busy_start or not busy_end:
                continue

            if busy_end <= day_start or busy_start >= day_end:
                continue

            busy_slots.append((
                max(busy_start, day_start),
                min(busy_end, day_end),
            ))

        busy_slots.sort(key=lambda slot: slot[0])

        available_slots = []

        for busy_start, busy_end in busy_slots:
            if busy_end <= cursor:
                continue

            if busy_start > cursor:
                gap = busy_start - cursor

                if gap >= duration:
                    available_slots.append((cursor, cursor + duration))

                    if len(available_slots) >= limit:
                        return available_slots

            if busy_end > cursor:
                cursor = busy_end

        if cursor + duration <= day_end:
            available_slots.append((cursor, cursor + duration))

        return available_slots[:limit]

    def _render_conflict_message(
        self,
        title: str,
        date: str,
        start_dt: datetime,
        end_dt: datetime,
        conflicts: List[Dict[str, Any]],
    ) -> str:
        lines = [
            "⚠️ 요청한 시간대에 이미 등록된 일정이 있어 새 일정을 등록하지 않았습니다.",
            "",
            "[요청한 일정]",
            f"- 제목: {title}",
            f"- 시간: {start_dt.strftime('%Y-%m-%d %H:%M')} ~ {end_dt.strftime('%H:%M')}",
            "",
            "[겹치는 기존 일정]",
        ]

        for i, event in enumerate(conflicts, 1):
            summary = event.get("summary", "제목 없음")
            event_id = event.get("id", "")

            event_start = self._format_datetime(
                event.get("start", {}).get("dateTime")
                or event.get("start", {}).get("date")
            )
            event_end = self._format_datetime(
                event.get("end", {}).get("dateTime")
                or event.get("end", {}).get("date")
            )

            lines.append(f"{i}. {summary}")
            lines.append(f"   - 시간: {event_start} ~ {event_end}")
            lines.append(f"   - event_id: {event_id}")

        suggested_slots = self._suggest_available_slots(
            date=date,
            start_dt=start_dt,
            end_dt=end_dt,
            limit=3,
        )

        if suggested_slots:
            lines.append("")
            lines.append("[대체 가능한 시간]")
            for i, (slot_start, slot_end) in enumerate(suggested_slots, 1):
                lines.append(
                    f"{i}. {slot_start.strftime('%Y-%m-%d %H:%M')} ~ "
                    f"{slot_end.strftime('%H:%M')}"
                )
        else:
            lines.append("")
            lines.append("같은 날짜에서 바로 추천할 수 있는 대체 시간대를 찾지 못했습니다.")

        return "\n".join(lines)

    def list_events(self, range_type: str = "today") -> str:
        events, range_title = self._fetch_events(range_type=range_type)
        return self._render_event_lines(events, f"{range_title} 일정")

    def search_events(
        self,
        keyword: str,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        if not keyword:
            return "검색할 일정 키워드가 필요합니다."

        events, range_title = self._fetch_events(
            range_type=range_type,
            date=date,
            start_date=start_date,
            end_date=end_date,
            keyword=keyword,
            max_results=100,
        )

        if not events:
            return f"'{keyword}'와 비슷한 일정을 찾지 못했습니다. 검색 범위: {range_title}"

        return self._render_event_lines(events, f"'{keyword}'와 비슷한 일정 검색 결과")

    def create_event(
        self,
        title: str,
        date: str,
        start_time: str,
        end_time: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        reminder_minutes: Optional[int] = None,
    ) -> str:
        self._ensure_initialized()

        start_dt = self._parse_date_time(date, start_time)

        if end_time:
            end_dt = self._parse_date_time(date, end_time)
        else:
            end_dt = start_dt + timedelta(hours=1)

        if end_dt <= start_dt:
            return "일정 종료 시간이 시작 시간보다 늦어야 합니다."

        conflicts = self._find_conflicting_events(
            start_dt=start_dt,
            end_dt=end_dt,
        )

        if conflicts:
            return self._render_conflict_message(
                title=title,
                date=date,
                start_dt=start_dt,
                end_dt=end_dt,
                conflicts=conflicts,
            )

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

        if description:
            event_body["description"] = description

        if location:
            event_body["location"] = location

        if reminder_minutes is not None:
            event_body["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {
                        "method": "popup",
                        "minutes": reminder_minutes,
                    }
                ],
            }

        created_event = self.service.events().insert(
            calendarId="primary",
            body=event_body,
        ).execute()

        link = created_event.get("htmlLink", "")
        event_id = created_event.get("id", "")

        lines = [
            "✅ Google Calendar에 일정을 등록했습니다.",
            "",
            f"- 제목: {title}",
            f"- 날짜: {date}",
            f"- 시간: {start_time} ~ {end_time or end_dt.strftime('%H:%M')}",
            f"- event_id: {event_id}",
        ]

        if description:
            lines.append(f"- 설명: {description}")

        if location:
            lines.append(f"- 장소: {location}")

        if reminder_minutes is not None:
            lines.append(f"- 알림: 시작 {reminder_minutes}분 전")

        if link:
            lines.append(f"- 링크: {link}")

        return "\n".join(lines)

    def update_event(
        self,
        keyword: str,
        new_title: Optional[str] = None,
        new_date: Optional[str] = None,
        new_start_time: Optional[str] = None,
        new_end_time: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        reminder_minutes: Optional[int] = None,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        self._ensure_initialized()

        event, error = self._find_one_event(
            keyword=keyword,
            range_type=range_type,
            date=date,
        )

        if error:
            return error

        event_id = event["id"]

        before_summary = event.get("summary", "제목 없음")
        before_start = self._format_datetime(
            event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
        )
        before_end = self._format_datetime(
            event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
        )

        patched = {}

        if new_title:
            patched["summary"] = new_title

        original_start_dt = self._event_start_dt(event)
        original_end_dt = self._event_end_dt(event)

        if new_date or new_start_time or new_end_time:
            if not original_start_dt:
                return "기존 일정의 시작 시간을 확인할 수 없어 수정할 수 없습니다."

            target_date = new_date or original_start_dt.strftime("%Y-%m-%d")
            target_start_time = new_start_time or original_start_dt.strftime("%H:%M")

            if new_end_time:
                target_end_time = new_end_time
            elif original_end_dt:
                target_end_time = original_end_dt.strftime("%H:%M")
            else:
                target_end_time = None

            new_start_dt = self._parse_date_time(target_date, target_start_time)

            if target_end_time:
                new_end_dt = self._parse_date_time(target_date, target_end_time)
            else:
                new_end_dt = new_start_dt + timedelta(hours=1)

            if new_end_dt <= new_start_dt:
                return "수정할 일정의 종료 시간이 시작 시간보다 늦어야 합니다."

            patched["start"] = {
                "dateTime": new_start_dt.isoformat(),
                "timeZone": TIMEZONE,
            }
            patched["end"] = {
                "dateTime": new_end_dt.isoformat(),
                "timeZone": TIMEZONE,
            }

        if description is not None:
            patched["description"] = description

        if location is not None:
            patched["location"] = location

        if reminder_minutes is not None:
            patched["reminders"] = {
                "useDefault": False,
                "overrides": [
                    {
                        "method": "popup",
                        "minutes": reminder_minutes,
                    }
                ],
            }

        if not patched:
            return "수정할 내용이 없습니다. 제목, 날짜, 시간, 설명, 장소, 알림 중 변경할 내용을 알려주세요."

        updated_event = self.service.events().patch(
            calendarId="primary",
            eventId=event_id,
            body=patched
        ).execute()

        after_summary = updated_event.get("summary", "제목 없음")
        after_start = self._format_datetime(
            updated_event.get("start", {}).get("dateTime") or updated_event.get("start", {}).get("date")
        )
        after_end = self._format_datetime(
            updated_event.get("end", {}).get("dateTime") or updated_event.get("end", {}).get("date")
        )

        link = updated_event.get("htmlLink", "")

        return (
            "✅ Google Calendar 일정을 수정했습니다.\n\n"
            "[수정 전]\n"
            f"- 제목: {before_summary}\n"
            f"- 시간: {before_start} ~ {before_end}\n\n"
            "[수정 후]\n"
            f"- 제목: {after_summary}\n"
            f"- 시간: {after_start} ~ {after_end}\n"
            f"- event_id: {event_id}\n"
            f"- 링크: {link}"
        )

    def delete_event(
        self,
        keyword: str,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
    ) -> str:
        self._ensure_initialized()

        event, error = self._find_one_event(
            keyword=keyword,
            range_type=range_type,
            date=date,
        )

        if error:
            return error

        event_id = event["id"]
        summary = event.get("summary", "제목 없음")
        start_text = self._format_datetime(
            event.get("start", {}).get("dateTime") or event.get("start", {}).get("date")
        )
        end_text = self._format_datetime(
            event.get("end", {}).get("dateTime") or event.get("end", {}).get("date")
        )

        self.service.events().delete(
            calendarId="primary",
            eventId=event_id
        ).execute()

        return (
            "🗑️ Google Calendar 일정을 삭제했습니다.\n\n"
            f"- 제목: {summary}\n"
            f"- 시간: {start_text} ~ {end_text}\n"
            f"- event_id: {event_id}"
        )

    def find_free_time(
        self,
        range_type: Optional[str] = None,
        date: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        duration_minutes: Optional[int] = None,
        time_period: Optional[str] = None,
    ) -> str:
        events, range_title = self._fetch_events(
            range_type=range_type,
            date=date,
            start_date=start_date,
            end_date=end_date,
            max_results=100,
        )

        duration = duration_minutes or 60
        tz = self._tz()

        start_dt, end_dt, _ = self._date_range(
            range_type=range_type,
            date=date,
            start_date=start_date,
            end_date=end_date,
        )

        period = time_period or "business_hours"

        if period == "morning":
            day_start_time = time(9, 0)
            day_end_time = time(12, 0)
            period_name = "오전"
        elif period == "afternoon":
            day_start_time = time(12, 0)
            day_end_time = time(18, 0)
            period_name = "오후"
        elif period == "evening":
            day_start_time = time(18, 0)
            day_end_time = time(21, 0)
            period_name = "저녁"
        elif period == "all_day":
            day_start_time = time(9, 0)
            day_end_time = time(21, 0)
            period_name = "하루"
        else:
            day_start_time = time(9, 0)
            day_end_time = time(18, 0)
            period_name = "업무 시간"

        busy_by_date: Dict[str, List[Tuple[datetime, datetime]]] = {}

        for event in events:
            s = self._event_start_dt(event)
            e = self._event_end_dt(event)

            if not s or not e:
                continue

            date_key = s.astimezone(tz).strftime("%Y-%m-%d")
            busy_by_date.setdefault(date_key, []).append((s, e))

        free_slots = []
        current_day = start_dt

        while current_day < end_dt:
            date_key = current_day.strftime("%Y-%m-%d")

            window_start = datetime.combine(
                current_day.date(),
                day_start_time,
                tzinfo=tz
            )
            window_end = datetime.combine(
                current_day.date(),
                day_end_time,
                tzinfo=tz
            )

            busy_slots = sorted(
                busy_by_date.get(date_key, []),
                key=lambda x: x[0]
            )

            cursor = window_start

            for busy_start, busy_end in busy_slots:
                busy_start = max(busy_start, window_start)
                busy_end = min(busy_end, window_end)

                if busy_end <= window_start or busy_start >= window_end:
                    continue

                if busy_start > cursor:
                    gap_minutes = int((busy_start - cursor).total_seconds() // 60)
                    if gap_minutes >= duration:
                        free_slots.append((cursor, busy_start))

                if busy_end > cursor:
                    cursor = busy_end

            if cursor < window_end:
                gap_minutes = int((window_end - cursor).total_seconds() // 60)
                if gap_minutes >= duration:
                    free_slots.append((cursor, window_end))

            current_day += timedelta(days=1)

        if not free_slots:
            return (
                f"⏰ {range_title}의 {period_name} 중 "
                f"{duration}분 이상 비어 있는 시간을 찾지 못했습니다."
            )

        lines = [
            f"⏰ **{range_title}의 {period_name} 중 {duration}분 이상 가능한 시간대**\n"
        ]

        for i, (slot_start, slot_end) in enumerate(free_slots[:10], 1):
            lines.append(
                f"{i}. {slot_start.strftime('%Y-%m-%d %H:%M')} ~ "
                f"{slot_end.strftime('%H:%M')}"
            )

        return "\n".join(lines)

    def summarize_events(
        self,
        range_type: str = "today",
        date: Optional[str] = None,
    ) -> str:
        events, range_title = self._fetch_events(
            range_type=range_type,
            date=date,
            max_results=100,
        )

        if not events:
            return f"📌 {range_title} 등록된 일정이 없어 요약할 내용이 없습니다."

        events_sorted = sorted(
            events,
            key=lambda event: self._event_start_dt(event) or datetime.max.replace(tzinfo=self._tz())
        )

        total_count = len(events_sorted)
        first_event = events_sorted[0]
        last_event = events_sorted[-1]

        day_counts: Dict[str, int] = {}

        for event in events_sorted:
            s = self._event_start_dt(event)
            if not s:
                continue

            key = s.strftime("%Y-%m-%d")
            day_counts[key] = day_counts.get(key, 0) + 1

        busiest_day = max(day_counts.items(), key=lambda x: x[1]) if day_counts else None

        lines = [
            f"📌 **{range_title} 일정 요약**",
            "",
            f"- 총 일정 수: {total_count}개",
            f"- 첫 일정: {first_event.get('summary', '제목 없음')} "
            f"({self._format_datetime(first_event.get('start', {}).get('dateTime') or first_event.get('start', {}).get('date'))})",
            f"- 마지막 일정: {last_event.get('summary', '제목 없음')} "
            f"({self._format_datetime(last_event.get('start', {}).get('dateTime') or last_event.get('start', {}).get('date'))})",
        ]

        if busiest_day:
            lines.append(f"- 일정이 가장 많은 날: {busiest_day[0]} ({busiest_day[1]}개)")

        lines.append("")
        lines.append("주요 일정:")

        for i, event in enumerate(events_sorted[:5], 1):
            lines.append(
                f"{i}. {self._format_datetime(event.get('start', {}).get('dateTime') or event.get('start', {}).get('date'))} "
                f"- {event.get('summary', '제목 없음')}"
            )

        return "\n".join(lines)


class CalendarAgent:
    """
    Google Calendar Agent

    지원 기능:
    1. 오늘/이번 주 일정 조회
    2. 유사도 기반 일정 검색
    3. 일정 등록
    4. 일정 수정
    5. 일정 삭제
    6. 빈 시간 확인
    7. 일정 요약
    8. 일정 등록 전 충돌 방지
    """

    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
        self.calendar_client = GoogleCalendarClient()

    def parse_request(self, query: str) -> CalendarRequest:
        tz = ZoneInfo(TIMEZONE)
        now = datetime.now(tz)

        system_prompt = f"""
당신은 Google Calendar 요청 분석기입니다.
사용자 요청을 CalendarRequest 스키마에 맞게 구조화하세요.

현재 기준:
- 오늘 날짜: {now.strftime("%Y-%m-%d")}
- 현재 시간: {now.strftime("%H:%M")}
- 시간대: {TIMEZONE}

action 선택:
- list_events: 오늘/이번 주 일정 조회
- search_events: 특정 일정 검색
- create_event: 일정 등록/추가/회의 잡기
- update_event: 일정 수정/변경/시간 변경/제목 변경
- delete_event: 일정 삭제/취소/지우기
- find_free_time: 빈 시간/가능한 시간/회의 가능한 시간 확인
- summarize_events: 오늘/이번 주 일정 요약/브리핑/정리
- unsupported: 위 기능에 해당하지 않음

핵심 추출 규칙:
1. 날짜가 있으면 반드시 date에 YYYY-MM-DD 형식으로 넣습니다.
   - "2026-06-27" → date="2026-06-27"
   - "내일", "다음 주 월요일"은 현재 날짜 기준으로 계산합니다.
2. 시간은 반드시 HH:MM 24시간 형식으로 변환합니다.
   - "오후 2시" → "14:00"
   - "18시" → "18:00"
3. search_events/update_event/delete_event에서 keyword는 일정 이름만 넣습니다.
   - 날짜, 시간, "일정", "찾아줘", "삭제해줘", "수정해줘", "변경해줘" 같은 명령어는 keyword에서 제외합니다.
4. create_event에서 title은 새로 만들 일정 제목입니다.
5. update_event에서 변경값은 아래 필드에 넣습니다.
   - 새 제목: new_title
   - 새 날짜: new_date
   - 새 시작 시간: new_start_time
   - 새 종료 시간: new_end_time
6. find_free_time에서 시간대는 아래 값으로 넣습니다.
   - 오전: morning
   - 오후: afternoon
   - 저녁: evening
   - 업무 시간: business_hours
   - 하루 전체: all_day
7. "이번주", "이번 주"는 range_type="week"입니다.
8. "오늘"은 range_type="today"입니다.
9. 사용자가 수정/삭제를 요청하면 확인 없이 바로 실행하는 것으로 분류합니다.

중요 예시:
- "2026-06-27 AI 에이전트 시연 준비 회의 일정 삭제해줘"
  → action="delete_event", date="2026-06-27", keyword="AI 에이전트 시연 준비 회의"

- "2026-06-27 AI 에이전트 시연 준비 회의를 18:00부터 19:00까지로 변경해줘"
  → action="update_event", date="2026-06-27", keyword="AI 에이전트 시연 준비 회의",
     new_start_time="18:00", new_end_time="19:00"

- "내일 오후에 비어있는 시간 알려줘"
  → action="find_free_time", date=내일 날짜, time_period="afternoon"

반드시 CalendarRequest 스키마에 맞게 응답하세요.
"""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ]

        structured_llm = self.llm.with_structured_output(CalendarRequest)
        return structured_llm.invoke(messages)

    def run(self, query: str) -> str:
        logger.info(f"[CALENDAR AGENT] 요청 수신: {query}")

        parsed = self.parse_request(query)
        logger.info(f"[CALENDAR AGENT] 분석 결과: {parsed.model_dump()}")

        if parsed.action == "list_events":
            return self.calendar_client.list_events(
                range_type=parsed.range_type or "today"
            )

        if parsed.action == "search_events":
            keyword = parsed.keyword or parsed.title

            if not keyword:
                return "검색할 일정 키워드가 필요합니다."

            return self.calendar_client.search_events(
                keyword=keyword,
                range_type=parsed.range_type,
                date=parsed.date,
                start_date=parsed.start_date,
                end_date=parsed.end_date,
            )

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
                description=parsed.description,
                location=parsed.location,
                reminder_minutes=parsed.reminder_minutes,
            )

        if parsed.action == "update_event":
            keyword = parsed.keyword or parsed.title

            if not keyword:
                return "수정할 일정의 제목 또는 키워드가 필요합니다."

            return self.calendar_client.update_event(
                keyword=keyword,
                new_title=parsed.new_title,
                new_date=parsed.new_date,
                new_start_time=parsed.new_start_time or parsed.start_time,
                new_end_time=parsed.new_end_time or parsed.end_time,
                description=parsed.description,
                location=parsed.location,
                reminder_minutes=parsed.reminder_minutes,
                range_type=parsed.range_type,
                date=parsed.date,
            )

        if parsed.action == "delete_event":
            keyword = parsed.keyword or parsed.title

            if not keyword:
                return "삭제할 일정의 제목 또는 키워드가 필요합니다."

            return self.calendar_client.delete_event(
                keyword=keyword,
                range_type=parsed.range_type,
                date=parsed.date,
            )

        if parsed.action == "find_free_time":
            return self.calendar_client.find_free_time(
                range_type=parsed.range_type,
                date=parsed.date,
                start_date=parsed.start_date,
                end_date=parsed.end_date,
                duration_minutes=parsed.duration_minutes,
                time_period=parsed.time_period,
            )

        if parsed.action == "summarize_events":
            return self.calendar_client.summarize_events(
                range_type=parsed.range_type or "today",
                date=parsed.date,
            )

        return (
            "현재 Calendar Agent는 다음 기능을 지원합니다.\n\n"
            "1. 오늘/이번 주 일정 조회\n"
            "2. 유사도 기반 일정 검색\n"
            "3. 일정 등록\n"
            "4. 일정 수정\n"
            "5. 일정 삭제\n"
            "6. 빈 시간 확인\n"
            "7. 오늘/이번 주 일정 요약\n"
            "8. 일정 등록 전 충돌 방지"
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
