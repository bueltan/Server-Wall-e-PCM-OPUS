"""Bounded Calendar polling plus local ten-minute reminder scheduling."""
import asyncio
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from bridge.diagnostics.logging import event
from bridge.mcp_servers.calendar import service, timestamp, api_failure


@dataclass(frozen=True)
class Reminder:
    key: str
    start: datetime
    title: str
    calendar_id: str

    def message(self, zone):
        return {'title': self.title, 'start': self.start.astimezone(zone).isoformat(),
                'calendar_id': self.calendar_id}


class CalendarReminders:
    def __init__(self, config, deliver, fetch=None, clock=None):
        self.config, self.deliver = config, deliver
        self.fetch = fetch or self.fetch_events
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.path = Path(config.calendar_reminder_state).expanduser()
        self.reminders = []
        self.claimed = {}
        self.refreshed_at = None
        self.tasks = []
        if self.path.exists():
            # Fail startup on corrupt state instead of silently replaying reminders.
            self.claimed = json.loads(self.path.read_text())
            if not isinstance(self.claimed, dict):
                raise ValueError('invalid_calendar_reminder_state')
            for key, value in self.claimed.items():
                if not isinstance(key, str):
                    raise ValueError('invalid_calendar_reminder_state')
                timestamp(value)

    def fetch_events(self):
        now = self.clock()
        api = service()
        options = dict(calendarId=self.config.calendar_reminder_calendar,
                       timeMin=now.isoformat(), timeMax=(now + timedelta(days=1)).isoformat(),
                       singleEvents=True, orderBy='startTime', maxResults=1000,
                       timeZone=self.config.calendar_time_zone)
        items = []
        for _ in range(10):
            page = api.events().list(**options).execute(num_retries=0)
            items.extend(page.get('items', []))
            if not page.get('nextPageToken'):
                return items
            options['pageToken'] = page['nextPageToken']
        raise RuntimeError('calendar_reminder_page_limit')

    async def refresh(self):
        items = await asyncio.to_thread(self.fetch)
        reminders = []
        for item in items:
            if item.get('status') == 'cancelled' or not item.get('start', {}).get('dateTime'):
                continue  # All-day notices need an explicit morning-hour policy.
            if any(a.get('self') and a.get('responseStatus') == 'declined' for a in item.get('attendees', [])):
                continue
            start = timestamp(item['start']['dateTime'])
            key = hashlib.sha256(json.dumps([self.config.calendar_reminder_calendar,
                item['id'], start.isoformat()], separators=(',', ':')).encode()).hexdigest()
            reminders.append(Reminder(key, start, (item.get('summary') or 'Evento sin título')[:1000],
                                      self.config.calendar_reminder_calendar))
        self.reminders = sorted(reminders, key=lambda reminder: reminder.start)
        self.refreshed_at = self.clock()
        event('CALENDAR_REMINDERS_SYNCED', events=len(reminders))

    def claim(self, reminder):
        if reminder.key in self.claimed:
            return False
        cutoff = self.clock() - timedelta(days=2)
        updated = {key: value for key, value in self.claimed.items() if timestamp(value) > cutoff}
        updated[reminder.key] = reminder.start.isoformat()
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd, temporary = tempfile.mkstemp(dir=self.path.parent, prefix='.reminders-')
        try:
            with os.fdopen(fd, 'w') as stream:
                json.dump(updated, stream)
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        self.claimed = updated
        return True

    async def tick(self):
        now = self.clock()
        # Stop stale announcements after API errors: a cancelled event may be cached.
        if self.refreshed_at is None or (now - self.refreshed_at).total_seconds() > max(300, self.config.calendar_poll_sec * 2):
            return
        for reminder in list(self.reminders):
            due = reminder.start - timedelta(minutes=self.config.calendar_reminder_minutes)
            if due <= now < reminder.start and reminder.key not in self.claimed:
                # Revalidate against the latest refresh after waiting for CORE START.
                def claim_current(value):
                    return (self.refreshed_at is not None
                            and (self.clock() - self.refreshed_at).total_seconds() <= max(300, self.config.calendar_poll_sec * 2)
                            and any(r.key == value.key for r in self.reminders)
                            and self.clock() < value.start and self.claim(value))
                if await self.deliver(reminder, claim_current):
                    event('CALENDAR_REMINDER_SUBMITTED', reminder=reminder.key[:12])
                    break

    async def poll(self):
        errors = 0
        while True:
            try:
                await self.refresh()
                errors = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors += 1
                event('CALENDAR_REMINDERS_SYNC_FAILED', error=api_failure(exc)['error'])
            cap = max(1800, self.config.calendar_poll_sec)
            delay = min(cap, self.config.calendar_poll_sec * 2 ** min(errors, 4))
            # Jitter avoids aligned requests when more devices are deployed.
            await asyncio.sleep(min(cap, delay * random.uniform(1, 1.05)))

    async def run_clock(self):
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                event('CALENDAR_REMINDER_FAILED', error=type(exc).__name__)
            await asyncio.sleep(1)

    def start(self):
        self.tasks = [asyncio.create_task(self.poll(), name='calendar-poll'),
                      asyncio.create_task(self.run_clock(), name='calendar-reminders')]

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
