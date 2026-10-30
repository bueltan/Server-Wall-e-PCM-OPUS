"""Single local task-manager writer. State transitions and outbox are atomic."""
import json
import sqlite3
import time
import uuid
from pathlib import Path
from bridge.codex.policy import PolicyError
from bridge.diagnostics.logging import event

TERMINAL = {'completed', 'failed', 'cancelled'}
EVENTS = {'queued': 'CODEX_TASK_CREATED', 'running': 'CODEX_TASK_STARTED',
          'completed': 'CODEX_TASK_COMPLETED', 'failed': 'CODEX_TASK_FAILED',
          'cancelled': 'CODEX_TASK_CANCELLED', 'interrupted': 'CODEX_TASK_INTERRUPTED'}


class Store:
    def __init__(self, path):
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db = sqlite3.connect(path)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript('''
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;
        CREATE TABLE IF NOT EXISTS tasks (
          task_id TEXT PRIMARY KEY, project TEXT NOT NULL, instruction TEXT NOT NULL,
          idempotency_key TEXT UNIQUE NOT NULL, parent TEXT, thread_id TEXT, turn_id TEXT,
          status TEXT NOT NULL, progress_message TEXT NOT NULL DEFAULT '',
          created REAL NOT NULL, updated REAL NOT NULL, started REAL,
          cancel_requested INTEGER NOT NULL DEFAULT 0, result TEXT);
        CREATE TABLE IF NOT EXISTS events (
          sequence INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
          project TEXT NOT NULL, status TEXT NOT NULL, message TEXT NOT NULL,
          timestamp REAL NOT NULL, duration REAL);
        ''')

    def get(self, task_id):
        row = self.db.execute('SELECT * FROM tasks WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise PolicyError('unknown_task')
        return dict(row)

    def recent(self, limit=20):
        return [dict(row) for row in self.db.execute('SELECT * FROM tasks ORDER BY created DESC LIMIT ?', (limit,))]

    def pending(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM tasks WHERE status='queued' ORDER BY created")]

    def uncertain(self):
        return [dict(row) for row in self.db.execute("SELECT * FROM tasks WHERE status='interrupted'")]

    def create(self, project, instruction, key, parent=None):
        existing = self.db.execute('SELECT * FROM tasks WHERE idempotency_key=?', (key,)).fetchone()
        if existing:
            if (existing['project'], existing['instruction'], existing['parent']) != (project, instruction, parent):
                raise PolicyError('idempotency_conflict')
            return dict(existing)
        if self.db.execute("SELECT count(*) FROM tasks WHERE status='queued'").fetchone()[0] >= 100:
            raise PolicyError('queue_full')
        task_id, now = 'task_' + uuid.uuid4().hex, time.time()
        with self.db:
            self.db.execute('INSERT INTO tasks(task_id,project,instruction,idempotency_key,parent,status,created,updated) '
                            "VALUES(?,?,?,?,?,'queued',?,?)", (task_id, project, instruction, key, parent, now, now))
            self._event(task_id, project, 'queued', 'Tarea en cola', now, None)
        return self.get(task_id)

    def _event(self, task_id, project, status, message, now, duration):
        self.db.execute('INSERT INTO events(task_id,project,status,message,timestamp,duration) VALUES(?,?,?,?,?,?)',
                        (task_id, project, status, message, now, duration))
        event(EVENTS.get(status, 'CODEX_TASK_PROGRESS'), task_id=task_id, project_id=project,
              status=status, message=message, duration=duration)

    def update(self, task_id, **fields):
        allowed = {'status', 'progress_message', 'thread_id', 'turn_id', 'started', 'cancel_requested', 'result'}
        if not fields or set(fields) - allowed:
            raise ValueError('invalid_update')
        old, now = self.get(task_id), time.time()
        if 'result' in fields:
            fields['result'] = json.dumps(fields['result'], ensure_ascii=False)
        fields['updated'] = now
        with self.db:
            self.db.execute('UPDATE tasks SET ' + ','.join(key + '=?' for key in fields) + ' WHERE task_id=?',
                            (*fields.values(), task_id))
            if 'status' in fields and fields['status'] != old['status']:
                self._event(task_id, old['project'], fields['status'], fields.get('progress_message', ''), now,
                            now - old['started'] if old['started'] else None)

    def recover(self):
        rows = list(self.db.execute("SELECT task_id FROM tasks WHERE status='running'"))
        for row in rows:
            self.update(row['task_id'], status='interrupted', progress_message='Ejecución interrumpida; pendiente de reconciliación')

    def events(self, after=0, limit=100):
        return [dict(row) for row in self.db.execute('SELECT * FROM events WHERE sequence>? ORDER BY sequence LIMIT ?', (after, limit))]

    @staticmethod
    def public(task, result=False):
        data = {key: task[key] for key in ('task_id', 'project', 'status', 'progress_message', 'created', 'updated')}
        data['cancel_requested'] = bool(task['cancel_requested'])
        if result:
            data.update(json.loads(task['result']) if task['result'] else
                        dict(summary=None, files_modified=None, tests_passed=None, build_success=None))
        return data

    def close(self):
        self.db.close()
