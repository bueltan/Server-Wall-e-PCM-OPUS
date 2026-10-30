#!/usr/bin/env python3
"""Explicit opt-in real model smoke test, only in a new temporary project.

Uses the configured dedicated home (user login required); does not register or
modify firmware projects. Run with PYTHONPATH=src and PIUU_CODEX_HOME/BINARY set.
"""
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
from bridge.codex.adapter import CodexAdapter
from bridge.codex.policy import Project


async def main():
    with tempfile.TemporaryDirectory(prefix='piuu-smoke-', dir=os.environ.get('PIUU_CODEX_SMOKE_PARENT')) as directory:
        root = Path(directory).resolve()
        (root / 'test_example.py').write_text('import unittest\nclass Example(unittest.TestCase):\n def test_value(self):\n  self.assertEqual(2 + 2, 4)\n')
        project = Project('smoke', 'Disposable smoke test', root)
        adapter = CodexAdapter(os.getenv('PIUU_CODEX_BINARY') or shutil.which('codex'), os.environ['PIUU_CODEX_HOME'])
        async def emit(**fields):
            if 'progress_message' in fields:
                print(fields['progress_message'], flush=True)
        async with asyncio.timeout(180):
            status, result = await adapter.run(dict(task_id='smoke', instruction=
                'Create hello.txt containing PIUU_OK, then run python3 -m unittest -v. '
                'Do not use git or access anything outside this temporary project. '
                'Finish with a short Spanish summary.'), project, emit, asyncio.Event())
        if status != 'completed' or not (root / 'hello.txt').exists():
            print('Smoke evidence:', result, flush=True)
        assert status == 'completed', status
        assert (root / 'hello.txt').read_text().strip() == 'PIUU_OK', 'File was not created'
        assert result['tests_passed'] is True, 'No successful test command was reported'
        print('PASS: real Codex modified the disposable project and ran its test.')


if __name__ == '__main__':
    asyncio.run(main())
