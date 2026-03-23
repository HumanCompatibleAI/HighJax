'''Log metadata handling for ozette.'''
from __future__ import annotations

import datetime
import os
import re
import socket
import sys
from typing import Optional

LOG_HEADER_END = '--- LOG START ---\n'


def sanitize_for_filename(s: str, max_len: int = 50) -> str:
    '''Encode a string to be filename-safe.'''
    sanitized = re.sub(r'[^a-zA-Z0-9.\-_]+', '_', s)
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized


def make_log_filename(timestamp: datetime.datetime, argv: Optional[list[str]] = None) -> str:
    '''Create log filename from timestamp and optional argv.'''
    ts_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')

    if argv and len(argv) > 1:
        cmd_str = ' '.join(argv[1:])
        sanitized = sanitize_for_filename(cmd_str, max_len=40)
        if sanitized:
            return f'{ts_str}_{sanitized}.log'

    return f'{ts_str}.log'


def make_yaml_header(
    timestamp: datetime.datetime,
    argv: Optional[list[str]] = None,
) -> str:
    '''Create YAML header for log file.'''
    lines = []

    lines.append(f'datetime: {timestamp.isoformat()}')

    if argv:
        cmd_str = ' '.join(argv)
        lines.append(f'command: {_yaml_quote(cmd_str)}')

    try:
        cwd = os.getcwd()
    except OSError:
        cwd = None
    if cwd:
        lines.append(f'cwd: {_yaml_quote(cwd)}')

    user = os.environ.get('USER') or os.environ.get('USERNAME')
    if user:
        lines.append(f'user: {_yaml_quote(user)}')

    machine = os.environ.get('NAME') or socket.gethostname()
    if machine:
        lines.append(f'machine: {_yaml_quote(machine)}')

    lines.append(f'pid: {os.getpid()}')

    lines.append(LOG_HEADER_END.rstrip())

    return '\n'.join(lines) + '\n'


def _yaml_quote(s: str) -> str:
    '''Quote a string for YAML only when necessary, using single quotes.'''
    if not s:
        return "''"
    needs_quote = (
        s[0] in '&*!|>\'"@`%{}[]-' or
        ': ' in s or
        ' #' in s or
        '\n' in s
    )
    if needs_quote:
        escaped = s.replace("'", "''")
        return f"'{escaped}'"
    return s


def parse_yaml_header(content: str) -> tuple[dict, str]:
    '''Parse YAML header from log content.'''
    try:
        if LOG_HEADER_END not in content:
            return {}, content

        header_end_pos = content.index(LOG_HEADER_END)
        header_str = content[:header_end_pos]
        log_content = content[header_end_pos + len(LOG_HEADER_END):]

        metadata = {}
        for line in header_str.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if ': ' in line:
                key, value = line.split(': ', 1)
                value = _yaml_unquote(value)
                metadata[key] = value

        if 'datetime' in metadata:
            try:
                metadata['datetime'] = datetime.datetime.fromisoformat(metadata['datetime'])
            except ValueError:
                pass

        return metadata, log_content
    except Exception:
        return {}, content


def _yaml_unquote(s: str) -> str:
    '''Unquote a YAML string.'''
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
        s = s.replace('\\"', '"').replace('\\\\', '\\')
    elif s.startswith("'") and s.endswith("'"):
        s = s[1:-1]
        s = s.replace("''", "'")
    return s
