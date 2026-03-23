'''Text processing utilities for ozette log output.'''
from __future__ import annotations

import re


def _strip_ansi(s: str) -> str:
    '''Remove ANSI escape sequences from string.'''
    return re.sub(r'\x1b\[\??[0-9;]*[A-Za-z~]', '', s)
