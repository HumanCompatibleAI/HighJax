'''Ozette - TTY-preserving logging for trainer.'''
from __future__ import annotations

from .core import (
    Ozette,
    install,
    uninstall,
    is_installed,
    get_log_path,
    materialize,
    log,
    log_debug,
    log_info,
    log_warning,
    log_error,
)

__all__ = [
    'Ozette',
    'install',
    'uninstall',
    'is_installed',
    'get_log_path',
    'materialize',
    'log',
    'log_debug',
    'log_info',
    'log_warning',
    'log_error',
]
