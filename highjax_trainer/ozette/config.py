'''Configuration loading and log rotation for ozette.'''
from __future__ import annotations

import json
import pathlib
from typing import Optional


def _load_config(config_path: Optional[pathlib.Path] = None) -> dict:
    '''Load config from ~/.ozette/config.json or specified path.'''
    if config_path is None:
        config_path = pathlib.Path.home() / '.ozette' / 'config.json'
    default = {
        'max_folder_size_mb': 10,
        'disabled_apps': [],
    }
    if config_path.exists():
        try:
            with open(config_path) as f:
                user_config = json.load(f)
                default.update(user_config)
        except (json.JSONDecodeError, IOError):
            pass
    return default


def _cleanup_old_logs(log_dir: pathlib.Path, max_size_bytes: int) -> None:
    '''Delete oldest log files if folder exceeds max size.'''
    if not log_dir.exists():
        return

    files = sorted(log_dir.glob('*.log'), key=lambda f: f.stat().st_mtime)
    total_size = sum(f.stat().st_size for f in files)

    while total_size > max_size_bytes and len(files) > 1:
        oldest = files.pop(0)
        total_size -= oldest.stat().st_size
        oldest.unlink()
