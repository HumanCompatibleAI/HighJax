from __future__ import annotations

import sys
import os
import pathlib
import tempfile
import time

import pytest

from highjax_trainer.ozette import core as ozette
from highjax_trainer.ozette.metadata import LOG_HEADER_END


class TestDirectLogging:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_log_writes_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_log_write', base_dir=pathlib.Path(tmpdir))
            ozette.log('Direct message')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_log_write' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Direct message' in body

    def test_log_debug_has_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_debug', base_dir=pathlib.Path(tmpdir))
            ozette.log_debug('Debug info')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_debug' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '[DEBUG]' in body
            assert 'Debug info' in body

    def test_log_info_has_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_info', base_dir=pathlib.Path(tmpdir))
            ozette.log_info('Info message')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_info' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '[INFO]' in body
            assert 'Info message' in body

    def test_log_warning_has_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_warning', base_dir=pathlib.Path(tmpdir))
            ozette.log_warning('Warning message')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_warning' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '[WARNING]' in body
            assert 'Warning message' in body

    def test_log_error_has_prefix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_error', base_dir=pathlib.Path(tmpdir))
            ozette.log_error('Error message')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_error' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '[ERROR]' in body
            assert 'Error message' in body

    def test_log_has_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_log_ts', base_dir=pathlib.Path(tmpdir))
            ozette.log('Timestamped')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_log_ts' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) >= 1
            assert lines[0][:4].isdigit()
            assert 'T' in lines[0][:20]

    def test_log_not_on_screen(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_no_screen', base_dir=pathlib.Path(tmpdir))
            ozette.log('Silent message')
            ozette.uninstall()
            time.sleep(0.1)
            captured = capsys.readouterr()
            assert 'Silent message' not in captured.out
            assert 'Silent message' not in captured.err

    def test_log_safe_when_not_installed(self):
        ozette.log('This should not crash')
        ozette.log_debug('Nor this')
        ozette.log_info('Nor this')
        ozette.log_warning('Nor this')
        ozette.log_error('Nor this')

    def test_log_with_pid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_pid', base_dir=pathlib.Path(tmpdir), include_pid=True)
            ozette.log('PID message')
            ozette.uninstall()
            time.sleep(0.1)
            log_path = pathlib.Path(tmpdir) / '.test_pid' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            pid = str(os.getpid())
            assert f'[{pid}]' in body
            assert 'PID message' in body
