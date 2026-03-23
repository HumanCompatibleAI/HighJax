'''Tests for ozette core functionality: install, uninstall, TTY wrapper, config.'''
from __future__ import annotations

import sys
import io
import pathlib
import tempfile
import time

import pytest

from highjax_trainer.ozette import core as ozette
from highjax_trainer.ozette.processing import _strip_ansi
from highjax_trainer.ozette.config import _cleanup_old_logs, _load_config
from highjax_trainer.ozette.wrappers import _FakeTTYWrapper
from highjax_trainer.ozette.core import IS_WINDOWS, Ozette


class TestStripAnsi:

    def test_strips_basic_color_codes(self):
        text = '\x1b[31mRed\x1b[0m \x1b[32mGreen\x1b[0m'
        assert _strip_ansi(text) == 'Red Green'

    def test_strips_bold(self):
        text = '\x1b[1mBold\x1b[0m'
        assert _strip_ansi(text) == 'Bold'

    def test_strips_underline(self):
        text = '\x1b[4mUnderline\x1b[0m'
        assert _strip_ansi(text) == 'Underline'

    def test_preserves_plain_text(self):
        assert _strip_ansi('Hello World') == 'Hello World'

    def test_strips_multiple_codes_combined(self):
        text = '\x1b[1;31;42mComplex\x1b[0m'
        assert _strip_ansi(text) == 'Complex'

    def test_strips_256_color_codes(self):
        text = '\x1b[38;5;196mRed256\x1b[0m'
        assert _strip_ansi(text) == 'Red256'

    def test_strips_rgb_color_codes(self):
        text = '\x1b[38;2;255;0;0mTrueColor\x1b[0m'
        assert _strip_ansi(text) == 'TrueColor'

    def test_handles_empty_string(self):
        assert _strip_ansi('') == ''

    def test_handles_only_escape_codes(self):
        assert _strip_ansi('\x1b[31m\x1b[0m') == ''

    def test_preserves_newlines(self):
        text = '\x1b[32mLine1\x1b[0m\nLine2'
        assert _strip_ansi(text) == 'Line1\nLine2'


class TestFakeTTYWrapper:

    def test_isatty_returns_true(self):
        wrapper = _FakeTTYWrapper(io.StringIO(), lambda x: None)
        assert wrapper.isatty() is True

    def test_write_passes_through(self):
        wrapped = io.StringIO()
        wrapper = _FakeTTYWrapper(wrapped, lambda x: None)
        wrapper.write('Hello')
        assert wrapped.getvalue() == 'Hello'

    def test_write_returns_length(self):
        wrapper = _FakeTTYWrapper(io.StringIO(), lambda x: None)
        assert wrapper.write('Hello') == 5

    def test_calls_log_processor(self):
        logged = []
        wrapper = _FakeTTYWrapper(io.StringIO(), lambda x: logged.append(x))
        wrapper.write('Test')
        assert logged == ['Test']

    def test_flush_passes_through(self):
        wrapper = _FakeTTYWrapper(io.StringIO(), lambda x: None)
        wrapper.write('Test')
        wrapper.flush()


class TestCleanupOldLogs:

    def test_keeps_files_under_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = pathlib.Path(tmpdir)
            for i in range(3):
                (log_dir / f'{i}.log').write_text('x' * 100)
            _cleanup_old_logs(log_dir, 1000)
            assert len(list(log_dir.glob('*.log'))) == 3

    def test_deletes_oldest_when_over_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = pathlib.Path(tmpdir)
            for i in range(5):
                (log_dir / f'{i}.log').write_text('x' * 100)
                time.sleep(0.02)
            _cleanup_old_logs(log_dir, 300)
            remaining = list(log_dir.glob('*.log'))
            assert len(remaining) <= 3
            assert not (log_dir / '0.log').exists()

    def test_handles_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _cleanup_old_logs(pathlib.Path(tmpdir), 1000)

    def test_handles_nonexistent_directory(self):
        _cleanup_old_logs(pathlib.Path('/nonexistent/path'), 1000)

    def test_keeps_at_least_one_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = pathlib.Path(tmpdir)
            (log_dir / 'only.log').write_text('x' * 1000)
            _cleanup_old_logs(log_dir, 100)
            assert len(list(log_dir.glob('*.log'))) == 1


class TestConfigLoading:

    def test_returns_defaults_when_no_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = _load_config(config_path=pathlib.Path(tmpdir) / 'config.json')
            assert config['max_folder_size_mb'] == 10
            assert config['disabled_apps'] == []

    def test_loads_custom_max_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / 'config.json'
            config_path.write_text('{"max_folder_size_mb": 50}')
            config = _load_config(config_path=config_path)
            assert config['max_folder_size_mb'] == 50

    def test_handles_invalid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = pathlib.Path(tmpdir) / 'config.json'
            config_path.write_text('not valid json {{{')
            config = _load_config(config_path=config_path)
            assert config['max_folder_size_mb'] == 10


class TestOzetteInstall:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_install_preserves_stdout_isatty(self, temp_base_dir):
        original_isatty = sys.stdout.isatty()
        ozette.install('test_install', base_dir=temp_base_dir)
        assert sys.stdout.isatty() is original_isatty

    def test_install_preserves_stderr_isatty(self, temp_base_dir):
        original_isatty = sys.stderr.isatty()
        ozette.install('test_stderr', base_dir=temp_base_dir)
        assert sys.stderr.isatty() is original_isatty

    def test_is_installed_reflects_state(self, temp_base_dir):
        assert ozette.is_installed() is False
        ozette.install('test_state', base_dir=temp_base_dir)
        assert ozette.is_installed() is True
        ozette.uninstall()
        assert ozette.is_installed() is False

    def test_get_log_path_returns_path(self, temp_base_dir):
        ozette.install('test_log_path', base_dir=temp_base_dir)
        log_path = ozette.get_log_path()
        assert log_path is not None
        assert 'test_log_path' in str(log_path)
        assert log_path.suffix == '.log'

    def test_get_log_path_none_when_not_installed(self):
        assert ozette.get_log_path() is None

    def test_double_install_raises(self, temp_base_dir):
        ozette.install('test_double', base_dir=temp_base_dir)
        with pytest.raises(RuntimeError, match='already installed'):
            ozette.install('test_double_2', base_dir=temp_base_dir)

    def test_creates_log_file(self, temp_base_dir):
        ozette.install('test_creates_log', base_dir=temp_base_dir)
        log_path = ozette.get_log_path()
        print('Test message')
        sys.stdout.flush()
        ozette.uninstall()
        time.sleep(0.2)
        assert log_path.exists()
        content = log_path.read_text()
        assert 'Test message' in content

    def test_creates_log_directory(self, temp_base_dir):
        ozette.install('test_creates_dir', base_dir=temp_base_dir)
        expected = temp_base_dir / '.test_creates_dir' / 'logs'
        assert expected.exists()

    def test_log_path_format(self, temp_base_dir):
        import re
        ozette.install('test_path_fmt', base_dir=temp_base_dir)
        log_path = ozette.get_log_path()
        assert log_path.parent.name == 'logs'
        assert log_path.parent.parent.name == '.test_path_fmt'
        assert re.match(r'\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}', log_path.name)
        assert log_path.name.endswith('.log')

    def test_install_returns_instance(self, temp_base_dir):
        instance = ozette.install('test_returns', base_dir=temp_base_dir)
        assert isinstance(instance, Ozette)
        assert instance.app_name == 'test_returns'


class TestOzetteClass:

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_app_name_stored(self, temp_base_dir):
        instance = Ozette('myapp', base_dir=temp_base_dir)
        assert instance.app_name == 'myapp'
        instance.close()

    def test_log_dir_path(self, temp_base_dir):
        instance = Ozette('testapp', base_dir=temp_base_dir)
        assert instance.log_dir == temp_base_dir / '.testapp' / 'logs'
        instance.close()

    def test_close_idempotent(self, temp_base_dir):
        instance = Ozette('test_close', base_dir=temp_base_dir)
        instance.close()
        instance.close()
        instance.close()
