from __future__ import annotations

import sys
import time
import pathlib
import tempfile

import pytest

from highjax_trainer.ozette import core as ozette
from highjax_trainer.ozette.core import Ozette, IS_WINDOWS
from highjax_trainer.ozette.metadata import LOG_HEADER_END


class TestLogContent:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_adds_timestamp_to_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_ts', base_dir=pathlib.Path(tmpdir))
            print('Hello world')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_ts' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) >= 1
            assert lines[0][:4].isdigit()
            assert 'Hello world' in lines[0]

    def test_multiple_lines_each_get_timestamp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_multi_ts', base_dir=pathlib.Path(tmpdir))
            print('Line one')
            print('Line two')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_multi_ts' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) >= 2
            for line in lines:
                assert line[:4].isdigit()

    def test_empty_lines_not_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_empty', base_dir=pathlib.Path(tmpdir))
            print('')
            print('')
            print('Visible')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_empty' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 1
            assert 'Visible' in lines[0]

    def test_whitespace_only_lines_not_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_ws', base_dir=pathlib.Path(tmpdir))
            print('   ')
            print('\t')
            print('Content')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_ws' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 1
            assert 'Content' in lines[0]


class TestProgressBarCollapse:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_collapses_progress_to_final(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_progress', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('Step 1/3\r')
            sys.stdout.write('Step 2/3\r')
            sys.stdout.write('Step 3/3\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_progress' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 1
            assert 'Step 3/3' in lines[0]
            assert 'Step 1/3' not in body
            assert 'Step 2/3' not in body

    def test_multiple_progress_bars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_multi_prog', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('Progress A 1/2\r')
            sys.stdout.write('Progress A 2/2\n')
            sys.stdout.write('Progress B 1/2\r')
            sys.stdout.write('Progress B 2/2\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_multi_prog' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 2
            assert 'Progress A 2/2' in lines[0]
            assert 'Progress B 2/2' in lines[1]

    def test_normal_lines_between_progress_bars(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_normal_between', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('Progress 1/2\r')
            sys.stdout.write('Progress 2/2\n')
            print('Normal line')
            sys.stdout.write('Progress2 1/2\r')
            sys.stdout.write('Progress2 2/2\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_normal_between' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 3
            assert 'Progress 2/2' in lines[0]
            assert 'Normal line' in lines[1]
            assert 'Progress2 2/2' in lines[2]

    def test_collapses_nested_progress_bars_with_cursor_up(self):
        from highjax_trainer.ozette import core as ozette_core
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_cursor_up', base_dir=pathlib.Path(tmpdir))
            instance = ozette_core._instance
            log_path = ozette.get_log_path()

            # Simulate nested tqdm: outer bar, then inner updates with cursor-up
            instance._process_for_log('\rOuter: 0%\n')
            instance._process_for_log('\rInner: 0%\x1b[A')
            for i in range(1, 5):
                instance._process_for_log(f'\n\rInner: {i * 25}%\x1b[A')
            instance._process_for_log('\n\rInner: 100%\n')

            time.sleep(0.1)
            ozette.uninstall()
            time.sleep(0.1)

            content = log_path.read_text()
            assert 'Outer: 0%' in content
            assert 'Inner: 25%' not in content
            assert 'Inner: 50%' not in content
            assert 'Inner: 100%' in content

    def test_cursor_up_strips_ansi_escape_from_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_ansi_strip', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('\x1b[32mGreen text\x1b[0m\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_ansi_strip' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '\x1b[' not in body
            assert 'Green text' in body

    def test_handles_crlf_line_endings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_crlf', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('CRLF line\r\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_crlf' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'CRLF line' in body


class TestAnsiStrippingInLogs:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_strips_ansi_from_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_strip_ansi', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('\x1b[31mRed text\x1b[0m\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_strip_ansi' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '\x1b[' not in body
            assert 'Red text' in body

    def test_strips_complex_ansi(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_complex_ansi', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('\x1b[1;31;42mBold red on green\x1b[0m\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_complex_ansi' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert '\x1b[' not in body
            assert 'Bold red on green' in body

    def test_preserves_text_with_brackets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_brackets', base_dir=pathlib.Path(tmpdir))
            print('Array [1, 2, 3] here')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_brackets' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Array [1, 2, 3] here' in body


class TestAlternateScreenMode:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    @pytest.mark.skipif(not IS_WINDOWS, reason='alt screen test requires Windows')
    def test_alt_screen_content_not_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_alt_screen', base_dir=pathlib.Path(tmpdir))
            print('Before alt screen')
            sys.stdout.write('\x1b[?1049h')
            sys.stdout.write('Secret alt content\n')
            sys.stdout.write('\x1b[?1049l')
            print('After alt screen')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_alt_screen' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Before alt screen' in body
            assert 'After alt screen' in body
            assert 'Secret alt content' not in body

    @pytest.mark.skipif(not IS_WINDOWS, reason='alt screen test requires Windows')
    def test_multiple_alt_screen_transitions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_multi_alt', base_dir=pathlib.Path(tmpdir))
            print('Visible 1')
            sys.stdout.write('\x1b[?1049h')
            sys.stdout.write('Hidden 1\n')
            sys.stdout.write('\x1b[?1049l')
            print('Visible 2')
            sys.stdout.write('\x1b[?1049h')
            sys.stdout.write('Hidden 2\n')
            sys.stdout.write('\x1b[?1049l')
            print('Visible 3')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_multi_alt' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Visible 1' in body
            assert 'Visible 2' in body
            assert 'Visible 3' in body
            assert 'Hidden 1' not in body
            assert 'Hidden 2' not in body

    def test_alt_screen_detection_in_process_for_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            instance = Ozette('test_alt_detect', base_dir=pathlib.Path(tmpdir))
            instance._process_for_log('Normal text\n')
            instance._process_for_log('\x1b[?1049h')
            instance._process_for_log('Hidden text\n')
            instance._process_for_log('\x1b[?1049l')
            instance._process_for_log('Visible again\n')
            instance.close()
            time.sleep(0.1)
            log_file = next((pathlib.Path(tmpdir) / '.test_alt_detect' / 'logs').glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Normal text' in body
            assert 'Visible again' in body
            assert 'Hidden text' not in body


class TestStderr:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_stderr_logged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_stderr', base_dir=pathlib.Path(tmpdir))
            sys.stderr.write('Error message\n')
            sys.stderr.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_stderr' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Error message' in body

    def test_mixed_stdout_stderr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_mixed', base_dir=pathlib.Path(tmpdir))
            print('Stdout line')
            sys.stderr.write('Stderr line\n')
            sys.stdout.flush()
            sys.stderr.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_mixed' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Stdout line' in body
            assert 'Stderr line' in body


class TestUnicode:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_handles_unicode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_unicode', base_dir=pathlib.Path(tmpdir))
            print('Hello \u4e16\u754c \u00e9\u00e8\u00ea')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_unicode' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            assert '\u4e16\u754c' in content
            assert '\u00e9\u00e8\u00ea' in content

    def test_handles_emoji(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_emoji', base_dir=pathlib.Path(tmpdir))
            print('\U0001f680 Launch \U0001f3c1 Finish')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_emoji' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            assert '\U0001f680' in content
            assert '\U0001f3c1' in content


class TestEdgeCases:

    def setup_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def teardown_method(self):
        if ozette.is_installed():
            ozette.uninstall()

    def test_very_long_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_long', base_dir=pathlib.Path(tmpdir))
            long_text = 'A' * 10000
            print(long_text)
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_long' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            assert long_text in content

    def test_rapid_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_rapid', base_dir=pathlib.Path(tmpdir))
            for i in range(100):
                print(f'Rapid line {i}')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_rapid' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            lines = [l for l in body.strip().splitlines() if l.strip()]
            assert len(lines) == 100

    def test_special_characters(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_special', base_dir=pathlib.Path(tmpdir))
            print('Tabs\there\tand\tthere')
            print('Backslash \\ and slash /')
            print('Quotes " and \'')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_special' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            assert 'Tabs' in content
            assert 'Backslash' in content
            assert 'Quotes' in content

    def test_partial_writes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ozette.install('test_partial', base_dir=pathlib.Path(tmpdir))
            sys.stdout.write('Part')
            sys.stdout.write('ial ')
            sys.stdout.write('line\n')
            sys.stdout.flush()
            ozette.uninstall()
            time.sleep(0.2)
            log_path = pathlib.Path(tmpdir) / '.test_partial' / 'logs'
            log_file = next(log_path.glob('*.log'))
            content = log_file.read_text()
            body = content.split(LOG_HEADER_END, 1)[1]
            assert 'Partial line' in body
