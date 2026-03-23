'''Ozette core implementation - cross-platform TTY-preserving logging.

Linux: Uses real PTY (pty.openpty) with reader thread.
Windows: Uses fake-TTY wrapper that fakes isatty()=True.
'''
from __future__ import annotations

import sys
import os
import datetime
import pathlib
import atexit
from typing import Optional

from .metadata import make_log_filename, make_yaml_header
from .config import _load_config, _cleanup_old_logs
from .processing import _strip_ansi
from .wrappers import _FakeBufferWrapper, _NullWriter, _TeeWrapper, _FakeTTYWrapper

IS_WINDOWS = sys.platform == 'win32'

if not IS_WINDOWS:
    import pty
    import select
    import threading

# Global instance
_instance: Optional[Ozette] = None


class Ozette:
    '''TTY-preserving logging that works on both Linux and Windows.

    Can start in buffer-only mode (no file), then materialize to a file later.
    '''

    _ALT_SCREEN_ENTER = '\x1b[?1049h'
    _ALT_SCREEN_EXIT = '\x1b[?1049l'

    def __init__(self, app_name: str, include_pid: bool = False, show_path: bool = False,
                 base_dir: Optional[pathlib.Path] = None, log_dir: Optional[pathlib.Path] = None,
                 log_path: Optional[pathlib.Path] = None, rotate: bool = True,
                 buffer_only: bool = False):
        self.app_name = app_name
        self.include_pid = include_pid
        self.show_path = show_path
        self._pid = os.getpid() if include_pid else None
        self.config = _load_config()

        env_base_dir = os.environ.get('OZETTE_BASE_DIR')
        if base_dir is not None:
            self._base_dir = base_dir
        elif env_base_dir:
            self._base_dir = pathlib.Path(env_base_dir)
        else:
            self._base_dir = pathlib.Path.home()

        self._closed = False
        if app_name in self.config.get('disabled_apps', []):
            self._disabled = True
            self.log_path = None
            return
        self._disabled = False

        self._timestamp = datetime.datetime.now()
        self._argv = list(sys.argv)

        # Buffer-only mode: capture to memory, materialize to file later
        self._buffering = buffer_only
        self._log_buffer = []  # list of strings (complete lines)
        self._log_file = None
        self.log_path = None
        self.log_dir = None

        if not buffer_only:
            self._open_log_file(log_dir=log_dir, log_path=log_path, rotate=rotate)

        self._line_buffer = ''
        self._pending_cr = False
        self._in_alt_screen = False
        self._in_escape_seq = False
        self._escape_buffer = ''
        self._cursor_up_count = 0

        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr

        if IS_WINDOWS:
            self._init_windows()
        else:
            self._init_linux()

        if self.show_path and self.log_path is not None:
            self._print_log_path()

        atexit.register(self.close)

    def _open_log_file(self, *, log_dir=None, log_path=None, rotate=True):
        '''Open the log file (either from explicit path or auto-generated).'''
        if log_dir is not None:
            self.log_dir = log_dir
        elif self.log_dir is None:
            self.log_dir = self._base_dir / f'.{self.app_name}' / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)

        if rotate:
            max_size = self.config.get('max_folder_size_mb', 10) * 1024 * 1024
            _cleanup_old_logs(self.log_dir, max_size)

        if log_path is not None:
            self.log_path = log_path
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            filename = make_log_filename(self._timestamp, self._argv)
            self.log_path = self.log_dir / filename

        self._log_file = open(self.log_path, 'w', encoding='utf-8')

        header = make_yaml_header(self._timestamp, self._argv)
        self._log_file.write(header)
        self._log_file.flush()

    def materialize(self, log_path: pathlib.Path) -> None:
        '''Switch from buffer-only mode to writing to a file.

        Flushes all buffered log lines to the file, then continues
        writing directly to it.
        '''
        if self._disabled or self._closed:
            return
        if self._log_file is not None:
            return  # already materialized

        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = open(self.log_path, 'w', encoding='utf-8')

        header = make_yaml_header(self._timestamp, self._argv)
        self._log_file.write(header)

        # Flush buffer
        for line in self._log_buffer:
            self._log_file.write(line)
        self._log_file.flush()
        self._log_buffer.clear()
        self._buffering = False

        if self.show_path:
            self._print_log_path()

    def _write_log_line(self, line: str) -> None:
        '''Write a complete log line to file or buffer.'''
        if self._buffering:
            self._log_buffer.append(line)
        elif self._log_file is not None:
            self._log_file.write(line)
            self._log_file.flush()

    def _init_windows(self):
        '''Windows: Use fake-TTY wrapper for TTY streams, tee wrapper for pipes.'''
        if self._orig_stdout is not None and self._orig_stdout.isatty():
            sys.stdout = _FakeTTYWrapper(self._orig_stdout, self._process_for_log)
        else:
            sys.stdout = _TeeWrapper(self._orig_stdout, self._process_for_log)
        if self._orig_stderr is not None and self._orig_stderr.isatty():
            sys.stderr = _FakeTTYWrapper(self._orig_stderr, self._process_for_log)
        else:
            sys.stderr = _TeeWrapper(self._orig_stderr, self._process_for_log)

    def _init_linux(self):
        '''Linux: Use PTYs for TTY streams, tee wrappers for pipes.'''
        import termios

        self._master_out_fd = None
        self._master_err_fd = None
        self._slave_out_fd = None
        self._slave_err_fd = None

        stdout_is_tty = self._orig_stdout is not None and self._orig_stdout.isatty()
        stderr_is_tty = self._orig_stderr is not None and self._orig_stderr.isatty()

        if stdout_is_tty:
            self._master_out_fd, self._slave_out_fd = pty.openpty()
            attrs = termios.tcgetattr(self._slave_out_fd)
            attrs[1] &= ~termios.ONLCR
            termios.tcsetattr(self._slave_out_fd, termios.TCSANOW, attrs)
            self._slave_out = os.fdopen(self._slave_out_fd, 'w', buffering=1, closefd=False)
            sys.stdout = self._slave_out
        else:
            sys.stdout = _TeeWrapper(self._orig_stdout, self._process_for_log)

        if stderr_is_tty:
            self._master_err_fd, self._slave_err_fd = pty.openpty()
            attrs = termios.tcgetattr(self._slave_err_fd)
            attrs[1] &= ~termios.ONLCR
            termios.tcsetattr(self._slave_err_fd, termios.TCSANOW, attrs)
            self._slave_err = os.fdopen(self._slave_err_fd, 'w', buffering=1, closefd=False)
            sys.stderr = self._slave_err
        else:
            sys.stderr = _TeeWrapper(self._orig_stderr, self._process_for_log)

        if self._master_out_fd is not None or self._master_err_fd is not None:
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
        else:
            self._stop = None
            self._thread = None

    def _print_log_path(self):
        '''Print log path in dim gray to stderr.'''
        msg = f'\x1b[90m{self.log_path}\x1b[0m\n'
        self._orig_stderr.write(msg)
        self._orig_stderr.flush()

    def _reader_loop(self):
        '''Linux: Read from PTY masters, forward to original streams and log.'''
        master_fds = [fd for fd in (self._master_out_fd, self._master_err_fd)
                      if fd is not None]
        while not self._stop.is_set():
            ready, _, _ = select.select(master_fds, [], [], 0.02)
            for fd in ready:
                try:
                    data = os.read(fd, 4096)
                    if data:
                        text = data.decode('utf-8', errors='replace')
                        if fd == self._master_out_fd:
                            self._orig_stdout.write(text)
                            self._orig_stdout.flush()
                        else:
                            self._orig_stderr.write(text)
                            self._orig_stderr.flush()
                        self._process_for_log(text)
                except OSError:
                    break

    def _process_for_log(self, text: str | bytes) -> None:
        '''Process text for logging: handle \\r, strip ANSI, detect alt screen,
        collapse progress bars.'''
        if isinstance(text, bytes):
            text = text.decode('utf-8', errors='replace')
        for char in text:
            if self._in_escape_seq:
                self._escape_buffer += char
                if len(self._escape_buffer) == 1 and char != '[':
                    self._in_escape_seq = False
                    self._escape_buffer = ''
                elif len(self._escape_buffer) > 1 and (char.isalpha() or char == '~'):
                    self._in_escape_seq = False
                    seq = self._escape_buffer
                    if seq == '[?1049h':
                        self._in_alt_screen = True
                    elif seq == '[?1049l':
                        self._in_alt_screen = False
                    elif char == 'A':
                        count_str = seq[1:-1]
                        count = int(count_str) if count_str.isdigit() else 1
                        self._cursor_up_count += count
                    self._escape_buffer = ''
                continue

            if char == '\x1b':
                self._in_escape_seq = True
                self._escape_buffer = ''
                continue

            if self._in_alt_screen:
                continue

            if char == '\r':
                self._pending_cr = True
            elif char == '\n':
                if self._pending_cr:
                    self._pending_cr = False
                if self._cursor_up_count > 0:
                    self._cursor_up_count -= 1
                    self._line_buffer = ''
                else:
                    clean = _strip_ansi(self._line_buffer)
                    if clean.strip():
                        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                        if self._pid is not None:
                            self._write_log_line(f'{ts} [{self._pid}] {clean}\n')
                        else:
                            self._write_log_line(f'{ts} {clean}\n')
                    self._line_buffer = ''
            else:
                if self._pending_cr:
                    self._line_buffer = ''
                    self._pending_cr = False
                self._line_buffer += char

    def close(self) -> None:
        '''Cleanup and restore original stdout/stderr.'''
        if self._closed or self._disabled:
            return
        self._closed = True

        try:
            atexit.unregister(self.close)
        except Exception:
            pass

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        if self._line_buffer.strip():
            clean = _strip_ansi(self._line_buffer)
            if clean.strip():
                ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
                if self._pid is not None:
                    self._write_log_line(f'{ts} [{self._pid}] {clean}\n')
                else:
                    self._write_log_line(f'{ts} {clean}\n')

        if IS_WINDOWS:
            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr
        else:
            if self._thread is not None:
                import time
                time.sleep(0.1)
                self._stop.set()
                self._thread.join(timeout=1.0)

                master_fds = [fd for fd in (self._master_out_fd, self._master_err_fd)
                              if fd is not None]
                while master_fds:
                    ready, _, _ = select.select(master_fds, [], [], 0.02)
                    if ready:
                        for fd in ready:
                            try:
                                data = os.read(fd, 4096)
                                if data:
                                    self._process_for_log(data.decode('utf-8', errors='replace'))
                            except Exception:
                                pass
                    else:
                        break

            sys.stdout = self._orig_stdout
            sys.stderr = self._orig_stderr

            for fd in (self._master_out_fd, self._slave_out_fd,
                       self._master_err_fd, self._slave_err_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except Exception:
                        pass

        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass

        global _instance
        if _instance is self:
            _instance = None


def install(app_name: str, include_pid: bool = False, show_path: bool = True,
            base_dir: Optional[pathlib.Path] = None, log_dir: Optional[pathlib.Path] = None,
            log_path: Optional[pathlib.Path] = None, rotate: bool = True,
            buffer_only: bool = False) -> Ozette:
    '''Install ozette logging for the current process.'''
    global _instance
    if _instance is not None:
        raise RuntimeError('Ozette already installed. Call uninstall() first.')

    _instance = Ozette(app_name, include_pid=include_pid, show_path=show_path,
                       base_dir=base_dir, log_dir=log_dir, log_path=log_path,
                       rotate=rotate, buffer_only=buffer_only)
    return _instance


def uninstall() -> None:
    '''Uninstall ozette and restore original stdout/stderr.'''
    global _instance
    if _instance is not None:
        _instance.close()
        _instance = None


def is_installed() -> bool:
    '''Check if ozette is currently installed.'''
    return _instance is not None


def get_log_path() -> Optional[pathlib.Path]:
    '''Get the current log file path, or None if not installed.'''
    if _instance is None:
        return None
    return _instance.log_path


def materialize(log_path: pathlib.Path) -> None:
    '''Materialize a buffered ozette instance to a file.'''
    if _instance is not None and not _instance._disabled:
        _instance.materialize(log_path)


def _log_direct(msg: str, level: Optional[str] = None) -> None:
    '''Write a message directly to the log file (internal helper).'''
    if _instance is None or _instance._disabled or _instance._closed:
        return
    ts = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]
    parts = [ts]
    if _instance._pid is not None:
        parts.append(f'[{_instance._pid}]')
    if level:
        parts.append(f'[{level}]')
    parts.append(msg)
    _instance._write_log_line(' '.join(parts) + '\n')


def log(msg: str) -> None:
    '''Write a message directly to the log file without showing on screen.'''
    _log_direct(msg)


def log_debug(msg: str) -> None:
    '''Write a debug message directly to the log file.'''
    _log_direct(msg, 'DEBUG')


def log_info(msg: str) -> None:
    '''Write an info message directly to the log file.'''
    _log_direct(msg, 'INFO')


def log_warning(msg: str) -> None:
    '''Write a warning message directly to the log file.'''
    _log_direct(msg, 'WARNING')


def log_error(msg: str) -> None:
    '''Write an error message directly to the log file.'''
    _log_direct(msg, 'ERROR')
