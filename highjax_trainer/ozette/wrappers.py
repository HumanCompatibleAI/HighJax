'''Stream wrappers for ozette logging.'''
from __future__ import annotations


class _FakeBufferWrapper:
    '''Wrap a binary buffer to intercept writes for logging.'''

    def __init__(self, wrapped_buffer, log_processor):
        self._wrapped = wrapped_buffer
        self._log_processor = log_processor

    def write(self, b):
        result = self._wrapped.write(b)
        try:
            text = b.decode('utf-8', errors='replace')
            self._log_processor(text)
        except Exception:
            pass
        return result

    def fileno(self):
        raise OSError('ozette wrapper has no fileno')

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _NullWriter:
    '''Null writer for when stdout/stderr is None.'''

    def write(self, s):
        return len(s) if isinstance(s, str) else len(s) if s else 0

    def flush(self):
        pass

    def fileno(self):
        raise OSError('null writer has no fileno')

    def isatty(self):
        return False

    @property
    def encoding(self):
        return 'utf-8'

    @property
    def buffer(self):
        return None


class _TeeWrapper:
    '''Wrap a stream to tee writes to a log processor.'''

    def __init__(self, wrapped, log_processor):
        if wrapped is None:
            wrapped = _NullWriter()
        self._wrapped = wrapped
        self._log_processor = log_processor

    def isatty(self):
        return self._wrapped.isatty()

    def write(self, s):
        result = self._wrapped.write(s)
        self._log_processor(s)
        return result

    def flush(self):
        return self._wrapped.flush()

    @property
    def encoding(self):
        return getattr(self._wrapped, 'encoding', 'utf-8')

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


class _FakeTTYWrapper:
    '''Wrap stdout and fake isatty()=True. Used for TTY streams on Windows.'''

    def __init__(self, wrapped, log_processor):
        if wrapped is None:
            wrapped = _NullWriter()
        self._wrapped = wrapped
        self._log_processor = log_processor
        if hasattr(wrapped, 'buffer') and wrapped.buffer is not None:
            self._buffer = _FakeBufferWrapper(wrapped.buffer, log_processor)
        else:
            self._buffer = None

    def isatty(self):
        return True

    def write(self, s):
        result = self._wrapped.write(s)
        self._log_processor(s)
        return result

    def flush(self):
        return self._wrapped.flush()

    def fileno(self):
        raise OSError('ozette wrapper has no fileno')

    @property
    def encoding(self):
        return getattr(self._wrapped, 'encoding', 'utf-8')

    @property
    def buffer(self):
        if self._buffer is not None:
            return self._buffer
        return getattr(self._wrapped, 'buffer', None)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)
