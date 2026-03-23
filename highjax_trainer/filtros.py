from __future__ import annotations

import sys
import os
import re


class FiltrosStream:
    def __init__(self, original_stream, patterns=()):
        self.original_stream = original_stream
        self.patterns = tuple(patterns)
        self.combined_pattern = re.compile(
            ('|'.join(f'(?:{pattern})' for pattern in patterns)) if patterns else r'(?!x)x'
        )
        self.last_printed = True

    def write(self, message):
        if message.isspace() and not self.last_printed:
            return
        if self.combined_pattern.search(message):
            self.last_printed = False
            return
        else:
            self.last_printed = True
            self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()

    def close(self):
        pass


def filtros():

    if filtros.called:
        return

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    sys.stderr = FiltrosStream(
        sys.stderr,
        (
        )
    )

    filtros.called = True


filtros.called = False
