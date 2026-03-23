from __future__ import annotations

import pathlib

import fastparquet
import pandas as pd


_DTYPES = {
    'epoch': 'int64',
    'e': 'int64',
    'alive_count': 'int64',
    'alive.count': 'int64',
}


class ParquetWriter:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self._columns: list[str] | None = None
        self._in_context = False

    def write(self, row: dict) -> None:
        if self._columns is None:
            self._columns = list(row.keys())
        else:
            new_keys = set(row.keys()) - set(self._columns)
            if new_keys:
                self._columns.extend(sorted(new_keys))

        df = pd.DataFrame([row], columns=self._columns)
        self._write_df(df)

    def write_batch(self, rows: list[dict]) -> None:
        if not rows:
            return
        if self._columns is None:
            self._columns = list(rows[0].keys())
            for row in rows[1:]:
                new_keys = set(row.keys()) - set(self._columns)
                if new_keys:
                    self._columns.extend(sorted(new_keys))
        else:
            for row in rows:
                new_keys = set(row.keys()) - set(self._columns)
                if new_keys:
                    self._columns.extend(sorted(new_keys))

        df = pd.DataFrame(rows, columns=self._columns)
        self._write_df(df)

    def _write_df(self, df: pd.DataFrame) -> None:
        for col, dtype in _DTYPES.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype(object)
        append = self.path.exists() and self.path.stat().st_size > 0
        fastparquet.write(
            str(self.path), df, compression='zstd', append=append,
        )

    def flush(self) -> None:
        pass

    def __enter__(self):
        self._in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._in_context = False
        return False
