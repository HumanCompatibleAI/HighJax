from __future__ import annotations

import pathlib

import fastparquet
import pandas as pd

ACTION_NAMES = {0: 'left', 1: 'idle', 2: 'right', 3: 'faster', 4: 'slower'}

_TRAINER_COLUMNS = ('v', 'tendency', 'advantage', 'nz_advantage')
_EGO_COLUMNS = ('state.ego_x', 'state.ego_y', 'state.ego_heading', 'state.ego_speed')
_ACTION_PROB_COLUMNS = ('p.left', 'p.idle', 'p.right', 'p.faster', 'p.slower')


def _build_columns(n_npcs: int) -> list[str]:
    cols = ['epoch', 'e', 't', 'reward', 'state.crashed',
            'action', 'action_name']
    cols.extend(_ACTION_PROB_COLUMNS)
    cols.extend(_TRAINER_COLUMNS)
    cols.extend(_EGO_COLUMNS)
    for i in range(n_npcs):
        for suffix in ('x', 'y', 'heading', 'speed'):
            cols.append(f'state.npc{i}_{suffix}')
    return cols


class EsParquetWriter:
    def __init__(self, path: pathlib.Path, n_npcs: int):
        '''Parquet writer for sample_es.parquet using fastparquet append.

        Each write_epoch appends a row group with a valid footer,
        so the file is always readable mid-training.
        '''
        self.path = path
        self.n_npcs = n_npcs
        self._columns = _build_columns(n_npcs)

    def write_epoch(self, epoch: int, rows: list[dict]):
        '''Write all rows for one epoch as a single row group.'''
        df = pd.DataFrame(rows, columns=self._columns)
        df['epoch'] = df['epoch'].astype('int64')
        df['e'] = df['e'].astype('int64')
        # Convert string columns to object for fastparquet compat
        for col in ('action', 'action_name'):
            df[col] = df[col].astype(object)
        append = self.path.exists() and self.path.stat().st_size > 0
        fastparquet.write(str(self.path), df, compression='zstd', append=append)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
