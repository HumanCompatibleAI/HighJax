from __future__ import annotations

import pyarrow.parquet as pq

from highjax_trainer.parquet_writing import ParquetWriter


def test_write_and_read_back(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write({'epoch': 0, 'value': 1.0})
    w.write({'epoch': 1, 'value': 2.0})
    w.write({'epoch': 2, 'value': 3.0})

    table = pq.read_table(path)
    assert table.num_rows == 3
    assert table.column('epoch').to_pylist() == [0, 1, 2]
    assert table.column('value').to_pylist() == [1.0, 2.0, 3.0]


def test_write_batch(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write_batch([
        {'epoch': 0, 'x': 1.0},
        {'epoch': 0, 'x': 2.0},
        {'epoch': 0, 'x': 3.0},
    ])

    table = pq.read_table(path)
    assert table.num_rows == 3


def test_zstd_compression(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write({'epoch': 0, 'value': 1.0})

    pf = pq.ParquetFile(path)
    row_group = pf.metadata.row_group(0)
    for i in range(row_group.num_columns):
        col = row_group.column(i)
        assert col.compression == 'ZSTD', (
            f'{col.path_in_schema} uses {col.compression}'
        )


def test_integer_types(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write({'epoch': 5, 'alive.count': 100, 'x': 1.5})

    table = pq.read_table(path)
    assert table.schema.field('epoch').type == 'int64'
    assert table.schema.field('alive.count').type == 'int64'


def test_readable_mid_writing(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write({'epoch': 0, 'value': 1.0})

    table = pq.read_table(path)
    assert table.num_rows == 1

    w.write({'epoch': 1, 'value': 2.0})
    table = pq.read_table(path)
    assert table.num_rows == 2


def test_context_manager(tmp_path):
    path = tmp_path / 'test.pq'
    with ParquetWriter(path) as w:
        w.write({'epoch': 0, 'value': 1.0})

    table = pq.read_table(path)
    assert table.num_rows == 1


def test_multiple_row_groups(tmp_path):
    path = tmp_path / 'test.pq'
    w = ParquetWriter(path)
    w.write({'epoch': 0, 'a': 1.0, 'b': 2.0})
    w.write({'epoch': 1, 'a': 3.0, 'b': 4.0})

    pf = pq.ParquetFile(path)
    assert pf.metadata.num_row_groups == 2

    table = pq.read_table(path)
    assert table.num_rows == 2
    assert table.column('a').to_pylist() == [1.0, 3.0]
