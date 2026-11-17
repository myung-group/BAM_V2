"""StreamLoader._block_result: a worker task that does not answer in time is submitted again; after the last
retry the loader raises instead of waiting for ever (no JAX, no data files)."""
import concurrent.futures as cf
import threading

import pytest

from race_v2.data import stream_data as SD


class _Loader:
    _block_result = SD.StreamLoader._block_result

    def __init__(self):
        self.pool = cf.ThreadPoolExecutor(max_workers=4)


def _run(monkeypatch, n_stuck, retries):
    calls, release = [], threading.Event()

    def block(task):
        calls.append(task[0])
        if len(calls) <= n_stuck:
            release.wait(20)                  # a read that does not return
        return task[0], ['graphs of', task[0]]

    monkeypatch.setattr(SD, '_worker_block', block)
    monkeypatch.setenv('BAM_LOADER_TIMEOUT', '0.2')
    monkeypatch.setenv('BAM_LOADER_RETRIES', str(retries))
    ld = _Loader()
    task = (7, 0, [1, 2, 3])
    try:
        return ld._block_result(task, ld.pool.submit(block, task)), calls, ld
    finally:
        release.set(); ld.pool.shutdown()


def test_answer_in_time(monkeypatch):
    gs, calls, ld = _run(monkeypatch, 0, 3)
    assert gs == ['graphs of', 7] and calls == [7] and not getattr(ld, '_stuck', False)


def test_stuck_worker_is_replaced(monkeypatch, capsys):
    gs, calls, ld = _run(monkeypatch, 2, 3)
    assert gs == ['graphs of', 7] and calls == [7, 7, 7] and ld._stuck
    assert capsys.readouterr().out.count('submitted again') == 2


def test_gives_up_with_an_error(monkeypatch):
    with pytest.raises(RuntimeError, match='not returned after 2 waits'):
        _run(monkeypatch, 5, 1)
