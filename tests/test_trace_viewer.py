import json

import numpy as np
import pytest

from scripts.view_generation_trace import verify_trace
from src.generation_trace import TraceStore, prediction_positions


def test_readonly_viewer_verifies_inventory_and_rejects_corruption(tmp_path):
    path = tmp_path / 'run'
    store = TraceStore(path, {'model': 'fixture'})
    codes = [128266 + 4096 * p for p in range(7)]
    store.context('d', 'default', 'default', prediction_positions([128257], codes), {})
    np.savez_compressed(path / 'states.npz', x=np.arange(8))
    store.artifact('states.npz', 'fixture')
    store.close()
    before = (path / 'trace.sqlite').read_bytes()
    result = verify_trace(path)
    assert result['verified_artifacts'] == 1 and result['context_positions'] == {'d': 7}
    assert before == (path / 'trace.sqlite').read_bytes()
    with (path / 'states.npz').open('ab') as f:
        f.write(b'corruption')
    with pytest.raises(ValueError, match='Altered/missing'):
        verify_trace(path)
