import pickle
import pytest
from scripts.inspect_local_masac import read_pandas


@pytest.mark.parametrize('protocol', [2, 4])
def test_pickle_inspection_rejects_unapproved_globals_before_loading(tmp_path, protocol):
    path = tmp_path / 'unapproved.p'
    path.write_bytes(pickle.dumps(complex(1, 2), protocol=protocol))
    with pytest.raises(ValueError, match='global'):
        read_pandas(path)


def test_inspector_rejects_unexpected_container(tmp_path):
    path = tmp_path / 'list.p'
    path.write_bytes(pickle.dumps([1, 2, 3], protocol=3))
    with pytest.raises(ValueError, match='pandas container'):
        read_pandas(path)
