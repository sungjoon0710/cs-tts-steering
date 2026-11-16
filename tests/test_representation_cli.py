import csv
from types import SimpleNamespace
import numpy as np
import pytest
from scripts.inspect_representations import analyze
from src.evaluation.common import digest, write_jsonl


def fixture_index(source):
    source.mkdir()
    records = []
    for i, split in enumerate(['calibration', 'calibration', 'heldout']):
        for condition in ['default', 'anger', 'happy']:
            values = np.array([[1., 2, 3], [3., 2, 1]])
            if condition != 'default':
                values += (i + 1) * (.1 if condition == 'anger' else .2)
            path = source / f'{i}_{condition}.npz'
            np.savez_compressed(path, residual=values)
            records.append({'id': str(i), 'condition': condition, 'split': split,
                            'source_document': 'doc' + str(i), 'text_sha256': 'text' + str(i),
                            'model_revision': 'fixture', 'codec_revision': 'fixture_codec', 'voice': 'fixture_voice', 'path': path.name,
                            'sha256': digest(path.read_bytes())})
    write_jsonl(source / 'probes.jsonl', records)
    return records


def test_analysis_offline_counts_splits_and_source_hashes(tmp_path):
    source = tmp_path / 'source'
    fixture_index(source)
    output = tmp_path / 'analysis'
    analyze(SimpleNamespace(source=source, output=output, reference='default', pca_layers=[0]))
    rows = list(csv.DictReader((output / 'per_text_geometry.csv').open()))
    assert len(rows) == 3 * 2 * 2
    with np.load(output / 'calibration_directions.npz') as values:
        np.testing.assert_allclose(values['anger'], .15)
    with pytest.raises(FileExistsError):
        analyze(SimpleNamespace(source=source, output=output, reference='default', pca_layers=[]))
    (source / '0_anger.npz').write_bytes(b'changed fixture')
    with pytest.raises(ValueError, match='changed'):
        analyze(SimpleNamespace(source=source, output=tmp_path / 'other', reference='default', pca_layers=[]))


def test_analysis_rejects_missing_or_mixed_checkpoint_pairs(tmp_path):
    source = tmp_path / 'source'
    records = fixture_index(source)
    write_jsonl(source / 'probes.jsonl', records[:-1])
    args = SimpleNamespace(source=source, output=tmp_path / 'output', reference='default', pca_layers=[])
    with pytest.raises(ValueError, match='Incomplete'):
        analyze(args)
    records[-1]['model_revision'] = 'different'
    write_jsonl(source / 'probes.jsonl', records)
    with pytest.raises(ValueError, match='pool checkpoints'):
        analyze(args)


def test_equal_length_control_analysis_can_select_emotion_conditions(tmp_path):
    source = tmp_path / 'source'
    fixture_index(source)
    output = tmp_path / 'analysis'
    analyze(SimpleNamespace(source=source, output=output, reference='default', pca_layers=[], conditions=['anger']))
    rows = list(csv.DictReader((output / 'summary.csv').open()))
    assert {r['condition'] for r in rows} == {'anger'}
