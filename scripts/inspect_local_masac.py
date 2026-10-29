"""Inspect the cloned MaSaC pickles and seek evidence in existing local downloads.

No repository scripts are executed, audio altered, or fuzzy match accepted.
"""
import argparse
import builtins
import csv
import hashlib
import io
import json
import pickle
import pickletools
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from numpy._core.multiarray import _reconstruct
from pandas.core.indexes.base import Index, _new_Index
from pandas.core.indexes.range import RangeIndex
from pandas.core.internals.managers import BlockManager, SingleBlockManager

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.expanded import romanize
from src.evaluation.provenance import recover_sources
from src.evaluation.text import normalize


def read_pandas(path):
    allowed = {('pandas.core.frame', 'DataFrame'): pd.DataFrame,
        ('pandas.core.series', 'Series'): pd.Series,
        ('pandas.core.internals.managers', 'BlockManager'): BlockManager,
        ('pandas.core.internals.managers', 'SingleBlockManager'): SingleBlockManager,
        ('pandas.core.indexes.base', '_new_Index'): _new_Index,
        ('pandas.core.indexes.base', 'Index'): Index,
        ('pandas.core.indexes.range', 'RangeIndex'): RangeIndex,
        ('numpy.core.multiarray', '_reconstruct'): _reconstruct,
        ('numpy', 'ndarray'): np.ndarray, ('numpy', 'dtype'): np.dtype,
        ('builtins', 'slice'): builtins.slice}
    data = path.read_bytes()
    if len(data) > 9_000_000:
        raise ValueError('Exceeds inspected pickle size bound')
    globals_seen = Counter()
    for op, arg, _ in pickletools.genops(data):
        if op.name == 'STACK_GLOBAL':
            raise ValueError('Unreviewed dynamic global')
        if op.name == 'GLOBAL':
            key = tuple(arg.split(' ', 1))
            if key not in allowed:
                raise ValueError('Unapproved pickle global: ' + arg)
            globals_seen[arg] += 1
    class Restricted(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError('Unapproved global')
            return allowed[module, name]
    value = Restricted(io.BytesIO(data)).load()
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        raise ValueError('Not a reviewed pandas container')
    return value, dict(globals_seen)


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@lru_cache(maxsize=20000)
def roman_key(text):
    return normalize(romanize(text))


def main(a):
    if a.output.exists():
        raise ValueError('Preserve previous inspection: use a new output path')
    pickles = a.repo / 'Data/Pickles'
    frame, globals_frame = read_pandas(pickles / 'Dataset.p')
    series, globals_series = read_pandas(pickles / 'Audio_features.p')
    assert frame.shape == (15576, 7) and len(series) == len(frame)
    feature_matrix = np.stack(series.to_list())
    assert feature_matrix.shape == (15576, 128) and np.isfinite(feature_matrix).all()
    baseline = [json.loads(s) for s in a.baseline.read_text(encoding='utf-8').splitlines()]
    raw_dir = a.repo / 'Data/Raw_Utterances'
    associations = recover_sources(baseline, frame,
        (raw_dir / 'Utterances.txt').read_text(encoding='utf-8').splitlines(),
        (raw_dir / 'Audio_Filename.txt.txt').read_text(encoding='utf-8').splitlines(),
        (raw_dir / 'Episodewise_dialoguelabels.txt').read_text(encoding='utf-8').splitlines())
    exact, roman = defaultdict(list), defaultdict(list)
    def add(text, metadata):
        if not isinstance(text, str) or not normalize(text):
            return
        exact[normalize(text)].append(metadata)
        roman[roman_key(text)].append(metadata)
    table_inventory = []
    for path in sorted(a.workspace.glob('*.csv')):
        table = pd.read_csv(path, keep_default_na=False, low_memory=False)
        if 'Transcript' not in table:
            continue
        table_inventory.append(dict(path=path.name, rows=len(table), columns=list(table.columns)))
        for i, row in table.iterrows():
            add(row['Transcript'], dict(source=path.name, row=int(i), local_id=str(row.get('name', '')),
                episode='', speaker=str(row.get('Speaker', '')), origin='CSV transcript; mapping not assumed'))
    json_records = 0
    for split in ['train', 'val', 'test']:
        path = a.workspace / f'{split}_data_final_plain.json'
        records = json.loads(path.read_text(encoding='utf-8'))
        for identifier, row in records.items():
            episode_match = re.match(r'^(\d+)', row.get('episode_name', ''))
            meta = dict(source=path.name, row=identifier, local_id=f'{split}_{identifier}',
                episode='E' + episode_match[1] if episode_match else '',
                episode_name=row.get('episode_name', ''), origin='JSON conversation; no assumed timestamp slicing')
            texts = list(row['context_utterances']) + [row['target_utterance']]
            speakers = list(row['context_speakers']) + [row['target_speaker']]
            assert len(texts) == len(speakers)
            for turn, (text, speaker) in enumerate(zip(texts, speakers)):
                add(text, dict(meta, turn=turn, speaker=speaker,
                    timing_status='target timing is conversation-level; no context timing inferred'))
                json_records += 1
    inventories, paths_by_stem = [], defaultdict(list)
    for folder in ['ML_audio', 'wavfiles_conv_level', 'wavfiles_utt_level']:
        files = list((a.workspace / folder).glob('*.wav'))
        inventories.append(dict(folder=folder, wavs=len(files)))
        for p in files:
            paths_by_stem[p.stem].append(p.relative_to(a.workspace).as_posix())
    matches = []
    candidate_records = []
    for source, intended in zip(associations, baseline):
        assert source['utterance_id'] == str(intended['line'])
        text = intended['text']
        candidates = exact.get(normalize(text), [])
        transliterated = roman.get(roman_key(text), [])
        matching_paths = paths_by_stem.get(Path(source['source_audio_filename']).stem, [])
        same_episode = [r for r in candidates if r['episode'] == source['source_episode']]
        same_episode_roman = [r for r in transliterated if r['episode'] == source['source_episode']]
        matches.append(dict(source, direct_original_basename_audio_paths=json.dumps(matching_paths),
            exact_text_candidate_records=len(candidates), same_episode_exact_text_candidates=len(same_episode),
            romanized_candidate_records=len(transliterated), same_episode_romanized_candidates=len(same_episode_roman),
            status='candidate_needs_audio_verification' if matching_paths or same_episode or same_episode_roman else 'no_verified_local_audio_match',
            feature_vector_available=True, feature_dimensions=128,
            feature_provenance='same integer index only; no independently keyed audio-feature mapping',
            audio_verified=False))
        for method, rows in [('normalized_text', candidates), ('uroman_sensitivity_only', transliterated)]:
            for r in rows:
                candidate_records.append(dict(utterance_id=source['utterance_id'], method=method,
                    source_episode=source['source_episode'], candidate=json.dumps(r, ensure_ascii=False),
                    accepted_audio_match=False, reason='text/romanization candidate is not acoustic identity proof'))
    a.output.mkdir(parents=True)
    write_csv(a.output / 'local_source_crosswalk.csv', matches)
    metadata = []
    for source, intended in zip(associations, baseline):
        row = frame.iloc[source['source_row']]
        metadata.append(dict(utterance_id=source['utterance_id'], intended_text=intended['text'],
            source_speaker=source['source_speaker'], episode_dialogue=source['source_dialogue'],
            expected_audio_filename=source['source_audio_filename'],
            sarcasm_label=int(row['Sarcasm']), humour_label=int(row['Humour']),
            context_json=json.dumps(row['Context'], ensure_ascii=False),
            audio_content_verified=False))
    write_csv(a.output / 'baseline_source_metadata.csv', metadata)
    selected_features = feature_matrix[[r['source_row'] for r in associations]]
    feature_table = pd.DataFrame(selected_features, columns=[f'feature_{i:03}' for i in range(128)])
    feature_table.insert(0, 'utterance_id', [r['utterance_id'] for r in associations])
    with pd.ExcelWriter(a.output / 'Baseline_Source_Data.xlsx', engine='openpyxl') as writer:
        pd.DataFrame(metadata).to_excel(writer, sheet_name='100 source inputs', index=False)
        feature_table.to_excel(writer, sheet_name='128 numbers per input', index=False)
    if candidate_records:
        write_csv(a.output / 'text_candidates_not_verified.csv', candidate_records)
    summary = dict(repository=str(a.repo), dataset_rows=len(frame), dataset_columns=list(frame.columns),
        feature_rows=len(series), feature_dimensions=128, features_finite=True,
        feature_kind='fixed-length acoustic features; no waveform, sample rate or duration arrays',
        feature_extraction_caveat='extraction script requests 256 mean MFCCs; released vectors have128 values; exact reconstruction method unresolved',
        source_metadata_associations=len(associations), original_audio_recovered=0,
        direct_original_basename_matches=sum(bool(json.loads(r['direct_original_basename_audio_paths'])) for r in matches),
        exact_candidate_inputs=sum(r['exact_text_candidate_records'] > 0 for r in matches),
        same_episode_exact_candidate_inputs=sum(r['same_episode_exact_text_candidates'] > 0 for r in matches),
        same_episode_roman_candidate_inputs=sum(r['same_episode_romanized_candidates'] > 0 for r in matches),
        local_audio_inventory=inventories, json_utterance_records=json_records, table_inventory=table_inventory,
        pickle_globals={'Dataset.p': globals_frame, 'Audio_features.p': globals_series},
        sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [pickles / 'Dataset.p', pickles / 'Audio_features.p']},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repo_scripts_executed=False, raw_audio_evaluation_executed=False,
        interpretation='No new matched-reference metric is licensed by metadata or a 128-number feature vector')
    (a.output / 'inspection.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('table_inventory', 'pickle_globals')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['repo', 'workspace', 'baseline', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    main(parser.parse_args())
