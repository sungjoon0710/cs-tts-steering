"""Prompt anchors, neutral-prompt speech phases, and saved-state geometry.

Extraction loads only explicitly pinned local models. Analysis reads NPZ/JSONL
outputs and requires neither a GPU nor a checkpoint. Never generates audio.
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest, read_jsonl, write_json, write_jsonl
from src.representation_geometry import (calibration_mean, contrasts, cosine,
    mean_direction_consistency, project_calibration_pca)


def csvsave(path, rows):
    if not rows:
        return
    with path.open('x', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def new_output(path):
    path.mkdir(parents=True, exist_ok=False)


def load_model(args, config=None):
    if config is None:
        config = json.loads(args.config.read_text(encoding='utf-8'))
    if args.snapshot.name != config['model_revision'] or args.codec_snapshot.name != config['codec_revision']:
        raise ValueError('Snapshot names do not match configured revisions')
    if digest((args.snapshot / 'inference.py').read_bytes()) != config['vendor_inference_sha256']:
        raise ValueError('Pinned inference source differs')
    codec = args.codec_snapshot
    if codec.parent.name != 'snapshots' or codec.parent.parent.name != 'models--hubertsiuzdak--snac_24khz':
        raise ValueError('Pass the pinned codec snapshot in its existing HF cache')
    if (codec.parent.parent / 'refs/main').read_text().strip() != config['codec_revision']:
        raise ValueError('Vendor codec cache ref differs from the pinned codec')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='1', HF_HUB_CACHE=str(codec.parents[2]))
    import torch
    from src.steering import SpeechLM
    torch.manual_seed(config['seed'])
    model = SpeechLM('indic-speak', args.snapshot, codec, device=args.device)
    if len(model.layers) != config['expected_layers'] or model.lm.config.hidden_size != config['expected_hidden_size']:
        raise ValueError('Loaded architecture differs from configured model')
    return config, model


def prompts(args):
    from src.representation_probe import collect_anchor
    config = json.loads(args.config.read_text(encoding='utf-8'))
    manifest_bytes = args.manifest.read_bytes()
    rows = [json.loads(line) for line in manifest_bytes.decode('utf-8-sig').splitlines() if line.strip()]
    manifest_hash = digest(manifest_bytes)
    ids = [r['id'] for r in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Expected nonempty, unique manifest IDs')
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', r['id']) or not r.get('text', '').strip():
            raise ValueError('IDs must be safe filename components and text must be nonempty')
        if r.get('split') not in ('calibration', 'heldout', 'validation'):
            raise ValueError('Specify calibration, heldout or validation for every input')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+', c) for c in config['styles']):
        raise ValueError('Invalid condition ID')
    new_output(args.output)
    config, model = load_model(args, config)
    write_json(args.output / 'config.json', config)
    index = []
    for row in rows:
        for condition, style in config['styles'].items():
            prompt = model._build_prompt(model.tok, row['text'], config['voice'], style=style)
            arrays, metadata = collect_anchor(model.lm, prompt)
            name = row['id'] + '_' + condition + '.npz'
            target = args.output / name
            np.savez_compressed(target, **arrays)
            record = {'id': row['id'], 'condition': condition, 'style': style, 'split': row['split'],
                      'source_document': row.get('source_document', row['id']), 'path': name,
                      'sha256': digest(target.read_bytes()), 'text_sha256': digest(row['text'].encode()),
                      'source_manifest_sha256': manifest_hash,
                      'model_revision': config['model_revision'], 'codec_revision': config['codec_revision'],
                      'voice': config['voice'], 'prompt_ids': list(prompt), **metadata}
            index.append(record)
            write_jsonl(args.output / 'probes.jsonl', index)


def phases(args):
    from src.representation_probe import collect_phases
    neutral = json.loads(args.neutral_tokens.read_text(encoding='utf-8'))
    continuation = json.loads(args.continuation_tokens.read_text(encoding='utf-8'))
    prompt = neutral.get('prompt_ids', neutral.get('prompt_token_ids'))
    if prompt is None:
        raise ValueError('Neutral token record has no prompt_ids/prompt_token_ids')
    new_output(args.output)
    config, model = load_model(args)
    for record in (neutral, continuation):
        if record.get('model_revision', config['model_revision']) != config['model_revision']:
            raise ValueError('Token checkpoint revision differs')
    arrays, metadata = collect_phases(model.lm, prompt, continuation['generated_token_ids'],
                                     layers=tuple(args.layers), chunk_size=args.chunk_size)
    np.savez_compressed(args.output / 'phase_statistics.npz', **arrays)
    write_json(args.output / 'metadata.json', {**metadata, 'configuration': config,
               'neutral_tokens_sha256': digest(args.neutral_tokens.read_bytes()),
               'continuation_tokens_sha256': digest(args.continuation_tokens.read_bytes())})


def analyze(args):
    records = read_jsonl(args.source / 'probes.jsonl')
    index = {(r['id'], r['condition']): r for r in records}
    if not records or len(index) != len(records):
        raise ValueError('Empty or duplicate probe records')
    if any(len({r[k] for r in records}) != 1 for k in ('model_revision', 'codec_revision', 'voice')):
        raise ValueError('Do not pool checkpoints, codecs or voices in one geometry analysis')
    if any(r['split'] not in ('calibration', 'heldout', 'validation') for r in records):
        raise ValueError('Unknown split label')
    conditions = sorted({r['condition'] for r in records})
    selected_conditions = getattr(args, 'conditions', None)
    if selected_conditions and (not set(selected_conditions) <= set(conditions) or args.reference in selected_conditions):
        raise ValueError('Choose existing non-reference conditions')
    ids = sorted({r['id'] for r in records})
    if args.reference not in conditions:
        raise ValueError('Missing reference condition')
    arrays = {}
    base = args.source.resolve()
    for uid in ids:
        if any((uid, c) not in index for c in conditions):
            raise ValueError('Incomplete matched conditions for ' + uid)
        ref = index[uid, args.reference]
        for c in conditions:
            r = index[uid, c]
            if any(r[k] != ref[k] for k in ('split', 'source_document', 'text_sha256', 'model_revision')):
                raise ValueError('Text/split/checkpoint association mismatch')
            p = (base / r['path']).resolve()
            if not p.is_relative_to(base) or digest(p.read_bytes()) != r['sha256']:
                raise ValueError('Unsafe or changed probe file')
            with np.load(p, allow_pickle=False) as data:
                arrays[uid, c] = data['residual'].copy()
    new_output(args.output)
    splits = [index[uid, args.reference]['split'] for uid in ids]
    documents = [index[uid, args.reference]['source_document'] for uid in ids]
    paired, summary, directions, pca_inputs = [], [], {}, []
    for c in conditions:
        if c == args.reference or (selected_conditions and c not in selected_conditions):
            continue
        default = np.stack([arrays[uid, args.reference] for uid in ids])
        changed = np.stack([arrays[uid, c] for uid in ids])
        delta, magnitude = contrasts(default, changed)
        direction = calibration_mean(delta, splits) if 'calibration' in splits else None
        if direction is not None:
            directions[c] = direction
        alignment = cosine(delta, np.broadcast_to(direction, delta.shape)) if direction is not None else np.full(magnitude.shape, np.nan)
        for i, uid in enumerate(ids):
            for layer in range(delta.shape[1]):
                paired.append({'id': uid, 'split': splits[i], 'source_document': documents[i],
                    'condition': c, 'reference': args.reference, 'layer': layer,
                    'normalized_magnitude': magnitude[i, layer], 'cosine_to_calibration': alignment[i, layer]})
        for split in sorted(set(splits)):
            selected = [i for i, s in enumerate(splits) if s == split]
            groups = sorted({documents[i] for i in selected})
            grouped = np.stack([delta[[i for i in selected if documents[i] == doc]].mean(axis=0) for doc in groups])
            consistency, counts = mean_direction_consistency(grouped)
            for layer in range(delta.shape[1]):
                summary.append({'condition': c, 'reference': args.reference, 'split': split, 'layer': layer,
                    'n_texts': len(selected), 'n_documents': len(groups), 'valid_direction_documents': counts[layer],
                    'mean_normalized_magnitude_per_text': magnitude[selected, layer].mean(),
                    'mean_document_pairwise_cosine': consistency[layer]})
        pca_inputs.extend((uid, c, splits[i], delta[i]) for i, uid in enumerate(ids))
    csvsave(args.output / 'per_text_geometry.csv', paired)
    csvsave(args.output / 'summary.csv', summary)
    np.savez_compressed(args.output / 'calibration_directions.npz', **directions)
    pca_rows = []
    for layer in args.pca_layers:
        x = np.stack([r[3][layer] for r in pca_inputs])
        result = project_calibration_pca(x, [r[2] for r in pca_inputs])
        np.savez_compressed(args.output / f'pca_layer_{layer}.npz', mean=result['mean'], axes=result['axes'])
        for row, scores in zip(pca_inputs, result['scores']):
            pca_rows.append({'id': row[0], 'condition': row[1], 'split': row[2], 'layer': layer,
                             'PC1': scores[0], 'PC2': scores[1]})
    csvsave(args.output / 'pca_projections.csv', pca_rows)
    write_json(args.output / 'analysis.json', {'source_index_sha256': digest((args.source / 'probes.jsonl').read_bytes()),
               'reference': args.reference, 'calibration_texts': splits.count('calibration'),
               'conditions': [c for c in conditions if c != args.reference and (not selected_conditions or c in selected_conditions)],
               'undefined_zero_direction_cosines': 'NaN, never interpreted as perfect consistency',
               'emotion_validated': False, 'audio_generated': False})


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    for mode, handler in [('prompts', prompts), ('phases', phases), ('analyze', analyze)]:
        q = sub.add_parser(mode)
        q.add_argument('--output', type=Path, required=True)
        q.set_defaults(handler=handler)
        if mode != 'analyze':
            q.add_argument('--config', type=Path, default=Path(__file__).parents[1] / 'configs/representation_pilot.json')
            q.add_argument('--snapshot', type=Path, required=True)
            q.add_argument('--codec-snapshot', type=Path, required=True)
            q.add_argument('--device', default='cpu')
        if mode == 'prompts':
            q.add_argument('--manifest', type=Path, required=True)
        elif mode == 'phases':
            q.add_argument('--neutral-tokens', type=Path, required=True)
            q.add_argument('--continuation-tokens', type=Path, required=True)
            q.add_argument('--layers', type=int, nargs='+', default=[8, 14, 20])
            q.add_argument('--chunk-size', type=int, default=256)
        else:
            q.add_argument('--source', type=Path, required=True)
            q.add_argument('--reference', default='default')
            q.add_argument('--pca-layers', type=int, nargs='*', default=[])
            q.add_argument('--conditions', nargs='+', help='Optional non-reference condition subset')
    return p


if __name__ == '__main__':
    args = parser().parse_args()
    args.handler(args)
