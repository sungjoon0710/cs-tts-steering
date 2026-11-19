"""Read-only M1 integrity/association verification. No model, GPU or fitting.

Writes only a NEW verification directory. Omitted tensors are never counted as
verified. Partial runs remain explicitly interim, even if all available files pass.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.tts_settings import token_budget

MODEL_REVISION = '76e0814f189321efe550011850cd288dae366aad'
CODEC_REVISION = 'd73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec'
SITES = ('resid', 'attn', 'mlp')
STYLES = ('default', 'anger', 'happy', 'sad', 'news')
CONTRASTS = [('news', c) for c in STYLES[1:4]] + [('default', c) for c in STYLES[1:]] + [('anger', 'happy'), ('anger', 'sad'), ('happy', 'sad')]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def readcsv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_path(root, *names):
    for name in names:
        p = root / name
        if p.is_file():
            return p
    raise FileNotFoundError('Missing ' + ' or '.join(names))


def cohort_checks(root):
    path = source_path(root, 'inputs/cohort.jsonl', 'dataset/cohort.jsonl')
    rows = jsonl(path)
    require(len(rows) == 120, 'Expected 120 requested utterances')
    require(len({r['id'] for r in rows}) == 120, 'Duplicate study ID')
    require(len({r['source_document'] for r in rows}) == 120, 'Duplicate source document')
    require(len({r['text_sha256'] for r in rows}) == 120, 'Duplicate text hash')
    require(all(hashlib.sha256(r['text'].encode()).hexdigest() == r['text_sha256'] for r in rows), 'Text hash differs')
    counts = Counter(r['study_split'] for r in rows)
    require(counts == {'development': 30, 'confirmation': 90}, 'Study split differs from 30/90')
    for role, n in [('development', 10), ('confirmation', 30)]:
        strata = Counter(r['category'] for r in rows if r['study_split'] == role)
        require(strata == {c: n for c in ('emotion_marked', 'referential_context', 'lexical_control')}, 'Category balance differs')
        require(sum(r['second_history'] for r in rows if r['study_split'] == role) == (10 if role == 'development' else 30), 'History subset count differs')
    report = {'requested': len(rows), 'roles': dict(counts), 'unique_documents': 120, 'unique_text_hashes': 120,
              'cohort_sha256': sha(path), 'prior_exposure_verified': False}
    warnings = []
    try:
        selection = load(source_path(root, 'inputs/selection.json', 'dataset/selection.json'))
        require(selection['cohort_sha256'] == sha(path), 'Selection manifest cohort hash differs')
        prior_path = source_path(root, 'inputs/prior_cohort.jsonl', 'dataset/prior_cohort.jsonl')
        require(sha(prior_path) == selection['prior_manifest_sha256'], 'Prior cohort manifest hash differs')
        prior = jsonl(prior_path)
        old_docs = {r['source_document'] for r in prior}
        old_text = {r.get('text_SHA256', r.get('text_sha256')) for r in prior}
        confirmation = [r for r in rows if r['study_split'] == 'confirmation']
        require(not any(r['source_document'] in old_docs or r['text_sha256'] in old_text for r in confirmation), 'Confirmation overlaps previously exposed documents/texts')
        report.update(prior_exposure_verified=True, prior_manifest_sha256=sha(prior_path))
    except FileNotFoundError:
        warnings.append('Prior-exposure exclusion could not be independently checked: selection/prior manifest unavailable.')
    return rows, report, warnings


def gate_checks(root, cohort, protocol, allow_missing=False, allow_partial=False):
    path = root / 'fit/confirmation_gate.json'
    if not path.is_file():
        require(allow_partial, 'Final verification requires the frozen confirmation gate')
        return None, {'available': False, 'interim': True}, []
    gate = load(path)
    require(gate['cohort_sha256'] == sha(cohort), 'Gate cohort hash differs')
    require(gate['protocol_sha256'] == sha(protocol), 'Gate protocol hash differs')
    rows = jsonl(cohort)
    expected = {r['id'] for r in rows if r['study_split'] == 'development'}
    training = set(gate['development_ids'])
    require(len(training) == len(gate['development_ids']) and training <= expected, 'Gate includes non-development or duplicate training IDs')
    require(set(gate['development_tensor_sha256']) == training, 'Training hash manifest differs from training IDs')
    require(set(gate.get('excluded_development_ids', [])) == expected - training, 'Development exclusions are not accounted for')
    require(not gate.get('fitted_on_confirmation', True) and gate.get('primary_histories_only') is True, 'Gate does not establish development-only primary-history fitting')
    require(gate.get('rank_shared_consensus') == 1, 'Unexpected shared subspace rank')
    report = {'available': True, 'gate_sha256': sha(path), 'development_n': len(training), 'training_ids': sorted(training),
              'basis_verified': False, 'shortlist_verified': False, 'development_tensors_verified': 0}
    skipped = []
    basis = root / 'fit/basis.npz'
    if basis.is_file():
        require(sha(basis) == gate['basis_sha256'], 'Frozen basis hash differs')
        report['basis_verified'] = True
    else:
        require(allow_missing, 'Frozen basis file is missing')
        skipped.append(str(basis.relative_to(root)))
    shortlist = root / 'fit/M2_shortlist.json'
    require(shortlist.is_file() and sha(shortlist) == gate['shortlist_sha256'], 'Frozen M2 shortlist is missing or changed')
    report['shortlist_verified'] = True
    for uid, expected_hash in gate['development_tensor_sha256'].items():
        p = root / 'study' / uid / 'h0/phase_contrasts.npz'
        if not p.exists():
            require(allow_missing, 'Development fit tensor missing: ' + uid)
            skipped.append(str(p.relative_to(root)))
        else:
            require(sha(p) == expected_hash, 'Development fit tensor changed: ' + uid)
            report['development_tensors_verified'] += 1
    return gate, report, skipped


def full_codes(record):
    ids = record['generated_token_ids']
    require(all(type(t) is int for t in ids), 'Reference IDs must be integers')
    require(len(ids) >= 8 and ids[-1] in (128258, 128001), 'Reference lacks normal EOS')
    codes = ids[:-1]
    require(len(codes) % 7 == 0, 'Reference has an incomplete code frame')
    require(all(128266 + (j % 7) * 4096 <= t < 128266 + (j % 7 + 1) * 4096 for j, t in enumerate(codes)), 'Reference code phase/range is invalid')
    return codes


def reference_failure(root, row, protocol):
    """Recognize a documented capped reference, never an arbitrary code failure.

    A failed reference makes this subject unavailable to the paired analysis.
    Its attempted continuation is retained; settings and selection stay frozen.
    """
    subject = Path(root) / 'study' / row['id']
    failure_path = subject / 'failure.json'
    if not failure_path.exists():
        return None
    failure = load(failure_path)
    require(failure == {'id': row['id'], 'type': 'ValueError',
            'message': 'Reference continuation lacks normal EOS or has trailing/invalid codes'},
            'Unresolved technical failure, not an established reference cap')
    require(not (subject / 'subject_complete.json').exists(), 'Failure conflicts with subject completion')
    for k in range(2 if row['second_history'] else 1):
        history_path = subject / f'history_{k}.json'
        if not history_path.exists():
            continue
        history = load(history_path)
        ids = history['generated_token_ids']
        if ids and ids[-1] in (128258, 128001):
            full_codes(history)
            continue
        require(history['source'] == 'generated_missing_reference_codes_only'
                and history['text_sha256'] == row['text_sha256']
                and history['model_revision'] == MODEL_REVISION and history['history_index'] == k,
                'Capped reference identity differs')
        require(history['seed'] == protocol['history_seed' if k == 0 else 'secondary_history_seed'],
                'Capped reference seed differs')
        require(history['hit_token_cap'] is True and len(ids) == history['max_new_tokens'] == token_budget(row['text'])
                and all(type(t) is int and 128266+(j%7)*4096 <= t < 128266+(j%7+1)*4096 for j,t in enumerate(ids)),
                'Reference failure is not a verified phase-valid token-cap attempt')
        require(history['audio_generated'] is False and history['decoder_executed'] is False
                and history['prompt_ids'][-1] == 128257, 'Capped reference scope/prompt differs')
        require(not (subject / f'h{k}/completion.json').exists(), 'Capped reference has a completed replay')
        return {'id': row['id'], 'failed_history': k, 'status': 'reference_token_cap',
                'tokens': len(ids), 'max_new_tokens': history['max_new_tokens'],
                'history_sha256': sha(history_path), 'failure_sha256': sha(failure_path),
                'message': failure['message']}
    raise ValueError('Failure has no corresponding capped reference evidence')


def check_positions(positions, prompt, codes):
    require(len(positions) == len(codes), 'Position count differs from inspected history')
    require(prompt and prompt[-1] == 128257, 'Prompt does not end at speech-start')
    for j, row in enumerate(positions):
        expected = {'j': j, 'absolute_position': len(prompt) - 1 + j,
                    'processed_phase': -1 if j == 0 else (j - 1) % 7,
                    'predicted_phase': j % 7, 'frame': j // 7,
                    'input_token': prompt[-1] if j == 0 else codes[j - 1], 'target_token': codes[j]}
        require(all(row.get(k) == value for k, value in expected.items()), f'Position/phase association differs at j={j}')


def allowed_omission(name):
    return name == 'phase_contrasts.npz' or name in {c + '.npz' for c in STYLES} or name.endswith('_heatmaps.npz')


def check_artifacts(directory, completion, allow_missing=False):
    verified = 0
    skipped = []
    for relative, expected in completion['artifacts'].items():
        p = (directory / relative).resolve()
        require(p.is_relative_to(directory.resolve()) and not p.is_symlink(), 'Unsafe declared artifact path')
        if not p.is_file():
            require(allow_missing and allowed_omission(relative), 'Declared artifact missing: ' + relative)
            skipped.append(relative)
            continue
        require(sha(p) == expected, 'Declared artifact hash differs: ' + relative)
        verified += 1
    return verified, skipped


def number(value):
    return float(value)


def check_phase_counts(rows, n, is_layer=False):
    for r in rows:
        phase = int(r['phase'])
        require(0 <= phase < 7, 'Unknown phase')
        require(int(r['n_positions']) == n // 7 - (1 if phase == 0 else 0), 'Primary phase counts include j0 or omit codes')
        if is_layer:
            require(r['site'] in SITES and 0 <= int(r['layer']) < 28, 'Unknown layer/site')


def verify_history(root, row, k, protocol, protocol_hash, gate, allow_missing=False):
    directory = root / 'study' / row['id'] / f'h{k}'
    completion = load(directory / 'completion.json')
    require(completion.get('completed') is True and completion['id'] == row['id'] and completion['history'] == k, 'Completion identity differs')
    require(completion['protocol_sha256'] == protocol_hash, 'Completed protocol hash differs')
    required = {'metadata.json', 'geometry_metrics.npz', 'historical_alignment_positions.npz',
                'layer_phase_metrics.csv', 'probability_phase_metrics.csv', 'comparison_summary.csv', 'residual_decomposition.csv'}
    required.update(c + suffix for c in STYLES for suffix in ('.json', '.npz'))
    required.update(c + '_minus_' + a + suffix for a, c in CONTRASTS for suffix in ('_predictions.csv', '_heatmaps.npz'))
    require(required <= set(completion['artifacts']), 'Completion inventory omits required artifacts')
    count, skipped = check_artifacts(directory, completion, allow_missing)
    meta = load(directory / 'metadata.json')
    require(meta['id'] == row['id'] and meta['text_sha256'] == row['text_sha256'] and meta['study_split'] == row['study_split'], 'Capture/cohort association differs')
    require(meta['category'] == row['category'] and meta['history_index'] == k, 'Category/history association differs')
    require(meta['checkpoint'] == MODEL_REVISION and meta['codec'] == CODEC_REVISION and meta['voice'] == protocol['voice'], 'Checkpoint/codec/voice differs')
    require(meta['protocol_sha256'] == protocol_hash and meta['decoder_replayed'] is False, 'Capture protocol/decoder scope differs')
    expected_basis = gate['basis_sha256'] if row['study_split'] == 'confirmation' and gate else None
    require(meta['basis_sha256'] == expected_basis, 'Capture basis association differs')
    history = load(root / 'study' / row['id'] / f'history_{k}.json')
    require(history == meta['history_provenance'], 'History provenance differs from source record')
    require(history['text_sha256'] == row['text_sha256'] and history['model_revision'] == MODEL_REVISION and history['history_index'] == k, 'History identity differs')
    allcodes = full_codes(history)
    n = meta['n_positions']
    require(n == completion['positions'] and protocol['min_frames'] * 7 <= n <= protocol['max_frames'] * 7 and n % 7 == 0 and n <= len(allcodes), 'Invalid inspected history length')
    codes = allcodes[:n]
    lengths = {}
    for condition in STYLES:
        context = load(directory / (condition + '.json'))
        prompt = context['prompt_ids']
        lengths[condition] = len(prompt)
        require(context['prompt_length'] == len(prompt) and len(context['prompt_token_pieces']) == len(prompt), 'Prompt metadata length differs')
        require(context['history_tokens'] == codes, 'Conditions do not share the same forced history')
        check_positions(context['positions'], prompt, codes)
        require(context['checks'] == meta['checks'][condition] and context['evidence'] == 'observational_replay', 'Context checks/evidence differ')
        require(all(context['checks'][key] is True for key in ('hooks_removed', 'fresh_uncached', 'residual_reconstruction_exact')), 'Capture engineering check failed')
        require(context['checks'].get('no_hook_logits_equal') is not False, 'No-hook logits check failed')
        tops = context['top_predictions']
        require(len(tops) == n, 'Top prediction coverage differs')
        for j, top in enumerate(tops):
            require(top['j'] == j and len(top['ids']) == len(top['probabilities']) == len(top['tokens']) == 5, 'Top prediction association differs')
            require(all(math.isfinite(p) and 0 <= p <= 1 for p in top['probabilities']), 'Invalid top probabilities')
            require(sum(top['probabilities']) <= 1.00001, 'Top-five probability mass exceeds one')
        tensor = directory / (condition + '.npz')
        if tensor.is_file():
            with np.load(tensor, allow_pickle=False) as arrays:
                selected = arrays['prediction_rows']
                require(selected.tolist() == meta['tensor_rows'] and len(set(selected.tolist())) == len(selected) and np.all((selected >= 0) & (selected < n)), 'Saved vector rows differ')
                for site in SITES:
                    require(arrays[site].shape == (28, len(selected), 3072) and np.isfinite(arrays[site]).all(), 'Invalid saved residual/branch tensor')
                require(arrays['prompt_embeddings'].shape == (len(prompt), 3072) and arrays['final_norm'].shape == (len(selected), 3072), 'Embedding/final norm shape differs')
    require(lengths == meta['style_prompt_lengths'] and len({lengths[c] for c in STYLES[1:]}) == 1 and lengths['news'] - lengths['default'] == 4, 'Equal-format prompt gate failed')
    require(meta['generation_rows'] == 'j>=1' and meta['phase_weighting'] == 'equal', 'Primary aggregation specification differs')
    layer = readcsv(directory / 'layer_phase_metrics.csv')
    probability = readcsv(directory / 'probability_phase_metrics.csv')
    require(len(layer) == 10 * 3 * 28 * 7 and len(probability) == 70, 'Aggregate row coverage differs')
    check_phase_counts(layer, n, True)
    check_phase_counts(probability, n)
    decomposition = readcsv(directory / 'residual_decomposition.csv')
    require(len(decomposition) == 3 * 28 * 7, 'Residual decomposition coverage differs')
    decomposition_keys = set()
    for entry in decomposition:
        key = (entry['condition'], int(entry['layer']), int(entry['phase']))
        require(key[0] in STYLES[1:4] and 0 <= key[1] < 28 and 0 <= key[2] < 7 and key not in decomposition_keys, 'Residual decomposition key differs')
        decomposition_keys.add(key)
        terms = [float(entry[s]) for s in ('incoming', 'attention', 'mlp', 'rounding')]
        if int(entry['n_nonzero']) > 0:
            require(np.isfinite(terms).all() and np.isclose(sum(terms), 1, atol=2e-4), 'Residual decomposition does not reconstruct projected difference')
    layer_index = {(r['reference'], r['condition'], r['site'], int(r['layer']), int(r['phase'])): r for r in layer}
    probability_index = {(r['reference'], r['condition'], int(r['phase'])): r for r in probability}
    require(len(layer_index) == len(layer) and len(probability_index) == len(probability), 'Duplicate aggregate keys')
    summaries = readcsv(directory / 'comparison_summary.csv')
    summary_index = {(r['reference'], r['condition']): r for r in summaries}
    require(set(summary_index) == set(CONTRASTS) and len(summaries) == 10, 'Comparison summary coverage differs')
    for reference, changed in CONTRASTS:
        prefix = changed + '_minus_' + reference
        predictions = readcsv(directory / (prefix + '_predictions.csv'))
        check_predictions(predictions, codes)
        for phase in range(7):
            subset = [r for r in predictions if int(r['j']) > 0 and int(r['predicted_phase']) == phase]
            aggregate = probability_index[(reference, changed, phase)]
            for key in ('jsd_nats', 'phase_jsd_nats', 'delta_target_logp', 'argmax_changed', 'reference_entropy_nats', 'changed_entropy_nats'):
                require(np.isclose(float(aggregate[key]), np.mean([float(r[key]) for r in subset]), rtol=1e-6, atol=1e-8), 'Probability phase mean differs: ' + key)
        summary = summary_index[(reference, changed)]
        require(int(summary['positions']) == n and np.isclose(float(summary['generation_JSD_nats']), np.mean([float(r['jsd_nats']) for r in predictions[1:]])), 'Position-weighted generation summary differs')
        maps = directory / (prefix + '_heatmaps.npz')
        if maps.is_file():
            with np.load(maps, allow_pickle=False) as arrays:
                for site in SITES:
                    for key in ('normalized_l2', 'cosine_distance'):
                        values = arrays[site + '_' + key]
                        require(values.shape == (28, n) and np.isfinite(values).all(), 'Invalid comparison heatmap')
                        for phase in range(7):
                            selected = (np.arange(n) % 7 == phase) & (np.arange(n) > 0)
                            expected = [float(layer_index[(reference, changed, site, l, phase)][key]) for l in range(28)]
                            require(np.allclose(values[:, selected].mean(axis=1), expected, rtol=3e-6, atol=3e-7), 'Heatmap/phase aggregate differs')
    with np.load(directory / 'historical_alignment_positions.npz', allow_pickle=False) as arrays:
        require(all(arrays[key].shape == (3, 28, n) and np.isfinite(arrays[key]).all() for key in ('news', 'default')), 'Historical alignment coverage/finite values failed')
    with np.load(directory / 'geometry_metrics.npz', allow_pickle=False) as arrays:
        for key in ('historical_news', 'historical_default'):
            require(arrays[key].shape == (3, 28, 7) and np.isfinite(arrays[key]).all(), 'Historical phase geometry failed')
        if row['study_split'] == 'confirmation':
            for key in ('shared_cosine', 'specific_cosine', 'specific_margin', 'shared_energy_fraction'):
                require(arrays[key].shape == (3, 3, 28, 7) and np.isfinite(arrays[key]).all(), 'Confirmation projection failed')
    if (directory / 'identical_control.json').is_file():
        control = load(directory / 'identical_control.json')
        require(control['states_exact'] is True and control['logits_exact'] is True and control['max_JSD_nats'] <= 2e-6 and control['n_positions'] == n, 'Identical-context numerical control failed')
    elif row.get('numerical_repeat'):
        raise ValueError('Prespecified identical-context control missing')
    return {'id': row['id'], 'split': row['study_split'], 'category': row['category'], 'history': k,
            'status': 'available_files_verified' if skipped else 'fully_verified', 'positions': n,
            'artifacts_verified': count, 'omitted_tensors': ';'.join(skipped),
            'completion_sha256': sha(directory / 'completion.json'), 'metadata_sha256': sha(directory / 'metadata.json'),
            'history_sha256': sha(root / 'study' / row['id'] / f'history_{k}.json')}


def check_predictions(rows, codes):
    require(len(rows) == len(codes), 'Prediction CSV coverage differs')
    for j, r in enumerate(rows):
        require(int(r['j']) == j and int(r['predicted_phase']) == j % 7 and int(r['target_token']) == codes[j], 'Prediction row identity differs')
        for key in ('jsd_nats', 'phase_jsd_nats'):
            value = float(r[key])
            require(math.isfinite(value) and -2e-6 <= value <= math.log(2) + 2e-6, 'Invalid Jensen-Shannon divergence')
        for side in ('reference', 'changed'):
            p, lp = float(r[side + '_eos_probability']), float(r[side + '_eos_logp'])
            require(math.isfinite(p) and 0 <= p <= 1 and math.isfinite(lp) and lp <= 1e-6 and np.isclose(p, math.exp(lp), rtol=3e-5, atol=1e-10), 'EOS field is not the actual probability')
            require(0 <= float(r[side + '_phase_mass']) <= 1.00001, 'Invalid expected-phase mass')
            require(math.isfinite(float(r[side + '_entropy_nats'])) and float(r[side + '_entropy_nats']) >= 0, 'Invalid entropy')
        require(np.isclose(float(r['delta_target_logp']), float(r['changed_target_logp']) - float(r['reference_target_logp']), rtol=2e-5, atol=2e-5), 'Target likelihood delta differs')
        require(float(r['argmax_changed']) in (0, 1), 'Invalid argmax change flag')


def check_pair(root, row):
    subject = root / 'study' / row['id']
    records = [load(subject / f'history_{k}.json') for k in (0, 1)]
    meta = [load(subject / f'h{k}/metadata.json') for k in (0, 1)]
    n = meta[0]['n_positions']
    require(meta[1]['n_positions'] == n, 'History comparison has different position counts')
    a, b = [np.asarray(full_codes(record)[:n]) for record in records]
    comparison = load(subject / 'history_comparison.json')
    duplicate = bool(np.array_equal(a, b))
    require(comparison['inspected_tokens'] == n and comparison['identical_inspected_prefix'] == duplicate and np.isclose(comparison['token_agreement_fraction'], np.mean(a == b)), 'History duplicate/agreement status differs')
    path = subject / 'history_robustness.npz'
    with np.load(path, allow_pickle=False) as arrays:
        values = arrays['cosine']
        require(values.shape == (3, 3, 28, 7) and np.isfinite(values).all() and np.all(np.abs(values) <= 1.00001), 'Missing/invalid paired-history robustness')
    return {'id': row['id'], 'identical_inspected_prefix': duplicate, 'independent_history_evidence': not duplicate,
            'token_agreement_fraction': comparison['token_agreement_fraction'], 'robustness_sha256': sha(path)}


def verify(root, output, allow_missing=False, allow_partial=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(not output.exists(), 'Output directory must be new')
    require(not root.is_relative_to(output), 'Output cannot contain the source root')
    report = {'source_root': str(root), 'interim': bool(allow_partial), 'allow_missing_tensors': bool(allow_missing),
              'errors': [], 'warnings': [], 'skipped_unverified_artifacts': [], 'paired_histories': [], 'reference_failures': []}
    manifest = []
    try:
        cohort, report['cohort'], warnings = cohort_checks(root)
        report['warnings'].extend(warnings)
        cohort_path = source_path(root, 'inputs/cohort.jsonl', 'dataset/cohort.jsonl')
        protocol_path = source_path(root, 'inputs/protocol.json', 'protocol.json')
        protocol = load(protocol_path)
        protocol_hash = sha(protocol_path)
        require(protocol['styles'] == {'default': '', 'anger': 'ANGER', 'happy': 'happy', 'sad': 'sad', 'news': 'news'} and protocol['voice'] == 'Amit', 'Frozen prompt conditions differ')
        gate, report['gate'], skipped = gate_checks(root, cohort_path, protocol_path, allow_missing, allow_partial)
        report['skipped_unverified_artifacts'].extend(skipped)
        report['protocol_sha256'] = protocol_hash
        for row in cohort:
            try:
                failed = reference_failure(root, row, protocol)
            except (ValueError, KeyError, OSError) as error:
                report['errors'].append(row['id'] + ': ' + str(error))
                failed = None
            if failed:
                report['reference_failures'].append(failed)
                for k in range(2 if row['second_history'] else 1):
                    manifest.append({'id':row['id'], 'split':row['study_split'], 'category':row['category'],
                                     'history':k, 'status':'unavailable_reference_token_cap' if k == failed['failed_history']
                                     else 'unavailable_paired_reference_dependency', 'error':failed['message']})
                continue
            statuses = []
            for k in range(2 if row['second_history'] else 1):
                directory = root / 'study' / row['id'] / f'h{k}'
                item = {'id': row['id'], 'split': row['study_split'], 'category': row['category'], 'history': k}
                if not (directory / 'completion.json').is_file() and allow_partial:
                    item.update(status='not_completed_interim')
                else:
                    try:
                        item = verify_history(root, row, k, protocol, protocol_hash, gate, allow_missing)
                        report['skipped_unverified_artifacts'].extend(str(directory.relative_to(root) / f) for f in item['omitted_tensors'].split(';') if f)
                    except (ValueError, KeyError, OSError) as error:
                        item.update(status='failed', error=str(error))
                        report['errors'].append(row['id'] + f'/h{k}: ' + str(error))
                manifest.append(item)
                statuses.append(item['status'])
            if all(s in ('fully_verified', 'available_files_verified') for s in statuses):
                try:
                    subject = root / 'study' / row['id']
                    done = load(subject / 'subject_complete.json')
                    require(done == {'id': row['id'], 'histories': len(statuses), 'status': 'complete'}, 'Subject completion differs')
                    require(not (subject / 'failure.json').exists(), 'Unresolved subject failure exists')
                    if row['second_history']:
                        report['paired_histories'].append(check_pair(root, row))
                except (ValueError, KeyError, OSError) as error:
                    report['errors'].append(row['id'] + ': ' + str(error))
        report['history_status_counts'] = dict(Counter(item['status'] for item in manifest))
        report['paired_history_count'] = len(report['paired_histories'])
        report['duplicate_history_count'] = sum(p['identical_inspected_prefix'] for p in report['paired_histories'])
        if not allow_partial:
            failed_paired = sum(r['second_history'] for r in cohort if r['id'] in {f['id'] for f in report['reference_failures']})
            require(len(report['paired_histories']) + failed_paired == 40, 'Paired-history requests are not fully accounted for')
        report['requested_subjects'] = len(cohort)
        report['completed_subjects'] = sum((root/'study'/r['id']/'subject_complete.json').exists() for r in cohort)
        report['all_requests_accounted_for'] = not allow_partial and not report['errors']
        report['warnings'].append('Zero-norm contrasts and exact development-selection ties require a separate numerical diagnostic; they are not resolved by artifact/hash verification.')
        report['warnings'].append('Generation summaries are position-weighted; primary phase tables exclude j0 and use equal phase weighting.')
    except (ValueError, KeyError, OSError) as error:
        report['errors'].append(str(error))
    report['status'] = ('failed' if report['errors'] else 'interim_available_files_verified' if allow_partial else
                        'available_files_verified_with_omissions' if report['skipped_unverified_artifacts'] else 'fully_verified')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'verification.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    columns = ['id', 'split', 'category', 'history', 'status', 'positions', 'artifacts_verified', 'omitted_tensors', 'completion_sha256', 'metadata_sha256', 'history_sha256', 'error']
    with (output / 'verified_manifest.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(manifest)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--allow-missing-tensors', action='store_true', help='Explicitly mark omitted large tensor files as unverified')
    parser.add_argument('--allow-partial', action='store_true', help='Interim check only; never reports study completion')
    args = parser.parse_args()
    report = verify(args.root, args.output, args.allow_missing_tensors, args.allow_partial)
    print(json.dumps({k: report.get(k) for k in ('status', 'history_status_counts', 'paired_history_count', 'errors')}))
    raise SystemExit(2 if report['errors'] else 0)


if __name__ == '__main__':
    main()
