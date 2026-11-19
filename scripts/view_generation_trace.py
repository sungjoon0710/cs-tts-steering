"""Verify saved traces and build an offline hypothesis-oriented viewer. CPU only.

No checkpoint is loaded and no inference is performed. Optional existing acoustic
tables are joined by generation ID AND exact audio hash, not by row order.
"""
import argparse
import csv
import json
import shutil
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest, private_directory, write_json
from src.snac_encode import codes_to_tokens


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def verify_trace(path):
    # Read-only SQLite URI: the viewer never mutates extraction outputs.
    db = sqlite3.connect(path.resolve().as_uri() + '/trace.sqlite?mode=ro', uri=True)
    try:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok' or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('SQLite integrity failure')
        inventory = db.execute('SELECT path,sha256,bytes FROM artifact').fetchall()
        for relative, expected, size in inventory:
            p = (path / relative).resolve()
            if not p.is_relative_to(path.resolve()) or p.stat().st_size != size or digest(p.read_bytes()) != expected:
                raise ValueError('Altered/missing trace artifact: ' + relative)
        counts = {cid: n for cid, n in db.execute('SELECT context_id,count(*) FROM position GROUP BY context_id')}
        return {'verified_artifacts': len(inventory), 'context_positions': counts,
                'sqlite_integrity': 'ok', 'sqlite_sha256': digest((path / 'trace.sqlite').read_bytes())}
    finally:
        db.close()


def build(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = private_directory(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'audio').mkdir()
    cached = read_csv(args.metrics) if args.metrics else []
    configuration = json.loads(args.source_configuration.read_text()) if args.source_configuration else None
    data, comparisons, phases, acoustics, verified, layer_summaries, input_checks = [], [], [], [], [], [], []
    for path in args.trace:
        check = verify_trace(path)
        meta = json.loads((path / 'metadata.json').read_text())
        completion = json.loads((path / 'completion.json').read_text())
        if not completion['completed'] or completion['new_lm_generations'] or completion['causal_interventions']:
            raise ValueError('Unexpected execution scope')
        uid = meta['source_generations']['default']['scenario_id']
        if uid in [d['id'] for d in data]:
            raise ValueError('Duplicate scenario')
        case = {'id': uid, 'text': meta['source_generations']['default']['input_text'],
                'n': completion['positions_per_context'], 'heatmaps': {}, 'predictions': {}, 'contexts': {}, 'decoder': {}, 'acoustics': {}}
        verified.append({'id': uid, 'trace_path': str(path.resolve()), **check})
        for row in read_csv(path / 'comparison_summary.csv'):
            comparisons.append({'id': uid, **row})
            contrast = row['contrast']
            with np.load(path / (contrast + '_heatmaps.npz'), allow_pickle=False) as z:
                case['heatmaps'][contrast] = {k: np.round(v, 6).tolist() for k, v in z.items()}
            rows = read_csv(path / (contrast + '_predictions.csv'))
            case['predictions'][contrast] = [{k: float(v) for k, v in r.items()} for r in rows]
            layer_summaries.extend({'id': uid, 'contrast': contrast, **r} for r in read_csv(path / (contrast + '_layers.csv')))
            for phase in range(7):
                selected = [r for r in rows if int(r['predicted_phase']) == phase]
                phases.append({'id': uid, 'contrast': contrast, 'predicted_phase': phase,
                    'n_positions': len(selected),
                    **{key: float(np.mean([float(r[key]) for r in selected]))
                       for key in ('jsd_nats', 'phase_jsd_nats', 'delta_target_logp', 'reference_phase_mass', 'changed_phase_mass')}})
        for p in ('default', 'anger'):
            for h in ('default', 'anger'):
                cid = p + '__' + h
                ctx = json.loads((path / (cid + '.json')).read_text())
                case['contexts'][cid] = ctx
                if not ctx['checks']['hooks_removed'] or not ctx['checks']['residual_reconstruction_exact']:
                    raise ValueError('Incomplete engineering checks')
                if args.source_run:
                    tp = args.source_run / 'tokens' / (uid + '_' + p + '.json')
                    hp = args.source_run / 'tokens' / (uid + '_' + h + '.json')
                    original_prompt, original_history = json.loads(tp.read_text()), json.loads(hp.read_text())
                    if ctx['prompt_ids'] != original_prompt['prompt_ids'] or ctx['history_tokens'] != original_history['generated_token_ids'][:case['n']]:
                        raise ValueError('Trace does not match saved source token IDs')
                    input_checks.append({'id': uid, 'context': cid, 'prompt_file': str(tp), 'prompt_sha256': digest(tp.read_bytes()),
                                         'history_file': str(hp), 'history_sha256': digest(hp.read_bytes()), 'exact_prefix_match': True})
        if configuration:
            for key in ('model_revision', 'voice', 'seed'):
                value = configuration['model'][1] if key == 'model_revision' else configuration[key]
                if value != meta['source_generations']['default'][key]:
                    raise ValueError('Cached configuration mismatch: ' + key)
        for c in ('default', 'anger'):
            rec = meta['source_generations'][c]
            sha = digest((path / 'audio' / (c + '.wav')).read_bytes())
            if sha != rec['audio_SHA256']:
                raise ValueError('Source generation audio hash mismatch')
            shutil.copyfile(path / 'audio' / (c + '.wav'), output / 'audio' / (uid + '_' + c + '.wav'))
            rows = [r for r in cached if r['generation_id'] == rec['generation_id']]
            if cached and (len(rows) != 1 or rows[0]['audio_SHA256'] != sha):
                raise ValueError('Acoustic cache does not match this exact WAV')
            if rows:
                acoustics.append(rows[0])
                case['acoustics'][c] = {k: float(rows[0][k]) if rows[0][k] else None
                    for k in ('duration_s', 'f0_median_hz', 'f0_range_st', 'pause_proportion', 'WER_proxy')}
            dec = json.loads((path / (c + '_decoder.json')).read_text())
            with np.load(path / (c + '_decoder.npz'), allow_pickle=False) as z:
                if args.source_run:
                    original = json.loads((args.source_run / 'tokens' / (uid + '_' + c + '.json')).read_text())
                    reconstructed = codes_to_tokens(*(z[f'codebook_{i}_ids'] for i in range(3)), base=128266, dedup=False)
                    if reconstructed != original['generated_token_ids'][:len(reconstructed)]:
                        raise ValueError('Decoded codes differ from original generated tokens')
                series = {}
                for k in z.files:
                    if k.endswith('.rms_per_frame'):
                        name = k.removesuffix('.rms_per_frame')
                        if name.startswith('snac.'):
                            step = 2048 / 24000 / (2 ** int(name.split('.')[2]))
                        elif name in ('vocos.stem', 'vocos.pre_last'):
                            step = 512 / 24000
                        else:
                            step = dec['hop'] / 24000
                        series[name] = {'dt': step, 'y': np.round(z[k], 6).tolist()}
                series['snac.z_q'] = {'dt': 512 / 24000, 'y': np.round(np.sqrt(np.mean(z['z_q'][0] ** 2, axis=0)), 6).tolist()}
                # Summary waveform preserves peaks in each 10 ms bin.
                wave = z['waveform_replay']
                bins = [wave[i:i + 240] for i in range(0, len(wave), 240)]
                series['waveform_absolute_peak_10ms'] = {'dt': .01, 'y': [float(np.max(np.abs(x))) for x in bins]}
            case['decoder'][c] = {'series': series, 'summary': dec,
                                  'audio': 'audio/' + uid + '_' + c + '.wav'}
        data.append(case)
    write_json(output / 'verification.json', {'traces': verified,
        'acoustic_cache': str(args.metrics) if args.metrics else None,
        'acoustic_cache_sha256': digest(args.metrics.read_bytes()) if args.metrics else None,
        'source_configuration': configuration, 'source_token_verification': input_checks,
        'source_configuration_sha256': digest(args.source_configuration.read_bytes()) if args.source_configuration else None,
        'note': 'Supplementary original generation settings; raw trace directories remain unchanged.'})
    save_csv(output / 'comparison_summary.csv', comparisons)
    save_csv(output / 'phase_summary.csv', phases)
    save_csv(output / 'layer_phase_summary.csv', layer_summaries)
    save_csv(output / 'linked_acoustic_results.csv', acoustics)
    payload = json.dumps(data, separators=(',', ':'), allow_nan=False).replace('<', chr(92) + 'u003c')
    template = Path(__file__).with_name('generation_trace_viewer.html').read_text(encoding='utf-8')
    (output / 'index.html').write_text(template.replace('__TRACE_DATA__', payload), encoding='utf-8')
    # Two compact scientific figures; no model inference or reused old rankings.
    fig, axes = plt.subplots(len(data), 2, figsize=(12, 3.6 * len(data)), squeeze=False, layout='constrained')
    for i, d in enumerate(data):
        for col, contrast in enumerate(('prompt_on_default', 'prompt_on_anger')):
            values = np.array(d['heatmaps'][contrast]['resid_normalized_l2'])
            image = axes[i, col].imshow(values, aspect='auto', origin='lower', cmap='viridis', vmin=0, vmax=.30,
                extent=(-.5, d['n'] - .5, -.5, values.shape[0] - .5))
            axes[i, col].set(xlabel='Predicted audio-token index j', ylabel='Zero-based layer',
                title=f"{d['id']}: same {contrast.removeprefix('prompt_on_')} history; ANGER − default prompt")
            fig.colorbar(image, ax=axes[i, col], label='||Δ residual|| / ||default residual||', shrink=.8, extend='max')
    fig.suptitle('Prompt-conditioned residual differences on IDENTICAL audio histories\nRow j=0 is the speech-start anchor; all seven predicted phases retained', fontsize=12)
    fig.savefig(output / 'layer_position.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
    for i, d in enumerate(data):
        for condition, style in [('prompt_on_default', '-'), ('prompt_on_anger', '--')]:
            rows = [r for r in phases if r['id'] == d['id'] and r['contrast'] == condition]
            axes[0].plot(range(7), [r['jsd_nats'] for r in rows], style, marker='o', label=d['id'] + ' / ' + condition.removeprefix('prompt_on_'))
            axes[1].plot(range(7), [r['delta_target_logp'] for r in rows], style, marker='o', label=d['id'] + ' / ' + condition.removeprefix('prompt_on_'))
    axes[0].set(ylabel='Mean full-vocabulary JSD (nats)', xlabel='Predicted SNAC phase', title='Prompt effects reach next-token distributions')
    axes[1].set(ylabel='Mean Δ log p(same saved target), nats', xlabel='Predicted SNAC phase', title='ANGER prompt minus empty-style prompt')
    axes[1].axhline(0, color='black', lw=.7)
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8); ax.set_xticks(range(7))
    fig.savefig(output / 'phase_predictions.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'output': str(output), 'cases': len(data), 'verified_artifacts': sum(r['verified_artifacts'] for r in verified),
                      'linked_acoustic_recordings': len(acoustics), 'viewer_bytes': (output / 'index.html').stat().st_size}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--metrics', type=Path)
    p.add_argument('--source-configuration', type=Path)
    p.add_argument('--source-run', type=Path, help='Optional local mirror for exact prompt/history and decoded-code verification')
    build(p.parse_args())
