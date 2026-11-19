"""Replay a saved Indic-Speak default/ANGER pair; never sample new speech.

Four contexts isolate prompt changes on a fixed history and history changes on
a fixed prompt. See docs/GENERATION_TRACING.md for definitions and limitations.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_representations import load_model
from src.evaluation.common import digest, private_directory, read_jsonl, write_json
from src.generation_trace import (TraceStore, prediction_differences, state_differences,
                                  top_predictions, trace_context, trace_decoder)
from src.representation_probe import code_phases


def csvsave(path, rows):
    with path.open('x', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Existing run with generations.jsonl and tokens/')
    p.add_argument('--id', required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--codec-snapshot', type=Path, required=True)
    p.add_argument('--config', type=Path, default=Path(__file__).resolve().parents[1] / 'configs/representation_pilot.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-frames', type=int, default=64)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--base-commit', required=True)
    args = p.parse_args()
    if not 1 <= args.max_frames <= 128:
        p.error('Use 1-128 frames for a bounded small trace')
    torch.set_num_threads(2)
    if args.device == 'cuda':
        torch.cuda.set_per_process_memory_fraction(0.65)
    started = time.time()
    config = json.loads(args.config.read_text())
    records, tokens = {}, {}
    for condition in ('default', 'anger'):
        matches = [r for r in read_jsonl(args.source / 'generations.jsonl')
                   if r['scenario_id'] == args.id and r['condition'] == condition]
        if len(matches) != 1:
            raise ValueError('Expected one saved generation per condition')
        r = records[condition] = matches[0]
        if not r.get('technical_pass') or r['model_revision'] != config['model_revision'] or r['voice'] != config['voice']:
            raise ValueError('Generation status/checkpoint/voice mismatch')
        if r['style'] != {'default': '', 'anger': 'ANGER'}[condition]:
            raise ValueError('Unexpected saved style')
        if digest(r['input_text'].encode()) != r['text_SHA256']:
            raise ValueError('Text hash mismatch')
        audio_path = (args.source / r['audio_path']).resolve()
        if not audio_path.is_relative_to(args.source.resolve()) or digest(audio_path.read_bytes()) != r['audio_SHA256']:
            raise ValueError('Unsafe/altered source audio')
        tp = args.source / 'tokens' / (r['generation_id'] + '.json')
        tokens[condition] = json.loads(tp.read_text())
    if any(records['default'][k] != records['anger'][k] for k in
           ('scenario_id', 'split', 'input_text', 'text_SHA256', 'voice', 'seed', 'max_new_tokens', 'model_revision')):
        raise ValueError('Unmatched source conditions')
    all_histories = {c: t['generated_token_ids'] for c, t in tokens.items()}
    valid_lengths = {}
    for c, history in all_histories.items():
        phases = code_phases(history)
        n = sum(x >= 0 for x in phases)
        if not n or phases[:n] != [i % 7 for i in range(n)] or any(x >= 0 for x in phases[n:]):
            raise ValueError('Invalid/non-contiguous code frames')
        valid_lengths[c] = n
    n = min(*valid_lengths.values(), args.max_frames * 7)
    output = private_directory(args.output)
    source_files = [args.config, Path(__file__), Path(__file__).resolve().parents[1] / 'src/generation_trace.py',
                    args.source / 'generations.jsonl']
    metadata = {'schema_version': 1, 'base_commit': args.base_commit,
        'source_sha256': {str(x): digest(x.read_bytes()) for x in source_files},
        'checkpoint': config, 'source_root': str(args.source), 'source_generations': records,
        'targets_per_context': n, 'common_frames': n // 7, 'valid_history_lengths': valid_lengths,
        'scope': 'One observational English Amit text; no vectors fitted, no new LM sampling',
        'torch_version': torch.__version__, 'selected_tensor_positions': 'first seven and last seven prediction rows',
        'probabilities': 'raw LM softmax; no sampling warpers; natural-log units',
        'alignment': 'j predicts audio token j; row 0 is speech-start, row j>0 processes audio token j-1',
        'position_control': 'semantic/relative alignment, original absolute positions; style prompt is four tokens longer'}
    store = TraceStore(output, metadata)
    try:
        write_json(output / 'metadata.json', metadata)
        store.artifact('metadata.json', 'run_metadata')
        config, model = load_model(args, config)
        write_json(output / 'architecture.json', {
            'lm_class': type(model.lm).__name__, 'config': model.lm.config.to_dict(),
            'sites': {'resid': 'model.layers[i] output before final model RMSNorm',
                      'attn': 'model.layers[i].self_attn.o_proj output before residual addition',
                      'mlp': 'model.layers[i].mlp output before residual addition'},
            'decoder_modules': {name: type(m).__name__ for name, m in model._t.vocos.named_modules()},
            'quantizer_modules': {name: type(m).__name__ for name, m in model.snac.quantizer.named_modules()},
            'gpu': torch.cuda.get_device_name() if args.device == 'cuda' else None})
        store.artifact('architecture.json', 'architecture')
        contexts = {}
        for history_condition in ('default', 'anger'):
            for prompt_condition in ('default', 'anger'):
                cid = prompt_condition + '__' + history_condition
                style = config['styles'][prompt_condition]
                prompt = list(model._build_prompt(model.tok, records['default']['input_text'], config['voice'], style=style))
                if prompt != tokens[prompt_condition].get('prompt_ids', tokens[prompt_condition].get('prompt_token_ids')):
                    raise ValueError('Rebuilt prompt differs from saved exact input')
                target = all_histories[history_condition][:n]
                trace = trace_context(model.lm, prompt, target, check_no_hook=not contexts)
                contexts[cid] = trace
                cm = {'prompt_ids': prompt, 'prompt_token_pieces': model.tok.convert_ids_to_tokens(prompt),
                      'prompt_text': model.tok.decode(prompt), 'history_tokens': target,
                      'checks': trace['checks'], 'prompt_length': len(prompt)}
                store.context(cid, prompt_condition, history_condition, trace['positions'], cm)
                write_json(output / (cid + '.json'), {**cm, 'top_predictions': top_predictions(trace['logits'], model.tok)})
                selected = np.unique(np.r_[np.arange(min(7, n)), np.arange(max(0, n - 7), n)])
                arrays = {k: v[:, selected] for k, v in trace['states'].items()}
                arrays.update(prediction_rows=selected,
                    prompt_embeddings=trace['extras']['embeddings'][:len(prompt)],
                    final_norm=trace['extras']['final_norm'][selected])
                np.savez_compressed(output / (cid + '.npz'), **arrays)
                store.artifact(cid + '.npz', 'selected_hidden_tensors', {'dtype': 'float32 lossless BF16 conversion', 'rows': selected.tolist()})
                store.artifact(cid + '.json', 'tokens_predictions_checks')
                print(json.dumps({'context': cid, 'positions': n, 'layers': model.n_layers, 'checks': trace['checks']}), flush=True)
        plans = [('prompt_on_default', 'default__default', 'anger__default', 'default', 'prompt'),
                 ('prompt_on_anger', 'default__anger', 'anger__anger', 'anger', 'prompt'),
                 ('history_under_default', 'default__default', 'default__anger', 'default', 'history'),
                 ('history_under_anger', 'anger__default', 'anger__anger', 'default', 'history')]
        summaries = []
        for contrast, ref, changed, target_name, kind in plans:
            a, b = contexts[ref], contexts[changed]
            rows = prediction_differences(a['logits'], b['logits'], all_histories[target_name][:n])
            if kind == 'history' and rows[0]['jsd_nats'] > 1e-7:
                raise ValueError('History affected the anchor before any history was supplied')
            csvsave(output / (contrast + '_predictions.csv'), rows)
            store.artifact(contrast + '_predictions.csv', 'next_token_comparison', {'units': 'nats', 'target_history': target_name})
            layer_rows, heatmaps = [], {}
            for site in ('resid', 'attn', 'mlp'):
                cos, l2 = state_differences(a['states'][site], b['states'][site])
                heatmaps[site + '_cosine_distance'], heatmaps[site + '_normalized_l2'] = cos, l2
                for layer in range(model.n_layers):
                    for phase in range(7):
                        mask = np.arange(n) % 7 == phase
                        layer_rows.append({'site': site, 'layer': layer, 'predicted_phase': phase,
                            'n_positions': int(mask.sum()), 'mean_cosine_distance': float(cos[layer, mask].mean()),
                            'mean_normalized_l2': float(l2[layer, mask].mean())})
            np.savez_compressed(output / (contrast + '_heatmaps.npz'), **heatmaps)
            csvsave(output / (contrast + '_layers.csv'), layer_rows)
            for suffix, artifact_kind in [('_heatmaps.npz', 'layer_position_comparisons'), ('_layers.csv', 'phase_layer_summary')]:
                store.artifact(contrast + suffix, artifact_kind)
            summary = {'contrast': contrast, 'kind': kind, 'n_positions': n,
                       'jsd_mean_nats': float(np.mean([r['jsd_nats'] for r in rows])),
                       'jsd_median_nats': float(np.median([r['jsd_nats'] for r in rows])),
                       'argmax_changed_fraction': float(np.mean([r['argmax_changed'] for r in rows])),
                       'mean_delta_target_logp': float(np.mean([r['delta_target_logp'] for r in rows])),
                       'residual_peak_layer_mean_l2': int(np.argmax(heatmaps['resid_normalized_l2'].mean(axis=1))),
                       'prompt_reference': ref, 'changed': changed, 'target_history': target_name}
            summaries.append(summary)
            store.comparison(contrast, ref, changed, summary)
            print(json.dumps(summary), flush=True)
        csvsave(output / 'comparison_summary.csv', summaries)
        store.artifact('comparison_summary.csv', 'comparison_summary')
        # Full histories decoded separately. Their unequal lengths are not aligned
        # speech content and must not be subtracted frame-by-frame.
        import soundfile as sf
        decoder_checks = {}
        (output / 'audio').mkdir()
        for condition in ('default', 'anger'):
            wave, arrays, info = trace_decoder(model, all_histories[condition])
            source_audio = args.source / records[condition]['audio_path']
            original, sr = sf.read(source_audio, dtype='float32')
            if sr != 24000 or original.ndim != 1 or wave.shape != original.shape:
                raise ValueError('Unexpected original/decoded waveform format')
            info.update(waveform_max_abs_error_vs_original=float(np.max(np.abs(wave - original))),
                        original_pcm_quantization_step=1 / 32768,
                        duration_s=len(wave) / sr, original_sha256=digest(source_audio.read_bytes()))
            # Copy the already generated original WAV, never replace it with replay.
            shutil.copyfile(source_audio, output / 'audio' / (condition + '.wav'))
            arrays['waveform_replay'] = wave
            np.savez_compressed(output / (condition + '_decoder.npz'), **arrays)
            write_json(output / (condition + '_decoder.json'), info)
            for name, kind in [(condition + '_decoder.npz', 'decoder_tensors'), (condition + '_decoder.json', 'decoder_summary'),
                               ('audio/' + condition + '.wav', 'copied_original_audio')]:
                store.artifact(name, kind)
            decoder_checks[condition] = info
        completion = {'completed': True, 'contexts': len(contexts), 'comparisons': len(plans),
                      'positions_per_context': n, 'layers': model.n_layers, 'runtime_s': time.time() - started,
                      'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                      'decoder_checks': decoder_checks, 'new_lm_generations': 0, 'causal_interventions': 0}
        write_json(output / 'completion.json', completion)
        store.artifact('completion.json', 'completion')
        print(json.dumps({k: v for k, v in completion.items() if k != 'decoder_checks'}), flush=True)
    except Exception as exc:
        write_json(output / 'failure.json', {'error_type': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        store.close()


if __name__ == '__main__':
    main()
