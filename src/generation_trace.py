"""Read-only, position-aligned Llama tracing; no generation or interventions.

Row j is the state predicting audio token j: speech-start for j=0, otherwise
audio token j-1. A full valid target frame is required, including its last phase.
Full distributions are temporary CPU arrays; persist summaries/top-k, not logits.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch

from src.representation_probe import code_phases
from src.steering import _out_tensor
from src.tts_settings import AUDIO_BASE, CODEBOOK_SIZE


def prediction_positions(prompt, targets):
    if not prompt or prompt[-1] != 128257:
        raise ValueError('Expected a final speech-start prompt')
    if not targets or len(targets) % 7 or code_phases(targets) != [i % 7 for i in range(len(targets))]:
        raise ValueError('Targets must contain complete valid seven-phase frames')
    return [{'j': j, 'absolute_position': len(prompt) - 1 + j,
             'processed_phase': (j - 1) % 7 if j else -1,
             'predicted_phase': j % 7, 'frame': j // 7,
             'input_token': prompt[-1] if j == 0 else targets[j - 1],
             'target_token': t} for j, t in enumerate(targets)]


@torch.inference_mode()
def trace_context(lm, prompt, targets, *, check_no_hook=False):
    """One fresh uncached teacher-forced pass, all layers and three real sites.

    Branches are attention o_proj and MLP outputs BEFORE residual addition;
    residual is the block output BEFORE final model RMSNorm. SDPA attention
    probabilities are not requested or fabricated. Hooks never change outputs.
    """
    positions = prediction_positions(prompt, targets)
    if lm.training:
        raise ValueError('Expected eval mode')
    layers = lm.model.layers
    modules = [lm, lm.model.embed_tokens, lm.model.norm]
    modules += [m for b in layers for m in (b, b.self_attn.o_proj, b.mlp)]
    if any(m._forward_hooks or m._forward_pre_hooks for m in modules):
        raise ValueError('Unexpected existing hooks')
    start, stop = len(prompt) - 1, len(prompt) + len(targets) - 1
    states = {s: [None] * len(layers) for s in ('resid', 'attn', 'mlp')}
    extras, handles = {}, []

    def capture(destination, key, full=False):
        def hook(module, args, output):
            h = _out_tensor(output)
            if not torch.isfinite(h).all():
                raise ValueError('Nonfinite activation at ' + str(key))
            selected = h[0] if full else h[0, start:stop]
            # BF16 values are exactly representable in FP32. No lossy storage
            # conversion is used while computing comparisons.
            destination[key] = selected.detach().float().cpu().numpy().copy()
        return hook

    ids = torch.tensor([list(prompt) + list(targets[:-1])], device=next(lm.parameters()).device)
    old_cache = lm.config.use_cache
    try:
        handles.append(lm.model.embed_tokens.register_forward_hook(capture(extras, 'embeddings', True)))
        handles.append(lm.model.norm.register_forward_hook(capture(extras, 'final_norm')))
        for i, block in enumerate(layers):
            for site, module in [('resid', block), ('attn', block.self_attn.o_proj), ('mlp', block.mlp)]:
                handles.append(module.register_forward_hook(capture(states[site], i)))
        lm.config.use_cache = False
        out = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                 logits_to_keep=len(targets))
        logits = out.logits[0].detach().float().cpu().numpy().copy()
        if logits.shape[0] != len(targets) or not np.isfinite(logits).all() or out.past_key_values is not None:
            raise ValueError('Unexpected logits/cache')
        del out
    finally:
        lm.config.use_cache = old_cache
        for handle in handles:
            handle.remove()
    if any(m._forward_hooks or m._forward_pre_hooks for m in modules):
        raise RuntimeError('Hook cleanup failed')
    states = {s: np.stack(x) for s, x in states.items()}
    previous = extras['embeddings'][start:stop]
    residual_exact = True
    for i in range(len(layers)):
        # Reproduce the actual BF16 two-add residual path, including rounding.
        a = torch.tensor(previous).to(lm.dtype)
        b = torch.tensor(states['attn'][i]).to(lm.dtype)
        c = torch.tensor(states['mlp'][i]).to(lm.dtype)
        expected = ((a + b) + c).float().numpy()
        residual_exact &= np.array_equal(expected, states['resid'][i])
        previous = states['resid'][i]
    if not residual_exact:
        raise ValueError('Residual branch reconstruction differs; inspect architecture')
    no_hook_equal = None
    if check_no_hook:
        repeat = lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                    logits_to_keep=len(targets))
        no_hook_equal = np.array_equal(logits, repeat.logits[0].float().cpu().numpy())
        del repeat
        if not no_hook_equal:
            raise ValueError('Hook/no-hook logits differ')
    return {'states': states, 'extras': extras, 'logits': logits, 'positions': positions,
            'checks': {'hooks_removed': True, 'fresh_uncached': True,
                       'residual_reconstruction_exact': residual_exact,
                       'no_hook_logits_equal': no_hook_equal}}


def state_differences(reference, changed):
    """Position-wise cosine distance and ||changed-reference||/||reference||."""
    a, b = np.asarray(reference, np.float32), np.asarray(changed, np.float32)
    if a.shape != b.shape or a.ndim < 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Matched finite state arrays required')
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    cosine = np.full(na.shape, np.nan)
    np.divide((a * b).sum(axis=-1), na * nb, out=cosine, where=na * nb > 0)
    ratio = np.full(na.shape, np.nan)
    np.divide(np.linalg.norm(b - a, axis=-1), na, out=ratio, where=na > 0)
    return np.clip(1 - cosine, 0, 2), ratio


def prediction_differences(reference, changed, targets, *, audio_base=AUDIO_BASE,
                           codebook_size=CODEBOOK_SIZE, eos=128258):
    """Full-vocabulary JSD in nats, plus a conditional next-phase JSD.

    Raw model probabilities, WITHOUT temperature/repetition/top-k/top-p warping.
    The likelihood difference always scores the SAME reference target token.
    Context j can have a different preceding history only in history contrasts.
    """
    a, b = np.asarray(reference), np.asarray(changed)
    if a.shape != b.shape or len(targets) != len(a) or a.ndim != 2:
        raise ValueError('Matched logit rows/targets required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite logits')
    rows = []
    for j, target in enumerate(targets):
        x, y = torch.from_numpy(a[j]).float(), torch.from_numpy(b[j]).float()
        lp, lq = x.log_softmax(-1), y.log_softmax(-1)
        lm = torch.logaddexp(lp, lq) - np.log(2)
        js = 0.5 * ((lp.exp() * (lp - lm)).sum() + (lq.exp() * (lq - lm)).sum())
        lo = audio_base + (j % 7) * codebook_size
        xp, yp = lp[lo:lo + codebook_size], lq[lo:lo + codebook_size]
        phase_mass_a, phase_mass_b = xp.logsumexp(0), yp.logsumexp(0)
        cp, cq = xp - phase_mass_a, yp - phase_mass_b
        cm = torch.logaddexp(cp, cq) - np.log(2)
        cjs = 0.5 * ((cp.exp() * (cp - cm)).sum() + (cq.exp() * (cq - cm)).sum())
        rows.append({'j': j, 'predicted_phase': j % 7, 'target_token': int(target),
                     'jsd_nats': max(0., float(js)), 'phase_jsd_nats': max(0., float(cjs)),
                     'reference_target_logp': float(lp[target]), 'changed_target_logp': float(lq[target]),
                     'delta_target_logp': float(lq[target] - lp[target]),
                     'reference_phase_mass': float(phase_mass_a.exp()),
                     'changed_phase_mass': float(phase_mass_b.exp()),
                     'reference_eos_probability': float(lp[eos]), 'changed_eos_probability': float(lq[eos]),
                     'reference_entropy_nats': float(-(lp.exp() * lp).sum()),
                     'changed_entropy_nats': float(-(lq.exp() * lq).sum()),
                     'argmax_changed': int(x.argmax() != y.argmax())})
    return rows


def top_predictions(logits, tokenizer, k=5):
    rows = []
    for j, x in enumerate(torch.from_numpy(logits)):
        lp = x.log_softmax(-1)
        values, ids = lp.topk(k)
        rows.append({'j': j, 'ids': ids.tolist(), 'probabilities': values.exp().tolist(),
                     'tokens': tokenizer.convert_ids_to_tokens(ids.tolist())})
    return rows


@torch.inference_mode()
def trace_decoder(model, tokens, selected_channels=16):
    """Decode existing codes once; hook only modules on the real Vocos path.

    F.embedding does not call the Embedding module's forward: capture its
    returned vectors at out_proj's input instead. Never export codebook weights.
    """
    from src.tts import snac_codes
    phases = code_phases(tokens)
    valid = sum(p >= 0 for p in phases)
    if not valid or phases[:valid] != [i % 7 for i in range(valid)] or any(p >= 0 for p in phases[valid:]):
        raise ValueError('Non-contiguous/invalid decoder frames')
    codes = snac_codes(tokens[:valid], model.device)
    quantizer, vocos = model.snac.quantizer, model._t.vocos
    if vocos.head.__class__.__name__ != 'ISTFTHead':
        raise ValueError('This decoder tracer requires the inspected ISTFT head')
    modules = [('vocos.stem', vocos.stem), ('vocos.pre_last', vocos.pre[-1]),
               ('vocos.up_conv', vocos.up_conv), ('vocos.post_last', vocos.post[-1]),
               ('vocos.final_norm', vocos.final_norm), ('vocos.head.out', vocos.head.out)]
    modules += [(f'snac.quantizer.{i}.out_proj', q.out_proj) for i, q in enumerate(quantizer.quantizers)]
    if any(m._forward_hooks or m._forward_pre_hooks for _, m in modules):
        raise ValueError('Unexpected existing decoder hooks')
    captured, handles = {}, []
    def capture(name, pre=False):
        def hook(module, args, output=None):
            x = args[0] if pre else _out_tensor(output)
            if not torch.isfinite(x).all():
                raise ValueError('Nonfinite decoder output')
            captured[name] = x.detach().float().cpu().numpy().copy()
        return hook
    try:
        for name, module in modules:
            handles.append(module.register_forward_hook(capture(name)))
            if name.startswith('snac.'):
                handles.append(module.register_forward_pre_hook(capture(name + '.lookup', True)))
        z = quantizer.from_codes(codes)
        waveform = vocos(z.float())[0, 0].clamp(-1, 1).float().cpu().numpy()
    finally:
        for h in handles:
            h.remove()
    if any(m._forward_hooks or m._forward_pre_hooks for _, m in modules):
        raise RuntimeError('Decoder hooks leaked')
    reconstructed = torch.zeros_like(z)
    arrays, summaries = {'z_q': z.float().cpu().numpy()}, []
    for i, (code, q) in enumerate(zip(codes, quantizer.quantizers)):
        arrays[f'codebook_{i}_ids'] = code.cpu().numpy()
        branch = captured[f'snac.quantizer.{i}.out_proj']
        reconstructed += torch.from_numpy(branch).to(z.device).repeat_interleave(q.stride, dim=-1)
    if not torch.equal(reconstructed, z):
        raise ValueError('Quantizer sum/stride reconstruction failed')
    for name, original in captured.items():
        # Actual LayerNorm and Linear outputs are channels-last; all others BCT.
        channels_last = name in ('vocos.final_norm', 'vocos.head.out')
        x = original[0].T if channels_last else original[0]
        arrays[name + '.channels'] = x[:selected_channels]
        arrays[name + '.rms_per_frame'] = np.sqrt(np.mean(x ** 2, axis=0))
        summaries.append({'site': name, 'shape': list(original.shape),
                          'channel_axis': -1 if channels_last else 1,
                          'mean': float(x.mean()), 'std': float(x.std()),
                          'rms': float(np.sqrt(np.mean(x ** 2))),
                          'selected_channels': min(selected_channels, len(x))})
    spec = vocos.head.last_spec.detach().cpu().numpy()
    arrays['vocos.predicted_stft_log_magnitude'] = np.log(np.maximum(np.abs(spec[0]), 1e-8))
    arrays['vocos.predicted_stft_phase'] = np.angle(spec[0])
    return waveform, arrays, {'hooks_removed': True, 'quantizer_reconstruction_exact': True,
        'valid_code_tokens': valid, 'snac_frames': valid // 7,
        'quantizer_strides': [q.stride for q in quantizer.quantizers],
        'vocos_upsample': vocos.upsample, 'head': type(vocos.head).__name__,
        'n_fft': vocos.head.n_fft, 'hop': vocos.head.hop,
        'noise_post_executed': vocos.noise_post is not None,
        'noise_head_executed': vocos.noise_head is not None,
        'stock_snac_decoder_executed': False, 'summaries': summaries}


class TraceStore:
    """Small inspectable SQLite index; bulk numeric data live in hashed NPZ/CSV.

    Caller owns a new experiment directory. No updates to historical runs.
    """
    def __init__(self, directory, metadata):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.db = sqlite3.connect(self.directory / 'trace.sqlite')
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
            CREATE TABLE run(id INTEGER PRIMARY KEY CHECK(id=1), metadata TEXT NOT NULL);
            CREATE TABLE context(id TEXT PRIMARY KEY, prompt TEXT NOT NULL, history TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE position(context_id TEXT REFERENCES context(id), j INTEGER, absolute_position INTEGER,
                processed_phase INTEGER, predicted_phase INTEGER, frame INTEGER, input_token INTEGER, target_token INTEGER,
                PRIMARY KEY(context_id,j));
            CREATE TABLE artifact(path TEXT PRIMARY KEY, kind TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE comparison(id TEXT PRIMARY KEY, reference_id TEXT REFERENCES context(id), changed_id TEXT REFERENCES context(id), metadata TEXT NOT NULL);
        ''')
        self.db.execute('INSERT INTO run VALUES(1,?)', (json.dumps(metadata),))
        self.db.commit()

    def context(self, cid, prompt, history, positions, metadata):
        self.db.execute('INSERT INTO context VALUES(?,?,?,?)', (cid, prompt, history, json.dumps(metadata)))
        self.db.executemany('INSERT INTO position VALUES(?,?,?,?,?,?,?,?)',
                            [(cid, *[r[k] for k in ('j', 'absolute_position', 'processed_phase',
                                                   'predicted_phase', 'frame', 'input_token', 'target_token')]) for r in positions])
        self.db.commit()

    def artifact(self, relative, kind, metadata=None):
        path = (self.directory / relative).resolve()
        if not path.is_relative_to(self.directory.resolve()):
            raise ValueError('Artifact outside experiment')
        content = path.read_bytes()
        self.db.execute('INSERT INTO artifact VALUES(?,?,?,?,?)',
                        (str(relative), kind, hashlib.sha256(content).hexdigest(), len(content), json.dumps(metadata or {})))
        self.db.commit()

    def comparison(self, cid, ref, changed, metadata):
        self.db.execute('INSERT INTO comparison VALUES(?,?,?,?)', (cid, ref, changed, json.dumps(metadata)))
        self.db.commit()

    def close(self):
        self.db.close()
