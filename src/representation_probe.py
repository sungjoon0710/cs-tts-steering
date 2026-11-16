"""Read-only representation collection and zero-strength engineering checks.

The upstream CAA model/steering API remains in src.steering. These utilities
inspect prompt anchors and seven-phase continuations without fitting vectors.
"""
from collections import Counter
import torch
from src.tts_settings import AUDIO_BASE,CODEBOOK_SIZE


@torch.inference_mode()
def _collect_span(lm, ids, span):
    """Reuse upstream extraction with an explicit fresh, uncached sequence."""
    from types import SimpleNamespace
    import numpy as np
    from src.steering import extract

    if lm.training:
        raise ValueError('Collect in eval mode')
    start, end = span
    if not 0 <= start < end <= len(ids):
        raise ValueError('Invalid representation span')
    layers = lm.model.layers
    modules = [lm] + [m for layer in layers for m in (layer, layer.self_attn.o_proj)]
    before = [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules]
    if any(pre or post for pre, post in before):
        raise ValueError('Unexpected existing hooks; collect unmodified states explicitly')
    calls = []

    def guard(module, args, kwargs):
        actual = kwargs.get('input_ids', args[0] if args else None)
        cache = kwargs.get('past_key_values')
        if actual is None or actual.shape != (1, len(ids)) or actual[0].tolist() != list(ids):
            raise ValueError('Unexpected input sequence')
        if cache is not None and cache.get_seq_length() != 0:
            raise ValueError('Collection requires a fresh cache')
        calls.append(True)

    handle = lm.register_forward_pre_hook(guard, with_kwargs=True)
    old_cache = lm.config.use_cache
    lm.config.use_cache = False
    try:
        proxy = SimpleNamespace(lm=lm, layers=layers, n_layers=len(layers),
                                device=next(lm.parameters()).device)
        result = extract(proxy, list(ids), span)
        if not all(np.isfinite(v).all() for v in result['pooled'].values()):
            raise ValueError('Nonfinite hidden state')
    finally:
        lm.config.use_cache = old_cache
        handle.remove()
    after = [(len(m._forward_pre_hooks), len(m._forward_hooks)) for m in modules]
    if before != after or len(calls) != 1:
        raise RuntimeError('Unexpected forward count or leaked hook')
    return result


@torch.inference_mode()
def collect_anchor(lm, prompt, speech_start_id=128257):
    """Collect the same final speech-start PROMPT token across all layers.

    Semantic anchor alignment does not remove instruction-length/context effects.
    No generated audio or unrelated sampled-token positions are compared.
    """
    if not prompt or prompt[-1] != speech_start_id:
        raise ValueError('Expected final speech-start anchor')
    result = _collect_span(lm, prompt, (len(prompt) - 1, len(prompt)))
    device = next(lm.parameters()).device
    embedding = lm.model.embed_tokens(torch.tensor([prompt[-1]], device=device))[0]
    arrays = {'residual': result['pooled']['resid', 'last'],
              'attention': result['pooled']['attn', 'last'],
              'embedding': embedding.float().cpu().numpy()}
    return arrays, {'anchor_token_id': prompt[-1], 'anchor_absolute_position': len(prompt) - 1,
                    'prompt_tokens': len(prompt), 'layers': list(range(len(lm.model.layers))),
                    'site': 'post-block residual; attention o_proj output also saved',
                    'fresh_cache': True, 'hooks_removed': True,
                    'context': 'prompt only; semantic anchor, absolute indices may differ'}


def collect_prefix(lm, prompt, prefix, speech_start_id=128257):
    """Inspect a fixed, complete audio-code prefix under a chosen prompt.

    Reuse identical prefix IDs across prompts for a context control. This does
    not extract an acoustic-emotion vector from independently sampled outputs.
    """
    if not prompt or prompt[-1] != speech_start_id:
        raise ValueError('Expected speech-start prompt')
    phases = code_phases(prefix)
    if not prefix or len(prefix) % 7 or phases != [i % 7 for i in range(len(prefix))]:
        raise ValueError('Prefix must contain complete valid seven-code frames only')
    result = _collect_span(lm, list(prompt) + list(prefix), (len(prompt), len(prompt) + len(prefix)))
    arrays = {f'{site}_{pool}': result['pooled'][site, pool]
              for site in ('resid', 'attn') for pool in ('mean', 'last')}
    return arrays, {'prefix_tokens': len(prefix), 'frames': len(prefix) // 7,
                    'position_convention': 'processed audio token predicts next token',
                    'fresh_cache': True, 'hooks_removed': True,
                    'interpretation': 'prompt context on a fixed teacher-forced audio history'}


class ZeroSteeringProbe:
    """Check masked block-output hooks during prefill and cached decoding.

    Saves shapes/counts/finite checks, never hidden activations or fitted vectors.
    A fixed all-ones readiness fixture is multiplied by alpha=0.
    """
    def __init__(self,lm,prompt_length,layers=(8,14,20),alpha=0.0):
        if alpha!=0:raise ValueError('Stage 1 permits alpha=0 only')
        self.lm=lm;self.prompt_length=prompt_length;self.layers=layers
        self.handles=[];self.calls=0;self.prefill_calls=0;self.selected_tokens=0
        self.shapes={str(i):Counter() for i in layers};self.finite={};self.unmasked={}
        self.fresh_cache=None;self.mask=None

    def _before(self,module,args,kwargs):
        ids=kwargs.get('input_ids',args[0] if args else None)
        if ids is None:raise ValueError('Expected token IDs, not embedding-only input')
        cache=kwargs.get('past_key_values');seen=cache.get_seq_length() if cache is not None else 0
        if self.calls==0:
            self.fresh_cache=seen==0
            if not self.fresh_cache:raise ValueError('Probe requires a fresh cache')
        positions=kwargs.get('position_ids')
        if positions is None:positions=torch.arange(seen,seen+ids.shape[1],device=ids.device)[None,:]
        self.mask=(positions>=self.prompt_length)&(ids>=AUDIO_BASE)&(ids<AUDIO_BASE+7*CODEBOOK_SIZE)
        if self.calls==0 and bool(self.mask.any()):raise ValueError('Prefill would be modified')
        self.calls+=1;self.prefill_calls+=int(seen==0)
        self.selected_tokens+=int(self.mask.sum())

    def _hook(self,layer):
        def apply(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            if h.ndim!=3 or self.mask.shape!=h.shape[:2]:raise ValueError('Unexpected block/mask shape')
            self.shapes[str(layer)][tuple(h.shape)]+=1
            check=torch.isfinite(h).all()
            self.finite[layer]=check if layer not in self.finite else self.finite[layer]&check
            fixture=torch.ones(h.shape[-1],device=h.device,dtype=h.dtype)
            new=torch.where(self.mask[...,None],h+0.0*fixture,h)
            equal=(new==h).all()
            self.unmasked[layer]=equal if layer not in self.unmasked else self.unmasked[layer]&equal
            return (new,*output[1:]) if isinstance(output,tuple) else new
        return apply

    def __enter__(self):
        self.original_counts=[len(self.lm._forward_pre_hooks)]+[len(self.lm.model.layers[i]._forward_hooks) for i in self.layers]
        try:
            self.handles.append(self.lm.register_forward_pre_hook(self._before,with_kwargs=True))
            for layer in self.layers:self.handles.append(self.lm.model.layers[layer].register_forward_hook(self._hook(layer)))
        except Exception:
            for h in self.handles:h.remove()
            raise
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        counts=[len(self.lm._forward_pre_hooks)]+[len(self.lm.model.layers[i]._forward_hooks) for i in self.layers]
        self.removed=counts==self.original_counts
        if not self.removed:raise RuntimeError('Hook removal failed')

    def result(self):
        return {'forward_calls':self.calls,'prefill_calls':self.prefill_calls,'audio_positions_checked':self.selected_tokens,
            'fresh_cache':self.fresh_cache,'hooks_removed':self.removed,
            'finite_layers':{str(k):bool(v) for k,v in self.finite.items()},
            'all_outputs_unchanged':all(bool(v) for v in self.unmasked.values()),
            'shapes':{k:{str(s):n for s,n in v.items()} for k,v in self.shapes.items()},
            'alpha':0,'vectors_fitted':False,'hidden_activations_saved':False}


def code_phases(tokens):
    """Mask complete valid seven-code frames without altering replay histories."""
    phases=[-1]*len(tokens);n=0
    for token in tokens:
        if not AUDIO_BASE<=token<AUDIO_BASE+7*CODEBOOK_SIZE:break
        n+=1
    for i in range(0,n-n%7,7):
        if all(AUDIO_BASE+p*CODEBOOK_SIZE<=tokens[i+p]<AUDIO_BASE+(p+1)*CODEBOOK_SIZE for p in range(7)):
            phases[i:i+7]=range(7)
    return phases


@torch.inference_mode()
def collect_phases(lm,neutral_prompt,continuation,layers=(8,14,20),chunk_size=256):
    """Teacher-force unmodified history; collect post-block speech-token states.

    Input-token states predict the NEXT token, matching the readiness hook site.
    No labels are subtracted and no vector is fitted. Prompt/EOS/invalid frames
    stay in the causal history but contribute nothing to the phase statistics.
    """
    import numpy as np
    if not neutral_prompt or neutral_prompt[-1]!=128257 or any(t in (156940,156941) for t in neutral_prompt):
        raise ValueError('Replay requires the empty-style speech-start prompt')
    if chunk_size<1 or not continuation or not layers or len(set(layers))!=len(layers):raise ValueError('Invalid collector inputs')
    if any(i < 0 or i >= len(lm.model.layers) for i in layers):raise ValueError('Layer indices must be zero-based and in range')
    if lm.training:raise ValueError('Collect in eval mode')
    phases=code_phases(continuation)
    if max(phases,default=-1)<0:raise ValueError('No complete valid code frames')
    device=next(lm.parameters()).device;stream=neutral_prompt+continuation
    phase_ids=np.asarray([-1]*len(neutral_prompt)+phases)
    sums={};counts={};handles=[];start=0;end=0;cache=None;calls=0
    old=[len(lm.model.layers[i]._forward_hooks) for i in layers]
    def hook(layer):
        def collect(module,args,output):
            h=output[0] if isinstance(output,tuple) else output
            if h.ndim!=3 or h.shape[:2]!=(1,end-start):raise ValueError('Unexpected replay residual shape')
            mask=phase_ids[start:end]>=0
            if not mask.any():return
            values=h[0,torch.as_tensor(mask,device=h.device)].float().cpu().numpy().astype(np.float64)
            if not np.isfinite(values).all():raise ValueError('Nonfinite replay activations')
            sums.setdefault(layer,np.zeros((7,h.shape[-1]),np.float64));counts.setdefault(layer,np.zeros(7,np.int64))
            selected=phase_ids[start:end][mask]
            for p in range(7):
                hit=selected==p;sums[layer][p]+=values[hit].sum(axis=0);counts[layer][p]+=int(hit.sum())
            # Returning None leaves the model output unchanged.
        return collect
    try:
        for i in layers:handles.append(lm.model.layers[i].register_forward_hook(hook(i)))
        for start in range(0,len(stream),chunk_size):
            end=min(len(stream),start+chunk_size)
            ids=torch.tensor([stream[start:end]],device=device)
            out=lm(input_ids=ids,attention_mask=torch.ones((1,end),device=device,dtype=torch.long),
                position_ids=torch.arange(start,end,device=device)[None,:],past_key_values=cache,
                use_cache=True,logits_to_keep=1)
            cache=out.past_key_values;calls+=1
            if cache is None or cache.get_seq_length()!=end:raise ValueError('Replay cache position mismatch')
            del out
    finally:
        for handle in handles:handle.remove()
        if old!=[len(lm.model.layers[i]._forward_hooks) for i in layers]:raise RuntimeError('Collector hook cleanup failed')
    c=np.stack([counts[i] for i in layers]);s=np.stack([sums[i] for i in layers])
    expected=sum(p==0 for p in phases)
    if not np.all(c==expected):raise ValueError('Incomplete phase/layer coverage')
    return {'sums':s,'counts':c,'means':s/c[...,None]}, {
        'layers':list(layers),'site':'decoder_block_output_post_residual','hidden_size':s.shape[-1],
        'phase_counts':c.tolist(),'processed_tokens':len(stream),'prompt_tokens':len(neutral_prompt),
        'continuation_tokens':len(continuation),'selected_audio_tokens':sum(p>=0 for p in phases),
        'forward_calls':calls,'chunk_size':chunk_size,'fresh_cache':True,'hooks_removed':True,
        'position_convention':'processed audio-token state predicts next token',
        'history':'unmodified prompt plus full continuation, including masked EOS/invalid tokens',
        'vectors_fitted':False,'nonzero_steering_run':False}
