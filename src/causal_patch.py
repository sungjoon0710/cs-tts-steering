"""Single-position, fresh-cache causal replay for the existing Llama speech LM.

Native replacements preserve the actual branch/residual dtype. Projected and
random additions are separate norm-matched off-manifold diagnostics. No training,
sampling, decoding, or architectural changes occur here.
"""
from dataclasses import dataclass
import math,time
import numpy as np
import torch
from src.generation_trace import prediction_positions
from src.steering import _out_tensor


@dataclass
class Patch:
    layer: int
    site: str
    j: int
    expected: np.ndarray
    donor: np.ndarray
    variant: str = 'native'
    alpha: float = 1.
    direction: np.ndarray | None = None
    seed: int = 1234
    dtype: str | None = None


def site_module(lm,layer,site):
    if not 0<=layer<len(lm.model.layers):raise ValueError('Invalid zero-based layer')
    block=lm.model.layers[layer]
    if site=='resid':return block
    if site=='attn':return block.self_attn.o_proj
    if site=='mlp':return block.mlp
    raise ValueError('Site must be resid, attn or mlp')


def intervention_vector(expected,donor,variant,direction=None,seed=1234):
    a,b=np.asarray(expected,np.float64),np.asarray(donor,np.float64)
    if a.ndim!=1 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Expected matched finite activation vectors')
    native=b-a;norm=float(np.linalg.norm(native))
    if variant=='native':return native.astype(np.float32)
    if variant=='random':v=np.random.default_rng(seed).standard_normal(len(a))
    elif variant in ('shared','specific'):
        v=np.asarray(direction,np.float64)
        if v.shape!=a.shape or not np.isfinite(v).all() or np.linalg.norm(v)==0:raise ValueError('Invalid frozen projection direction')
        coefficient=float(native@v)
        if coefficient==0:return np.zeros_like(a,dtype=np.float32)
        v=v*np.sign(coefficient)
    else:raise ValueError('Unknown intervention variant')
    return (v*(norm/np.linalg.norm(v))).astype(np.float32)


def norm_matched_add(original,delta):
    """Match the realized post-rounding perturbation norm, without model calls.

    Dense BF16 rounding can otherwise make nominally matched controls unequal.
    A bounded scalar search uses activation norms only, never output outcomes.
    """
    norm=float(delta.norm())
    if norm==0:return original
    unit=delta/norm;low,high=0.,2*norm;best=None;error=float('inf')
    for _ in range(16):
        scale=(low+high)/2;changed=(original.float()+scale*unit).to(original.dtype)
        actual=float((changed.float()-original.float()).norm());difference=abs(actual-norm)
        if difference<error:best,error=changed,difference
        if actual<norm:low=scale
        else:high=scale
    if error>max(1e-7,.01*norm):raise ValueError('Cannot match realized perturbation norm within 1%')
    return best


@torch.inference_mode()
def replay(lm,prompt,targets,rows,capture_sites=(),patch=None,capture_rows=None):
    """One uncached B=1 pass; return only requested prediction logits/states.

    LM-head row indices are absolute; capture indices are relative prediction j.
    Register the patch before read-only observers, so captured states include it.
    Existing unrelated hooks are rejected and never removed by this function.
    """
    positions=prediction_positions(prompt,targets)
    rows=list(rows);capture_rows=rows if capture_rows is None else list(capture_rows)
    if lm.training or rows!=sorted(set(rows)) or not rows or any(j<0 or j>=len(targets) for j in rows):
        raise ValueError('Expected eval mode and unique sorted valid prediction rows')
    if capture_rows!=sorted(set(capture_rows)) or any(j not in rows for j in capture_rows):raise ValueError('Invalid capture rows')
    modules={tuple(s):site_module(lm,*s) for s in capture_sites}
    if patch:
        if patch.j<=0 or patch.j not in rows:raise ValueError('M2 patches require a non-anchor captured prediction row')
        if not math.isfinite(patch.alpha):raise ValueError('Nonfinite intervention coefficient')
        module=site_module(lm,patch.layer,patch.site)
    if any(m._forward_hooks or m._forward_pre_hooks for m in lm.modules()):raise ValueError('Unexpected existing hooks')
    device=next(lm.parameters()).device;start=len(prompt)-1
    selected=torch.tensor([start+j for j in rows],device=device)
    captures=torch.tensor([start+j for j in capture_rows],device=device)
    ids=torch.tensor([list(prompt)+list(targets[:-1])],device=device)
    handles=[];states={};record={'hook_calls':0,'selected_prediction_row':patch.j if patch else None};old_cache=lm.config.use_cache
    def apply(mod,args,out):
        h=_out_tensor(out);record['hook_calls']+=1
        if h.ndim!=3 or h.shape[0]!=1 or h.shape[1]!=ids.shape[1]:raise ValueError('Unexpected intervention tensor/batch shape')
        if patch.dtype and str(h.dtype)!=patch.dtype:raise ValueError('Patch dtype differs from matched baseline')
        expected=torch.as_tensor(patch.expected,device=h.device,dtype=h.dtype)
        donor=torch.as_tensor(patch.donor,device=h.device,dtype=h.dtype)
        original=h[0,start+patch.j]
        if original.shape!=expected.shape or donor.shape!=expected.shape or not torch.equal(original,expected):
            raise ValueError('Recipient activation is not the saved matched context')
        delta=intervention_vector(patch.expected,patch.donor,patch.variant,patch.direction,patch.seed)
        if patch.alpha==0:changed=original
        elif patch.variant=='native' and patch.alpha==1:changed=donor
        else:changed=norm_matched_add(original,patch.alpha*torch.as_tensor(delta,device=h.device))
        if not torch.isfinite(changed).all():raise ValueError('Nonfinite patched activation')
        record.update(dtype=str(h.dtype),absolute_position=start+patch.j,predicted_phase=patch.j%7,
            processed_phase=(patch.j-1)%7,native_delta_norm=float((donor.float()-original.float()).norm()),
            actual_delta_norm=float((changed.float()-original.float()).norm()),original_norm=float(original.float().norm()),
            actual_changed_states=int(not torch.equal(changed,original)),expected_recipient_exact=True)
        if patch.alpha==0:return None
        result=h.clone();result[0,start+patch.j]=changed
        return (result,*out[1:]) if isinstance(out,tuple) else result
    def capture(key):
        def hook(mod,args,out):
            h=_out_tensor(out)
            if not torch.isfinite(h).all():raise ValueError('Nonfinite observed state')
            states[key]=h[0,captures].detach().float().cpu().numpy().copy()
        return hook
    started=time.time()
    try:
        if patch:handles.append(module.register_forward_hook(apply))
        for key,mod in modules.items():handles.append(mod.register_forward_hook(capture(key)))
        lm.config.use_cache=False
        output=lm(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=selected)
        if output.past_key_values is not None:raise ValueError('Unintended KV-cache output')
        logits=output.logits[0].detach().float()
        if logits.shape[0]!=len(rows) or not torch.isfinite(logits).all():raise ValueError('Invalid selected logits')
        if patch and record['hook_calls']!=1:raise ValueError('Patch hook must execute exactly once')
    finally:
        lm.config.use_cache=old_cache
        for handle in handles:handle.remove()
    if any(m._forward_hooks or m._forward_pre_hooks for m in lm.modules()):raise RuntimeError('Hook cleanup failed')
    record.update(hooks_removed=True,fresh_uncached=True,runtime_s=time.time()-started)
    return {'logits':logits,'states':states,'rows':rows,'capture_rows':capture_rows,'positions':[positions[j] for j in rows],
            'dtype':str(next(lm.parameters()).dtype),'checks':record}


def jsd(la,lb):
    middle=torch.logaddexp(la,lb)-math.log(2.)
    return (.5*((la.exp()*(la-middle)).sum(-1)+(lb.exp()*(lb-middle)).sum(-1))).clamp_min(0)


def code_logp(logp,phase,base=128266,size=4096):
    block=logp[...,base+phase*size:base+(phase+1)*size]
    if block.shape[-1]!=size:raise ValueError('Expected full vocabulary covering the selected SNAC phase')
    return block-block.logsumexp(-1,keepdim=True)


def score_patch(baselines,changed,recipient,donor,phase,target,floor):
    """Distribution recovery and own-minus-competing recovery; no top-1 primary.

    Inputs are full-vocabulary FP32 log-softmax rows. Normalized selectivity uses
    the own-target baseline distance as one common denominator. Small baselines
    remain in absolute outcomes; undefined ratios are None, never zero.
    """
    if recipient==donor or recipient not in baselines or donor not in baselines:raise ValueError('Invalid recipient/donor pair')
    result={};distances={}
    for metric,transform in [('full',lambda x:x),('code',lambda x:code_logp(x,phase))]:
        before=transform(baselines[recipient]);after=transform(changed);values={}
        for label,lp in baselines.items():
            original=float(jsd(before,transform(lp)));patched=float(jsd(after,transform(lp)))
            values[label]=original-patched
            result[f'{metric}_baseline_to_{label}']=original;result[f'{metric}_patched_to_{label}']=patched
            result[f'{metric}_absolute_recovery_to_{label}']=values[label]
        base=result[f'{metric}_baseline_to_{donor}'];own=values[donor]
        competitors=[c for c in baselines if c not in (recipient,donor)]
        selective=own-sum(values[c] for c in competitors)/len(competitors)
        valid=base>=floor
        result.update({f'{metric}_own_absolute_recovery':own,f'{metric}_selective_absolute_recovery':selective,
                       f'{metric}_baseline_above_floor':valid,f'{metric}_normalized_recovery':own/base if valid else None,
                       f'{metric}_normalized_selectivity':selective/base if valid else None})
    p=changed.exp();offset=128266+phase*4096
    result.update(target_token=int(target),target_logp=float(changed[target]),recipient_target_logp=float(baselines[recipient][target]),
        delta_target_logp=float(changed[target]-baselines[recipient][target]),valid_phase_mass=float(p[offset:offset+4096].sum()),
        eos_128258_probability=float(p[128258]),eos_128001_probability=float(p[128001]),
        total_stop_probability=float(p[128258]+p[128001]),entropy_nats=float(-(p*changed).sum()),
        raw_argmax_id=int(changed.argmax()),valid_phase_argmax_id=int(changed[offset:offset+4096].argmax())+offset)
    return result
