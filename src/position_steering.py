"""Speech-start position mask around the upstream norm-preserving CAA hook."""
from math import isfinite
from types import SimpleNamespace
import torch
from src.steering import Steerer


class AnchorIntervention:
    """Alter the final prompt state once; leave cached audio-token states alone.

    epsilon is a fraction of the selected state's L2 norm BEFORE rescaling.
    This prompt-position protocol is different from upstream all/generated mode.
    """
    def __init__(self, lm, prompt, layer, direction, epsilon, speech_start_id=128257):
        direction = torch.as_tensor(direction)
        if not prompt or prompt[-1] != speech_start_id:
            raise ValueError('Expected final speech-start prompt')
        if direction.ndim != 1 or not torch.isfinite(direction).all() or float(direction.norm()) == 0:
            raise ValueError('Expected a finite nonzero direction')
        if not isfinite(epsilon) or not 0 <= layer < len(lm.model.layers):
            raise ValueError('Invalid epsilon or layer')
        self.lm, self.prompt, self.layer = lm, list(prompt), layer
        self.direction, self.epsilon = direction, float(epsilon)
        self.handles, self.calls, self.selected, self.ratios = [], 0, 0, []
        self.active, self.removed = False, False
        self.reused = Steerer(SimpleNamespace(layers=lm.model.layers), mode='all')

    def before(self, module, args, kwargs):
        ids = kwargs.get('input_ids', args[0] if args else None)
        cache = kwargs.get('past_key_values')
        seen = 0 if cache is None else cache.get_seq_length()
        if self.calls == 0:
            if ids is None or seen != 0 or ids.shape != (1, len(self.prompt)) or ids[0].tolist() != self.prompt:
                raise ValueError('Requires the exact prompt and a fresh cache')
        self.active = self.calls == 0
        self.calls += 1

    def apply(self, module, args, out):
        if not self.active:
            return None
        h = out[0] if isinstance(out, tuple) else out
        if h.shape[:2] != (1, len(self.prompt)) or h.shape[-1] != len(self.direction):
            raise ValueError('Unexpected prefill/direction shape')
        self.selected += 1
        if self.epsilon == 0:
            return None  # Bit-exact zero behavior, including BF16.
        anchor = h[:, -1:, :]
        norm = anchor.float().norm(dim=-1)
        if not torch.isfinite(norm).all() or bool((norm == 0).any()):
            raise ValueError('Invalid selected state norm')
        unit = self.direction.to(h.device, dtype=torch.float32)
        unit = unit / unit.norm()
        delta = self.epsilon * norm * unit[None, :]
        changed = self.reused._hook(delta)(module, args, anchor)
        if not torch.isfinite(changed).all():
            raise ValueError('Nonfinite intervention')
        self.ratios.append(float((changed.float() - anchor.float()).norm() / norm.squeeze()))
        new = h.clone()
        new[:, -1:, :] = changed
        return (new, *out[1:]) if isinstance(out, tuple) else new

    def __enter__(self):
        block = self.lm.model.layers[self.layer]
        self.before_counts = (len(self.lm._forward_pre_hooks), len(block._forward_hooks))
        if any(self.before_counts) or self.calls:
            raise ValueError('Use a new context with no existing hooks')
        try:
            self.handles.append(self.lm.register_forward_pre_hook(self.before, with_kwargs=True))
            self.handles.append(block.register_forward_hook(self.apply))
        except Exception:
            for handle in self.handles:
                handle.remove()
            raise
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.removed = self.before_counts == (len(self.lm._forward_pre_hooks),
                                             len(self.lm.model.layers[self.layer]._forward_hooks))
        if not self.removed:
            raise RuntimeError('Hook cleanup failed')

    def result(self):
        return {'site': 'post-block final speech-start prompt token only', 'layer': self.layer,
                'epsilon_requested': self.epsilon, 'selected_states': self.selected,
                'forward_calls': self.calls, 'actual_relative_L2': self.ratios,
                'hooks_removed': self.removed}
