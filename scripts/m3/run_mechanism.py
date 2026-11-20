"""Read saved M2 contexts and intervene on the exact final-block computation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_updates(calibration, labels, rows):
    result = {}
    for style in labels:
        item = np.stack([capture[style] - capture["news"] for capture in calibration])
        result[(style, "global")] = item.mean((0, 1))
        result.update({(style, row): update for row, update in zip(rows, item.mean(0))})
    return result


def magnitude_rows(before, after):
    return (after.float() - before.float()).norm(dim=-1)


def load_context(root, uid, design):
    folder = root / "baselines" / uid / f'h{design["history"]}'
    archive = folder / "baseline.npz"
    metadata = json.loads((folder / "metadata.json").read_text())
    if (metadata["array_sha256"] != sha(archive) or metadata["id"] != uid
            or metadata["history"] != design["history"]
            or metadata["checkpoint"] != design["checkpoint"]
            or metadata["codec"] != design["codec"]
            or metadata["dtype"] != "torch.bfloat16"
            or metadata["study_split"] != "development"):
        raise ValueError(f"Invalid baseline identity for {uid}")
    return metadata, archive


@torch.inference_mode()
def run(args):
    sys.path.insert(0, str(args.source))
    from scripts.inspect_representations import load_model
    from src.causal_patch import code_logp, jsd, norm_matched_add, replay, score_patch
    from src.generation_trace import prediction_positions
    from src.steering import _out_tensor

    started = time.time()
    design = json.loads(args.design.read_text())
    donor_labels = design["donors"]
    labels = [design["recipient"], *donor_labels]
    rows = design["prediction_rows"]
    args.output.mkdir(mode=0o700)
    manifest = json.loads((args.source / "source_manifest.json").read_text())
    if manifest["release"] != design["source_release"]:
        raise ValueError("Unexpected M2 release")
    if any(sha(args.source / name) != expected for name, expected in manifest["sha256"].items()):
        raise ValueError("Changed M2 source release")
    provenance = {"source_manifest_sha256": sha(args.source / "source_manifest.json"),
                  "design_sha256": sha(args.design), "script_sha256": sha(Path(__file__)),
                  "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__), "contexts": {}}

    def context(uid):
        metadata, archive = load_context(args.m2, uid, design)
        count = len(metadata["inspected_codes"])
        prompts = metadata["prompts"]
        if count != 168 or len({len(prompts[s]) for s in labels}) != 1:
            raise ValueError("Expected matched prompt lengths and 24 full frames")
        for label in labels:
            prediction_positions(prompts[label], metadata["inspected_codes"])
        indices = [metadata["prediction_rows"].index(row) for row in rows]
        provenance["contexts"][uid] = metadata["array_sha256"]
        return metadata, archive, indices

    archived_calibration = []
    for uid in design["calibration"]:
        _, archive, indices = context(uid)
        with np.load(archive, allow_pickle=False) as data:
            archived_calibration.append({label: data[label + "__27__mlp"][indices] for label in labels})
    _, model = load_model(args)
    lm = model.lm
    if lm.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 operations")
    final = lm.model.layers[27]
    device = lm.device
    controls = []
    scores = []
    sites = ((26, "resid"), (27, "attn"), (27, "mlp"), (27, "resid"))

    def live_replays(metadata, selection, sites):
        codes = metadata["inspected_codes"]
        result = {}
        for label in labels:
            trace = replay(lm, metadata["prompts"][label], codes, selection, sites)
            result[label] = trace
        return result

    calibration = []
    for uid in design["calibration"]:
        metadata, _, _ = context(uid)
        traces = live_replays(metadata, rows, [(27, "mlp")])
        calibration.append({label: traces[label]["states"][(27, "mlp")] for label in labels})
    updates = mean_updates(calibration, donor_labels, rows)
    archived_updates = mean_updates(archived_calibration, donor_labels, rows)
    provenance["m2_archive_vs_m3_calibration_cosine"] = {
        f"{s}__{k}": float(np.dot(v, archived_updates[s, k]) /
                          (np.linalg.norm(v)*np.linalg.norm(archived_updates[s, k])))
        for (s,k), v in updates.items()}
    np.savez_compressed(args.output / "frozen_updates.npz",
                        **{f"{style}__{key}": v for (style, key), v in updates.items()})

    def capture_checks(uid, label, computed, saved, selected, enforce):
        base = saved.log_softmax(-1)
        value = computed.log_softmax(-1)
        difference = float((computed - saved).abs().max())
        full = float(jsd(base, value).max())
        conditional = max(float(jsd(code_logp(base[k], j % 7),
                                    code_logp(value[k], j % 7))) for k, j in enumerate(selected))
        if (not torch.isfinite(computed).all() or (enforce and (
            difference != 0 or max(full, conditional) > design["baseline_jsd_tolerance_nats"]))):
            raise ValueError(f"{uid}/{label} readout validation failed {difference=} {full=} {conditional=}")
        controls.append(dict(id=uid, label=label, max_logit_delta=difference,
                             max_full_jsd=full, max_code_jsd=conditional, enforced=enforce))

    for uid in design["assessment"][:args.limit]:
        metadata, archive, indices = context(uid)
        recorded_rows = metadata["prediction_rows"]
        codes = metadata["inspected_codes"]
        prompt = metadata["prompts"]["news"]
        absolute = torch.tensor([len(prompt)-1+j for j in recorded_rows], device=device)
        dtype = lm.dtype
        tokens = torch.tensor([prompt+codes[:-1]], device=device)
        unpatched = [i for i in range(len(absolute)) if i not in indices]

        def tensor(arr):
            return torch.as_tensor(arr, device=device, dtype=dtype)

        def checked_logits(output, expected_logits):
            values = output.logits[0].float()
            if (output.past_key_values is not None or not torch.isfinite(values).all()
                    or not torch.equal(values[unpatched], expected_logits[unpatched])):
                raise ValueError("Unexpected cache, nonfinite logits or change at an unpatched row")
            return values

        def mlp_replay(residual, expected_input, expected_logits, indices):
            """Patch only what enters the final MLP normalization, preserving the skip."""
            subset = absolute[indices]
            calls = []
            captured = []

            def substitute(module, inputs):
                state = inputs[0]
                if not torch.equal(state[0, absolute], expected_input):
                    raise ValueError("Live News MLP input mismatches exact BF16 residual sum")
                calls.append(True)
                result = state.clone()
                result[0, subset] = residual[indices]
                return (result, *inputs[1:])

            def observe(module, inputs, output):
                captured.append(output[0, absolute].clone())

            with final.post_attention_layernorm.register_forward_pre_hook(substitute), \
                    final.mlp.register_forward_hook(observe):
                output = lm(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                            use_cache=False, logits_to_keep=absolute)
            if len(calls)!=1 or len(captured)!=1:
                raise ValueError("Unintended processing of live MLP patch")
            checked_logits(output, expected_logits)
            return captured[0]

        def readout_replay(replacement, expected_state, expected_logits):
            """Replace rows before native final RMSNorm and native head on News."""
            calls = []

            def substitute(module, inputs, output):
                state = _out_tensor(output)
                if not torch.equal(state[0, absolute], expected_state):
                    raise ValueError("Unmodified News residual differs from baseline")
                calls.append(True)
                result = state.clone()
                result[0, absolute[indices]] = replacement[indices]
                return (result, *output[1:]) if isinstance(output, tuple) else result

            with final.register_forward_hook(substitute):
                output = lm(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                            use_cache=False, logits_to_keep=absolute)
            if len(calls)!=1:
                raise ValueError("Expected one final-block hook call")
            return checked_logits(output, expected_logits)

        traces = live_replays(metadata, recorded_rows, sites)
        components = {label: [tensor(traces[label]["states"][site]) for site in sites] for label in labels}
        logits = {label: traces[label]["logits"] for label in labels}
        with np.load(archive, allow_pickle=False) as data:
            for label in labels:
                capture_checks(uid, label+"__historical", logits[label],
                               torch.from_numpy(data[label+"__logits"]).to(device), recorded_rows, False)
        for label, (r, a, m, output) in components.items():
            if not torch.equal((r+a)+m, output):
                raise ValueError("Saved BF16 composition does not reconstruct exactly")
        n_r, n_a, n_m, n_output = components["news"]
        n_bypass = n_r + n_a
        capture_checks(uid,"news__identity",readout_replay(n_output,n_output,logits["news"]),
                       logits["news"],recorded_rows,True)
        if not torch.equal(mlp_replay(n_bypass, n_bypass, logits["news"], indices), n_m):
            raise ValueError("Identical MLP pre-normalization patch must preserve the original branch")
        baseline_lp = {label: values.log_softmax(-1) for label, values in logits.items()}
        for style, (d_r, d_a, d_m, _) in components.items():
            if style == "news":
                continue
            d_bypass = d_r + d_a
            donor_at_mlp_input = mlp_replay(d_bypass, n_bypass, logits["news"], indices)
            if not torch.equal(donor_at_mlp_input[indices], d_m[indices]):
                raise ValueError("Donor MLP input must reproduce the pointwise donor branch")
            variants = {
                "native_mlp": (n_bypass, d_m),
                "bypass_only": (d_bypass, n_m),
                "joint": (d_bypass, d_m),
                "upstream_residual_via_mlp": (n_bypass, mlp_replay(d_r+n_a, n_bypass, logits["news"], indices)),
                "attention_via_mlp": (n_bypass, mlp_replay(n_r+d_a, n_bypass, logits["news"], indices)),
            }
            for kind in ("global", "position"):
                after = n_m.clone()
                random_after = n_m.clone()
                for j, k in zip(rows, indices):
                    update = updates[style, "global" if kind == "global" else j]
                    delta = torch.from_numpy(update).to(device)*design["controller_strength"]
                    after[k] = (n_m[k].float() + delta).to(dtype)
                    realized = magnitude_rows(n_m[k], after[k])
                    key = f'{design["control_seed"]}/{uid}/{style}/{kind}/{j}'
                    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
                    rng = np.random.default_rng(seed)
                    v = torch.as_tensor(rng.standard_normal(delta.shape), dtype=torch.float32, device=device)
                    random_after[k] = norm_matched_add(n_m[k], v*(realized / v.norm()))
                variants[kind+"_fit"] = (n_bypass, after)
                variants[kind+"_random"] = (n_bypass, random_after)
            for name, (bypass, branch) in variants.items():
                # The final-block output has no cross-position operation downstream.
                outgoing = bypass + branch
                output = readout_replay(outgoing,n_output,logits["news"])
                if name=="joint":
                    capture_checks(uid,style+"__donor_joint",output[indices],logits[style][indices],rows,True)
                for j, k in zip(rows, indices):
                    changed = output[k].log_softmax(-1)
                    metrics = score_patch({s: lp[k] for s, lp in baseline_lp.items()}, changed,
                                          "news", style, j % 7, codes[j],
                                          design["normalization_floor_nats"])
                    scores.append(dict(id=uid, history=design["history"], style=style, variant=name,
                                       j=j, predicted_phase=j%7, consumed_phase=(j-1)%7,
                                       mlp_delta_norm=float(magnitude_rows(n_m[k], branch[k])),
                                       bypass_delta_norm=float(magnitude_rows(n_bypass[k], bypass[k])),
                                       output_delta_norm=float(magnitude_rows(n_output[k], outgoing[k])), **metrics))
        print(uid, len(scores), round(time.time()-started, 1), flush=True)
    for filename, values in (("mechanism.csv", scores), ("numerical_controls.csv", controls)):
        with (args.output / filename).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    provenance.update(elapsed_s=time.time()-started, max_allocated_bytes=torch.cuda.max_memory_allocated(),
                      output_hashes={p.name: sha(p) for p in args.output.iterdir()})
    (args.output / "receipt.json").write_text(json.dumps(provenance, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("design", "source", "m2", "config", "snapshot", "codec_snapshot", "output"):
        parser.add_argument("--"+key.replace("_","-"), required=True, type=Path)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--device", default="cuda:0")
    run(parser.parse_args())
