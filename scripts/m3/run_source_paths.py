"""Patch the exact changed style-token K/V slots, then clamp the late MLP."""
import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
import time

import torch

from run_mechanism import load_context, sha


BANDS = {"early": list(range(0,9)), "middle":list(range(9,19)), "late":list(range(19,28)),
         "all":list(range(28))}
TEXTS = ("d005","d006","d007","d008","d009","d010")


@torch.inference_mode()
def main(a):
    sys.path.insert(0,str(a.source))
    from scripts.inspect_representations import load_model
    from src.causal_patch import score_patch
    from src.generation_trace import prediction_positions

    tick=time.time()
    design=json.loads(a.design.read_text())
    _, speech=load_model(a)
    lm=speech.lm
    a.output.mkdir(mode=0o700)
    labels=["news",*design["donors"]]
    arow=design["prediction_rows"]
    records=[]
    info=[]
    max_checks={"identity":0,"earlier":0}
    for uid in TEXTS[:a.limit]:
        metadata,_=load_context(a.m2,uid,design)
        prompts=metadata["prompts"]
        codes=metadata["inspected_codes"]
        rec=prompts["news"]
        for label in labels:
            prediction_positions(prompts[label],codes)
        rows=sorted({0,*arow})
        idx=torch.tensor([len(rec)-1+j for j in rows],device=lm.device)
        anchor=torch.tensor([len(rec)-1+j for j in arow],device=lm.device)
        # Style fields have equal lengths. Restrict to the actual differing token span.
        changed_slots={label:[i for i,(r,d) in enumerate(zip(rec,prompts[label])) if r!=d]
                       for label in labels[1:]}
        from itertools import chain
        positions=sorted(set(chain.from_iterable(changed_slots.values())))
        if not positions or len(positions)>8 or any(len(p)!=len(rec) for p in prompts.values()):
            raise ValueError("Expected a small, equal-format style span")
        scored=[]

        def forward(label, capture=False, donor=None, layers=(), original=None, freeze=None):
            prompt=prompts[label]
            ids=torch.tensor([prompt+codes[:-1]],device=lm.device)
            captured={}
            received=[]

            def observer(key, position):
                def save(mod,args,out):
                    captured[key]=out[0,position].clone()
                return save

            def intervention(key, position, selected):
                def replace(mod,args,out):
                    if not torch.equal(out[0,position],original[key]):
                        if key[0] == 0:
                            raise ValueError("Original first-layer K/V has changed")
                    result=out.clone()
                    result[0,selected]=donor[key][[positions.index(k) for k in selected]]
                    received.append(key)
                    return result
                return replace

            def clamp(mod,args,out):
                result=out.clone()
                result[0,anchor]=freeze
                received.append("clamp")
                return result

            with ExitStack() as stack:
                for k,layer in enumerate(lm.model.layers):
                    for projection in ("k_proj","v_proj"):
                        module=getattr(layer.self_attn,projection)
                        key=(k,projection)
                        if donor is not None and k in layers:
                            stack.enter_context(module.register_forward_hook(
                                intervention(key,positions,scored)))
                        if capture:
                            stack.enter_context(module.register_forward_hook(observer(key,positions)))
                if freeze is not None:
                    stack.enter_context(lm.model.layers[27].mlp.register_forward_hook(clamp))
                elif capture:
                    stack.enter_context(lm.model.layers[27].mlp.register_forward_hook(observer("mlp",anchor)))
                out=lm(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,logits_to_keep=idx)
            values=out.logits[0].float()
            if not torch.isfinite(values).all() or out.past_key_values is not None:
                raise ValueError("Invalid K/V edit")
            if len(received)!=2*len(layers)+int(freeze is not None):
                raise ValueError("Unexpected K/V hook coverage")
            return values,captured

        baseline={label:forward(label,capture=True) for label in labels}
        for donor in labels[1:]:
            scored=changed_slots[donor]
            if not scored:
                raise ValueError("Donor matches News tokenization exactly")
            news_logits, news_state=baseline["news"]
            donor_logits, donor_state=baseline[donor]
            identity,_=forward("news",donor=news_state,layers=BANDS["all"],original=news_state)
            err=float((identity-news_logits).abs().max())
            max_checks["identity"]=max(max_checks["identity"],err)
            if err!=0:
                raise ValueError("K/V identity patch changed logits")
            restored={k:forward("news",donor=donor_state,layers=v,original=news_state)[0]
                      for k,v in BANDS.items()}
            restored["all_mlp_clamped"]=forward("news",donor=donor_state,layers=BANDS["all"],
                                                original=news_state,freeze=news_state["mlp"])[0]
            noised,_=forward(donor,donor=news_state,layers=BANDS["all"],original=donor_state)
            restored["all_reverse"] = noised
            # Check strict causal isolation in a separate pre-style prefix; the analysis
            # rows above are after the source tokens and may legitimately change.
            before_idx = torch.tensor([min(scored)-1],device=lm.device)
            if before_idx.item()<0:
                raise ValueError("Style must follow an untouched prompt position")
            end_rows=idx
            idx=before_idx
            over=forward("news",donor=donor_state,layers=BANDS["all"],original=news_state)[0]
            clean=forward("news")[0]
            idx=end_rows
            err=float((over-clean).abs().max())
            max_checks["earlier"]=max(max_checks["earlier"],err)
            if err!=0:
                raise ValueError("Changed prediction at a causal position before source patch")
            for variant,logits in restored.items():
                receiver,target=("news",donor) if variant!="all_reverse" else (donor,"news")
                # Score the reverse separately. Competing labels omit the receiver and target.
                for j in arow:
                    k=rows.index(j)
                    result=score_patch({s: pair[0][k].log_softmax(-1) for s,pair in baseline.items()},
                                      logits[k].log_softmax(-1),receiver,target,j%7,codes[j],.001)
                    records.append(dict(id=uid, donor=donor, variant=variant, recipient=receiver,
                                        target=target, j=j, phase=j%7, **result))
            info.append(dict(id=uid,donor=donor,source_token_positions=scored,
                             donor_style_tokens=[prompts[donor][i] for i in scored],
                             news_style_tokens=[rec[i] for i in scored],
                             archive_sha256=metadata["array_sha256"]))
        print(uid,len(records),round(time.time()-tick,2),flush=True)
    with (a.output/"source_path_effects.csv").open("x",newline="") as out:
        writer=csv.DictWriter(out,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (a.output/"source_positions.json").write_text(json.dumps(info,indent=2)+"\n")
    receipt=dict(script_sha256=sha(Path(__file__)),design_sha256=sha(a.design),
                 documents=len({r["id"] for r in records}),checks=max_checks,bands=BANDS,
                 runtime_s=time.time()-tick,max_allocated_bytes=torch.cuda.max_memory_allocated(),
                 output_hashes={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("design","source","m2","config","snapshot","codec_snapshot","output"):
        parser.add_argument("--"+key.replace("_","-"),required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--limit",type=int,default=6)
    main(parser.parse_args())
