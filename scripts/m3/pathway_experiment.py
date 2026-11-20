"""Query-specific early style K/V replay using the installed attention interface."""
import argparse
from contextlib import ExitStack, contextmanager
import csv
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import numpy as np
import torch

from run_mechanism import load_context, sha


def prompt_layout(model, prompt, text, total, style):
    tok = model.tok
    tid = tok.convert_tokens_to_ids
    encode = lambda value: tok.encode(value, add_special_tokens=False)
    anchor = len(prompt) - 1
    text_end = anchor - 3
    text_start = text_end - len(encode(text))
    if (prompt[text_start:text_end] != encode(text)
            or prompt[text_end:] != [tid(t) for t in
                ("<|eot_id|>", "<|end_of_human|>", "<|start_of_ai|>", "<|start_of_speech|>")]):
        raise ValueError("Prompt suffix does not match the pinned builder")
    speaker = prompt.index(tid("<|speaker>")) + 1
    if prompt[speaker:speaker+len(encode("Amit"))] != encode("Amit"):
        raise ValueError("Wrong speaker field")
    result = dict(speaker=speaker, anchor=anchor, text_start=text_start, text_end=text_end,
                  text=list(range(text_start,text_end)), audio=list(range(anchor,total)))
    if style:
        start = prompt.index(tid("<|style>")) + 1
        end = prompt.index(tid("<style|>"))
        if prompt[start:end] != encode(style):
            raise ValueError("Wrong style field")
        boundaries = list(range(end,text_start)) + list(range(text_end,anchor))
        result.update(style_slots=list(range(start,end)), boundaries=boundaries)
    else:
        result.update(style_slots=[],boundaries=list(range(text_end,anchor)))
    return result


def mix_query_outputs(native, donor, positions):
    if native.shape != donor.shape or native.dtype != donor.dtype:
        raise ValueError("Incompatible native SDPA outputs")
    out = native.clone()
    out[:,positions] = donor[:,positions]
    return out


def replace_slot(keys, values, slot, replacement):
    k, v = replacement
    if (keys[:,:,slot].shape!=k.shape or values[:,:,slot].shape!=v.shape
            or keys.dtype!=k.dtype or values.dtype!=v.dtype):
        raise ValueError("K/V replacement shape or dtype mismatch")
    new_k, new_v = keys.clone(), values.clone()
    new_k[:,:,slot] = k
    new_v[:,:,slot] = v
    return new_k, new_v


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as out:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_checks(output, checks, name, delta, **details):
    checks.append(dict(check=name, max_abs_logit_difference=delta, passed=delta==0, **details))
    with (output/"engineering.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(checks[-1])+"\n")
    if delta != 0:
        raise ValueError(f"Exact engineering control failed: {name} delta={delta}")


@contextmanager
def sdpa_dispatch(registry, interface_name, modules, slots, capture, edit, positions, seen):
    get_interface = registry.get_interface

    def interface(implementation, default):
        native = get_interface(implementation, default)
        if implementation!=interface_name:
            raise ValueError(f"Expected {interface_name}, got {implementation}")

        def apply(module, query, key, value, mask, **kwargs):
            index = modules.get(id(module))
            if index is None:
                return native(module,query,key,value,mask,**kwargs)
            if (query.ndim!=4 or query.shape[0]!=1 or query.shape[1]!=24
                    or key.shape[1]!=8 or value.shape!=key.shape or query.shape[2]!=key.shape[2]
                    or query.shape[-1]!=128 or query.dtype!=torch.bfloat16
                    or key.dtype!=query.dtype or value.dtype!=query.dtype
                    or kwargs.get("cache") is not None or kwargs.get("dropout",0)!=0):
                raise ValueError("Unexpected SDPA shapes, dtype, cache or dropout")
            seen.append(index)
            if capture is not None:
                for slot in slots:
                    capture[index,slot] = (key[:,:,slot].clone(), value[:,:,slot].clone())
            if edit is None:
                return native(module,query,key,value,mask,**kwargs)
            slot, states = edit
            new_k,new_v = replace_slot(key,value,slot,states[index])
            donor, donor_weights = native(module,query,new_k,new_v,mask,**kwargs)
            if positions is None:
                return donor, donor_weights
            original, weights = native(module,query,key,value,mask,**kwargs)
            if weights is not None or donor_weights is not None or original.shape[1]!=query.shape[2]:
                raise ValueError("Expected fused output in B,T,H,D layout")
            return mix_query_outputs(original,donor,positions), None
        return apply

    with patch.object(registry,"get_interface",new=interface):
        yield


def matched_control(donor, news, origin, seed):
    from src.causal_patch import norm_matched_add
    rng=np.random.default_rng(seed)
    result={}
    errors=[]
    for layer, states in donor.items():
        values=[]
        for d,n,o in zip(states,news[layer],origin[layer]):
            amount=(d.float()-n.float()).norm()
            v=torch.as_tensor(rng.standard_normal(o.numel()),dtype=torch.float32,device=o.device)
            patched=norm_matched_add(o.flatten(),v*(amount/v.norm())).reshape_as(o)
            actual=(patched.float()-o.float()).norm()
            errors.append(float((actual-amount).abs()/amount) if float(amount) else 0.)
            values.append(patched)
        result[layer]=tuple(values)
    return result,max(errors)


@torch.inference_mode()
def main(a):
    sys.path.insert(0,str(a.source))
    from scripts.inspect_representations import load_model
    from src.causal_patch import code_logp, jsd, score_patch
    from src.generation_trace import prediction_positions
    from transformers.models.llama import modeling_llama

    tick=time.time()
    old = json.loads(a.design.read_text())
    plan = json.loads(a.plan.read_text())
    a.output.mkdir(mode=0o700)
    torch.cuda.set_per_process_memory_fraction(plan["memory_limit_gib"]*1024**3/
                                              torch.cuda.get_device_properties(0).total_memory)
    manifest=a.source/"source_manifest.json"
    pinned=json.loads(manifest.read_text())
    if pinned["release"]!=old["source_release"] or any(sha(a.source/n)!=h for n,h in pinned["sha256"].items()):
        raise ValueError("Unverified historical source release")
    _,model=load_model(a)
    lm=model.lm
    layers=plan["layers"]
    backend=lm.config._attn_implementation
    if backend!="sdpa" or lm.dtype!=torch.bfloat16 or lm.training:
        raise ValueError("Expected eval-mode BF16 SDPA")
    modules={id(lm.model.layers[i].self_attn):i for i in layers}
    registry=modeling_llama.ALL_ATTENTION_FUNCTIONS
    get_interface=registry.get_interface
    donors=old["donors"]
    labels=["news",*donors]
    metadata_by_id = {d["id"]:d for d in
                      map(json.loads,(a.m2/"inputs/development_cohort.jsonl").read_text().splitlines())}
    scored=plan["score_rows"]
    observed=plan["observation_rows"]
    metrics=[]
    observations=[]
    prompts_table=[]
    checks=[]
    histories={}
    for uid in plan["documents"][:a.limit]:
        meta,_=load_context(a.m2,uid,old)
        original=metadata_by_id[uid]
        text=original["text"]
        if hashlib.sha256(text.encode()).hexdigest()!=meta["text_sha256"]:
            raise ValueError("Historical text checksum differs")
        codes=meta["inspected_codes"]
        if len(codes)!=168 or meta["history"]!=plan["history_index"]:
            raise ValueError("Unexpected history provenance")
        histories[uid]=meta["array_sha256"]
        prompts={name:list(model._build_prompt(model.tok,text,"Amit",style=value))
                 for name,value in plan["styles"].items()}
        layouts={name:prompt_layout(model,p,text,len(p)+len(codes)-1,plan["styles"][name])
                 for name,p in prompts.items()}
        layout=layouts["news"]
        source,=layout["style_slots"]
        speaker=layout["speaker"]
        if not 0 < speaker < source < layout["text_start"] < layout["anchor"]:
            raise ValueError("Prompt query boundaries are invalid")
        for label in labels:
            actual=prompts[label]
            diffs=[i for i,(n,d) in enumerate(zip(prompts["news"],actual)) if n!=d]
            if (actual!=meta["prompts"][label] or len(actual)!=len(prompts["news"])
                    or diffs!=([] if label=="news" else [source])):
                raise ValueError("Not an equal-format one-slot comparison")
        for label,p in prompts.items():
            l=layouts[label]
            recorded=prediction_positions(p,codes)
            for j in observed:
                if recorded[j]["absolute_position"]!=l["anchor"]+j:
                    raise ValueError("Semantic prediction alignment failed")
            prompts_table.append(dict(id=uid,label=label,style=plan["styles"][label],prompt_tokens=len(p),
                tokens_vs_news=len(p)-len(prompts["news"]),style_token_ids=p[l["style_slots"][0]:l["style_slots"][-1]+1]
                    if l["style_slots"] else [], style_token_positions=l["style_slots"],
                speaker_slot=speaker,text_start=l["text_start"],text_end_exclusive=l["text_end"],
                speech_start=l["anchor"],absolute_scored_positions=[l["anchor"]+j for j in observed],
                prompt_ids=p))

        def forward(label, *, capture=False, native_edit=None, edit=None, queries=None,
                    clamp=None, validate_bf16=False):
            prompt=prompts[label]
            row_layout=layouts[label]
            absolute=[row_layout["anchor"]+j for j in observed]
            probes=sorted({speaker-1,source-1,source,layout["text_start"]-1,layout["anchor"]-1}) if label in labels else []
            seq=sorted(set(probes+absolute))
            indices=[seq.index(pos) for pos in absolute]
            ids=torch.tensor([prompt+codes[:-1]],device=lm.device)
            head=torch.tensor(seq,device=lm.device)
            mlp_rows=torch.tensor([row_layout["anchor"]+j for j in scored],device=lm.device)
            states={} if capture else None
            projections={}
            round_trip={}
            seen=[]
            captured_mlp=[]

            def take_projection(i, which):
                def keep(mod, args, out):
                    projections[i,which]={k:out[:,k].clone() for k in (speaker,source)}
                return keep

            def change_projection(slot, replacement):
                def substitute(mod,args,out):
                    if out.dtype!=replacement.dtype or out[:,slot].shape!=replacement.shape:
                        raise ValueError("Native pre-RoPE replacement shape mismatch")
                    result=out.clone()
                    result[:,slot]=replacement
                    return result
                return substitute

            def control_mlp(mod, args, out):
                if clamp is not None:
                    value=out.clone()
                    value[0,mlp_rows]=clamp
                    return value
                captured_mlp.append(out[0,mlp_rows].clone())

            with ExitStack() as scope:
                if capture:
                    for i in layers:
                        attn=lm.model.layers[i].self_attn
                        for which in ("k_proj","v_proj"):
                            scope.enter_context(getattr(attn,which).register_forward_hook(take_projection(i,which)))
                if native_edit:
                    slot, sv=native_edit
                    for (i,which),values in sv.items():
                        mod=getattr(lm.model.layers[i].self_attn,which)
                        scope.enter_context(mod.register_forward_hook(change_projection(slot,values[slot])))
                if capture or clamp is not None:
                    scope.enter_context(lm.model.layers[27].mlp.register_forward_hook(control_mlp))
                if capture or edit is not None:
                    scope.enter_context(sdpa_dispatch(registry,backend,modules,(speaker,source),states,edit,queries,seen))
                if validate_bf16:
                    def see(i, key, pre=False):
                        def observer(mod,args,out=None):
                            value=args[0] if pre else out
                            if value.dtype!=torch.bfloat16:
                                raise ValueError("Non-BF16 decoder residual arithmetic")
                            round_trip[i,key]=value.detach().clone()
                        return observer
                    for i in (0,8,27):
                        block=lm.model.layers[i]
                        for obj,k,pre in ((block.input_layernorm,"in",True),(block.self_attn.o_proj,"attn",False),
                                          (block.post_attention_layernorm,"skip",True),(block.mlp,"mlp",False),
                                          (block,"out",False)):
                            hook=obj.register_forward_pre_hook if pre else obj.register_forward_hook
                            scope.enter_context(hook(see(i,k,pre)))
                output=lm(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                          logits_to_keep=head)
            if ((capture or edit is not None) and seen!=layers or output.past_key_values is not None
                    or registry.get_interface!=get_interface):
                raise ValueError("SDPA coverage, hook restoration or cache check failed")
            if any(m._forward_hooks or m._forward_pre_hooks for m in lm.modules()):
                raise ValueError("Leaked torch hooks")
            logits=output.logits[0].float()
            if not torch.isfinite(logits).all():
                raise ValueError("Non-finite replay")
            if validate_bf16:
                for i in (0,8,27):
                    skip=round_trip[i,"in"]+round_trip[i,"attn"]
                    out=skip+round_trip[i,"mlp"]
                    err=max(float((skip-round_trip[i,"skip"]).abs().max()),
                            float((out-round_trip[i,"out"]).abs().max()))
                    save_checks(a.output,checks,"bf16_residual_arithmetic",err,id=uid,layer=i)
            return dict(logits=logits,probes=seq,indices=indices,states=states,projections=projections,
                        mlp=captured_mlp[0] if captured_mlp else None)

        normal={s:forward(s,capture=s in labels) for s in prompts}
        news=normal["news"]
        baseline=forward("news",validate_bf16=True)
        save_checks(a.output,checks,"capture_identity",float((news["logits"]-baseline["logits"]).abs().max()),id=uid)

        def score_rows(result):
            return result["logits"][result["indices"]].log_softmax(-1)

        lp={s:score_rows(v) for s,v in normal.items()}
        observation_pairs=[*( (s,"news") for s in donors), *((s,"default") for s in donors),
                           ("news","default")]
        if len(prompts["neutral_candidate"])==len(prompts["news"]) and layouts["neutral_candidate"]["style_slots"]==[source]:
            observation_pairs.append(("neutral_candidate","news"))
        for target,reference in observation_pairs:
            for k,j in enumerate(observed):
                prob_a,prob_b=lp[target][k],lp[reference][k]
                phase=j%7
                observations.append(dict(id=uid,target=target,reference=reference,j=j,phase=phase,
                    target_absolute=layouts[target]["anchor"]+j,reference_absolute=layouts[reference]["anchor"]+j,
                    conditional_jsd_nats=float(jsd(code_logp(prob_a,phase),code_logp(prob_b,phase))),
                    full_jsd_nats=float(jsd(prob_a,prob_b))))

        def slot_states(baseline_state, slot):
            return {i:baseline_state["states"][i,slot] for i in layers}

        def assert_early(out, before, name, style):
            unedited=[k for k,pos in enumerate(out["probes"]) if pos<before]
            if not unedited:
                raise ValueError("No earlier causal positions observed")
            e=float((out["logits"][unedited]-news["logits"][unedited]).abs().max())
            save_checks(a.output,checks,"earlier_logits",e,id=uid,style=style,variant=name,
                        positions=[out["probes"][i] for i in unedited])

        text_queries = layout["text"]
        audio_queries = layout["audio"]
        joint=sorted(text_queries+audio_queries)
        boundaries=layout["boundaries"]
        paths={"text_only":text_queries,"audio_only":audio_queries,"joint":joint}
        self_scope=sorted(joint+boundaries)
        if len(set(self_scope))!=len(self_scope) or self_scope!=list(range(source+1,len(prompts["news"])+len(codes)-1)):
            raise ValueError("Literal text/audio/boundary spans do not tile positions after style")
        news_style=slot_states(news,source)
        news_field=slot_states(news,speaker)
        identity=forward("news",edit=(source,news_style),queries=joint)
        save_checks(a.output,checks,"query_gated_news_identity",
                    float((identity["logits"]-news["logits"]).abs().max()),id=uid)
        for d in donors:
            donor=normal[d]
            donor_style=slot_states(donor,source)
            donor_field=slot_states(donor,speaker)
            unrestricted=forward("news",native_edit=(source,donor["projections"]))
            via_interface=forward("news",edit=(source,donor_style))
            save_checks(a.output,checks,"unrestricted_reference_equivalence",
                        float((via_interface["logits"]-unrestricted["logits"]).abs().max()),id=uid,style=d)
            assert_early(unrestricted,source,"unrestricted",d)
            result={"unrestricted":unrestricted}
            for name,queries in {**paths,"boundaries_only":boundaries,"joint_plus_boundaries":self_scope}.items():
                result[name]=forward("news",edit=(source,donor_style),queries=queries,validate_bf16=name=="joint")
                assert_early(result[name],min(queries),name,d)
            for name,queries in {**paths,"unrestricted":None}.items():
                result[name+"_clamped"]=forward("news",edit=(source,donor_style),queries=queries,clamp=news["mlp"])
                assert_early(result[name+"_clamped"],min(queries) if queries else source,name+"_clamped",d)
            same_field=forward("news",edit=(speaker,donor_field),queries=joint)
            save_checks(a.output,checks,"unchanged_earlier_field_donor",
                        float((same_field["logits"]-news["logits"]).abs().max()),id=uid,style=d)
            seed=int(hashlib.sha256(f'{plan["random_seed"]}/{uid}/{d}'.encode()).hexdigest()[:8],16)
            random_style,err=matched_control(donor_style,news_style,news_style,seed)
            random_field,ferr=matched_control(donor_style,news_style,news_field,seed)
            if max(err,ferr)>.01:
                raise ValueError("Random norms must match within 1%")
            control_errors={}
            for name,queries in paths.items():
                key=name+"_random"
                result[key]=forward("news",edit=(source,random_style),queries=queries)
                assert_early(result[key],min(queries),key,d)
                control_errors[key]=err
            result["nonstyle_donor"]=same_field
            result["nonstyle_random"]=forward("news",edit=(speaker,random_field),queries=joint)
            assert_early(result["nonstyle_random"],min(joint),"nonstyle_random",d)
            control_errors["nonstyle_random"]=ferr
            if a.delimiter_focus:
                groups={"style_close_only":[boundaries[0]],"metadata_newline_only":boundaries[1:2],
                        "style_close_newline":list(range(source+1,layout["text_start"])),
                        "end_text_controls":list(range(layout["text_end"],layout["anchor"])),
                        "boundaries_only":boundaries}
                if sorted(groups["style_close_newline"]+groups["end_text_controls"])!=boundaries or len(boundaries)!=5:
                    raise ValueError("Unexpected prompt delimiter inventory")
                for name,queries in groups.items():
                    if name not in result:
                        result[name]=forward("news",edit=(source,donor_style),queries=queries)
                        assert_early(result[name],min(queries),name,d)
                    if name in ("boundaries_only","style_close_newline","end_text_controls"):
                        key=name+"_clamped"
                        result[key]=forward("news",edit=(source,donor_style),queries=queries,clamp=news["mlp"])
                        assert_early(result[key],min(queries),key,d)
                for key,slot,norms,norm_error in (
                    ("boundaries_only_random",source,random_style,err),
                    ("boundaries_nonstyle_random",speaker,random_field,ferr)):
                    result[key]=forward("news",edit=(slot,norms),queries=boundaries)
                    assert_early(result[key],min(boundaries),key,d)
                    control_errors[key]=norm_error
                decomposition={
                    "joint":("text_only","audio_only"),
                    "joint_plus_boundaries":("joint","boundaries_only"),
                    "boundaries_only":("style_close_newline","end_text_controls")}
                decoded_logits = {n:score_rows(out) for n,out in result.items()}
            for name,out in result.items():
                current=score_rows(out)
                for j in scored:
                    k=observed.index(j)
                    s=score_patch({s:lp[s][k] for s in labels},current[k],"news",d,j%7,codes[j],
                                  plan["ratio_floor_nats"])
                    if a.delimiter_focus and name in decomposition:
                        first,second=decomposition[name]
                        code = lambda lp: code_logp(lp, j % 7)
                        additive=code(decoded_logits[first][k])+code(decoded_logits[second][k])-code(lp["news"][k])
                        additive=additive-additive.logsumexp(-1)
                        observed_joint=code(current[k])
                        s["joint_vs_log_additive_jsd_nats"]=float(jsd(observed_joint,additive))
                        s["joint_vs_news_conditional_jsd_nats"]=float(jsd(observed_joint,code(lp["news"][k])))
                    metrics.append(dict(id=uid,style=d,condition=name,j=j,phase=j%7,
                        random_norm_relative_error=control_errors.get(name),**s))
        print(uid,len(metrics),round(time.time()-tick,1),flush=True)

    for name,rows in (("pathways.csv",metrics),("prompt_tokens.csv",prompts_table),
                      ("prompt_observational.csv",observations)):
        write_csv(a.output/name,rows)
    receipt=dict(status="passed",documents=plan["documents"][:a.limit],
                 checkpoint=old["checkpoint"],archived_histories=histories,
                 torch=torch.__version__,gpu=torch.cuda.get_device_name(),runtime_s=time.time()-tick,
                 max_allocated_bytes=torch.cuda.max_memory_allocated(),sdpa_backend=backend,
                 installed_llama_sha256=sha(Path(modeling_llama.__file__)),
                 installed_interface_sha256=sha(Path(inspect.getfile(get_interface(backend,None)))),
                 plan_sha256=sha(a.plan),script_sha256=sha(Path(__file__)),source_manifest_sha256=sha(manifest),
                 engineering_checks=len(checks),output_hashes={p.name:sha(p) for p in a.output.iterdir()})
    (a.output/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("design","plan","source","m2","config","snapshot","codec_snapshot","output"):
        parser.add_argument("--"+key.replace("_","-"),required=True,type=Path)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--limit",type=int,choices=[1,6],required=True)
    parser.add_argument("--delimiter-focus",action="store_true")
    main(parser.parse_args())
