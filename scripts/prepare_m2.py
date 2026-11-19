"""Freeze the initial M2 development setup without inspecting causal outcomes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest,write_json,write_jsonl


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--m1',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
    source=a.m1/'dataset/cohort.jsonl';rows=[json.loads(s) for s in source.read_text().splitlines()]
    development=[r for r in rows if r['study_split']=='development']
    if len(development)!=30:raise ValueError('Expected original 30 M1 development texts')
    if (out/'protocol_development.json').exists():raise FileExistsError('Do not overwrite a frozen setup')
    write_jsonl(out/'development_cohort.jsonl',development)
    protocol=dict(study='M2 causal localization',version=1,checkpoint='76e0814f189321efe550011850cd288dae366aad',
        codec='d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec',speaker='Amit',language='English',dtype='bfloat16',
        styles=dict(news='news',anger='ANGER',happy='happy',sad='sad'),history_seeds=[1234,5678],window_frames=24,
        frames=[4,18],phases=[0,1,2,4,5],prediction_rows=sorted({0,1,27,125,167,*[f*7+p for f in (4,18) for p in (0,1,2,4,5)]}),
        decoding=dict(temperature=.6,top_p=.9,top_k=50,repetition_penalty=1.2),normalization_floor=1e-6,
        floor_policy='Engineering/development reports absolute values and provisional ratios. Before confirmation freeze max(1e-6,min(0.001,development_positive_code_JSD_5th_percentile/10)).',
        minimum_meaningful_normalized_selective_excess=.10,confirmation_target_range=[60,80],confirmation_history_subset=20,
        primary_candidate=dict(layer=5,site='attn',phase=0),exploratory_candidates=[dict(layer=11,site='attn'),dict(layer=11,site='mlp'),dict(layer=27,site='mlp')],
        residual_controls=[5,11,26,27],primary_endpoint='Phase-conditional own-minus-competing recovery, divided by own baseline distance, native minus realized-norm-matched random',
        normalized_recovery='(D(recipient,donor)-D(patched,donor))/D(recipient,donor), undefined below frozen floor',
        absolute_selectivity='own absolute recovery minus mean competing-style absolute recovery; common own-target denominator for normalization',
        precision=dict(identity_max_abs=0,earlier_prediction_max_abs=0,last_residual_donor_logit_max_abs=1e-5,norm_match_relative=.01),
        controls='No hook, self-identity, alpha zero, negative-add engineering, reverse recipient/donor, matched random, layer27 full residual donor-logit positive control',
        expected_phase_mass_boundary=.99,total_stop_probability_boundary=.01,
        missingness='No replacement. Failed/short reference histories remain in requested denominators. Technical/infinite/cache/identity failures halt a bounded job.',
        m1_cohort_sha256=digest(source.read_bytes()),m1_protocol_sha256=digest((a.m1/'protocol.json').read_bytes()),
        m1_basis_sha256=digest((a.m1/'remote_export_final_v1/fit/basis.npz').read_bytes()),
        constraints='No decoder, WAVs, training, package changes, M1 modifications or Git publishing.')
    write_json(out/'protocol_development.json',protocol)
    plan=dict(name='development_A',rules=[dict(layer=5,site='attn',variants=['native','random','reverse'])]+
        [dict(layer=l,site='resid',variants=['native']) for l in [5,11,26,27]])
    write_json(out/'plan_A.json',plan)
    write_json(out/'plan_B.json',dict(name='development_B',rules=[dict(layer=l,site=s,variants=['native','random','reverse']) for l,s in [(11,'attn'),(11,'mlp'),(27,'mlp')]]))
    print(json.dumps(dict(development_texts=30,protocol=str(out/'protocol_development.json'),plan_A_cases_per_history=42,plan_B_cases_per_history=54)))


if __name__=='__main__':main()
