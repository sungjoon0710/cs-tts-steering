"""Compact M2 descriptive diagnostics and figures from saved numerical results only."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.analyze_m2 import ci,BRANCHES


def summarize(frame,keys,metric,expected):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,);values=[];cats=[]
        for uid,text in g.groupby('id'):
            if len(text)!=expected or not text[metric].notna().all():continue
            values.append(float(text[metric].mean()));cats.append(text.category.iloc[0])
        rows.append(dict(zip(keys,key),metric=metric,**ci(values,cats)))
    return pd.DataFrame(rows)


def errorbars(ax,frame,labels,colors):
    for i,(_,r) in enumerate(frame.iterrows()):
        if not np.isfinite(r['mean']):continue
        ax.errorbar(i,r['mean'],yerr=[[max(0,r['mean']-r.ci_lower)],[max(0,r.ci_upper-r['mean'])]],fmt='o',color=colors[i],capsize=4,markersize=7)
    ax.set_xticks(range(len(labels)),labels);ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['analysis','export','gate','output']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--split',choices=['development','confirmation'],default='confirmation')
    p.add_argument('--development',type=Path)
    a=p.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=False);gate=json.loads(a.gate.read_text())
    data=pd.read_csv(a.analysis/'per_intervention.csv');pairs=pd.read_csv(a.analysis/'native_random_matched.csv');pt=pd.read_csv(a.analysis/'per_utterance_phase_effects.csv')
    d=data[(data.study_split==a.split)&(data.history==0)];m=pairs[(pairs.study_split==a.split)&(pairs.history==0)]
    branch=m[(m.phase==0)&m[['layer','site']].apply(tuple,axis=1).isin(BRANCHES)]
    tables={}
    tables['branch_by_donor']=summarize(branch,['layer','site','style'],'code_normalized_selectivity_excess',2)
    tables['branch_by_position']=summarize(branch,['layer','site','frame'],'code_normalized_selectivity_excess',3)
    tables['branch_phase0']=summarize(branch,['layer','site'],'code_normalized_selectivity_excess',6)
    selected=gate['selected_branch'];chosen=m[(m.layer==selected['layer'])&(m.site==selected['site'])]
    tables['selected_phase']=summarize(chosen,['phase'],'code_normalized_selectivity_excess',6)
    tables['selected_phase_absolute']=summarize(chosen,['phase'],'code_selective_absolute_recovery_excess',6)
    baseline_rows=d[(d.layer==selected['layer'])&(d.site==selected['site'])&(d.comparison=='native')].copy()
    baseline_rows['baseline_distance']=[r['code_baseline_to_'+r['donor']] for r in baseline_rows.to_dict('records')]
    tables['selected_phase_baseline']=summarize(baseline_rows,['phase'],'baseline_distance',6)
    native=d[(d.site=='resid')&(d.comparison=='native')&(d.phase==0)]
    tables['residual_depth']=summarize(native,['layer'],'code_normalized_recovery',6)
    # Boundary sensitivity is descriptive and uses only fully retained matched cells.
    tables['ordinary_code_sensitivity']=summarize(branch[~(branch.boundary_flag_native|branch.boundary_flag_random)],['layer','site'],'code_normalized_selectivity_excess',6)
    tables['ordinary_selected_phase']=summarize(chosen[~(chosen.boundary_flag_native|chosen.boundary_flag_random)],['phase'],'code_normalized_selectivity_excess',6)
    tables['full_vocabulary_branch']=summarize(branch,['layer','site'],'full_normalized_selectivity_excess',6)
    tables['absolute_branch']=summarize(branch,['layer','site'],'code_selective_absolute_recovery_excess',6)
    projections=d[(d.comparison.isin(['shared','specific','native','random']))&(d.frame==4)&(d.phase==0)&
        (((d.layer==5)&(d.site=='attn'))|((d.layer==selected['layer'])&(d.site==selected['site'])))]
    tables['projection_diagnostics']=summarize(projections,['layer','site','comparison'],'code_normalized_selectivity',3)
    tables['projection_own_recovery']=summarize(projections,['layer','site','comparison'],'code_normalized_recovery',3)
    tables['reverse_diagnostics']=summarize(d[(d.comparison=='reverse')&(d.phase==0)],['layer','site'],'code_normalized_recovery',6)
    all_forward=d[(d.recipient=='news')&(d.comparison.isin(['native','random']))]
    for metric in ['valid_phase_mass','total_stop_probability','delta_target_logp']:
        tables[metric]=summarize(all_forward,['layer','site','comparison','phase'],metric,6)
    h=pt[(pt.study_split==a.split)&(pt.phase==0)]
    h0=h[h.history==0];h1=h[h.history==1]
    hist=h0.merge(h1,on=['id','layer','site','phase','category','study_split'],suffixes=('_primary','_secondary'),validate='one_to_one')
    metric='code_normalized_selectivity_excess';hist['secondary_minus_primary']=hist[metric+'_secondary']-hist[metric+'_primary']
    tables['history_paired_per_text']=hist
    rows=[]
    for (layer,site),g in hist.groupby(['layer','site']):
        rows.append(dict(layer=layer,site=site,**ci(g.secondary_minus_primary,g.category),
            correlation=float(g[metric+'_primary'].corr(g[metric+'_secondary'])) if len(g)>2 else None))
    tables['history_paired_summary']=pd.DataFrame(rows)
    # Missingness, including all requested failures, stays outside complete-case statistics.
    requested=a.export/'verification/requested_coverage.json'
    coverage=json.loads(requested.read_text()) if requested.exists() else json.loads((a.analysis/'analysis_manifest.json').read_text())['lineage']
    tables['requested_coverage']=pd.DataFrame(coverage)
    exclusions=[]
    for path in (a.export/'runs').glob('*/*/h*/exclusions.json'):exclusions+=json.loads(path.read_text())
    tables['projection_exclusions']=pd.DataFrame(exclusions)
    boundary=[]
    for key,g in d.groupby(['layer','site','comparison','phase']):
        boundary.append(dict(zip(['layer','site','comparison','phase'],key),cells=len(g),texts=g.id.nunique(),baseline_boundary_cells=int(g.boundary_flag.sum()),
            below_floor_cells=int(g.code_normalized_recovery.isna().sum()),patched_low_phase_mass_cells=int((g.valid_phase_mass<.99).sum()),
            patched_high_stop_cells=int((g.total_stop_probability>.01).sum())))
    tables['boundary_and_floor_diagnostics']=pd.DataFrame(boundary)
    propagation=[]
    for path in (a.export/'runs').glob('*/*/h0/results.json'):
        for r in json.loads(path.read_text()):
            if r['study_split']!=a.split:continue
            for pr in r['propagation']:propagation.append({k:r[k] for k in ['id','category','layer','site','comparison','style','phase','frame']}|{'downstream_layer':pr['layer'],**{k:v for k,v in pr.items() if k!='layer'}})
    tables['downstream_per_intervention']=pd.DataFrame(propagation)
    propagated=tables['downstream_per_intervention']
    propagated=propagated[(propagated.phase==0)&propagated.comparison.isin(['native','random','reverse'])]
    for name in ['target_alignment','relative_change']:
        tables['downstream_'+name]=summarize(propagated,['layer','site','comparison','downstream_layer'],name,6)
    for name,table in tables.items():table.to_csv(out/(name+'.csv'),index=False)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    labels=['L5 attention','L11 attention','L11 MLP','L27 MLP'];colors=['#1f77b4','#9467bd','#ff7f0e','#b22222'];figures=[]
    def save(fig,name,caption):
        requested_n=gate['confirmation_n'] if a.split=='confirmation' else 30
        fig.suptitle(f'M2 {a.split}: {d.id.nunique()} usable / {requested_n} requested texts; token-distribution effects',fontsize=12)
        fig.tight_layout(rect=[0,.07,1,.91]);fig.text(.02,.012,caption,fontsize=9,va='bottom');fig.savefig(out/(name+'.png'),dpi=300);figures.append((fig,name))
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    b=tables['branch_phase0'].set_index(['layer','site']).reindex(BRANCHES).reset_index()
    errorbars(axes[0],b,labels,colors);axes[0].set_ylabel('Normalized selective recovery\n(native − matched random)');axes[0].set_title('Branch interventions: predicted phase 0')
    if a.development:
        dev=pd.read_csv(a.development/'development_branch_selection.csv').set_index(['layer','site']).reindex(BRANCHES)
        axes[0].errorbar(np.arange(len(BRANCHES))+.12,dev['mean'],yerr=[dev['mean']-dev.ci_lower,dev.ci_upper-dev['mean']],fmt='x',color='#777777',capsize=2,label='Development (n=30)')
        axes[0].plot([],[],marker='o',linestyle='',color='black',label='Confirmation');axes[0].legend(fontsize=8)
    r=tables['residual_depth'];errorbars(axes[1],r,[f'L{int(l)}' for l in r.layer],['#555555']*len(r));axes[1].set_ylabel('Normalized own-target recovery');axes[1].set_title('Matched-scope full residual swaps')
    save(fig,'01_causal_localization','Utterance-level means and 95% bootstrap CIs. Branch and full-state scopes differ; L27 residual is an engineering positive control.')
    fig,axes=plt.subplots(1,3,figsize=(15,4.9));phase=tables['selected_phase']
    errorbars(axes[0],phase,[str(int(k)) for k in phase.phase],['#b22222']*len(phase));axes[0].set_xlabel('Predicted SNAC phase');axes[0].set_ylabel('Native − random selective recovery');axes[0].set_title('Selected L27 MLP: phase specificity')
    absolute=tables['selected_phase_absolute'];baseline=tables['selected_phase_baseline']
    errorbars(axes[1],absolute,[str(int(k)) for k in absolute.phase],['#b22222']*len(absolute))
    axes[1].plot(range(len(baseline)),baseline['mean'],color='#777777',marker='s',ms=4,label='Original donor–News JSD')
    axes[1].plot([],[],color='#b22222',marker='o',label='Selective excess');axes[1].set_xlabel('Predicted SNAC phase');axes[1].set_ylabel('Conditional JSD / recovery (nats)');axes[1].set_title('Absolute effect and original distance');axes[1].legend(fontsize=8)
    pos=tables['branch_by_position'];x=np.arange(4)
    for offset,frame,c in [(-.12,4,'#20639b'),(.12,18,'#a52a2a')]:
        v=pos[pos.frame==frame].set_index(['layer','site']).reindex(BRANCHES)
        axes[2].errorbar(x+offset,v['mean'],yerr=[v['mean']-v.ci_lower,v.ci_upper-v['mean']],fmt='o',capsize=3,color=c,label=f'Frame {frame}')
    axes[2].set_xticks(x,labels,rotation=15);axes[2].axhline(0,color='black',lw=.8);axes[2].legend();axes[2].grid(axis='y',alpha=.2);axes[2].set_title('Phase-0 early versus later positions')
    save(fig,'02_phase_and_time','Fixed frames 4/18; phases 0 coarse, 1/4 middle, 2/5 fine. Larger normalized recovery need not mean a larger absolute effect. No acoustic function is established.')
    fig,ax=plt.subplots(figsize=(10,4.8));donors=tables['branch_by_donor']
    for offset,style,c in [(-.20,'anger','#b22222'),(0,'happy','#d79c13'),(.20,'sad','#4169e1')]:
        v=donors[donors['style']==style].set_index(['layer','site']).reindex(BRANCHES)
        ax.errorbar(x+offset,v['mean'],yerr=[v['mean']-v.ci_lower,v.ci_upper-v['mean']],fmt='o',capsize=3,color=c,label=style)
    ax.set_xticks(x,labels);ax.set_ylabel('Donor-selective recovery above random');ax.axhline(0,color='black',lw=.8);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_title('Does recovery prefer its own donor distribution?')
    save(fig,'03_donor_selectivity','Descriptive per-donor means, paired across the two positions. Styles are instruction labels; semantic emotion and audible expression remain untested.')
    if len(hist):
        fig,axes=plt.subplots(1,2,figsize=(11,4.8))
        for ax,(l,s) in zip(axes,[(5,'attn'),(selected['layer'],selected['site'])]):
            g=hist[(hist.layer==l)&(hist.site==s)];ax.scatter(g[metric+'_primary'],g[metric+'_secondary'],color='#4169e1',s=26)
            combined=np.r_[g[metric+'_primary'],g[metric+'_secondary']];combined=combined[np.isfinite(combined)]
            if len(combined):lo,hi=min(combined.min(),0),max(combined.max(),.1);ax.plot([lo,hi],[lo,hi],color='gray',ls='--',lw=1)
            ax.set_title(f'L{l} {s}: n={len(g)} paired texts');ax.set_xlabel('Primary default-code history');ax.set_ylabel('Second default-code history');ax.grid(alpha=.2)
        save(fig,'04_history_transfer','Each point is one text, averaging three donors and two positions. Histories are paired within text, not independent samples.')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for ax,(name,label) in zip(axes,[('projection_own_recovery','Normalized own-target recovery'),('projection_diagnostics','Normalized selective recovery')]):
        proj=tables[name];v=proj[(proj.layer==selected['layer'])&(proj.site==selected['site'])].set_index('comparison').reindex(['native','random','shared','specific']).reset_index()
        errorbars(ax,v,['Native','Random','Shared','Specific'],['#222222','#888888','#008b8b','#b22222']);ax.set_ylabel(label);ax.set_title(f'L{selected["layer"]} {selected["site"]}, frame 4 / phase 0')
    save(fig,'05_native_and_projected_directions','Same-position, norm-matched tests. Positive relative selectivity can coexist with negative donor recovery. Projections remain off-manifold diagnostics.')
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(out/'M2_figures.pdf') as pdf:
        for fig,_ in figures:pdf.savefig(fig)
    for fig,_ in figures:plt.close(fig)
    manifest={'source_analysis':str(a.analysis),'gate':str(a.gate),'study_split':a.split,'figures':[name+'.png' for _,name in figures],
        'statistics':'Utterance-level category-stratified bootstrap 2000; figure CIs descriptive, not additional confirmatory tests. Primary Holm family is separate.',
        'tables':list(tables),'causal_audio_generated':False}
    (out/'report_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
