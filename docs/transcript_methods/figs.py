import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, sys
import os; os.chdir(os.path.dirname(os.path.abspath(__file__))); d=pd.read_csv('transcript_audit_per_utterance.csv'); ok=d.audio_match_flag=='ok'
BLUE,ORANGE,AQUA,GRAY,INK,MUT='#2a78d6','#eb6834','#1baf7a','#c9c8c2','#1f1f1e','#6b6a64'
plt.rcParams.update({'font.size':10,'axes.edgecolor':MUT,'axes.labelcolor':INK,'xtick.color':MUT,'ytick.color':MUT,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
def lab(ax,bars,fmt):
    for b in bars:
        w=b.get_width() if b.get_height()<1 else None
        ax.annotate(fmt(b.get_height()),(b.get_x()+b.get_width()/2,b.get_height()),xytext=(0,3),textcoords='offset points',ha='center',fontsize=9,color=INK)
# Fig 1
names=['O\noriginal\nromanized','A\ncurrent\n(hinlang)','B\nLLM','C\nIndicXlit']
fig,ax=plt.subplots(figsize=(6.4,3.3)); import numpy as np; x=np.arange(4); w=.36
a=[d[f'cer_{v}_vs_D'].median() for v in 'OABC']; b=[d.loc[ok,f'cer_{v}_vs_D'].median() for v in 'OABC']
b1=ax.bar(x-w/2-.01,a,w,color=BLUE,label='All 100 clips'); b2=ax.bar(x+w/2+.01,b,w,color=ORANGE,label='71 clips that match their audio')
lab(ax,b1,lambda v:f'{v:.2f}'); lab(ax,b2,lambda v:f'{v:.2f}')
ax.set_xticks(x,names); ax.set_ylabel('Median normalised CER vs D\n(lower = closer)'); ax.grid(axis='y',color='#ebeae6'); ax.set_axisbelow(True); ax.set_ylim(0,.3); ax.legend(frameon=False,fontsize=9,ncol=2,loc='upper center')
fig.tight_layout(); fig.savefig('fig1_closeness.pdf')
# Fig 2
pairs=[('B','A','B (LLM) vs A (current)'),('B','C','B (LLM) vs C (IndicXlit)'),('C','A','C (IndicXlit) vs A (current)')]
fig,ax=plt.subplots(figsize=(6.4,2.4))
for i,(p,q,t) in enumerate(pairs[::-1]):
    P=d[f'cer_{p}_vs_D'];Q=d[f'cer_{q}_vs_D']; win,tie,loss=int((P<Q).sum()),int((P==Q).sum()),int((P>Q).sum())
    l=0
    for n,c,txt in [(win,BLUE,f'{p} closer: {win}'),(tie,GRAY,f'tie: {tie}'),(loss,ORANGE,f'{q} closer: {loss}')]:
        ax.barh(i,n-0.6,left=l+0.3,color=c,height=.6); 
        if n>=6: ax.text(l+n/2,i,txt if n>=15 else str(n),ha='center',va='center',fontsize=8.5,color='white' if c!=GRAY else INK)
        l+=n
ax.set_yticks(range(3),[t for _,_,t in pairs[::-1]]); ax.set_xlim(0,100); ax.set_xlabel('Number of clips (of 100) where each version is closer to D')
ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
fig.tight_layout(); fig.savefig('fig2_headtohead.pdf')
# Fig 3
cats=[('vowel','Vowel marks\n(length, final a,\nnasal)'),('consonant','Consonants\n(e.g. retroflex\nr as d)'),('latin_left','Hindi word\nleft in Latin'),('mein_as_main','"mein" (in) written\nas "main" (I)'),('en_to_dev','English word put\ninto Devanagari')]
vals=[int(d[c].sum()) for c,_ in cats]; aff=[int((d[c]>0).sum()) for c,_ in cats]
plt.rcParams['font.family']=['Helvetica','Kohinoor Devanagari']
fig,ax=plt.subplots(figsize=(7.2,3.2)); bb=ax.bar(range(5),vals,.6,color=BLUE)
for b,v,u in zip(bb,vals,aff): ax.annotate(f'{v} words\n({u} clips)',(b.get_x()+b.get_width()/2,v),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8.5,color=INK)
ax.set_xticks(range(5),[t for _,t in cats],fontsize=8.5); ax.set_ylabel('Misspelled Hindi words'); ax.set_ylim(0,max(vals)*1.3); ax.grid(axis='y',color='#ebeae6'); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig('fig3_errors.pdf')
# Fig 4
plt.rcParams['font.family']=['Helvetica']
f=d[d.audio_match_flag!='ok'].copy(); f['r']=f.audio_match_reason.replace({'missing-words':'wrong-word'})
types=[('segment-boundary','Clip boundary error',BLUE),('wrong-word','Word errors in transcript',ORANGE),('wrong-content','Audio says something else',AQUA),('garbled','Garbled transcript','#eda100'),('unverifiable','Unclear (needs listening)','#e87ba4')]
rows=[('ok','No mismatch'),('moderate','Small'),('severe','Large')]
seen=set()
fig,ax=plt.subplots(figsize=(6.8,3.0))
for yi,(flag,name) in enumerate(rows):
    left=0; x=f[f.audio_match_flag==flag]
    if flag=='ok':
        n=int((d.audio_match_flag=='ok').sum()); ax.barh(yi,n-0.3,left=0.15,color='#b4b2a9',height=.55,label='No mismatch')
        ax.text(n/2,yi,f'{n} clips ({n}%)',ha='center',va='center',fontsize=9,color=INK); continue
    for k,lab_,c in types:
        n=int((x.r==k).sum())
        if n==0: continue
        ax.barh(yi,n-0.3,left=left+0.15,color=c,height=.55,label=lab_ if lab_ not in seen else None); seen.add(lab_)
        ax.text(left+n/2,yi,str(n),ha='center',va='center',fontsize=8,color='white' if c in (BLUE,ORANGE) else INK)
        left+=n
    ax.text(left+0.6,yi,f'{left} clips ({left}%)',va='center',fontsize=9,color=INK)
ax.set_yticks(range(3),['No mismatch','Small mismatch','Large mismatch']); ax.invert_yaxis(); ax.set_xlim(0,80); ax.set_xticks(range(0,81,10))
ax.set_xlabel('Clips (of 100)'); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0)
h,l=ax.get_legend_handles_labels(); order=[l.index(t) for t in ['No mismatch']+[t for _,t,_ in types] if t in l]
ax.legend([h[i] for i in order],[l[i] for i in order],frameon=False,fontsize=9,loc='upper center',bbox_to_anchor=(0.45,-0.35),ncol=3,handlelength=1.2,columnspacing=1.2)
fig.tight_layout(); fig.savefig('fig4_match.pdf')
