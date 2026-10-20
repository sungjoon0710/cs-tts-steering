"""Small analysis helpers for learned diagnostics; no model loading here."""
import numpy as np

def speaker_consistency(vectors,seed=1234,repeats=1000):
    e=np.asarray(vectors,float)
    if e.ndim!=2 or len(e)<2 or not np.isfinite(e).all():raise ValueError('At least two finite embedding vectors required')
    norms=np.linalg.norm(e,axis=1)
    if np.any(norms<=0):raise ValueError('Zero embedding')
    e=e/norms[:,None];n=len(e);similarity=e@e.T
    pairs=similarity[np.triu_indices(n,1)]
    other=e.sum(axis=0)-e;other/=np.linalg.norm(other,axis=1)[:,None]
    loo=(e*other).sum(axis=1)
    # Resample utterances, never treat the n(n-1)/2 pairs as independent.
    off_diagonal=similarity-np.eye(n);rng=np.random.default_rng(seed);boot=[]
    for _ in range(repeats):
        w=np.bincount(rng.integers(0,n,n),minlength=n);den=w.sum()**2-w@w
        if den:boot.append(float(w@off_diagonal@w/den))
    return {'n_recordings':n,'embedding_dimension':e.shape[1],'pairwise_mean_cosine':float(pairs.mean()),
        'pairwise_q05_cosine':float(np.quantile(pairs,.05)),'loo_centroid_cosine':loo.tolist(),
        'pairwise_mean_bootstrap_95ci':np.quantile(boot,[.025,.975]).tolist(),
        'uncertainty':'Utterance bootstrap with same-original-index pairs excluded; unknown conversation clustering unresolved',
        'interpretation':'Within-voice embedding consistency; content/language/domain effects remain; not naturalness'}
