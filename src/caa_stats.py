"""Small statistics toolkit for the CAA pilot (numpy + torch only; scipy/sklearn are not in the env)."""
from __future__ import annotations

import math

import numpy as np
import torch


def auc(scores_pos, scores_neg) -> float:
    """Mann-Whitney AUC = P(pos > neg) + 0.5 P(tie)."""
    p, n = np.asarray(scores_pos, float), np.asarray(scores_neg, float)
    if not len(p) or not len(n):
        return float("nan")
    allv = np.concatenate([p, n])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):  # average ranks for ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    rp = ranks[: len(p)].sum()
    return float((rp - len(p) * (len(p) + 1) / 2) / (len(p) * len(n)))


def fit_logreg(X, y, l2=1.0, iters=50, tol=1e-8):
    """L2 logistic regression (bias unpenalized) by damped Newton/IRLS in float64. Returns (w, b)."""
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    n, d = X.shape
    A = np.hstack([X, np.ones((n, 1))])
    beta = np.zeros(d + 1)
    reg = np.full(d + 1, l2)
    reg[-1] = 0.0

    def loss(bt):
        z = A @ bt
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(reg * bt * bt)
    cur = loss(beta)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(A @ beta)))
        g = A.T @ (p - y) + reg * beta
        H = (A * (p * (1 - p))[:, None]).T @ A + np.diag(reg + 1e-9)
        step = np.linalg.solve(H, g)
        t = 1.0
        while t > 1e-4:
            new = loss(beta - t * step)
            if new <= cur:
                break
            t /= 2
        beta = beta - t * step
        if cur - new < tol * max(1.0, abs(cur)):
            cur = new
            break
        cur = new
    return beta[:-1], float(beta[-1])


class Standardize:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def pca_fit(X, k):
    mu = X.mean(0)
    _, _, vt = np.linalg.svd(X - mu, full_matrices=False)
    return mu, vt[:k]


def probe_cv(X, y, groups, folds, pca=64, l2=1.0):
    """Utterance-grouped CV accuracy and AUC of a standardized (+PCA) L2 logistic probe.

    folds: list of arrays of group ids held out in each fold."""
    acc, scores, labels = [], [], []
    for held in folds:
        te = np.isin(groups, held)
        tr = ~te
        st = Standardize().fit(X[tr])
        Xtr, Xte = st(X[tr]), st(X[te])
        if pca and pca < Xtr.shape[1]:
            mu, comp = pca_fit(Xtr, min(pca, Xtr.shape[0] - 1))
            Xtr, Xte = (Xtr - mu) @ comp.T, (Xte - mu) @ comp.T
            s2 = Standardize().fit(Xtr)
            Xtr, Xte = s2(Xtr), s2(Xte)
        w, b = fit_logreg(Xtr, y[tr], l2=l2)
        z = Xte @ w + b
        acc.append(np.mean((z > 0) == (y[te] == 1)))
        scores.append(z)
        labels.append(y[te])
    s, l = np.concatenate(scores), np.concatenate(labels)
    return float(np.mean(acc)), auc(s[l == 1], s[l == 0])


def projection_cv(X, y, groups, folds):
    """Held-out projection onto v = mean(nat) - mean(syn) estimated on the training folds.

    Returns AUC (pooled), paired win rate (nat proj > syn proj within utterance) and Cohen's d."""
    proj = np.zeros(len(y))
    for held in folds:
        te = np.isin(groups, held)
        tr = ~te
        v = X[tr & (y == 1)].mean(0) - X[tr & (y == 0)].mean(0)
        v = v / (np.linalg.norm(v) + 1e-12)
        proj[te] = X[te] @ v
    pn, ps = proj[y == 1], proj[y == 0]
    d = (pn.mean() - ps.mean()) / math.sqrt(0.5 * (pn.var(ddof=1) + ps.var(ddof=1)) + 1e-12)
    pairs = {}
    for g, lab, p in zip(groups, y, proj):
        pairs.setdefault(g, {})[int(lab)] = p
    win = np.mean([v[1] > v[0] for v in pairs.values() if 0 in v and 1 in v])
    return auc(pn, ps), float(win), float(d)


def pair_permute(y, groups, rng):
    """Swap the natural/synthetic labels within each utterance pair with probability 1/2."""
    y = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        if rng.random() < 0.5:
            y[idx] = 1 - y[idx]
    return y


def make_folds(group_ids, k, seed):
    g = np.array(sorted(set(group_ids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(g)
    return [g[i::k] for i in range(k)]


def bootstrap_ci(values_by_unit, stat=np.mean, n=2000, seed=0, alpha=0.05):
    """values_by_unit: 1-D array of per-utterance values (seeds already averaged)."""
    x = np.asarray(values_by_unit, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return (float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2)))


def norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2))


def wilcoxon(d):
    """Paired Wilcoxon signed-rank test (zeros dropped). Exact sign-flip p for n<=20, else normal approx."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    n = len(d)
    if n == 0:
        return dict(n=0, W=float("nan"), p=float("nan"))
    r = np.empty(n)
    a = np.abs(d)
    order = a.argsort()
    sv = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = r[d > 0].sum()
    if n <= 20:
        # exact distribution of W+ under random signs (enumerate via DP on doubled ranks)
        rr = (2 * r).astype(int)
        tot = rr.sum()
        dist = np.zeros(tot + 1)
        dist[0] = 1
        for x in rr:  # x >= 2; new[t] = old[t] + old[t-x]
            dist[x:] = dist[x:] + dist[:-x].copy()
        dist /= dist.sum()
        obs = int(round(2 * w_plus))
        center = tot / 2
        dev = abs(obs - center)
        vals = np.arange(tot + 1)
        p = float(dist[np.abs(vals - center) >= dev - 1e-9].sum())
    else:
        mu = n * (n + 1) / 4
        ties = np.unique(r, return_counts=True)[1]
        var = n * (n + 1) * (2 * n + 1) / 24 - (ties ** 3 - ties).sum() / 48
        z = (abs(w_plus - mu) - 0.5) / math.sqrt(var)
        p = 2 * norm_sf(z)
    return dict(n=n, W=float(w_plus), p=min(1.0, p))


def chi2_sf(x, k):
    return float(torch.special.gammaincc(torch.tensor(k / 2, dtype=torch.float64),
                                         torch.tensor(x / 2, dtype=torch.float64)))


def friedman(M):
    """M: (n_blocks, k_treatments) with no NaN. Returns chi2 statistic and p (ties averaged)."""
    M = np.asarray(M, float)
    n, k = M.shape
    R = np.zeros_like(M)
    for i in range(n):
        order = M[i].argsort()
        sv = M[i][order]
        j = 0
        while j < k:
            m = j
            while m + 1 < k and sv[m + 1] == sv[j]:
                m += 1
            R[i, order[j:m + 1]] = (j + m) / 2 + 1
            j = m + 1
    Rj = R.sum(0)
    stat = 12 / (n * k * (k + 1)) * (Rj ** 2).sum() - 3 * n * (k + 1)
    return dict(n=n, k=k, chi2=float(stat), p=chi2_sf(stat, k - 1))


def dtw_path(A, B):
    """DTW on frame features A (n,d), B (m,d), Euclidean cost. Returns list of (i,j)."""
    n, m = len(A), len(B)
    C = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    for i in range(1, n + 1):
        row = D[i]
        prev = D[i - 1]
        ci = C[i - 1]
        best = np.minimum(prev[1:], prev[:-1])  # from (i-1,j) and (i-1,j-1)
        for j in range(1, m + 1):
            v = best[j - 1]
            if row[j - 1] < v:
                v = row[j - 1]
            row[j] = ci[j - 1] + v
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
    return path[::-1]


def ccc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 5:
        return float("nan")
    mx, my = x.mean(), y.mean()
    return float(2 * np.mean((x - mx) * (y - my)) / (x.var() + y.var() + (mx - my) ** 2 + 1e-12))
