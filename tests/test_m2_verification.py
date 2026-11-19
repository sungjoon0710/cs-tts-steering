"""Independent probability-domain verifier against the executed log-domain method."""
import numpy as np
import pytest
import torch
from scripts.verify_m2 import jsd as probability_jsd
from src.causal_patch import jsd as log_jsd


def test_independent_jsd_matches_log_domain_and_handles_zero_support():
    generator=np.random.default_rng(21)
    a=torch.from_numpy(generator.normal(size=4096)).float().log_softmax(-1)
    b=torch.from_numpy(generator.normal(size=4096)).float().log_softmax(-1)
    assert probability_jsd(a.double().exp().numpy(),b.double().exp().numpy())==pytest.approx(float(log_jsd(a,b)),abs=2e-7)
    assert probability_jsd(np.array([0.,1.]),np.array([1.,0.]))==pytest.approx(np.log(2))
    assert probability_jsd(np.array([0.,1.]),np.array([0.,1.]))==0
