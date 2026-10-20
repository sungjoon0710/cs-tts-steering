import numpy as np
import pytest
from src.evaluation.learned import speaker_consistency

def test_speaker_consistency_matches_hand_calculated_pairs():
    result=speaker_consistency([[1,0],[0,1],[1,1]],repeats=100)
    assert result['pairwise_mean_cosine']==pytest.approx(np.sqrt(2)/3)
    assert len(result['loo_centroid_cosine'])==3
    assert speaker_consistency([[1,0],[1,0]],repeats=50)['pairwise_mean_cosine']==pytest.approx(1)
    with pytest.raises(ValueError):speaker_consistency([[0,0],[1,0]])
