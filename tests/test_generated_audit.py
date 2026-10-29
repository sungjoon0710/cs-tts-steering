import numpy as np
import pytest
from scripts.audit_generated_metrics import cluster_weights,index,ratio_difference,deleted_positions
from src.evaluation.text import edit_score


def test_corpus_error_weighting_is_not_mean_of_utterance_rates():
    # The long sentence contributes100 reference words, not one observation.
    result=ratio_difference([1,0],[0,20],[1,100],np.ones((4,2)))
    assert result['B_minus_A']==pytest.approx(19/101)
    assert result['bootstrap_95ci']==pytest.approx([19/101,19/101])
    assert np.mean([0,.2])-np.mean([1,0])<0  # Opposite macro direction is legitimate.
    with pytest.raises(ValueError):ratio_difference([1],[0],[0],np.ones((2,1)))


def test_cluster_weights_keep_source_records_together_and_preserve_pairing():
    weights=cluster_weights(['episode1','episode1','episode2'],repeats=100)
    assert np.array_equal(weights[:,0],weights[:,1])
    result=ratio_difference([1,2,3],[2,4,6],[1,2,3],weights)
    assert result['B_minus_A']==1 and result['bootstrap_95ci']==[1,1]
    with pytest.raises(ValueError):index([{'id':'x'},{'id':'x'}],'id','fixture')


def test_joint_deletions_use_reference_indices_not_hypothesis_positions():
    # The insertion shifts hypothesis indices; the omitted reference word is unchanged.
    a=edit_score('one two three four','one four')
    b=edit_score('one two three four','extra one two four')
    assert deleted_positions(a)=={1,2}
    assert deleted_positions(b)=={2}
    assert deleted_positions(a)&deleted_positions(b)=={2}
    assert deleted_positions(edit_score('one two','one new two'))==set()
