"""Prevent pseudo-replication and hidden partial-cell averaging in causal analysis."""
import numpy as np
import pandas as pd
import pytest
from scripts.analyze_m2 import per_text,matched,ratios
from scripts.run_m2 import case_list


def cells():
    return pd.DataFrame([dict(id='TEST',study_split='development',category='TEST',history=0,layer=5,site='attn',phase=0,
        style=style,frame=frame,code_normalized_selectivity_excess=(frame==18)*.2,
        boundary_flag_native=False,boundary_flag_random=False) for style in ['anger','happy','sad'] for frame in [4,18]])


def test_one_text_not_six_independent_patches_and_no_partial_denominator():
    d=cells();out=per_text(d)
    assert len(out)==1 and out.n_cells.iloc[0]==6 and out.code_normalized_selectivity_excess.iloc[0]==pytest.approx(.1)
    assert np.isnan(per_text(d.iloc[:-1]).code_normalized_selectivity_excess.iloc[0])
    d.loc[0,'code_normalized_selectivity_excess']=np.nan
    assert np.isnan(per_text(d).code_normalized_selectivity_excess.iloc[0])


def test_duplicate_style_frame_cannot_fake_complete_design():
    d=cells();d.iloc[0]=d.iloc[1]
    assert np.isnan(per_text(d).code_normalized_selectivity_excess.iloc[0])


def test_bounded_case_semantics_and_reverse_pair():
    plan={'rules':[{'layer':5,'site':'attn','variants':['native','random','reverse']}]}
    c=case_list(plan);assert len(c)==18 and len({r['case_id'] for r in c})==18
    assert all(r['j'] in [28,126] and r['phase']==0 for r in c)
    assert all(r['donor']=='news' for r in c if r['comparison']=='reverse')
    with pytest.raises(ValueError,match='generation positions'):case_list({'rules':[{'layer':5,'site':'attn','frames':[0],'variants':['native']}]})
