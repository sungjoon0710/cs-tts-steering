import pickle
import numpy as np
import pandas as pd
import pytest
from src.evaluation.provenance import recover_sources,restricted_dataframe,paired_cluster_intervals


def test_source_association_recovers_only_the_exact_malformed_row():
    baseline=[{'line':0,'speaker':'Sahil','text':'hello there'}]
    frame=pd.DataFrame([{'Speaker':'Sahil hello there','text':None,'Audio_Filename':'clip.mp3','Episode_label':'E1-D2'}])
    args=(baseline,frame,['Sahil hello there'],['Audio_Filename','clip.mp3'],['Episode_label','E1-D2'])
    row=recover_sources(*args)[0]
    assert row['source_association']=='malformed_source_fields_recovered_from_exact_raw_line'
    assert row['source_speaker']=='Sahil' and row['source_dialogue']=='E1-D2'
    frame.loc[0,'Speaker']='Sahil some other words'
    with pytest.raises(ValueError,match='fields'):recover_sources(*args)


def test_shifted_audio_sidecar_is_rejected():
    frame=pd.DataFrame([{'Speaker':'Maya','text':'hi','Audio_Filename':'a.mp3','Episode_label':'E2-D3'}])
    with pytest.raises(ValueError,match='sidecar mismatch'):
        recover_sources([{'line':0,'speaker':'Maya','text':'hi'}],frame,['Maya: hi'],['Audio_Filename','b.mp3'],['Episode_label','E2-D3'])


def test_unreviewed_pickle_global_is_rejected(tmp_path):
    path=tmp_path/'not_a_source.p';path.write_bytes(pickle.dumps(complex(1,2)))
    with pytest.raises(pickle.UnpicklingError,match='Unapproved'):restricted_dataframe(path)


def test_cluster_resampling_preserves_pairs_and_missing_cells():
    result=paired_cluster_intervals([[3,np.nan],[3,2],[3,2],[3,2]],['a','a','b','c'],repeats=200)
    assert result[0]['mean_B_minus_A']==3 and result[0]['bootstrap_95ci']==[3,3]
    assert result[0]['source_clusters_sampled']==3
    assert result[1]['matched_recordings']==3 and result[1]['bootstrap_95ci']==[2,2]
    with pytest.raises(ValueError,match='two source clusters'):paired_cluster_intervals([1,2],['a','a'])
