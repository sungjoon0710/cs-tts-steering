import numpy as np
import pytest
import os
from pathlib import Path
from scripts.lab_rate_evaluate import detector_body,measure


def test_unknown_lab_source_is_not_executed():
    with pytest.raises(ValueError,match='changed'):detector_body(b'unreviewed source')


def test_no_nuclei_is_not_reported_as_valid_zero_articulation(monkeypatch):
    variables={'voicedcount':0,'speakingtot':.2,'originaldur':1.,'speakingrate':0.,
        'articulationrate':0.,'peakcount':1,'validpeakcount':0}
    monkeypatch.setattr('parselmouth.praat.run',lambda *a,**k:([],'',variables))
    result=measure(np.zeros(24000),24000,'reviewed body')
    assert result['status']=='no_valid_detected_nuclei'
    assert result['nuclei_per_s'] is None
    assert result['nuclei_per_detected_sounding_s'] is None


@pytest.fixture
def pinned_source():
    path=os.environ.get('SPEECHLAB_RATE_SOURCE')
    if not path:pytest.skip('Set SPEECHLAB_RATE_SOURCE to the reviewed lab method; no network fetch')
    return Path(path).read_bytes()


@pytest.mark.parametrize('count',[1,10])
def test_terminal_candidate_is_counted_on_controlled_voiced_pulses(pinned_source,count):
    sr=24000;t=np.arange(sr*3)/sr
    centers=[1.5] if count==1 else np.linspace(.3,2.7,count)
    envelope=sum(np.exp(-.5*((t-c)/.045)**2) for c in centers)
    y=.2*envelope*np.sin(2*np.pi*180*t)
    old=measure(y,sr,detector_body(pinned_source))
    fixed=measure(y,sr,detector_body(pinned_source,'terminal_v2'),'terminal_v2')
    assert old['detected_nuclei']==count-1
    assert fixed['detected_nuclei']==count
    assert fixed['nuclei_per_s']==pytest.approx(count/3)


@pytest.mark.parametrize('silence',[True,False])
def test_terminal_policy_does_not_unconditionally_add_one(pinned_source,silence):
    sr=24000;t=np.arange(sr*3)/sr
    y=np.zeros(len(t)) if silence else .1*np.exp(-.5*((t-1.5)/.045)**2)*np.random.default_rng(42).normal(0,.3,len(t))
    fixed=measure(y,sr,detector_body(pinned_source,'terminal_v2'),'terminal_v2')
    assert fixed['detected_nuclei']==0
    assert fixed['nuclei_per_s'] is None
