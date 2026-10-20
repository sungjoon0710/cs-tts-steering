import numpy as np
import pytest
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
