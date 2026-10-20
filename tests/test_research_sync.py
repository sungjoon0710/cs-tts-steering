import io
import sys
import zipfile
from scripts import sync_evaluation


def test_report_transfer_includes_figures_but_excludes_gpt_and_audio(tmp_path,monkeypatch):
    for name in ['summary.json','figures/plot.png','GPT_PROMPT.md','HANDOFF.md','GPT_packet/summary.json','clip.wav']:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'test')
    captured=[]
    monkeypatch.setattr(sync_evaluation,'ssh',lambda code,payload:captured.append(payload) or b'{}')
    monkeypatch.setattr(sys,'argv',['sync_evaluation','research-push','--source',str(tmp_path),'--include-reports'])
    sync_evaluation.main()
    with zipfile.ZipFile(io.BytesIO(captured[0])) as archive:
        assert set(archive.namelist())=={'summary.json','figures/plot.png','artifact_manifest.json'}
