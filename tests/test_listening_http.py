"""HTTP and optional existing-browser smoke tests on synthetic data only."""
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.evaluation.common import write_json, write_jsonl
from src.evaluation.listening import ListeningStudy, handler


@pytest.fixture
def service(tmp_path):
    gs=[]
    for i in range(3):
        for c in ('fixture_a','fixture_b','fixture_c'):
            rel=c+'/'+str(i)+'.wav'
            g={'generation_id':str(i)+c,'utterance_id':str(i),'configuration_id':c,'relative_audio_path':rel}
            gs.append(g);p=tmp_path/'source/audio'/rel;p.parent.mkdir(parents=True,exist_ok=True)
            sf.write(p,.1*np.sin(2*np.pi*220*np.arange(2400)/24000),24000)
    write_json(tmp_path/'pilot.json',{'utterance_ids':['0','1','2']})
    write_jsonl(tmp_path/'tables/generations.jsonl',gs)
    write_jsonl(tmp_path/'tables/utterances.jsonl',[{'utterance_id':str(i),'text_raw':'Synthetic target '+str(i)} for i in range(3)])
    write_jsonl(tmp_path/'tables/metrics.jsonl',[{'generation_id':g['generation_id'],'audio':{'audio_sha256':'synthetic'}} for g in gs])
    study=ListeningStudy(tmp_path)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(study))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    yield 'http://127.0.0.1:'+str(server.server_port),study
    server.shutdown();server.server_close();thread.join(timeout=3)


def test_http_audio_and_blinding(service):
    url,study=service
    body=json.dumps({'listener_id':'fixture_http','profile':{'consent':True}}).encode()
    req=urllib.request.Request(url+'/api/enroll',data=body,headers={'Content-Type':'application/json'})
    assert json.load(urllib.request.urlopen(req))['saved']
    view=json.load(urllib.request.urlopen(url+'/api/trial?listener=fixture_http'))
    assert 'transcript' not in view and 'fixture_a' not in json.dumps(view)
    with urllib.request.urlopen(url+view['audio_A']) as response:
        assert response.headers['Content-Type']=='audio/wav'
        assert response.read(4)==b'RIFF'
    assert urllib.request.urlopen(url).status==200


def test_existing_browser_flow(service):
    playwright=pytest.importorskip('playwright.sync_api')
    edge=Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')
    if not edge.exists(): pytest.skip('No existing Edge browser; no download attempted')
    url,study=service
    with playwright.sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(edge),headless=True)
        page=browser.new_page(viewport={'width':1100,'height':1000})
        errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
        page.goto(url);page.fill('#listener','fixture_browser');page.check('#consent')
        page.select_option('#hi','Fluent');page.select_option('#en','Fluent');page.select_option('#frequency','Daily');page.click('#start')
        page.locator('#primary').wait_for(state='visible')
        assert not page.locator('#transcript').is_visible()
        assert page.locator('#lock').is_disabled()
        for aid in ('audioA','audioB'):
            page.locator('#'+aid).evaluate("e=>e.play()")
            page.wait_for_function("id=>document.getElementById(id).ended",arg=aid)
        page.select_option('#naturalness','a');page.click('#lock')
        page.locator('#diagnostics').wait_for(state='visible')
        assert page.locator('#transcript').is_visible()
        page.select_option('#pronunciation','both_acceptable');page.select_option('#smoothness','not_applicable')
        page.click('#finish');page.locator('#primary').wait_for(state='visible')
        assert len(study.trials('fixture_browser'))==3
        assert not errors
        screenshot=Path(__file__).resolve().parents[2]/'evaluation_runs/interface_preview.png'
        screenshot.parent.mkdir(parents=True,exist_ok=True);page.screenshot(path=str(screenshot),full_page=True)
        browser.close()
