"""CPU-only catalog and read-only adapters for saved model computations.

The catalog is separate from immutable experiment SQLite/NPZ files. It indexes
identities and capture availability, not inferred vectors or model weights.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np

from scripts.view_generation_trace import verify_trace
from src.representation_geometry import cosine
from src.caa_stats import pca_fit
from src.snac_encode import codes_to_tokens


def identity(kind, obj):
    return kind + '_' + hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:24]


def readjson(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def csvrows(p):
    with Path(p).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def prediction_rows(path):
    """Keep legacy raw columns while correcting their verified EOS encoding.

    Original generation_trace.py v1 wrote lp[eos] to *_eos_probability.
    Never mutate CSVs or infer a distribution from top-five predictions.
    """
    rows = [{k: float(v) for k, v in r.items()} for r in csvrows(path)]
    values = [r[k] for r in rows for k in ('reference_eos_probability', 'changed_eos_probability')]
    if not all(np.isfinite(v) for v in values):
        raise ValueError('Nonfinite EOS statistics')
    legacy_log = all(v <= 0 for v in values) and any(v < 0 for v in values)
    for r in rows:
        for prefix in ('reference', 'changed'):
            key = prefix + '_eos_probability'
            r[key + '_raw'] = r[key]
            if legacy_log:
                r[prefix + '_eos_logp'] = r[key]
                r[key] = float(np.exp(r[key]))
            if not 0 <= r[key] <= 1:
                raise ValueError('Invalid EOS probability encoding')
        r['eos_encoding'] = 'legacy log p; probability derived as exp(log p)' if legacy_log else 'native probability'
    return rows


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def token_history_codes(record):
    """Validate a saved SNAC continuation without decoding or loading a model."""
    ids=record.get('generated_token_ids')
    if not isinstance(ids,list) or len(ids)<8 or ids[-1] not in (128258,128001):
        raise ValueError('Token-only history needs complete audio frames and normal EOS')
    codes=ids[:-1]
    if len(codes)%7 or any(not isinstance(v,int) or isinstance(v,bool) or
        not 128266+(j%7)*4096<=v<128266+(j%7+1)*4096 for j,v in enumerate(codes)):
        raise ValueError('Invalid or incomplete seven-phase token-only history')
    return codes


def bf16_round(x):
    """IEEE round-to-nearest-even conversion, returned as exact FP32 BF16 values."""
    x = np.asarray(x, dtype=np.float32).copy()
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite residual arithmetic')
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)


def metrics(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Incompatible/nonfinite vectors')
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    c = float(cosine(a, b)) if na and nb else None
    return dict(dimension=a.size, norm_a=na, norm_b=nb,
                normalized_l2=float(np.linalg.norm(b - a) / na) if na else None,
                cosine_similarity=c, cosine_distance=1 - c if c is not None else None,
                range_a=[float(a.min()), float(a.max())], range_b=[float(b.min()), float(b.max())])


class Catalog:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA foreign_keys=ON')
        self.db.executescript('''
        CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY CHECK(version=1));
        INSERT OR IGNORE INTO schema_version VALUES(1);
        CREATE TABLE IF NOT EXISTS dataset(id TEXT PRIMARY KEY,label TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS model(id TEXT PRIMARY KEY,metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS experiment(id TEXT PRIMARY KEY,label TEXT,dataset_id TEXT REFERENCES dataset(id),model_id TEXT REFERENCES model(id));
        CREATE TABLE IF NOT EXISTS utterance(id TEXT PRIMARY KEY,experiment_id TEXT REFERENCES experiment(id),label TEXT,text TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS condition(id TEXT PRIMARY KEY,experiment_id TEXT REFERENCES experiment(id),label TEXT,metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS recording(id TEXT PRIMARY KEY,path TEXT,sha256 TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS history(id TEXT PRIMARY KEY,recording_id TEXT REFERENCES recording(id),metadata TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS source(id TEXT PRIMARY KEY,path TEXT UNIQUE,sha256 TEXT,verification TEXT);
        CREATE TABLE IF NOT EXISTS context(id TEXT PRIMARY KEY,utterance_id TEXT REFERENCES utterance(id),condition_id TEXT REFERENCES condition(id),history_id TEXT REFERENCES history(id),metadata TEXT);
        CREATE TABLE IF NOT EXISTS capture(id TEXT PRIMARY KEY,context_id TEXT UNIQUE REFERENCES context(id),source_id TEXT REFERENCES source(id),adapter TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS comparison(id TEXT PRIMARY KEY,a TEXT REFERENCES context(id),b TEXT REFERENCES context(id),source_id TEXT REFERENCES source(id),artifact_prefix TEXT,metadata TEXT);
        ''')
        self.db.commit()

    def insert(self, table, values):
        keys = list(values)
        self.db.execute(f'INSERT OR IGNORE INTO {table}({",".join(keys)}) VALUES({",".join("?" for _ in keys)})',
                        [json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for v in values.values()])

    def register_legacy(self, path, experiment, dataset, language, settings=None, acoustic_rows=()):
        path = Path(path).resolve()
        verification = verify_trace(path)
        old = sqlite3.connect((path / 'trace.sqlite').as_uri() + '?mode=ro', uri=True)
        old.row_factory = sqlite3.Row
        try:
            meta = json.loads(old.execute('SELECT metadata FROM run WHERE id=1').fetchone()[0])
            arch = readjson(path / 'architecture.json')['config']
            cfg = meta['checkpoint']
            model = {'revision': cfg['model_revision'], 'codec_revision': cfg['codec_revision'],
                     'model_repo': cfg['model_repo'], 'layers': arch['num_hidden_layers'],
                     'hidden_size': arch['hidden_size'], 'dtype': arch.get('dtype', arch.get('torch_dtype', 'bfloat16')),
                     'vendor_inference_sha256': cfg['vendor_inference_sha256']}
            mid, did = identity('model', model), identity('dataset', dataset)
            eid = identity('experiment', [experiment, did, mid])
            sid = identity('source', [verification['sqlite_sha256'], digest(path / 'metadata.json')])
            self.insert('dataset', dict(id=did, label=dataset))
            self.insert('model', dict(id=mid, metadata=model))
            self.insert('experiment', dict(id=eid, label=experiment, dataset_id=did, model_id=mid))
            self.insert('source', dict(id=sid, path=str(path), sha256=verification['sqlite_sha256'], verification=verification))
            gens = meta['source_generations']
            first = next(iter(gens.values()))
            if any(r['voice']!=first['voice'] or r['model_revision']!=cfg['model_revision'] for r in gens.values()):
                raise ValueError('Legacy bundle must contain one voice/checkpoint; use a capture manifest for mixed conditions')
            uid = identity('utterance', [eid, first['scenario_id'], first['text_SHA256']])
            if any(r['text_SHA256'] != first['text_SHA256'] for r in gens.values()):
                raise ValueError('Source texts are not matched')
            self.insert('utterance', dict(id=uid, experiment_id=eid, label=first['scenario_id'], text=first['input_text'],
                metadata={'text_sha256': first['text_SHA256'], 'split': first['split'], 'language': language,
                          'provenance': 'existing authored calibration sentence'}))
            mapping = {}
            for row in old.execute('SELECT * FROM context'):
                cm = json.loads(row['metadata'])
                prompt_key, history_key = row['prompt'], row['history']
                condition_meta = {'style': cfg['styles'][prompt_key], 'speaker': first['voice'], 'language': language,
                    'model_id': mid, 'model_revision': cfg['model_revision'], 'codec_revision': cfg['codec_revision'],
                    'prompt_template': {'name': 'Indic-Speak vendor build_prompt', 'sha256': cfg['vendor_inference_sha256']},
                    'settings': settings or {'status': 'not supplied; consult original run'}, 'intervention': None}
                cond = self.register_condition(eid, prompt_key, condition_meta)
                gen = gens[history_key]
                audio = path / 'audio' / (history_key + '.wav')
                if digest(audio) != gen['audio_SHA256']:
                    raise ValueError('Audio history hash differs')
                rec = identity('recording', [gen['audio_SHA256'], gen['generation_id'], gen['model_revision']])
                acoustic = [r for r in acoustic_rows if r['generation_id'] == gen['generation_id'] and r['audio_SHA256'] == gen['audio_SHA256']]
                self.insert('recording', dict(id=rec, path=str(audio), sha256=gen['audio_SHA256'],
                    metadata={'generation': gen, 'acoustics': acoustic[0] if len(acoustic) == 1 else None}))
                with np.load(path / (history_key + '_decoder.npz'), allow_pickle=False) as z:
                    full_codes = codes_to_tokens(*(z[f'codebook_{i}_ids'] for i in range(3)), base=128266, dedup=False)
                if cm['history_tokens'] != full_codes[:len(cm['history_tokens'])]:
                    raise ValueError('Forced history and decoder codes differ')
                hid = identity('history', [rec, full_codes])
                self.insert('history', dict(id=hid, recording_id=rec, metadata={'label': history_key,
                    'token_sha256': identity('tokens', full_codes), 'source_id': sid, 'legacy_key': history_key,
                    'audio_code_count': len(full_codes)}))
                cid = identity('context', [uid, cond, hid, cm['prompt_ids'], cm['history_tokens']])
                positions = [dict(r) for r in old.execute('SELECT * FROM position WHERE context_id=? ORDER BY j', (row['id'],))]
                with np.load(path / (row['id'] + '.npz'), allow_pickle=False) as z:
                    saved = z['prediction_rows'].tolist()
                    if z['resid'].shape != (model['layers'], len(saved), model['hidden_size']):
                        raise ValueError('Capture shape differs from model')
                self.insert('context', dict(id=cid, utterance_id=uid, condition_id=cond, history_id=hid,
                    metadata={**cm, 'positions': positions, 'evidence': 'observational_replay'}))
                self.insert('capture', dict(id=identity('capture', [sid, row['id']]), context_id=cid, source_id=sid,
                    adapter='legacy_npz_v1', metadata={'legacy_context': row['id'], 'layers': model['layers'],
                    'hidden_size': model['hidden_size'], 'n_positions': len(positions), 'full_vector_positions': saved,
                    'capture_levels': {'distance_maps': 'all positions', 'hidden_vectors': 'selective',
                    'attention_weights': 'not captured', 'normalization_inputs': 'not captured',
                    'projected_codec_channels': 'first 16 of 768', 'quantized_latent': 'all 768 channels'}}))
                mapping[row['id']] = cid
            for row in old.execute('SELECT * FROM comparison'):
                self.insert('comparison', dict(id=identity('comparison', [sid, row['id']]), a=mapping[row['reference_id']],
                    b=mapping[row['changed_id']], source_id=sid, artifact_prefix=row['id'], metadata=json.loads(row['metadata'])))
            self.db.commit()
            return {'experiment': eid, 'utterance': uid, 'contexts': len(mapping), 'verified': verification}
        except Exception:
            self.db.rollback()
            raise
        finally:
            old.close()

    def register_condition(self, experiment_id, label, metadata):
        for key in ('style', 'speaker', 'language', 'model_id', 'model_revision', 'prompt_template', 'settings'):
            if key not in metadata:
                raise ValueError('Condition lacks ' + key)
        if not self.db.execute('SELECT 1 FROM experiment WHERE id=?', (experiment_id,)).fetchone():
            raise ValueError('Unknown experiment')
        cid = identity('condition', [experiment_id, metadata])
        self.insert('condition', dict(id=cid, experiment_id=experiment_id, label=label, metadata=metadata))
        return cid

    def register_spec(self, manifest):
        """Register a condition-agnostic, hash-declared capture manifest.

        Existing array keys/CSV columns are reused; artifact filenames and
        condition names are unconstrained. No plugin code or model is loaded.
        """
        manifest = Path(manifest).resolve()
        spec = readjson(manifest)
        if spec.get('schema_version') != 1:
            raise ValueError('Unsupported capture manifest version')
        root = manifest.parent
        artifacts = {}
        def artifact(entry):
            path = (root / entry['path']).resolve()
            if not path.is_relative_to(root) or digest(path) != entry['sha256']:
                raise ValueError('Unsafe or changed declared artifact')
            artifacts[str(path)] = entry['sha256']
            return str(path)
        model = spec['experiment']['model']
        for key in ('revision','codec_revision','layers','hidden_size','dtype'):
            if key not in model:
                raise ValueError('Model metadata lacks ' + key)
        did = identity('dataset', spec['experiment']['dataset'])
        mid = identity('model', model)
        eid = identity('experiment', [spec['experiment']['name'], did, mid])
        sid = identity('source', digest(manifest))
        try:
            self.insert('dataset', dict(id=did,label=spec['experiment']['dataset']))
            self.insert('model', dict(id=mid,metadata=model))
            self.insert('experiment', dict(id=eid,label=spec['experiment']['name'],dataset_id=did,model_id=mid))
            u=spec['utterance']
            uid=identity('utterance',[eid,u['id'],hashlib.sha256(u['text'].encode()).hexdigest()])
            self.insert('utterance',dict(id=uid,experiment_id=eid,label=u['id'],text=u['text'],metadata=u['metadata']))
            conditions={}
            for key,value in spec['conditions'].items():
                cm={**value['metadata'],'model_id':mid,'model_revision':model['revision'],'codec_revision':model['codec_revision']}
                conditions[key]=self.register_condition(eid,value['label'],cm)
            # Source row is inserted before its foreign-key dependants; verification
            # is completed atomically with the rest of this transaction.
            self.insert('source',dict(id=sid,path=str(manifest),sha256=digest(manifest),verification={}))
            histories={}
            for key,value in spec.get('histories',{}).items():
                token_only=value.get('recording') is None
                if token_only:
                    if not value.get('tokens') or value.get('decoder_npz') or value.get('decoder_json'):
                        raise ValueError('Token-only history must declare tokens and no decoder artifacts')
                    token_file=artifact(value['tokens'])
                    tokens=token_history_codes(readjson(token_file));rec=None
                    history_files={'tokens':token_file}
                else:
                    audio=artifact(value['recording']['audio'])
                    decoder_npz=artifact(value['decoder_npz']);decoder_json=artifact(value['decoder_json'])
                    with np.load(decoder_npz,allow_pickle=False) as z:
                        tokens=codes_to_tokens(*(z[f'codebook_{i}_ids'] for i in range(3)),base=128266,dedup=False)
                    rec=identity('recording',[digest(audio),value['recording']['metadata']])
                    self.insert('recording',dict(id=rec,path=audio,sha256=digest(audio),metadata=value['recording']['metadata']))
                    history_files={'arrays':decoder_npz,'metadata':decoder_json}
                hid=identity('history',[rec,mid,tokens] if token_only else [rec,tokens])
                self.insert('history',dict(id=hid,recording_id=rec,metadata={'label':value['label'],'token_sha256':identity('tokens',tokens),
                    'source_id':sid,'audio_code_count':len(tokens),'files':history_files,'token_only':token_only,
                    'provenance':value.get('metadata',{})}))
                histories[key]=(hid,tokens)
            contexts={}
            for value in spec.get('contexts',[]):
                cm={**value['metadata']};cm.setdefault('evidence','observational_replay');cm.setdefault('prompt_length',len(cm['prompt_ids']))
                condition=conditions[value['condition']];hid,full_history=histories[value['history']]
                if cm['history_tokens']!=full_history[:len(cm['history_tokens'])]:
                    raise ValueError('Forced context and decoder history differ')
                positions=cm['positions']
                if any(r['j']!=j for j,r in enumerate(positions)) or len(positions)!=len(cm['history_tokens']):
                    raise ValueError('Invalid explicit prediction grid')
                for j,pos in enumerate(positions):
                    expected_input=cm['prompt_ids'][-1] if j==0 else cm['history_tokens'][j-1]
                    if (pos['predicted_phase']!=j%7 or pos['processed_phase']!=((j-1)%7 if j else -1) or
                        pos['frame']!=j//7 or pos['absolute_position']!=len(cm['prompt_ids'])-1+j or
                        pos['input_token']!=expected_input or pos['target_token']!=cm['history_tokens'][j]):
                        raise ValueError('Prediction position, phase, or token association differs')
                if cm.get('evidence','observational_replay')=='causal_intervention' and not cm.get('intervention',{}).get('executed'):
                    raise ValueError('Causal trace requires explicit executed intervention metadata')
                files={'arrays':artifact(value['arrays']),'predictions':artifact(value['predictions'])}
                with np.load(files['arrays'],allow_pickle=False) as z:
                    saved=z['prediction_rows'].tolist()
                    if z['resid'].shape!=(model['layers'],len(saved),model['hidden_size']):
                        raise ValueError('Declared architecture and capture shape differ')
                cid=identity('context',[uid,condition,hid,cm['prompt_ids'],cm['history_tokens'],sid])
                self.insert('context',dict(id=cid,utterance_id=uid,condition_id=condition,history_id=hid,metadata=cm))
                self.insert('capture',dict(id=identity('capture',[sid,value['key']]),context_id=cid,source_id=sid,adapter='capture_manifest_v1',metadata={
                    'files':files,'layers':model['layers'],'hidden_size':model['hidden_size'],'n_positions':len(positions),
                    'full_vector_positions':saved,'capture_levels':value['capture_levels']}))
                contexts[value['key']]=cid
            for value in spec.get('comparisons',[]):
                files={'arrays':artifact(value['arrays']),'predictions':artifact(value['predictions'])}
                self.insert('comparison',dict(id=identity('comparison',[sid,value['a'],value['b']]),a=contexts[value['a']],b=contexts[value['b']],
                    source_id=sid,artifact_prefix=None,metadata={**value['metadata'],'files':files}))
            self.db.execute('UPDATE source SET verification=? WHERE id=?',(json.dumps({'kind':'capture_manifest_v1','artifacts':artifacts}),sid))
            self.db.commit()
            return {'experiment':eid,'utterance':uid,'contexts':len(contexts),'verified_artifacts':len(artifacts)}
        except Exception:
            self.db.rollback()
            raise

    def verify_sources(self):
        out = []
        for row in self.db.execute('SELECT * FROM source'):
            verification=json.loads(row['verification'])
            if verification.get('kind')=='capture_manifest_v1':
                if digest(row['path'])!=row['sha256'] or any(digest(p)!=sha for p,sha in verification['artifacts'].items()):
                    raise ValueError('Registered capture manifest/artifact changed')
                out.append({'verified_artifacts':len(verification['artifacts']),'source_sha256':row['sha256']})
                continue
            if digest(Path(row['path']) / 'trace.sqlite') != row['sha256']:
                raise ValueError('Historical SQLite changed')
            out.append(verify_trace(Path(row['path'])))
        return out

    def close(self):
        self.db.close()


@lru_cache(maxsize=6)
def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: v.copy() for k, v in z.items()}


class Explorer:
    def __init__(self, catalog):
        self.catalog = Path(catalog).resolve()

    def query(self, sql, args=()):
        db=sqlite3.connect(self.catalog.as_uri() + '?mode=ro',uri=True)
        try:
            db.row_factory = sqlite3.Row
            out = [dict(r) for r in db.execute(sql, args)]
        finally:
            db.close()
        for r in out:
            if 'metadata' in r:
                r['metadata'] = json.loads(r['metadata'])
        return out

    def one(self, table, key):
        if table not in ('context', 'capture', 'source', 'condition', 'history', 'recording', 'utterance', 'model'):
            raise ValueError('Unknown entity')
        rows = self.query(f'SELECT * FROM {table} WHERE id=?', (key,))
        if len(rows) != 1:
            raise ValueError('Unknown ' + table)
        return rows[0]

    def context(self, cid):
        row = self.one('context', cid)
        cap = self.query('SELECT * FROM capture WHERE context_id=?', (cid,))
        if not cap:
            raise ValueError('No trace captured for this context')
        row['capture'] = cap[0]
        source=Path(self.one('source',cap[0]['source_id'])['path'])
        row['root']=source if source.is_dir() else source.parent
        return row

    def catalog_data(self):
        out = {name: self.query('SELECT * FROM ' + table) for name, table in
                [('datasets', 'dataset'), ('models', 'model'), ('experiments', 'experiment'),
                 ('utterances', 'utterance'), ('conditions', 'condition'), ('histories', 'history'),
                 ('contexts', 'context'), ('captures', 'capture'), ('comparisons', 'comparison')]}
        for context in out['contexts']:
            context['metadata'] = {k: context['metadata'][k] for k in ('prompt_length', 'evidence')}
        return out

    def describe(self, cid):
        c = self.context(cid)
        c['root'] = str(c['root'])
        c['condition'] = self.one('condition', c['condition_id'])
        c['history'] = self.one('history', c['history_id'])
        c['recording'] = self.one('recording', c['history']['recording_id']) if c['history']['recording_id'] else None
        c['model'] = self.one('model', c['condition']['metadata']['model_id'])
        return c

    def compatible(self, a, b):
        ca, cb = self.context(a), self.context(b)
        ma, mb = [self.one('condition', c['condition_id'])['metadata'] for c in (ca, cb)]
        if ca['utterance_id'] != cb['utterance_id'] or ma['model_id'] != mb['model_id']:
            raise ValueError('Different utterances/checkpoints are incompatible with this aligned comparison')
        pa, pb = ca['metadata']['positions'], cb['metadata']['positions']
        if len(pa) != len(pb) or any(x['j'] != y['j'] or x['predicted_phase'] != y['predicted_phase'] or x['frame'] != y['frame'] for x, y in zip(pa, pb)):
            raise ValueError('No matching generation-position/phase grid')
        same_history = ca['history_id'] == cb['history_id']
        same_condition = ca['condition_id'] == cb['condition_id']
        kind = 'same_context' if a == b else 'prompt_condition' if same_history else 'audio_history' if same_condition else 'mixed'
        warning = ('Identical forced audio history. Prompt length/context effects remain.' if same_history else
                   'Different audio histories: code indices/time grids are NOT word or phoneme alignments. Not an isolated emotion effect.')
        if ma['speaker'] != mb['speaker']:
            warning += ' Speaker identities also differ.'
        causal=any(c['metadata'].get('evidence')=='causal_intervention' and c['metadata'].get('intervention',{}).get('executed') for c in (ca,cb))
        if causal:
            warning+=' An executed intervention is registered; inspect other changed factors before making causal claims.'
        return ca, cb, {'type': kind, 'same_history': same_history, 'same_condition': same_condition,
                        'evidence': 'registered_intervention' if causal else 'observational_replay', 'warning': warning}

    def _vectors(self, context):
        files=context['capture']['metadata'].get('files')
        return arrays(files['arrays'] if files else str(context['root'] / (context['capture']['metadata']['legacy_context'] + '.npz')))

    def _vector(self, context, layer, j, site):
        if site not in ('resid','attn','mlp','incoming','after_attention','attention_norm','mlp_norm','final_norm','embedding'):
            raise ValueError('Unknown tensor site')
        z = self._vectors(context)
        if not 0 <= layer < context['capture']['metadata']['layers']:
            raise ValueError('Unknown layer')
        found = np.flatnonzero(z['prediction_rows'] == j)
        if site == 'embedding':
            return z['prompt_embeddings'][-1] if j == 0 and len(z.get('prompt_embeddings',[])) else None, 'captured prompt anchor only where available; audio embeddings were not saved'
        if not len(found):
            return None, 'Full vector not captured at this prediction position'
        k = int(found[0])
        if site in ('resid', 'attn', 'mlp'):
            return (z[site][layer,k], 'captured exactly (FP32 representation of BF16 values)') if site in z else (None,'This branch was not captured')
        if site == 'final_norm':
            return (z['final_norm'][k], 'captured final model RMSNorm output; independent of selected block') if 'final_norm' in z else (None,'Final normalization output was not captured')
        if site in ('attention_norm', 'mlp_norm'):
            return None, 'Normalization inputs/weights were not captured; cannot reconstruct exactly'
        incoming = z['resid'][layer - 1, k] if layer else z['prompt_embeddings'][-1] if j == 0 and len(z.get('prompt_embeddings',[])) else None
        if site == 'incoming':
            return incoming, 'previous block output' if layer else 'prompt embedding at j=0 only'
        if site == 'after_attention':
            if incoming is None:
                return None, 'Layer-0 audio embedding unavailable; subtraction cannot invert BF16 rounding'
            if 'attn' not in z or 'mlp' not in z:
                return None,'Missing branch capture prevents verified residual reconstruction'
            after = bf16_round(incoming + z['attn'][layer, k])
            check = bf16_round(after + z['mlp'][layer, k])
            if not np.array_equal(check, z['resid'][layer, k]):
                raise ValueError('BF16 residual reconstruction failed; no inferred tensor is exposed')
            return after, 'reconstructed BF16 addition; final residual verified bit-exact against capture'
        raise ValueError('Unknown tensor site')

    def vectors(self, a, b, layer, j, site, start=0, end=3072):
        ca, cb, relation = self.compatible(a, b)
        if not 0 <= j < len(ca['metadata']['positions']):
            raise ValueError('Unknown position')
        va, sa = self._vector(ca, layer, j, site)
        vb, sb = self._vector(cb, layer, j, site)
        out = {'available': va is not None and vb is not None, 'status_a': sa, 'status_b': sb,
               'site': site, 'layer': layer, 'j': j, 'relation': relation,
               'position_a': ca['metadata']['positions'][j], 'position_b': cb['metadata']['positions'][j],
               'saved_positions': ca['capture']['metadata']['full_vector_positions'],
               'provenance': [self._context_file(c,'arrays') for c in (ca, cb)]}
        if out['available']:
            end = min(end, va.size)
            if not 0 <= start < end <= va.size:
                raise ValueError('Invalid channel range')
            out.update(a=va[start:end].tolist(), b=vb[start:end].tolist(),
                       delta=(vb[start:end] - va[start:end]).tolist(), start=start, end=end, metrics=metrics(va, vb))
        return out

    def comparison(self, a, b, site='resid'):
        if site not in ('resid', 'attn', 'mlp'):
            raise ValueError('No full-position distance capture for that site')
        ca, cb, relation = self.compatible(a, b)
        rows = self.query('SELECT * FROM comparison WHERE a=? AND b=?', (a, b))
        if not rows:
            raise ValueError('No stored ordered comparison. Choose a registered comparison; do not reconstruct distributions from top-five entries.')
        row = rows[0]
        root = Path(self.one('source', row['source_id'])['path'])
        prefix = row['artifact_prefix'];files=row['metadata'].get('files')
        tensor_file=files['arrays'] if files else str(root/(prefix+'_heatmaps.npz'))
        prediction_file=files['predictions'] if files else str(root/(prefix+'_predictions.csv'))
        z=arrays(tensor_file);predictions=prediction_rows(prediction_file)
        n = len(predictions)
        if n != len(ca['metadata']['positions']):
            raise ValueError('Comparison grid differs from context')
        expected=(ca['capture']['metadata']['layers'],n)
        if z[site+'_cosine_distance'].shape!=expected or z[site+'_normalized_l2'].shape!=expected:
            raise ValueError('Comparison tensor shape differs from capture')
        return {'relation': relation, 'summary': row['metadata'], 'positions_a': ca['metadata']['positions'],
                'positions_b': cb['metadata']['positions'], 'predictions': predictions,
                'cosine_distance': z[site + '_cosine_distance'].tolist(),
                'normalized_l2': z[site + '_normalized_l2'].tolist(),
                'provenance': [tensor_file,prediction_file]}

    def _context_file(self,c,kind):
        files=c['capture']['metadata'].get('files')
        if files:return files[kind]
        return str(c['root']/(c['capture']['metadata']['legacy_context']+('.npz' if kind=='arrays' else '.json')))

    def predictions(self, a, b, j):
        ca, cb, relation = self.compatible(a, b)
        if not 0 <= j < len(ca['metadata']['positions']):
            raise ValueError('Unknown position')
        out = {'relation': relation, 'j': j, 'position_a': ca['metadata']['positions'][j],
               'position_b': cb['metadata']['positions'][j], 'probabilities': 'raw LM softmax, not warped sampling probabilities'}
        for key, ctx in [('a', ca), ('b', cb)]:
            out[key] = cached_json(self._context_file(ctx,'predictions'))['top_predictions'][j]
        try:
            out['statistics'] = self.comparison(a, b)['predictions'][j]
        except ValueError:
            out['statistics'] = None
        return out

    def embeddings(self, a, b, ia, ib):
        ca, cb, relation = self.compatible(a, b)
        za, zb = self._vectors(ca), self._vectors(cb)
        if not len(za.get('prompt_embeddings',[])) or not len(zb.get('prompt_embeddings',[])):
            return {'available':False,'status_a':'Prompt embeddings not included in this capture/export',
                    'status_b':'Prompt embeddings not included in this capture/export','relation':relation,
                    'note':'Summary-only exports do not fabricate missing embedding vectors.'}
        if not 0 <= ia < len(za['prompt_embeddings']) or not 0 <= ib < len(zb['prompt_embeddings']):
            raise ValueError('Invalid prompt-token index')
        va, vb = za['prompt_embeddings'][ia], zb['prompt_embeddings'][ib]
        return {'available': True, 'a': va.tolist(), 'b': vb.tolist(), 'delta': (vb-va).tolist(), 'start': 0,
                'metrics': metrics(va, vb), 'status_a': 'captured prompt embedding', 'status_b': 'captured prompt embedding',
                'tokens': [ca['metadata']['prompt_ids'][ia], cb['metadata']['prompt_ids'][ib]],
                'note': 'Manually selected prompt tokens; equal integer indices need not be semantically aligned.'}

    def geometry(self, a, b, layers, site):
        ca, cb, relation = self.compatible(a, b)
        if site not in ('resid','attn','mlp','incoming','after_attention'):
            raise ValueError('This PCA requires a layer-specific residual/branch site')
        if len(layers) > 2 or not layers:
            raise ValueError('Choose one or two layers for one shared PCA fit')
        points, values = [], []
        for ctx, label in [(ca, 'A'), (cb, 'B')]:
            for layer in layers:
                for j in ctx['capture']['metadata']['full_vector_positions']:
                    v, _ = self._vector(ctx, layer, j, site)
                    if v is not None and np.linalg.norm(v) > 0:
                        points.append({'context': ctx['id'], 'condition': label, 'layer': layer, 'j': j})
                        values.append(v / np.linalg.norm(v))
        if len(values) < 3:
            raise ValueError('Insufficient compatible captured vectors for PCA')
        x = np.asarray(values, np.float64)
        mean, axes = pca_fit(x, 2)
        scores = (x - mean) @ axes.T
        total = np.square(x - mean).sum()
        explained = np.square(scores).sum(axis=0) / total if total else np.zeros(2)
        return {'points': [{**p, 'x': float(s[0]), 'y': float(s[1])} for p, s in zip(points, scores)],
                'explained_variance_ratio': explained.tolist(), 'fit_id': identity('pca', [a, b, layers, site]),
                'method': 'One shared PCA fit on unit-normalized captured vectors from selected contexts/layers. Descriptive, not independent samples.',
                'relation': relation}

    def decoder_info(self, hid):
        h = self.one('history', hid)
        root = Path(self.one('source', h['metadata']['source_id'])['path'])
        files=h['metadata'].get('files');key=h['metadata'].get('legacy_key')
        if h['metadata'].get('token_only'):
            return {'available':False,'history':h,'stages':[],'root':str(root.parent if root.is_file() else root),
                    'metadata':None,'reason':'Token-only reference history: no decoder replay or waveform is registered in this capture.'}
        tensor_file=files['arrays'] if files else str(root/(key+'_decoder.npz'))
        metadata_file=files['metadata'] if files else str(root/(key+'_decoder.json'))
        meta=cached_json(metadata_file);z=arrays(tensor_file)
        stages = []
        for name, x in z.items():
            if name == 'z_q' or name.endswith('.channels') or name.startswith('vocos.predicted_stft_'):
                summary = next((r for r in meta['summaries'] if r['site'] + '.channels' == name), None)
                matrix = x[0] if name == 'z_q' else x
                original = summary['shape'] if summary else list(x.shape)
                stages.append({'key': name, 'channels': matrix.shape[0], 'time_frames': matrix.shape[1],
                    'original_shape': original, 'coverage':
                    f"selected first {matrix.shape[0]} of {summary['shape'][summary['channel_axis']]} channels"
                    if summary and matrix.shape[0] < summary['shape'][summary['channel_axis']] else 'all recorded channels'})
        return {'available':True,'history': h, 'metadata': meta, 'stages': stages, 'root': str(root), 'key': key,'tensor_file':tensor_file}

    def decoder(self, hid, site, frame, subframe=0, channel=0, chstart=0, chcount=32, tstart=None, tcount=160):
        info = self.decoder_info(hid)
        if not info.get('available',True):
            raise ValueError(info['reason'])
        z = arrays(info['tensor_file'])
        if site not in {s['key'] for s in info['stages']}:
            raise ValueError('Decoder tensor not captured')
        x = z[site][0] if site == 'z_q' else z[site]
        if not 0 <= frame < info['metadata']['snac_frames'] or not 0 <= subframe < 4:
            raise ValueError('Unknown SNAC frame/subframe')
        if site.startswith('snac.quantizer.'):
            branch = int(site.split('.')[2])
            index = frame * (2 ** branch) + subframe // (4 // 2 ** branch)
            dt = 2048 / 24000 / (2 ** branch)
        elif site == 'z_q' or site.startswith(('vocos.stem', 'vocos.pre_last')):
            index, dt = frame * 4 + subframe, 512 / 24000
        else:
            index = (frame * 4 + subframe) * info['metadata']['vocos_upsample']
            dt = info['metadata']['hop'] / 24000
        tstart = max(0, index - 40) if tstart is None else tstart
        if not 0 <= channel < x.shape[0] or not 0 <= chstart < x.shape[0] or not 0 <= tstart < x.shape[1]:
            raise ValueError('Unknown channel/time window')
        if not 1 <= chcount <= 256 or not 1 <= tcount <= 512:
            raise ValueError('Choose a bounded decoder slice')
        ids = codes_to_tokens(*(z[f'codebook_{i}_ids'] for i in range(3)), base=128266, dedup=False)[frame*7:frame*7+7]
        checks = {'original_full_quantizer_check': info['metadata']['quantizer_reconstruction_exact']}
        if site == 'z_q':
            terms = [np.repeat(z[f'snac.quantizer.{i}.out_proj.channels'], stride, axis=1)
                     for i, stride in enumerate(info['metadata']['quantizer_strides'])]
            check = (terms[0] + terms[1]) + terms[2]
            checks['saved_first_16_channel_sum_exact'] = bool(np.array_equal(check, x[:check.shape[0]]))
            checks['reconstructed_channels'] = check.shape[0]
        stat = next(s for s in info['stages'] if s['key'] == site)
        summary_key = site.removesuffix('.channels') + '.rms_per_frame'
        return {'site': site, 'shape': list(x.shape), 'original_shape': stat['original_shape'],
                'coverage': stat['coverage'], 'index': index, 'dt_s': dt, 'frame': frame,
                'subframe': subframe, 'time_s_approx': frame * 2048 / 24000 + subframe * 512 / 24000,
                'code_ids': ids, 'codebook_entries': [int(v - 128266 - p*4096) for p, v in enumerate(ids)],
                'matrix': x[chstart:chstart+chcount, tstart:tstart+tcount].tolist(),
                'channel_start': chstart, 'time_start': tstart, 'channel': channel,
                'profile': x[channel].tolist(), 'vector': x[:, min(index, x.shape[1]-1)].tolist(),
                'rms': z[summary_key].tolist() if summary_key in z else np.sqrt(np.mean(x*x, axis=0)).tolist() if site=='z_q' else None,
                'checks': checks, 'provenance': info['tensor_file'],
                'note': 'Recorded code/decoder frame grid, not word alignment. Spectra are predicted Vocos spectra, not a fresh waveform STFT.'}

    def waveform(self, hid):
        info = self.decoder_info(hid)
        if not info.get('available',True):
            return {'available':False,'reason':info['reason'],'acoustics':None,'audio_id':None}
        z = arrays(info['tensor_file'])
        wave = z['waveform_replay']
        chunks = [wave[i:i+240] for i in range(0, len(wave), 240)]
        rec = self.one('recording', info['history']['recording_id'])
        return {'available':True,'dt_s': .01, 'min': [float(v.min()) for v in chunks], 'max': [float(v.max()) for v in chunks],
                'sample_rate': 24000, 'samples': len(wave), 'duration_s': len(wave)/24000,
                'acoustics': rec['metadata']['acoustics'], 'audio_id': rec['id'],
                'provenance': info['tensor_file'],
                'replay_pcm_error': info['metadata']['waveform_max_abs_error_vs_original']}


@lru_cache(maxsize=12)
def cached_json(path):
    return readjson(path)
