"""Prepare a separate, audio-linked script cohort from existing local downloads.

Language tags, transcript equivalence and audible segmentation remain unverified.
Original texts/audio are retained; only Hindi token spans change between inputs.
"""
import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation.common import digest, stable_id, write_json, write_jsonl
from src.evaluation.language import aligned_labels
from src.evaluation.text import normalize


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = {r['name']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate source IDs: ' + str(path))
    return result


def script_pair(roman, deva, tags):
    rt, dt = normalize(roman).split(), normalize(deva).split()
    if tags is None or len(rt) != len(dt) or len(rt) != len(tags):
        raise ValueError('Unaligned text pair')
    if any(t not in ('HI', 'EN') for t in tags) or 'HI' not in tags:
        raise ValueError('Incomplete language labels')
    if any(x != y for x, y, t in zip(rt, dt, tags) if t == 'EN'):
        raise ValueError('Changed English words')
    # Reject obvious Hindi function words mislabelled as English. This is a
    # conservative exclusion rule, not a replacement language annotator.
    obvious_hi = {'mai', 'main', 'mein', 'hai', 'hain', 'tha', 'thi', 'toh',
                  'aap', 'tum', 'hum', 'aur', 'bhi', 'yeh', 'woh', 'kya', 'kyun',
                  'us', 'iss', 'hi', 'ho', 'par', 'per', 'na', 'ko', 'ka', 'ki',
                  'ke', 'se', 'ne', 'ye', 'vo', 'voh', 'ji', 'meri', 'mera',
                  'mere', 'mujhe', 'tumhe', 'aapko', 'apko', 'humko', 'unko',
                  'usko', 'usse', 'iske', 'uske', 'kyunki', 'phir', 'kab', 'ab',
                  'yahan', 'wahan', 'nahin', 'nahi'}
    if any(w in obvious_hi and t == 'EN' for w, t in zip(rt, tags)):
        raise ValueError('Obvious ambiguous language tag')
    for word, tag in zip(dt, tags):
        if tag == 'HI' and not any('DEVANAGARI' in unicodedata.name(c, '') for c in word):
            raise ValueError('Hindi span not converted to Devanagari')
    pieces = []; offset = 0; index = 0
    for match in re.finditer(r'\S+', roman):
        segment = match.group(); tokens = normalize(segment).split()
        if not tokens:
            continue
        if len(tokens) != 1 or tokens[0] != rt[index]:
            raise ValueError('Ambiguous raw token spans')
        if tags[index] == 'HI':
            positions = [i for i, c in enumerate(segment) if c.isalnum()]
            if not positions:
                raise ValueError('Empty Hindi word')
            start = match.start() + positions[0]; stop = match.start() + positions[-1] + 1
            if normalize(roman[start:stop]) != rt[index]:
                raise ValueError('Hindi word boundary mismatch')
            pieces.extend([roman[offset:start], dt[index]]); offset = stop
        index += 1
    if index != len(rt):
        raise ValueError('Incomplete token projection')
    paired = ''.join(pieces) + roman[offset:]
    if normalize(paired).split() != dt:
        raise ValueError('Projected canonical tokens changed')
    return paired


def choose(rows, n, seed, used_conversations=None, used_texts=None):
    conversations = used_conversations if used_conversations is not None else set()
    texts = used_texts if used_texts is not None else set()
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row['length_bin'], row['source_speaker'])].append(row)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: digest(f"{seed}:{r['source_name']}".encode()))
    chosen = []
    while len(chosen) < n:
        added = False
        for key in sorted(buckets):
            bucket = buckets[key]
            while bucket:
                row = bucket.pop(0)
                if row['conversation_id'] in conversations or row['canonical_hash'] in texts:
                    continue
                chosen.append(row); conversations.add(row['conversation_id']); texts.add(row['canonical_hash'])
                added = True; break
            if len(chosen) == n:
                break
        if not added:
            raise ValueError(f'Only {len(chosen)} nonoverlapping candidates available for requested {n}')
    return chosen


def main(a):
    if a.output.exists():
        raise ValueError('Use a new dataset output directory')
    local, shared, deva = table(a.local_csv), table(a.roman_csv), table(a.deva_csv)
    counts = Counter(); rejected = []; candidates = []
    for name, r in local.items():
        try:
            if name not in shared or name not in deva:
                raise ValueError('Missing counterpart')
            s, d = shared[name], deva[name]
            if r['Transcript'] != s['Transcript']:
                raise ValueError('Known local/Kedi transcript mismatch')
            if r['Speaker'] != s['Speaker'] or s['Speaker'] != d['Speaker']:
                raise ValueError('Speaker metadata mismatch')
            if r['Conversation'] != s['Conversation'] or s['Conversation'] != d['Conversation']:
                raise ValueError('Conversation metadata mismatch')
            speaker = s['Speaker']
            if speaker not in ('MAYA', 'MONISHA', 'SAHIL', 'INDRAVARDHAN', 'ROSESH', 'DUSHYANT'):
                raise ValueError('Outside fixed six-character sampling frame')
            tags = aligned_labels(s['Transcript'], s['LID_tags'])
            text = script_pair(s['Transcript'], d['Transcript'], tags)
            nt = len(tags)
            if not 4 <= nt <= 25:
                raise ValueError('Outside 4–25 words')
            if Path(name).name != name or not name.endswith('.wav'):
                raise ValueError('Unsafe source filename')
            audio = (a.audio / name).resolve()
            if audio.parent != a.audio.resolve() or not audio.is_file():
                raise ValueError('Missing named natural WAV')
            info = sf.info(audio)
            if not .8 <= info.duration <= 15 or info.channels not in (1, 2):
                raise ValueError('Outside 0.8–15 seconds or unsupported channels')
            row = {'utterance_id':Path(name).stem, 'source_name':name,
                'conversation_id':s['Conversation'], 'source_speaker':speaker,
                'romanized_text_original':s['Transcript'], 'devanagari_text_original':d['Transcript'],
                'input_romanized':s['Transcript'], 'input_devanagari':text,
                'canonical_text':text, 'canonical_hash':digest(text.encode()),
                'language_labels':tags, 'canonical_tokens':normalize(text).split(),
                'language_annotation_status':'provided_corpus_tags_unverified',
                'condition':'code_switched_provisional' if 'EN' in tags else 'Hindi_only_provisional',
                'token_count':nt, 'english_tokens':tags.count('EN'),
                'length_bin':'short' if nt < 10 else 'medium' if nt < 18 else 'long',
                'source_audio_local_path':str(audio), 'source_sample_rate':info.samplerate,
                'source_channels':info.channels, 'source_frames':info.frames,
                'source_duration_s':info.duration,
                'transcript_verification':'not_human_verified',
                'script_equivalence':'automatic_token_projection_not_human_verified',
                'background_audio_review':'not_human_reviewed', 'speaker_overlap_review':'not_human_reviewed',
                'segmentation_review':'header_only; audible boundaries not_human_reviewed',
                'text_policy':'user approved supplied spellings, provisional comparison; English/punctuation preserved'}
            candidates.append(row)
        except (ValueError, OSError, RuntimeError) as exc:
            reason = str(exc); counts[reason] += 1
            rejected.append({'source_name':name, 'reason':reason})
    conversations, texts = set(), set()
    cs = [r for r in candidates if r['english_tokens']]
    hi = [r for r in candidates if not r['english_tokens']]
    cohort = choose(cs, 90, a.seed, conversations, texts) + choose(hi, 10, a.seed, conversations, texts)
    pilot = choose(cohort[:90], 8, a.seed + 1) + choose(cohort[90:], 2, a.seed + 1)
    pilot_ids = {r['utterance_id'] for r in pilot}
    a.output.mkdir(parents=True)
    for row in cohort:
        source = Path(row['source_audio_local_path']); data = source.read_bytes()
        wave, sr = sf.read(source, always_2d=True, dtype='float64')
        if not len(wave) or not np.isfinite(wave).all() or not np.any(wave):
            raise ValueError('Selected audio failed integrity: ' + source.name)
        row['natural_sha256'] = digest(data); row['pilot'] = row['utterance_id'] in pilot_ids
        row['natural_audio_relative_path'] = 'natural/' + row['source_name']
        destination = a.output / row['natural_audio_relative_path']; destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        if digest(destination.read_bytes()) != row['natural_sha256']:
            raise ValueError('Natural audio copy hash mismatch')
        rms = np.sqrt(np.mean(wave ** 2, axis=0))
        row['technical_checks'] = {'finite_nonempty':True, 'channel_rms':rms.tolist(),
            'clipped_fraction':float(np.mean(np.abs(wave) >= 32767/32768)),
            'audible_quality_confirmed':False}
    write_jsonl(a.output / 'cohort.jsonl', cohort)
    write_jsonl(a.output / 'pilot.jsonl', [r for r in cohort if r['pilot']])
    write_jsonl(a.output / 'excluded_candidates.jsonl', rejected)
    write_json(a.output / 'dataset.json', {'experiment':'localdrive_matched_20261021', 'seed':a.seed,
        'cohort_inputs':100, 'pilot_inputs':10, 'code_switched_provisional':90, 'Hindi_controls_provisional':10,
        'selection':'fixed six-character frame; speaker/length round robin; deterministic hash order; one per conversation and normalized content',
        'source_recording_access':'owned copies of local downloaded WAVs; originals unchanged',
        'excluded_counts':dict(counts), 'eligible_candidates':len(candidates),
        'manual_validation_complete':False, 'reference_status':'metadata_linked_not_gold_transcribed',
        'channel_policy':'preserve original; explicit mean downmix/resampling only in derived model inputs',
        'source_csv_sha256':{str(p):digest(p.read_bytes()) for p in [a.local_csv,a.roman_csv,a.deva_csv]},
        'script_sha256':digest(Path(__file__).read_bytes())})
    print(json.dumps({'cohort':100,'pilot':10,'eligible':len(candidates),'excluded':dict(counts),'output':str(a.output)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['local-csv','roman-csv','deva-csv','audio','output']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--seed',type=int,default=1234)
    main(parser.parse_args())
