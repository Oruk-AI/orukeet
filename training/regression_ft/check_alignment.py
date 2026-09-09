"""Check a fixed sample of repaired human spans with an independent ASR model."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--alignment', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.root / 'unseen_20260907/code'))
    from metrics import counts
    a.output.mkdir(parents=True, exist_ok=True)
    samples, inputs = [], {}
    for split in ['eurospeech_el', 'eurospeech_it']:
        source = a.alignment / (split + '-accepted.jsonl')
        accepted = [json.loads(l) for l in source.open()]
        inputs[split] = sha(source)
        audio = {r['uid']: r for r in [json.loads(l) for l in (a.root / 'regression_ft_20260907/audio' / (split + '.jsonl')).open()]}
        for status in ['assigned_pair_verified', 'realigned_human_span']:
            part = sorted([r for r in accepted if r['status'] == status],
                          key=lambda r: hashlib.sha256(('alignment-check-20260918:' + r['uid']).encode()).hexdigest())[:32]
            for r in part:
                samples.append(dict(r, audio_filepath=audio[r['uid']]['audio_filepath'],
                                    old_reference=audio[r['uid']]['text'], lang=audio[r['uid']]['lang']))
    protocol = dict(selection='First 32 UIDs by SHA-256(alignment-check-20260918:UID) per language and acceptance type.',
                    input_sha256=inputs, model='openai/whisper-large-v3',
                    revision='06f233fe06e710322aca913c1bc4249a0d71fce1',
                    model_weights_sha256=sha(a.model / 'model.safetensors'),
                    script_sha256=sha(__file__), rows=len(samples),
                    samples=[dict(uid_sha256=hashlib.sha256(r['uid'].encode()).hexdigest(), split=r['split'], status=r['status']) for r in samples],
                    purpose='Independent alignment diagnostic. Whisper output is not used as a training label.')
    (a.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    torch.set_num_threads(8)
    torch.manual_seed(20260918)
    processor = AutoProcessor.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(a.model, local_files_only=True,
                torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()
    results = []
    for language in ['el', 'it']:
        rows = [r for r in samples if r['lang'] == language]
        for start in range(0, len(rows), 8):
            batch = rows[start:start + 8]
            audio = []
            for r in batch:
                data, rate = sf.read(r['audio_filepath'], dtype='float32')
                assert rate == 16000 and data.ndim == 1 and len(data) <= 480000
                audio.append(data)
            features = processor(audio, sampling_rate=16000, return_tensors='pt', return_attention_mask=True)
            features['input_features'] = features['input_features'].cuda().half()
            features['attention_mask'] = features['attention_mask'].cuda()
            with torch.inference_mode():
                ids = model.generate(**features, language=language, task='transcribe',
                                     do_sample=False, num_beams=1, max_length=448,
                                     return_timestamps=False, condition_on_prev_tokens=False)
            texts = processor.batch_decode(ids, skip_special_tokens=True)
            for r, text in zip(batch, texts):
                result = dict(uid=r['uid'], split=r['split'], status=r['status'], whisper_text=text,
                              aligned_reference=r['text'], original_reference=r['old_reference'],
                              aligned_counts=counts(r['text'], text), original_counts=counts(r['old_reference'], text))
                results.append(result)
            print('WHISPER_ALIGNMENT_CHECK', language, start + len(batch), '/', len(rows), flush=True)
    (a.output / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in results))
    grouped = defaultdict(list)
    for r in results:
        grouped[(r['split'], r['status'])].append(r)
    summary = []
    for (split, status), rows in grouped.items():
        item = dict(split=split, status=status, rows=len(rows))
        for key in ['aligned', 'original']:
            counts_ = [r[key + '_counts'] for r in rows]
            errors = sum(r['char_errors'] for r in counts_)
            chars = sum(r['chars'] for r in counts_)
            cers = [r['char_errors'] / max(r['chars'], 1) for r in counts_]
            item[key] = dict(cer=errors / chars, median_cer=float(np.median(cers)),
                             p90_cer=float(np.quantile(cers, .9)), pairs_over_35pct_cer=sum(v > .35 for v in cers))
        summary.append(item)
    report = dict(status='complete', rows=len(results), groups=summary,
                  protocol_sha256=sha(a.output / 'protocol.json'), predictions_sha256=sha(a.output / 'predictions.jsonl'),
                  note='Diagnostic evidence about automatic alignment; not independent human annotation of every pair.')
    (a.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print('ALIGNMENT_CHECK_COMPLETE', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
