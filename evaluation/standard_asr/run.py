"""Compare two fixed NeMo checkpoints on complete LibriSpeech and FLEURS tests.

Input is a JSONL manifest with uid, split, language, text, audio_filepath,
duration, and pcm_sha256. Every record is decoded by both checkpoints; empty
hypotheses remain in the score. Resume verifies model, manifest and code hashes.
"""
import argparse
from collections import Counter, defaultdict
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--parakeet', type=Path, required=True)
    p.add_argument('--orukeet', type=Path, required=True)
    p.add_argument('--parakeet-sha256', default='3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d')
    p.add_argument('--orukeet-sha256', default='0ccfefcd1894871cb0850bd3c464adf5397752840de2a76d1d2d075c4141a945')
    p.add_argument('--metric-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    import torch
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import logging
    sys.path.insert(0, str(a.metric_code))
    from metrics import counts, normalize, ENGLISH

    logging.set_verbosity(logging.ERROR)
    torch.set_num_threads(8)
    torch.manual_seed(20260908)
    np.random.seed(20260908)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    a.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in a.manifest.open()]
    assert len(rows) == len({r['uid'] for r in rows})
    rows.sort(key=lambda r: (r['duration'], r['uid']))
    identities = dict(manifest_sha256=sha(a.manifest), script_sha256=sha(__file__),
                      normalizer_sha256=sha(a.metric_code / 'metrics.py'))
    for row in rows:
        audio, rate = sf.read(row['audio_filepath'], dtype='float32')
        assert rate == 16000 and audio.ndim == 1
        assert abs(len(audio) / rate - row['duration']) < 1 / rate
        assert hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest() == row['pcm_sha256']
    print('AUDIO_VERIFIED', len(rows), flush=True)
    decoding, predictions, timings, models = None, {}, {}, {}
    for label, path in [('parakeet', a.parakeet), ('orukeet', a.orukeet)]:
        identity = dict(identities, model_sha256=sha(path))
        assert identity['model_sha256'] == getattr(a, label + '_sha256')
        models[label] = identity['model_sha256']
        target = a.output / (label + '.jsonl')
        prior = [json.loads(line) for line in target.open()] if target.exists() else []
        assert len(prior) == len({r['uid'] for r in prior})
        for record in prior:
            assert all(record[k] == v for k, v in identity.items())
        done = {r['uid']: r for r in prior}
        assert set(done) <= {r['uid'] for r in rows}
        pending = [r for r in rows if r['uid'] not in done]
        if pending:
            model = ASRModel.restore_from(str(path), map_location='cuda').eval()
            model.freeze()
            config = OmegaConf.to_container(model.cfg.decoding, resolve=True)
            assert config['strategy'] == 'greedy_batch' and config['greedy']['max_symbols'] == 10
            if decoding is None:
                decoding = config
                write(a.output / 'decoding.json', config)
            assert config == decoding
            model.change_decoding_strategy(OmegaConf.create(config))
            started = time.monotonic()

            def infer(batch):
                try:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        hypotheses = model.transcribe([r['audio_filepath'] for r in batch],
                                                      batch_size=16, num_workers=4, verbose=False)
                    assert len(hypotheses) == len(batch)
                    return [h.text if hasattr(h, 'text') else str(h) for h in hypotheses]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if len(batch) == 1:
                        raise
                    mid = len(batch) // 2
                    return infer(batch[:mid]) + infer(batch[mid:])

            with target.open('a') as stream:
                for offset in range(0, len(pending), 256):
                    batch = pending[offset:offset + 256]
                    for row, text in zip(batch, infer(batch)):
                        record = dict(uid=row['uid'], prediction=text, **identity)
                        done[row['uid']] = record
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                    stream.flush()
                    print('EVAL', label, len(done), '/', len(rows), flush=True)
            torch.cuda.synchronize()
            timings[label] = dict(seconds=time.monotonic() - started, rows=len(pending))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        predictions[label] = done
        assert len(done) == len(rows)
    totals, sizes, durations = defaultdict(lambda: defaultdict(Counter)), Counter(), Counter()
    with gzip.open(a.output / 'numeric-evidence.jsonl.gz', 'wt') as stream:
        for row in rows:
            split = row['split']
            sizes[split] += 1
            durations[split] += row['duration']
            normalizer = ENGLISH if row['language'] == 'en' else normalize
            values = {label: counts(row['text'], predictions[label][row['uid']]['prediction'], normalizer)
                      for label in predictions}
            for label, count in values.items():
                totals[split][label].update(count)
            stream.write(json.dumps(dict(record_sha256=hashlib.sha256(row['uid'].encode()).hexdigest(),
                                         split=split, language=row['language'],
                                         duration=row['duration'], counts=values)) + '\n')
    sets = {}
    for split in sorted(sizes):
        sets[split] = dict(rows=sizes[split], hours=durations[split] / 3600,
                           language=next(r['language'] for r in rows if r['split'] == split),
                           models={label: dict(c, wer=100*c['errors']/c['words'],
                                               cer=100*c['char_errors']/c['chars'])
                                   for label, c in totals[split].items()})
    result = dict(status='complete', publication_authorized=False, rows=len(rows), sets=sets,
                  models=models, **identities, decoding=json.loads((a.output/'decoding.json').read_text()),
                  precision='FP32 weights, BF16 CUDA autocast, TF32 matrix multiplication disabled',
                  normalizers={'en':'Whisper EnglishTextNormalizer', 'other':'Recorded multilingual normalizer'},
                  empty_output_handling='Every manifest record is scored; empty hypotheses remain.',
                  timings=timings, prediction_sha256={k:sha(a.output/(k+'.jsonl')) for k in models},
                  numeric_evidence_sha256=sha(a.output/'numeric-evidence.jsonl.gz'))
    write(a.output / 'comparison.json', result)
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
