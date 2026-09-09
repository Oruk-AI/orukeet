"""Matched three-model diagnostic on a sealed, role-labelled audio manifest."""
import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--metric-code', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.metric_code))
    from metrics import counts, normalize, ENGLISH
    from nemo.collections.asr.models import ASRModel
    from nemo.utils import logging
    logging.set_verbosity(logging.ERROR)
    torch.set_num_threads(8)
    torch.manual_seed(20260918)
    np.random.seed(20260918)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    a.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in a.manifest.open()]
    assert len(rows) == len({r['uid'] for r in rows})
    rows.sort(key=lambda r: (r['duration'], r['uid']))
    manifest_sha = sha(a.manifest)
    for r in rows:
        data, rate = sf.read(r['audio_filepath'], dtype='float32')
        assert rate == 16000 and data.ndim == 1
        assert hashlib.sha256(data.astype('<f4').tobytes()).hexdigest() == r['training_pcm_sha256']
    decode_path = a.output / 'decoding.json'
    hashes, all_predictions, timings = {}, {}, {}
    decode = json.loads(decode_path.read_text()) if decode_path.exists() else None
    for label, path in [('parakeet', a.base), ('parent', a.parent), ('candidate', a.candidate)]:
        hashes[label] = sha(path)
        target = a.output / (label + '.jsonl')
        prior = [json.loads(line) for line in target.open()] if target.exists() else []
        assert len(prior) == len({r['uid'] for r in prior})
        predictions = {r['uid']: r for r in prior}
        assert set(predictions) <= {r['uid'] for r in rows}
        for r in prior:
            assert r['model_sha256'] == hashes[label] and r['manifest_sha256'] == manifest_sha
        pending = [r for r in rows if r['uid'] not in predictions]
        if pending:
            model = ASRModel.restore_from(str(path), map_location='cuda').eval()
            model.freeze()
            current = OmegaConf.to_container(model.cfg.decoding, resolve=True)
            if decode is None:
                decode = current
                write_json(decode_path, decode)
            assert current == decode
            assert current['strategy'] == 'greedy_batch' and current['greedy']['max_symbols'] == 10
            model.change_decoding_strategy(OmegaConf.create(current))
            started = time.monotonic()

            def infer(batch):
                try:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        hyps = model.transcribe([r['audio_filepath'] for r in batch], batch_size=16,
                                                num_workers=4, verbose=False)
                    assert len(hyps) == len(batch)
                    return [(h.text if hasattr(h, 'text') else str(h), None) for h in hyps]
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    if len(batch) == 1:
                        raise
                    middle = len(batch) // 2
                    return infer(batch[:middle]) + infer(batch[middle:])

            with target.open('a') as f:
                for start in range(0, len(pending), 256):
                    batch = pending[start:start + 256]
                    for r, (text, failure) in zip(batch, infer(batch)):
                        result = dict(uid=r['uid'], pred_text=text, failure=failure,
                                      model_sha256=hashes[label], manifest_sha256=manifest_sha)
                        predictions[r['uid']] = result
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f.flush()
                    print('EVAL', label, len(predictions), '/', len(rows), flush=True)
            torch.cuda.synchronize()
            timings[label] = dict(seconds=time.monotonic() - started, rows=len(pending))
            del model
            gc.collect()
            torch.cuda.empty_cache()
        assert len(predictions) == len(rows)
        all_predictions[label] = predictions

    totals = defaultdict(lambda: defaultdict(Counter))
    sizes = Counter()
    hours = Counter()
    roles = {}
    numeric = []
    for r in rows:
        split = r['split']
        sizes[split] += 1
        hours[split] += r['duration'] / 3600
        roles[split] = r['evaluation_role']
        normalizers = {'legacy': normalize}
        if r['lang'] == 'en':
            normalizers['standard_english'] = ENGLISH
        item = dict(record_sha256=hashlib.sha256(r['uid'].encode()).hexdigest(), split=split,
                    evaluation_role=r['evaluation_role'], counts={})
        for variant, normalizer in normalizers.items():
            item['counts'][variant] = {}
            for label in all_predictions:
                values = counts(r['text'], all_predictions[label][r['uid']]['pred_text'], normalizer)
                totals[(split, variant)][label].update(values)
                item['counts'][variant][label] = values
        numeric.append(item)
    results = {}
    for (split, variant), models in totals.items():
        entry = results.setdefault(split, dict(rows=sizes[split], hours=hours[split], evaluation_role=roles[split]))
        rates = {}
        for label, counts_ in models.items():
            rates[label] = dict(counts_, wer=100 * counts_['errors'] / counts_['words'],
                                cer=100 * counts_['char_errors'] / counts_['chars'])
        rates['candidate_minus_parent_wer_pp'] = rates['candidate']['wer'] - rates['parent']['wer']
        rates['candidate_minus_parakeet_wer_pp'] = rates['candidate']['wer'] - rates['parakeet']['wer']
        entry[variant] = rates
    report = dict(status='complete', models=hashes, manifest_sha256=manifest_sha,
                  script_sha256=sha(__file__), normalizer_sha256=sha(a.metric_code / 'metrics.py'),
                  matched_input='Identical mono 16 kHz PCM16 FLAC files for all three models',
                  precision='BF16 autocast, original FP32 weights, TF32 matmul disabled',
                  decoding=decode, rows=len(rows), failures=0, sets=results, timings=timings,
                  scope='Fixed sample diagnostics. Training-exposed rows measure fit; they are not an unseen benchmark.')
    write_json(a.output / 'comparison.json', report)
    import gzip
    with gzip.open(a.output / 'numeric-evidence.jsonl.gz', 'wt') as f:
        for item in numeric:
            f.write(json.dumps(item) + '\n')
    print('EVALUATION_COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
