"""Pair pinned FLEURS references and official LibriSpeech transcripts with audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

FLEURS_REPO = 'hf-audio/open-asr-leaderboard-multilingual-datasets'
FLEURS_REVISION = 'b791fc9151221e5b7e59c6c2dfa4dee09dda3cb7'
LANGUAGES = 'bg cs da de el en es et fi fr hr hu it lt lv mt nl pl pt ro ru sk sl sv uk'.split()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import numpy as np
    import soundfile as sf
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    a.output.mkdir(parents=True, exist_ok=True)

    def fleurs(language):
        path = f'datasets/{FLEURS_REPO}@{FLEURS_REVISION}/data/fleurs/{language}_test.parquet'
        target = a.output / f'fleurs-{language}-references.json'
        if target.exists():
            return json.loads(target.read_text())
        with HfFileSystem().open(path, 'rb', block_size=1024*1024) as stream:
            parquet = pq.ParquetFile(stream)
            records = parquet.read(columns=['file_name', 'duration', 'text']).to_pylist()
            assert len(records) == parquet.metadata.num_rows
        target.write_text(json.dumps(records, ensure_ascii=False) + '\n')
        print('REFERENCES', language, len(records), flush=True)
        return records

    with ThreadPoolExecutor(max_workers=6) as pool:
        reference_sets = dict(zip(LANGUAGES, pool.map(fleurs, LANGUAGES)))
    rows, sources = [], {}

    def add(split, language, uid, path, text, duration=None):
        assert path.is_file(), path
        audio, rate = sf.read(path, dtype='float32')
        assert rate == 16000 and audio.ndim == 1 and np.isfinite(audio).all()
        seconds = len(audio) / rate
        if duration is not None:
            assert abs(seconds - duration) < 0.002, (path, seconds, duration)
        rows.append(dict(uid=uid, split=split, language=language, text=text,
                         audio_filepath=str(path), duration=seconds,
                         pcm_sha256=hashlib.sha256(audio.astype('<f4').tobytes()).hexdigest(),
                         reference_sha256=hashlib.sha256(text.encode()).hexdigest()))

    for language, records in reference_sets.items():
        split = f'fleurs_{language}'
        assert len(records) == len({r['file_name'] for r in records})
        for row in records:
            path = a.audio_root / 'data/fleurs' / language / 'test' / (Path(row['file_name']).stem + '.flac')
            add(split, language, split+':'+Path(row['file_name']).stem, path, row['text'], row['duration'])
        sources[split] = dict(repository=FLEURS_REPO, revision=FLEURS_REVISION,
                             path=f'data/fleurs/{language}_test.parquet', rows=len(records),
                             reference_file_sha256=sha(a.output/f'fleurs-{language}-references.json'))

    for split, folder, expected in [
        ('librispeech_test_clean', 'data/goal_v2_sources/libri/LibriSpeech/test-clean', 2620),
        ('librispeech_test_other', 'data/librispeech/LibriSpeech/test-other', 2939),
    ]:
        transcripts = sorted((a.audio_root / folder).rglob('*.trans.txt'))
        count = 0
        for transcript in transcripts:
            for line in transcript.read_text().splitlines():
                uid, text = line.split(' ', 1)
                add(split, 'en', split+':'+uid, transcript.parent/(uid+'.flac'), text)
                count += 1
        assert count == expected, (split, count)
        sources[split] = dict(source='Official LibriSpeech test partition and original .trans.txt transcripts',
                             rows=count, transcript_files_sha256={str(x.relative_to(a.audio_root)):sha(x) for x in transcripts})
    rows.sort(key=lambda r: (r['split'], r['uid']))
    assert len(rows) == len({r['uid'] for r in rows})
    manifest = a.output/'manifest.jsonl'
    with manifest.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')
    receipt = dict(status='prepared', publication_authorized=False, rows=len(rows), splits=len(sources),
                   selection='Complete published test partitions; all records retained, no selection by model output.',
                   sources=sources, manifest_sha256=sha(manifest), script_sha256=sha(Path(__file__)),
                   hours=sum(r['duration'] for r in rows)/3600,
                   audio='Existing mono 16 kHz FLAC files, decoded float32 PCM hashes checked before inference.',
                   abstract_comparisons=['librispeech_test_clean','librispeech_test_other','fleurs_en'],
                   additional_abstract_summary='Equal-language macro over de, es, fr, it, pt FLEURS test partitions')
    (a.output/'preparation.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('PREPARED', len(rows), receipt['hours'], flush=True)


if __name__ == '__main__':
    main()
