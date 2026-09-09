#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', choices=['english', 'voxpopuli', 'coral', 'ftspeech', 'rixvox', 'rixvox_dev'], required=True)
    args = p.parse_args()
    repo = {'english': 'ylacombe/english_dialects', 'voxpopuli': 'facebook/voxpopuli',
            'coral': 'alexandrainst/coral', 'ftspeech': 'alexandrainst/ftspeech',
            'rixvox': 'KBLab/rixvox', 'rixvox_dev': 'KBLab/rixvox'}[args.source]
    revision = 'ed3d69abb765fd304ccdcc646ec6ca3c49740d15' if args.source == 'english' else None
    info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
    names = [f.rfilename for f in info.siblings if f.rfilename.endswith('.parquet')]
    if args.source == 'voxpopuli':
        names = [n for n in names if '/test-' in n and not n.startswith('multilang/')]
    elif args.source == 'coral':
        names = [n for n in names if '/test-' in n and 'read_aloud' in n]
    elif args.source == 'ftspeech':
        names = [n for n in names if '/test_' in n]
    elif args.source == 'rixvox':
        names = [f.rfilename for f in info.siblings if f.rfilename.startswith('data/test')]
    elif args.source == 'rixvox_dev':
        names = [f.rfilename for f in info.siblings if f.rfilename.startswith('data/dev')]
    if not names:
        raise RuntimeError('No matching published Parquet files: ' + repr([f.rfilename for f in info.siblings][:30]))
    out = args.root / 'data/goal_v2_sources' / args.source
    print('FETCHING', repo, info.sha, len(names), flush=True)
    out.mkdir(parents=True, exist_ok=True)
    receipt = {'repository': repo, 'revision': info.sha, 'files': names,
               'bytes': sum(f.size or 0 for f in info.siblings if f.rfilename in names)}
    if receipt['bytes'] > 25 * 1024**3:
        raise RuntimeError('Unexpectedly large selected test source: ' + str(receipt['bytes']))
    (out / 'source_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    snapshot_download(repo, repo_type='dataset', revision=info.sha,
                      local_dir=out, allow_patterns=names + ['README.md'], max_workers=4)
    (out / '_READY').write_text(info.sha + '\n')
    print('HF_SOURCE_READY', args.source, flush=True)


if __name__ == '__main__':
    main()
