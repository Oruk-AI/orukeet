#!/usr/bin/env python3
"""Conservatively index every historical manifest, including unused reserves.

Run before any new inference. Store only fingerprints in the overlap index;
raw references and identities remain in their original private files.
"""
import argparse, hashlib, json, os, sqlite3, unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def text_key(s):
    s=unicodedata.normalize('NFC',s).casefold()
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in s).split())


def fingerprint(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def keys(row):
    out=[]
    for name in ['text','text_original_manifest','reference','ref_text']:
        value=row.get(name)
        if isinstance(value,str) and text_key(value): out.append(('text',fingerprint(text_key(value))))
    for name in ['audio_filepath','audio_path']:
        if row.get(name):
            out.append(('basename',fingerprint(Path(row[name]).name)))
    for name in ['source_row_id','sentence_id','fleurs_id','pcm_sha256','audio_sha256']:
        if row.get(name): out.append((name,fingerprint(row[name])))
    for name in ['speaker_id','client_id']:
        if row.get(name):out.append(('speaker',fingerprint(str(row.get('src',''))+':'+str(row[name]))))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    dest=a.out/'history.sqlite'
    if dest.exists():raise SystemExit('Refusing to replace a history index')
    db=sqlite3.connect(dest);db.execute('PRAGMA journal_mode=OFF');db.execute('PRAGMA synchronous=OFF');db.execute('PRAGMA cache_size=-1048576')
    db.execute('CREATE TABLE exposure (kind TEXT, fingerprint TEXT, PRIMARY KEY(kind,fingerprint)) WITHOUT ROWID')
    inventory=[];counts=Counter();duplicate_content={};indexed_keys=set()
    # Source caches are not proof of exposure; all actual manifests and saved
    # predictions outside them are indexed, including unused reserve manifests.
    skip={'data','models','shards','env','__pycache__','export-deps','unseen_20260907','.git'}
    paths=[]
    for base,dirs,files in os.walk(a.root):
        dirs[:]=sorted(d for d in dirs if d not in skip and not d.startswith('native-'))
        for n in files:
            if n.endswith(('.jsonl','.json')):paths.append(Path(base)/n)
    for path in sorted(paths):
        digest=sha(path);entry={'path':str(path.relative_to(a.root)),'bytes':path.stat().st_size,'sha256':digest}
        if digest in duplicate_content:
            entry['duplicate_of']=duplicate_content[digest];inventory.append(entry);continue
        duplicate_content[digest]=entry['path'];records=0;pending=[]
        try:
            with path.open() as f:
                for line in f:
                    try: row=json.loads(line)
                    except json.JSONDecodeError:continue
                    if not isinstance(row,dict) or not any(k in row for k in ['audio_filepath','audio_path']):continue
                    values=keys(row)
                    if not values:continue
                    records+=1
                    for value in values:
                        if value not in indexed_keys:indexed_keys.add(value);pending.append(value)
                    if len(pending)>10000:
                        db.executemany('INSERT OR IGNORE INTO exposure VALUES (?,?)',pending);pending=[]
            if pending:db.executemany('INSERT OR IGNORE INTO exposure VALUES (?,?)',pending)
        except UnicodeDecodeError:entry['not_utf8']=True
        entry['indexed_records']=records;inventory.append(entry);counts['records_with_repetition']+=records
        if records: print('INDEXED',entry['path'],records,flush=True)
    db.commit();summary=dict(db.execute('SELECT kind,COUNT(*) FROM exposure GROUP BY kind'));db.close()
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'policy':'All historical manifests and line-oriented saved predictions; conservative unused reserves included. Source audio caches alone are not treated as exposure.',
      'index_sha256':sha(dest),'unique_keys':summary,'counts':dict(counts),'files':inventory,
      'limitations':['Exact metadata/text fingerprints; not an acoustic near-duplicate or cross-corpus speaker-identity proof.',
        'NVIDIA aggregate pretraining sources do not provide complete record-level provenance.']}
    (a.out/'history-audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('AUDIT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
