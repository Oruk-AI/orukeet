#!/usr/bin/env python3
"""Describe writing-script errors without changing the Greek test or decoder."""
import argparse,hashlib,json,unicodedata
from datetime import datetime,timezone
from pathlib import Path


def greek_fraction(text):
    letters=[c for c in text if c.isalpha()]
    return sum('GREEK' in unicodedata.name(c,'') for c in letters)/max(1,len(letters))


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    seal=json.loads((a.experiment/'metadata/seal.json').read_text())
    manifest=Path(seal['sets']['lesbos_el']['path'])
    assert digest(manifest)==seal['sets']['lesbos_el']['sha256']
    rows=[json.loads(s) for s in manifest.read_text().splitlines()]
    refs={d['uid']:d for d in rows};models={}
    receipt=json.loads((a.experiment/'results/lesbos_el.complete.json').read_text())
    for model in ['parakeet','orukeet']:
        path=a.experiment/'results'/model/'lesbos_el.jsonl'
        assert digest(path)==receipt['prediction_sha256'][model]
        predictions=[json.loads(s) for s in path.read_text().splitlines()]
        assert len(predictions)==len(refs) and {d['uid'] for d in predictions}==set(refs)
        models[model]={'prediction_sha256':digest(path),
                       'empty_hypotheses':sum(not d['pred_text'].strip() for d in predictions),
                       'greek_reference_non_greek_script_hypotheses':sum(greek_fraction(refs[d['uid']]['text'])>=.9 and bool(d['pred_text'].strip()) and greek_fraction(d['pred_text'])<.1 for d in predictions)}
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
            'scope':'Post-inference descriptive script diagnostic on the complete 230-clip Greek dialect test; no row or reference changes.',
            'rows':len(rows),'reference_greek_letter_fraction_at_least_90pct':sum(greek_fraction(d['text'])>=.9 for d in rows),
            'method':'Count alphabetic Unicode characters whose Unicode name contains GREEK. A flagged nonempty hypothesis has less than 10% Greek letters while its reference has at least 90%.',
            'unicode_version':unicodedata.unidata_version,'manifest_sha256':digest(manifest),'models':models,'script_sha256':digest(Path(__file__)),
            'limits':'Script is a descriptive proxy, not a validated language-ID label. The models use their original automatic decoding without an oracle language constraint.'}
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('GREEK_SCRIPT_DIAGNOSTIC_COMPLETE',len(rows))


if __name__=='__main__':main()
