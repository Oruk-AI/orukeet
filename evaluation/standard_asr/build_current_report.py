"""Bind the short report to one checkpoint and reproduce every aggregate from counts."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
LANGUAGES = dict(bg='Bulgarian', cs='Czech', da='Danish', de='German', el='Greek', en='English',
    es='Spanish', et='Estonian', fi='Finnish', fr='French', hr='Croatian', hu='Hungarian',
    it='Italian', lt='Lithuanian', lv='Latvian', mt='Maltese', nl='Dutch', pl='Polish',
    pt='Portuguese', ro='Romanian', ru='Russian', sk='Slovak', sl='Slovenian', sv='Swedish', uk='Ukrainian')
MODEL = '031c8ddab4845aeced904a7cde8e8aa57993b2e344716cf83a545b079c473b56'
BASE = '3cbdc85877e668ca7b82d0d56770eb1fac76691f55d6b97545e8d61ca588d10d'
MODELS = ['parakeet', 'orukeet']
COUNTS = ['errors','words','substitutions','deletions','insertions','chars','char_errors','utterance_error']

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def write(path, value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def validate(folder, nrows, nsets):
    d=ROOT/'evidence'/folder; result=read(d/'comparison.json');audit=read(d/'hypotheses-audit.json')
    assert result['status']=='complete' and result['rows']==nrows and len(result['sets'])==nsets
    assert result['models']==dict(parakeet=BASE,orukeet=MODEL)
    assert result['timings']['parakeet']['rows']==result['timings']['orukeet']['rows']==nrows
    assert audit['status']=='passed' and audit['models']==result['models']
    assert audit['upstream_scoring_verified']['model_partition_pairs']==2*nsets
    assert result['numeric_evidence_sha256']==sha(d/'numeric-evidence.jsonl.gz')
    for name,digest in result['scoring']['code_sha256'].items(): assert sha(ROOT/name)==digest,name
    for name,digest in audit['inputs_sha256'].items(): assert sha(d/name)==digest,name
    seen=set();counts=defaultdict(Counter);sizes=Counter()
    with gzip.open(d/'numeric-evidence.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);uid=row['record_sha256'];assert uid not in seen;seen.add(uid)
            sizes[row['split']]+=1
            for model in MODELS:
                c=row['counts'][model];assert c['errors']==c['substitutions']+c['deletions']+c['insertions']
                counts[(row['split'],model)].update({k:c[k] for k in COUNTS})
    assert len(seen)==nrows
    for split,spec in result['sets'].items():
        assert sizes[split]==spec['rows']
        for model in MODELS:
            c=counts[(split,model)];reported=spec['models'][model]
            assert all(reported[k]==v for k,v in c.items())
            for metric,num,den in [('wer','errors','words'),('cer','char_errors','chars')]:
                assert abs(reported[metric]-100*c[num]/c[den])<1e-12
    return result

def pool(result, splits):
    members=[result['sets'][s] for s in splits];models={}
    for model in MODELS:
        c=Counter()
        for spec in members: c.update({k:spec['models'][model][k] for k in COUNTS})
        models[model]=dict(c,wer=100*c['errors']/c['words'],cer=100*c['char_errors']/c['chars'])
    wins=sum(s['models']['orukeet']['wer']<s['models']['parakeet']['wer'] for s in members)
    ties=sum(s['models']['orukeet']['wer']==s['models']['parakeet']['wer'] for s in members)
    return dict(splits=list(splits),rows=sum(s['rows'] for s in members),models=models,wins=wins,ties=ties,
                losses=len(members)-wins-ties,relative_wer_reduction_percent=100*(1-models['orukeet']['wer']/models['parakeet']['wer']))

def label(split):
    if split.startswith('librispeech_'):return 'LibriSpeech '+split.removeprefix('librispeech_').replace('_','-')
    if split.startswith('fleurs_'):return 'FLEURS '+LANGUAGES[split.rsplit('_',1)[1]]
    gsb=dict(agr='agriculture',ait='AI',art='arts',bio='biology',chn='Chinese accent',ecm='economics',
        eng='engineering',ent='entertainment',fin='finance',hum='humanities',ind='Indian accent',jpn='Japanese accent',law='law',
        med='medicine',mil='military',phl='Filipino accent',sct='Scottish accent',sgp='Singaporean accent')
    if split.startswith('gigaspeechbench_'):return 'GSB '+gsb[split.split('_')[1]]
    if split.startswith('eurospeech_'):return 'EuroSpeech '+split.rsplit('_',1)[1].upper()
    if split.startswith('voxpopuli_'):return 'VoxPopuli '+split.rsplit('_',1)[1].upper()
    return dict(golos_crowd_ru='Golos crowd RU',golos_farfield_ru='Golos far-field RU',
        nst_da_da='NST Danish',nst_sv_sv='NST Swedish',monsoon_en_in='Monsoon India',lesbos_el='Lesbos Greek')[split]

def wer_tex(spec,model):
    value=f"{spec['models'][model]['wer']:.2f}"
    other=MODELS[1-MODELS.index(model)]
    if spec['models'][model]['wer']<spec['models'][other]['wer']:value=r'\textbf{'+value+'}'
    return value

def main():
    standard=validate('standard-asr-r3-20260908',25705,27)
    domains=validate('domains-r3-20260908',12006,47)
    expected={'librispeech_test_clean','librispeech_test_other'}|{'fleurs_'+l for l in LANGUAGES}
    assert set(standard['sets'])==expected
    assert all(s['rows']==(230 if name=='lesbos_el' else 256) for name,s in domains['sets'].items())
    freeze=read(ROOT/'evidence/librispeech-ft-20260908/r3/export-audit.json')
    assert freeze['status']=='pass' and freeze['candidate_sha256']==MODEL
    assert freeze['frozen_gabor_rows_exact']==12288 and freeze['gabor_values_match_parent_and_original_functions']
    summaries={'fleurs':pool(standard,sorted(s for s in expected if s.startswith('fleurs_'))),
               'standard':pool(standard,sorted(expected)),
               'domains':pool(domains,sorted(domains['sets'])),
               'domain_english':pool(domains,sorted(s for s,v in domains['sets'].items() if v['language']=='en'))}
    assert summaries['fleurs']['rows']==20146 and summaries['domain_english']['rows']==5120
    macros={}
    for prefix,source,split in [('LibriClean',standard,'librispeech_test_clean'),('LibriOther',standard,'librispeech_test_other'),('FleursEnglish',standard,'fleurs_en')]:
        for model in MODELS:macros[prefix+model.title()+'WER']=f"{source['sets'][split]['models'][model]['wer']:.2f}"
    for prefix,key in [('FleursPooled','fleurs'),('DomainPooled','domains'),('DomainEnglish','domain_english')]:
        for model in MODELS:macros[prefix+model.title()+'WER']=f"{summaries[key]['models'][model]['wer']:.2f}"
    for model in MODELS:
        macros['FleursMacro'+model.title()+'WER']=f"{sum(standard['sets'][s]['models'][model]['wer'] for s in summaries['fleurs']['splits'])/25:.2f}"
    for name,key in [('FleursWins','fleurs'),('StandardWins','standard'),('DomainWins','domains'),('DomainEnglishWins','domain_english')]:macros[name]=str(summaries[key]['wins'])
    assert set(standard['sets']).isdisjoint(domains['sets'])
    macros['TestedWins']=str(summaries['standard']['wins']+summaries['domains']['wins'])
    macros['TestedSplits']=str(len(standard['sets'])+len(domains['sets']))
    macros['FleursPooledReduction']=f"{summaries['fleurs']['relative_wer_reduction_percent']:.1f}"
    (ROOT/'report/current-benchmark-values.tex').write_text('% Generated from audited r3 edit counts.\n'+''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    order=['librispeech_test_clean','librispeech_test_other']+sorted((s for s in expected if s.startswith('fleurs_')),key=label)
    table=[r'\begin{table}[!ht]',r'\centering',r'\caption{Complete LibriSpeech and FLEURS test partitions. WER and CER are percentages; bold identifies lower WER. The pooled FLEURS row sums errors and reference words over all 25 languages, including English. The macro row weights languages equally.}',r'\label{tab:current-standard}',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\renewcommand{\arraystretch}{1.02}',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Benchmark & Clips & \multicolumn{2}{c}{Parakeet} & \multicolumn{2}{c}{Orukeet}\\',r' & & WER & CER & WER & CER\\',r'\midrule']
    table_rows=[]
    for split in order:
        spec=standard['sets'][split];table_rows.append((label(split),spec))
    table_rows.append(('FLEURS pooled',summaries['fleurs']))
    macro=dict(rows=20146,models={m:{metric:sum(standard['sets'][s]['models'][m][metric] for s in summaries['fleurs']['splits'])/25 for metric in ['wer','cer']} for m in MODELS})
    table_rows.append(('FLEURS language macro',macro))
    for name,spec in table_rows:
        if name=='FLEURS pooled':table.append(r'\midrule')
        values=[wer_tex(spec,m)+' & '+f"{spec['models'][m]['cer']:.2f}" for m in MODELS]
        table.append(name+' & '+f"{spec['rows']:,}"+' & '+' & '.join(values)+r'\\')
        if name=='LibriSpeech test-other':table.append(r'\midrule')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'report/current-standard-table.tex').write_text('\n'.join(table)+'\n')
    domain_order=sorted(domains['sets'],key=label)
    table=[r'\begin{table}[!p]',r'\centering',r'\caption{WER (\%) on the fixed accent and domain sample. Each partition contains 256 recordings, except Lesbos (230). GSB denotes GigaSpeechBench; two-letter suffixes identify languages. Bold identifies lower WER. Both models use the same decoding and scoring protocol as Table~\ref{tab:current-standard}.}',r'\label{tab:current-domains}',r'\small']
    for idx,subset in enumerate([domain_order[:24],domain_order[24:]]):
        table += [r'\begin{minipage}[t]{0.49\linewidth}',r'\vspace{0pt}',r'\centering',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.10}',r'\begin{tabular}{lrr}',r'\toprule',r'Partition & Parakeet & Orukeet\\',r'\midrule']
        for split in subset:
            spec=domains['sets'][split]
            table.append(label(split)+' & '+' & '.join(wer_tex(spec,m) for m in MODELS)+r'\\')
        table += [r'\bottomrule',r'\end{tabular}',r'\end{minipage}'+(r'\hfill%' if idx==0 else '')]
    table += [r'\par\vspace{12pt}',r'\begin{tabular}{lrrr}',r'\toprule',r'Pooled comparison & Clips & Parakeet & Orukeet\\',r'\midrule']
    for name,key in [('All 47 partitions','domains'),('All 20 English partitions','domain_english')]:
        spec=summaries[key];table.append(name+' & '+f"{spec['rows']:,}"+' & '+' & '.join(wer_tex(spec,m) for m in MODELS)+r'\\')
    table += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'report/current-domains-table.tex').write_text('\n'.join(table)+'\n')
    md=['# Orukeet r3: paired recognition scores','',f'All Orukeet scores refer to NeMo SHA-256 `{MODEL}`. Parakeet is `{BASE}`. Both systems were decoded afresh on identical audio with FP32 weights, BF16 autocast and greedy-batch TDT. Lower WER is better.','',
        'Pooled WER is 100 times total substitutions, deletions and insertions divided by total normalized reference words. FLEURS pooling includes all 25 supported languages, including English. It is not an average of language WERs. Compound-boundary alignment can give each model a different reference-word denominator. CER uses normalized strings before compound alignment.','',
        'LibriSpeech test-other was used for final adaptation and checkpoint selection. The accent/domain comparison retains its prior fixed sample; 6,118 recordings were included in the preceding adaptation. Greek and Italian EuroSpeech retain the audited human transcript spans. No records are dropped from either comparison.','']
    for title,source,order,folder in [('Complete read-speech partitions',standard,order,'standard-asr-r3-20260908'),('Accent and domain sample',domains,domain_order,'domains-r3-20260908')]:
        md += ['## '+title,'','| Partition | Clips | Parakeet WER / CER | Orukeet WER / CER |','|:--|--:|--:|--:|']
        with (ROOT/'evidence'/folder/'scores.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['split','clips','hours']+[m+'_'+k for m in MODELS for k in ['wer','cer','errors','words']])
            for s in order:
                spec=source['sets'][s];md.append('| '+label(s)+' | '+str(spec['rows'])+' | '+' | '.join(f"{spec['models'][m]['wer']:.2f} / {spec['models'][m]['cer']:.2f}" for m in MODELS)+' |')
                writer.writerow([s,spec['rows'],spec['hours']]+[spec['models'][m][k] for m in MODELS for k in ['wer','cer','errors','words']])
        md += ['',f'[Full precision](../evidence/{folder}/scores.csv) · [Counts](../evidence/{folder}/numeric-evidence.jsonl.gz) · [Independent scoring audit](../evidence/{folder}/hypotheses-audit.json)','']
    md += ['## Pooled comparisons','','| Comparison | Clips | Parakeet errors / words | WER | Orukeet errors / words | WER | Wins / partitions |','|:--|--:|--:|--:|--:|--:|--:|']
    for name,key in [('FLEURS, 25 languages','fleurs'),('Accents/domains, 25 languages','domains'),('Accents/domains, English','domain_english')]:
        spec=summaries[key];md.append('| '+name+' | '+str(spec['rows'])+' | '+' | '.join(f"{spec['models'][m]['errors']:,} / {spec['models'][m]['words']:,} | {spec['models'][m]['wer']:.2f}" for m in MODELS)+f" | {spec['wins']} / {len(spec['splits'])} |")
    (ROOT/'docs/current-checkpoint-benchmarks.md').write_text('\n'.join(md)+'\n')
    receipt=dict(status='passed',publication_authorized=False,models=standard['models'],summaries=summaries,
                 checks=['Fresh matched decoding of both checkpoints','All 74 paired split results recomputed from per-record counts','Pooled WER recomputed from summed errors and reference words','Independent upstream scoring matches all 148 model/partition pairs'],
                 inputs_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ['standard-asr-r3-20260908','domains-r3-20260908'] for p in [ROOT/'evidence'/folder/f for f in ['comparison.json','hypotheses-audit.json','numeric-evidence.jsonl.gz']]})
    if '--verify-pdf' in sys.argv:
        text=unicodedata.normalize('NFKC',subprocess.check_output(['pdftotext','-layout',str(ROOT/'output/pdf/orukeet-technical-report.pdf'),'-'],text=True))
        lines=[' '.join(l.split()) for l in text.splitlines()]
        for name,spec in table_rows:
            target=' '.join([name,f"{spec['rows']:,}"]+[f"{spec['models'][m][metric]:.2f}" for m in MODELS for metric in ['wer','cer']])
            assert target in lines,target
        for s in domain_order:
            spec=domains['sets'][s];target=' '.join([label(s)]+[f"{spec['models'][m]['wer']:.2f}" for m in MODELS])
            assert any(target in line for line in lines),target
        abstract=text.split('Abstract',1)[1].split('A fixed structure',1)[0]
        assert all(macros[k] in abstract for k in ['FleursPooledParakeetWER','FleursPooledOrukeetWER','LibriCleanParakeetWER','LibriCleanOrukeetWER','LibriOtherParakeetWER','LibriOtherOrukeetWER','FleursEnglishParakeetWER','FleursEnglishOrukeetWER'])
        assert 'leaderboard' not in text.lower() and 'gabormer' not in text.lower()
        # Ignore line-break hyphenation when checking the rendered claim and setup.
        compact_abstract=''.join(abstract.split()).replace('-','')
        assert '031c8ddab484' in text
        for phrase in ['Final adaptation and checkpoint selection use LibriSpeech test-other.',
                       f"Orukeet outperforms Parakeet on {macros['TestedWins']} out of {macros['TestedSplits']} tested splits"]:
            assert ''.join(phrase.split()).replace('-','') in compact_abstract,phrase
        receipt['checks'].append('Rendered abstract and all 74 benchmark rows match the current checkpoint evidence')
        receipt['pdf_sha256']=sha(ROOT/'output/pdf/orukeet-technical-report.pdf')
    write(ROOT/'report/current-benchmark-validation.json',receipt)
    print(json.dumps({k:{'parakeet':v['models']['parakeet']['wer'],'orukeet':v['models']['orukeet']['wer'],'wins':v['wins'],'partitions':len(v['splits'])} for k,v in summaries.items()}))

if __name__=='__main__':main()
