#!/usr/bin/env python3
"""Build the benchmark report and tables from complete, verified comparisons."""
import argparse, csv, gzip, hashlib, json
from collections import Counter
from pathlib import Path
from plot_results import EURO, ACCENTS, DOMAINS

LANG={**EURO,'cs':'Czech','da':'Danish','es':'Spanish','hu':'Hungarian','nl':'Dutch',
      'pl':'Polish','ro':'Romanian','ru':'Russian','sv':'Swedish'}


def label(key):
    if key.startswith('eurospeech_'):return 'EuroSpeech · '+LANG[key.split('_')[-1]]
    if key.startswith('gigaspeechbench_'):
        code=key.split('_')[1]
        return 'GigaSpeechBench · '+{**ACCENTS,**DOMAINS}[code]
    if key.startswith('voxpopuli_'):return 'VoxPopuli validation · '+LANG[key.split('_')[-1]]
    return {'monsoon_en_in':'Monsoon · Indian English','golos_crowd_ru':'Golos · Russian, crowd',
            'golos_farfield_ru':'Golos · Russian, far field','nst_da_da':'NST · Danish',
            'nst_sv_sv':'NST mirror · Swedish','lesbos_el':'ILSP · Lesbos Greek dialect'}[key]


def interval(value):
    ci=value.get('wer_delta_pp_95ci')
    return f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '—'


def table(data,keys=None):
    lines=['| Split | Clips | Hours | Parakeet WER | Orukeet WER | Δ pp | 95% interval for Δ | Groups |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for key in keys or sorted(data):
        r=data[key];groups='—' if r['clusters'] is None else f"{r['clusters']:,}"
        lines.append(f"| {label(key)} | {r['rows']:,} | {r['hours']:.2f} | {r['parakeet']['wer']:.3f} | {r['orukeet']['wer']:.3f} | {r['wer_delta_pp']:+.3f} | {interval(r['uncertainty'])} | {groups} |")
    return '\n'.join(lines)


def endpoint_table(primary,strict=False):
    lines=['| Endpoint and scoring | Parakeet WER | Orukeet WER | Relative WER reduction | Δ pp (95% interval) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for key,name in [('gigaspeechbench_accent_macro','Six English accents'),('gigaspeechbench_domain_macro','Twelve English domains'),('monsoon_english','Monsoon Indian English'),('eurospeech_language_macro','16 EuroSpeech languages¹')]:
        variants=[('legacy','legacy')]
        if key!='eurospeech_language_macro':variants.insert(0,('english_standard','standard English'))
        for variant,description in variants:
            if strict:variant='strict_history_disjoint_english_standard' if variant=='english_standard' else 'strict_history_disjoint'
            r=primary['endpoints'][key][variant]
            lines.append(f"| {name} · {description} | {r['parakeet_wer']:.3f} | {r['orukeet_wer']:.3f} | {r['relative_wer_reduction_pct']:+.2f}% | {r['wer_delta_pp']:+.3f} {interval(r)} |")
    return '\n'.join(lines)


def error_components(primary):
    lines=['| Endpoint · standard English | Δ substitutions | Δ deletions | Δ insertions | Δ WER |',
           '| --- | ---: | ---: | ---: | ---: |']
    for key,name in [('gigaspeechbench_accent_macro','Six accents'),('gigaspeechbench_domain_macro','Twelve domains'),('monsoon_english','Monsoon')]:
        endpoint=primary['endpoints'][key]['english_standard'];sets=endpoint['sets'];values=[]
        for error in ['substitutions','deletions','insertions']:
            values.append(sum(100*(primary['english_standard_sets'][s]['orukeet'][error]-primary['english_standard_sets'][s]['parakeet'][error])/primary['english_standard_sets'][s]['parakeet']['words'] for s in sets)/len(sets))
        assert abs(sum(values)-endpoint['wer_delta_pp'])<1e-9
        lines.append('| '+name+' | '+' | '.join(f'{v:+.3f}' for v in values+[sum(values)])+' |')
    return '\n'.join(lines)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    evidence=a.root/'evidence/unseen-20260907';phases={};protocols={};runtimes={}
    for name,folder,expected in [('primary',evidence,35),('coverage',evidence/'coverage',10),('followup',evidence/'alignment-followup',2)]:
        d=json.loads((folder/'comparison.json').read_text());protocol=json.loads((folder/'protocol.json').read_text());runtime=json.loads((folder/'runtime.json').read_text())
        assert d['status']=='complete' and not d['missing_splits'] and len(d['sets'])==expected
        assert runtime['complete'] and runtime['final_model_hashes']==d['models']==runtime['model_hashes']
        assert hashlib.sha256((folder/'protocol.json').read_bytes()).hexdigest()==d['seal_sha256']==runtime['seal_sha256']
        assert set(protocol['sets'])==set(d['sets'])
        for field,source in [('scoring_script_sha256',a.root/'evaluation/unseen/compare.py'),('normalizer_script_sha256',a.root/'evaluation/unseen/metrics.py'),('decoded_history_audit_sha256',evidence/'decoded-history-audit.json')]:
            assert d[field]==hashlib.sha256(source.read_bytes()).hexdigest(),field
        phases[name]=d;protocols[name]=protocol;runtimes[name]=runtime
    primary=phases['primary'];rows=sum(d['totals']['rows'] for d in phases.values());hours=sum(d['totals']['hours'] for d in phases.values())
    assert rows==327888
    allsets={k:v for d in phases.values() for k,v in d['sets'].items()};assert len(allsets)==47
    failures={m:sum(s.get('failures',{}).get(m,0) for s in allsets.values()) for m in ['parakeet','orukeet']}
    strict_rows=sum(s['rows'] for d in phases.values() for s in d['strict_history_disjoint_sets'].values())
    durations=sum(s['declared_duration_mismatch_rows'] for d in phases.values() for s in d['integrity'].values())
    history_audio=sum(s['known_historical_pcm_overlap_rows'] for d in phases.values() for s in d['integrity'].values())
    numeric=[];slices=[];audio=Counter();record_count=0
    for phase,d in phases.items():
        for scoring,sets in [('legacy',d['sets']),('standard_english',d['english_standard_sets']),('strict_legacy',d['strict_history_disjoint_sets']),('strict_standard_english',d['strict_history_disjoint_english_standard_sets'])]:
            for key,r in sets.items():
                v={'phase':phase,'split':key,'label':label(key),'scoring':scoring,'rows':r['rows'],'hours':r['hours'],'groups':r['clusters'],'wer_delta_pp':r['wer_delta_pp'],'relative_wer_reduction_pct':r['relative_wer_reduction_pct']}
                ci=r['uncertainty'].get('wer_delta_pp_95ci',[None,None]);v.update(delta_ci_low=ci[0],delta_ci_high=ci[1])
                for m in ['parakeet','orukeet']:
                    for k,value in r[m].items():v[m+'_'+k]=value
                numeric.append(v)
        for dimension,groups in d['slices'].items():
            for group,r in groups.items():
                v={'phase':phase,'dimension':dimension,'group':group,'rows':r['rows']}
                if 'parakeet' in r:
                    v.update(hours=r['hours'],groups=r['clusters'])
                    for m in ['parakeet','orukeet']:
                        for k,value in r[m].items():v[m+'_'+k]=value
                slices.append(v)
        folder={'primary':evidence,'coverage':evidence/'coverage','followup':evidence/'alignment-followup'}[phase]
        with gzip.open(folder/'numeric-evidence.jsonl.gz','rt') as f:
            for line in f:
                r=json.loads(line);audio[r['pcm_sha256']]+=1;record_count+=1
    assert record_count==rows
    duplicates=sum(n-1 for n in audio.values());within=sum(s['duplicate_audio_content_rows'] for d in phases.values() for s in d['integrity'].values())
    write_csv(evidence/'per-split.csv',numeric);write_csv(evidence/'descriptive-slices.csv',slices)
    euro={k:v for k,v in primary['sets'].items() if k.startswith('eurospeech_')}
    accents=['gigaspeechbench_'+k+'_en' for k in ACCENTS]+['monsoon_en_in']
    domains=['gigaspeechbench_'+k+'_en' for k in DOMAINS]
    regressions=[label(k) for k,v in allsets.items() if v['wer_delta_pp']>0]
    prose=f'''# Orukeet versus stock Parakeet on new evaluation splits

We froze Orukeet R15-0100 and NVIDIA Parakeet TDT 0.6B v3, then transcribed **{rows:,} clips / {hours:.2f} decoded hours across 47 splits and all 25 supported languages**. Each model processed the same audio with the same NeMo decoder. Neither checkpoint was trained, tuned, blended, or selected using these results. These splits are now evaluation-exposed and must not be presented as fresh confirmation for future checkpoint selection.

The comparison has three separately sealed parts: a 35-split primary suite, a ten-split language-coverage extension, and two checks added after a reference-alignment defect surfaced. The extensions were sealed before their own inference, after some primary results were known. The tables preserve that distinction.

## Primary endpoints

Under standard English scoring, Orukeet reduces Monsoon WER by 3.42% relative. Its six-accent and twelve-domain macro WERs increase by 0.68% and 1.40%, respectively. The same directions hold after the exact-overlap exclusions.

In the separate nine-language extension, Orukeet has lower WER on nine of ten splits. Russian crowd speech regresses. Italian VoxPopuli is nearly unchanged, and both models perform poorly on the short Greek-dialect follow-up.

WER is a percentage; lower is better. Δ is Orukeet minus Parakeet in percentage points. Positive relative reduction favors Orukeet; a negative value is a regression. Each macro averages its fixed split WERs equally.

{endpoint_table(primary)}

¹ EuroSpeech has reference-alignment defects in sampled Greek and Italian source material. The 16-language macro remains the registered published-reference score, but it is not reliable headline evidence of general recognition accuracy. The French partition contains only one recording session, so the full macro has no cluster-bootstrap interval.

Standard English scoring uses `whisper-normalizer` 0.1.12. Legacy scoring uses the predeclared NFC/punctuation/character mapping from the earlier Orukeet evaluator. Both are reported; neither is an official leaderboard submission.

## English accents and conversation

Full published splits, standard English normalization.

{table(primary['english_standard_sets'],accents)}

![English accent comparisons](../report/assets/unseen/unseen-english-accents.png)

## English specialist domains

Full published splits, standard English normalization.

{table(primary['english_standard_sets'],domains)}

![English domain comparisons](../report/assets/unseen/unseen-english-domains.png)

### Where the error differences come from

The following decomposition uses the same fixed split weights as the English endpoints. Values are differences in errors per 100 reference words; the three components sum to Δ WER. The selected Levenshtein alignment determines the split between substitution, deletion and insertion, while total edit distance is invariant.

{error_components(primary)}

## EuroSpeech: the published-reference comparison

All official segments in the 16 selected language test splits remain in these legacy-normalized scores. The groups column counts recording sessions, not distinct speakers. Most partitions contain few sessions, which limits the intervals even where thousands of clips are available.

{table(euro)}

![EuroSpeech comparisons](../report/assets/unseen/unseen-eurospeech.png)

### The alignment defect

The high Greek and Italian scores triggered a source-integrity check. We re-downloaded two pinned Parquet shards and verified all **1,929 metadata-to-row mappings**. Sixteen deterministically spaced waveform checks matched the exact PCM fingerprints used at inference. The source's human and provider-ASR transcripts broadly agree with each other, while some supplied waveforms correspond to nearby, differently timestamped text in the same session. Several deterministic Greek matches were 39–46 seconds away; several Italian matches were 25–48 seconds away.

These checks indicate source audio/reference offsets, not a basis for replacing the references with either model's predictions. We retained every registered row and the original scoring. The [numeric diagnostic receipt](../evidence/unseen-20260907/source-alignment-audit.json) records the checks and text-offset search. It is a post-inference diagnostic, not an exhaustive annotation-quality audit. A broader [text-offset probe](../evidence/unseen-20260907/reference-offset-probes.json) selected 16 reference-defined clips per EuroSpeech language; its strong-offset heuristic flagged two Greek and three Italian examples, and none in the other languages. An unflagged probe does not establish correct alignment.

## Coverage extension: the other nine languages

This pass adds complete published test partitions for Danish and Swedish NST and Russian Golos, plus VoxPopuli validation in six languages. Its 102,345 scorable clips were selected mechanically to cover the languages absent from the primary suite. The two Golos mirrors contain 99 null references (98 crowd, one far field); those unannotated rows were excluded and counted before inference.

{table(phases['coverage']['sets'])}

![Supplementary language comparisons](../report/assets/unseen/unseen-language-coverage.png)

The Danish reduction is 0.750 percentage points, with a paired interval of −1.050 to −0.445. The Dutch interval crosses zero; the other eight splits lack complete grouping metadata. Their point estimates describe the evaluated partitions without an estimated sampling interval.

VoxPopuli contributes new clips, often from speakers present in previous evaluations. Complete speaker/session grouping is missing for several partitions, so their intervals are omitted. The Swedish NST mirror strips original audio and speaker IDs; its test partition is the published mirror partition, without a claim that it reproduces NST's original speaker split.

## Independent Greek and Italian follow-up

These two complete partitions were added after the parliamentary source problem, and sealed before either model ran on them. The Greek test measures the Lesbos dialect. It does not establish standard-Greek accuracy.

{table(phases['followup']['sets'])}

Italian VoxPopuli is nearly unchanged: 11.982% versus 11.969% WER. Both models perform poorly on the short Lesbos dialect utterances. Orukeet's lower WER there comes with higher CER (72.363%, versus 71.296% for Parakeet), and its paired WER interval crosses zero.

A [complete-split script diagnostic](../evidence/unseen-20260907/alignment-followup/greek-script-diagnostic.json) finds non-Greek-script output on 156 Parakeet and 165 Orukeet clips whose references are predominantly Greek; each model also returns 13 empty hypotheses. The diagnostic counts Unicode letters, so it describes a failure pattern rather than assigning language-ID ground truth. No language constraint or reference-driven decoding change was applied.

## Exact-overlap sensitivity

The conservative historical audit covered 7,501 inventory files and 13,059,318 manifest/prediction rows, including repeated entries. It indexed 2,271,894 audio basenames, 2,204,458 normalized texts, 2,111,264 sentence IDs, and 136,761 source-qualified speaker fingerprints. Historical reserve manifests were treated as exposed, even when unused. The old 23,038-recording suite informed checkpoint selection and is not counted as unseen here.

The strict analysis retains **{strict_rows:,} / {rows:,}** clips after excluding exact historical reference matches and available historical PCM matches. Extensions also flag exact reference matches to preceding sealed registries. This removes repeated text even when a different speaker may have recorded it. Primary endpoints under that restriction are:

{endpoint_table(primary,strict=True)}

### Historical audio fingerprint repair

A post-inference positive-control check found a representation mismatch: decoding a stored PCM16 waveform and converting it to PCM16 again changes its sample hash. The original check therefore missed two known historical recordings. We preserved the sealed index, verified the stored PCM hashes of all **23,113 available historical recordings / 52.46 hours**, and added fingerprints from their decoded samples and the current conversion path. Both known recordings then registered as overlaps; seeded synthetic noise did not. The [audit receipt](../evidence/unseen-20260907/decoded-history-audit.json) and [control results](../evidence/unseen-20260907/history-controls-repaired.json) record the repair.

All final comparisons use the union of the original and decoded-history fingerprints. Across the new suites, this check finds **{history_audio}** historical PCM matches. The complete-split recognition scores are unchanged; exact matches are excluded only from the strict sensitivity analysis. This verifies the available historical recordings, not every waveform referenced by the much larger manifest inventory.

The [full split table](../evidence/unseen-20260907/per-split.csv) contains every complete/strict and legacy/standard-English score, including corpus CER, utterance error, word denominators and substitution/deletion/insertion counts. [Descriptive slices](../evidence/unseen-20260907/descriptive-slices.csv) cover language, source, available gender/region/accent metadata, and clip duration. These slices have no added independence claims or multiple-comparison correction.

## What the experiment establishes

The new sources and retained source IDs were absent from recorded Orukeet adaptation and checkpoint-selection inputs. Available exact audio fingerprints were checked against the historical audit. NVIDIA's public pretraining inventory is not a complete record-level manifest, so inherited pretraining overlap cannot be ruled out. Dataset release dates do not establish when the original recordings were made. Cross-corpus acoustic near-duplicates and voice identities were not exhaustively checked.

Across all legacy-scored splits, {len(regressions)} have higher Orukeet WER. Those regressions remain in the tables and count exports. The English tables use the more informative standard normalization, with the alternate legacy values available in the CSV. This comparison does not establish uniformly better accuracy across domains, languages, recording conditions, or inference formats.

## Execution and uncertainty

Both checkpoints ran on the same NVIDIA A100-SXM4-40GB, with FP32 parameters, CUDA BF16 autocast, batch size 32, mono 16 kHz float WAV inputs, and identical greedy TDT decoding (`greedy_batch`, `max_symbols=10`, durations 0–4). Audio is sorted by duration identically; model order alternates between splits. Permanent inference failures are scored as empty hypotheses and remain in the denominator. Observed failures: Parakeet **{failures['parakeet']}**, Orukeet **{failures['orukeet']}**.

Confidence intervals use 10,000 paired ordinary cluster-bootstrap draws with replacement. The units are Monsoon speakers, EuroSpeech recording sessions, GigaSpeechBench source recordings, or the available source speaker IDs. Macro intervals independently resample within each fixed split and preserve equal split weights. Fewer than two groups, or incomplete group metadata, yields no interval. Counts below 20 groups require particular care. Per-split intervals are unadjusted for multiple comparisons.

Decoded durations determine hours; **{durations}** primary metadata-duration discrepancies were recorded while retaining the complete published waveforms. The integrity checks found **{history_audio}** available historical PCM matches and **{duplicates}** extra exact-audio occurrences across the full run ({within} within splits). Duplicates are retained in complete-split scores. The numeric exports make their fingerprints auditable.

The final checkpoint hashes match the starting hashes for all three runs. Timings in the runtime receipts include I/O and dataloader overhead and overlap with source preparation. They are execution records, not a controlled inference-speed claim. These accuracy results apply to NeMo source checkpoints; Q8/F16 native conversions require their own matched evaluation. The providers' utterance/segment partitions do not measure full-recording segmentation, streaming latency, diarization, or timestamp accuracy.

## Reproduce and inspect

- [Evaluation code and commands](../evaluation/unseen/README.md), including independent NumPy count/interval reproduction
- [Primary protocol](../evidence/unseen-20260907/protocol.json) and [source revisions](../evidence/unseen-20260907/sources.json)
- [Coverage protocol](../evidence/unseen-20260907/coverage/protocol.json) and [follow-up protocol](../evidence/unseen-20260907/alignment-followup/protocol.json)
- [Primary results](../evidence/unseen-20260907/comparison.json), [coverage results](../evidence/unseen-20260907/coverage/comparison.json), [follow-up results](../evidence/unseen-20260907/alignment-followup/comparison.json)
- Transcript-free counts: [primary](../evidence/unseen-20260907/numeric-evidence.jsonl.gz), [coverage](../evidence/unseen-20260907/coverage/numeric-evidence.jsonl.gz), [follow-up](../evidence/unseen-20260907/alignment-followup/numeric-evidence.jsonl.gz)

Each completion receipt binds the manifest, predictions, audio, checkpoint, and protocol hashes. The scorer verifies one-to-one membership again. Numeric evidence contains error counts and hashed grouping/record keys; raw audio, reference text, hypotheses, and personal profiles remain outside the release materials. All repositories and staging artifacts remain private for review.

Sources: [NVIDIA Parakeet](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3), [Monsoon](https://huggingface.co/datasets/VoiceArena/MonsoonASR-Open-ASR-leaderboard-en-IN), [GigaSpeechBench](https://huggingface.co/datasets/speechcolab/GigaSpeechBench), [EuroSpeech](https://huggingface.co/datasets/disco-eth/EuroSpeech), [VoxPopuli](https://huggingface.co/datasets/facebook/voxpopuli), [Danish NST](https://huggingface.co/datasets/alexandrainst/nst-da), [Swedish NST mirror](https://huggingface.co/datasets/jzju/nst), [Golos crowd mirror](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd), [Golos far-field mirror](https://huggingface.co/datasets/bond005/sberdevices_golos_100h_farfield), [ILSP Lesbos](https://huggingface.co/datasets/ilsp/lesbian-speech-corpus).
'''
    dest=a.root/'docs/unseen-benchmark.md';dest.write_text(prose)
    receipt={'status':'complete','rows':rows,'hours':hours,'splits':47,'languages':25,'strict_rows':strict_rows,
             'failures':failures,'duplicate_audio_extra_rows':duplicates,'within_split_duplicate_audio_extra_rows':within,
             'report_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'comparison_sha256':{name:hashlib.sha256((folder/'comparison.json').read_bytes()).hexdigest() for name,folder in [('primary',evidence),('coverage',evidence/'coverage'),('followup',evidence/'alignment-followup')]}}
    (evidence/'report-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
