# Data, provenance and license scope

Updated September 8, 2026 for private review of **Orukeet r3**.
The inventory covers the earlier adaptation and the final continuation.
Source revisions come from the saved download receipts; license descriptions
come from the providers' cards, papers and license files.

## Earlier adaptation

The continuation samples from **1,716,650 rows / 2,676.18 hours**. Common Voice
22 and FLEURS supply 2,663.80 hours. English Dialects adds 5,973 clips / 9.82
hours; SpeechOcean762 adds 2,254 clips / 2.57 hours. Those two extensions get
8% of sampling probability. The inherited stage-1/2 campaign used CV/FLEURS.
The final continuation uses the additional sources listed below.

| Source | Provider and attribution | Provider's license | Role |
| --- | --- | --- | --- |
| Common Voice 22 | Mozilla and volunteer contributors; [distribution](https://commonvoice.mozilla.org/data), [current data catalog API](https://mozilladatacollective.com/api-reference/docs) | CC0 | Historical training, continuation, evaluation |
| FLEURS | Google; Conneau et al.; [dataset card](https://huggingface.co/datasets/google/fleurs) | CC BY 4.0 | Historical training, continuation, evaluation |
| English Dialects | Google, OpenSLR 83 authors; [original corpus](https://www.openslr.org/83/), [HF restructuring](https://huggingface.co/datasets/ylacombe/english_dialects) | CC BY-SA 4.0 | Continuation and English guardrails |
| SpeechOcean762 | Beijing Kingline Data Technology and corpus authors; [OpenSLR 101](https://www.openslr.org/101/) | CC BY 4.0 | Continuation and English guardrails |

The English Dialects source revision is
`ed3d69abb765fd304ccdcc646ec6ca3c49740d15`. Recorded filenames, speaker IDs and
transcripts from the training manifest were spot-checked against that public
restructuring. Historical extension manifests retain their original
`conditional_contract` labels. We preserve that history and record the public
source findings here rather than silently rewriting provenance.

FLEURS in the inherited manifests is labeled `fleurs_recovered_2026_09_04`.
We preserve those manifests and their hashes; we do not invent a missing
original download revision. The Common Voice release is 22, not the newer
release currently shown by a download site. The private `training-inputs`
artifact contains the exact sampled-manifest inputs and configuration.

## Parent FT-4035 continuation

The parent pass used **223,452 recordings / 371.47 hours across 24 splits**.
The [sealed plan](../evidence/regression-ft-20260907/training-plan.json) records
every split's accepted count and manifest hash. Greek and Italian EuroSpeech
use the recorded human transcript spans; the
[formatting record](../evidence/regression-ft-20260907/text-normalization.json)
specifies the final labels. Source audio and transcripts retain their original
terms.

| Source used in the parent pass | Attribution and source record | Recorded terms |
| --- | --- | --- |
| GigaSpeechBench: 17 English accent/domain splits | SpeechColab, Yujie Tu and coauthors; [dataset](https://huggingface.co/datasets/speechcolab/GigaSpeechBench/tree/680d3057641b7507a1ef14974407c7b0a7964e64), [paper](https://arxiv.org/html/2606.28884v3) | The paper's ethics statement identifies Creative Commons source audio. The pinned dataset has no license file or card identifying the variants. |
| EuroSpeech: Bulgarian, Greek and Italian | DISCO at ETH Zurich and the EuroSpeech contributors; [pinned card](https://huggingface.co/datasets/disco-eth/EuroSpeech/blob/7a55bbcee3933f5d9817bb2d4c37399d596f543d/README.md#licensing-information) | Country-specific parliamentary terms; the card uses `license: other`. Its source table links the respective copyright and reuse records. |
| Golos Crowd | SberDevices; Alexander Denisenko, Angelina Kovalenko, Fedor Minkin and Nikolay Karpov; [pinned restructuring](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd/tree/e634b6b810e4d30c81b4c6d8262379fe8b9f708c) | [Public license with attribution and conditions reserved](https://github.com/sberdevices/golos/blob/master/license/en_us.pdf), a custom license requiring attribution, change notices and its stated adaptation-license conditions. |
| NST Danish | Nordisk Språkteknologi, National Library of Norway and Alexandra Institute; [pinned card](https://huggingface.co/datasets/alexandrainst/nst-da/blob/0f14ad2005e0aab8f56cf3213b7689da1faf23c2/README.md) | CC0 1.0. |
| NST Swedish | Nordisk Språkteknologi and National Library of Norway; `jzju/nst` restructuring; [pinned card](https://huggingface.co/datasets/jzju/nst/blob/ade45482b1fa163b34177963c1e6f4d29621e24f/README.md) | CC0 1.0. |
| Lesbian Speech Corpus: dialect speech from Lesbos | ILSP/Athena Research Center and corpus contributors; [pinned dataset](https://huggingface.co/datasets/ilsp/lesbian-speech-corpus/tree/6bb115fa67491dd74ba81f59e8a0fbb94384ea72), [institutional resource page](https://www.ilsp.gr/en/ai-for-modern-greek-dialects/) | The dataset card is empty and states no reuse license. The institutional page describes the speech resources as open access. |

The GigaSpeechBench subsets are AGR, AIT, ART, BIO, CHN, ECM, ENG, ENT, FIN,
HUM, IND, LAW, MED, MIL, PHL, SCT and SGP English. GigaSpeechBench is a separate
release from GigaSpeech; this inventory does not transfer the latter's license
badge to the benchmark. Golos's custom license is separate from the Creative
Commons licenses. The Lesbos card's missing license and GigaSpeechBench's
unspecified Creative Commons variants remain explicit source-record gaps.

Exact repository revisions and downloaded file hashes are in the
[initial source registry](../evidence/unseen-20260907/sources.json),
[coverage registry](../evidence/unseen-20260907/coverage/sources.json) and
[follow-up registry](../evidence/unseen-20260907/alignment-followup/sources.json).
These records preserve source attribution without distributing the training
audio or transcript archive.

## Final r3 continuation

The final pass uses **2,939 LibriSpeech test-other recordings** for 168 updates.
LibriSpeech is distributed by OpenSLR under CC BY 4.0; attribution is to
Panayotov and the corpus contributors. The targets preserve the model's native
casing and punctuation while correcting reference words. The same split is used
for checkpoint selection. [The r3 recipe](../training/librispeech_ft/README.md)
and [freeze audit](../evidence/librispeech-ft-20260908/r3/export-audit.json)
record the run. The release's LibriSpeech/FLEURS and accent/domain comparisons
are decoded afresh with both r3 and stock Parakeet.

## Earlier formatting and filtering

JSONL rows retain original text, normalized text, source, language, audio path,
duration and available speaker/sentence/accent metadata. Training text is NFC
normalized. The pipeline rejects invalid duration/text, duplicate audio IDs,
missing files and protected speaker or sentence identities. Duration is limited
to 0.4–20 seconds for training. A global multilingual Common Voice speaker
exclusion removes 74,618 additional rows before sampling.

Languages and sources receive square-root-duration weights. Free-text accent
labels are pooled when fewer than 200 clips are available; the sampler caps
upsampling at 3× within-source empirical frequency. Unknown labels remain
unknown. The input configuration and per-pool hashes are in the
[sampling report](../evidence/sampling_report.json).

All retained extension clips were decoded, resampled to 16 kHz mono PCM16 and
checked for finite, nonempty audio and normalized-PCM duplicates. The much
larger inherited pool received metadata and file-existence checks. It has not
received a complete acoustic near-duplicate audit. Accent labels are noisy and
self-reported; a count of label strings is not a count of distinct accents.
SpeechOcean includes children and adults speaking prompted learner English.

## Other evaluation sources

| Source | Terms and attribution | Handling in this candidate |
| --- | --- | --- |
| LibriSpeech | [OpenSLR 12](https://www.openslr.org/12/), CC BY 4.0; Panayotov et al. | IDs, references and predictions in private evidence |
| VoxPopuli | [Dataset card](https://huggingface.co/datasets/facebook/voxpopuli), CC0 metadata plus European Parliament raw-data terms | Retain the source/legal-notice link |
| FTSpeech | [Dataset card](https://huggingface.co/datasets/alexandrainst/ftspeech), [Folketinget custom terms](https://www.ft.dk/da/aktuelt/tv-fra-folketinget/deling-og-rettigheder) | No blanket open license assigned to its audio |
| RixVox | [KBLab card](https://huggingface.co/datasets/KBLab/rixvox), CC BY 4.0; cite the Swedish Parliament | Parliamentary references may not be verbatim |

The source evidence bundle also retains source-discovery material for rejected
or unused candidates. Membership in `sealed_metric_registry.json` determines
which recordings contribute to the historical deployment evaluation. The later
47-split benchmark uses its own saved protocols and source registries. Download
scripts do not, by themselves, establish evaluation membership.

The committed metric-evidence archives contain generated error/word counts,
reference fingerprints and pseudonymous cluster ranks. It omits audio, transcript
text and speaker names. These statistical records are offered under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), attributed to Oruk AI.
They reproduce the comparisons without redistributing the reference speech.

The listening demo includes three attributed CC BY 4.0 FLEURS examples, each
with its source and playback WAV. Other multilingual evaluation audio is
excluded. The separate proof demo includes an attributed public JFK benchmark
fixture. Private evidence archives
retain transcripts and source identities needed to reproduce the comparison.
The prepared public-copy allowlist includes the transcript-free metric package
and its source notices. Exact training manifests and raw prediction/reference
archives remain reviewer-only in the legacy private repository. The package
reproduces statistical comparisons; replaying ASR and training also requires
those inputs and separately obtained source audio. See the
[copy manifest](../release/copy-manifest.json) for this boundary.

## Release licenses

The prepared r3 NeMo, Q8 and F16 weights and fitted kernels use
[CC BY-SA 4.0](../LICENSE-WEIGHTS). They retain NVIDIA's CC BY 4.0 foundation
attribution and the source credits in [NOTICE.md](../NOTICE.md). Code is
[MIT](../LICENSE); the native SDK retains its own license files. Generated
metric records use CC BY 4.0. The report's affiliation marks remain the property
of their organizations, as recorded in the
[logo sources](../report/assets/affiliations/SOURCES.md).

These licenses identify the release artifacts. They do not replace the
source-specific terms in the inventory. The prepared package includes the
adaptation code, model parameters, available provenance and transcript-free
evaluation counts. Dataset audio is acquired from its providers; selected demo
fixtures carry their own source notices. Exact research manifests, raw
predictions and optimizer archives remain private.
