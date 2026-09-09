<!-- orukeet-brand:start -->
<p><img src="../../docs/assets/team/oruk.png" alt="Oruk AI" width="184"></p>
<!-- orukeet-brand:end -->

# Private multilingual listening kit

<!-- orukeet-team:start -->
<p>
Nathan Roll<sup>1,2</sup> · Irene Yi<sup>1,2</sup> · Büşra Marşan<sup>1,2</sup><br>
Vianney Grenez<sup>1</sup> · Gabriel Stein<sup>4</sup> · Momcilo Mrkaic<sup>5</sup><br>
Pavle Padjin<sup>5</sup> · Vladimir Zeljkovic<sup>5</sup> · Calbert Graham<sup>1,3</sup>
</p>

<p><strong><sup>1</sup> Oruk AI</strong></p>
<table>
<tr>
<td align="center" valign="middle"><img src="../../docs/assets/team/stanford.png" alt="Stanford University" width="144"><br><sup>2</sup> Stanford University</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/cambridge.png" alt="University of Cambridge" width="144"><br><sup>3</sup> University of Cambridge</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/openwhispr.png" alt="OpenWhispr" width="40"><br><sup>4</sup> OpenWhispr</td>
<td align="center" valign="middle"><img src="../../docs/assets/team/hoid.png" alt="Hoid" width="76"><br><sup>5</sup> Hoid</td>
</tr>
</table>
<!-- orukeet-team:end -->

Three actual **Orukeet r3 Q8** transcriptions, displayed beside the recordings and their provider references. Open [index.html](index.html) locally, play each clip and compare the text.

The examples are French, Spanish and Latvian read speech from FLEURS, a public evaluation source used in this project. The three recordings were fixed before inference, and all outputs are retained. The release’s benchmark tables provide the quantitative comparisons and evaluation protocol. The page uses local audio and has no hosted service, uploads or analytics.

## Selection fixed before inference

[selection.json](provenance/selection.json) records the selection before any
model outputs were produced: the first physical data row, zero-based row 0, of
each original `data/{config}/test.tsv` at the pinned source revision. This is
the original TSV order, not an assertion about Parquet or Dataset Viewer order.
No recording was replaced after its result was inspected.

| Language / source configuration | Original file | Provider ID |
| --- | --- | --- |
| French / `fr_fr` | `7105431834829365765.wav` | 1829 |
| Spanish, Latin America / `es_419` | `7285658688146080595.wav` | 1770 |
| Latvian / `lv_lv` | `8857195911787819845.wav` | 1889 |

References and every prediction are preserved without editorial corrections.
The page also exposes the provider's normalized reference. No native speaker
has independently verified these examples. The run receipt records individual
calls, including errors if present; there is no best-of selection or aggregate
quality score.

## Source, rights and changes

The source is Google's [FLEURS dataset](https://huggingface.co/datasets/google/fleurs)
at revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd`. Its
[pinned provider card](https://huggingface.co/datasets/google/fleurs/blob/70bb2e84b976b7e960aa89f1c648e09c59f894dd/README.md)
identifies **CC BY 4.0** in the license metadata. The
[card record](provenance/source-card.json) preserves its URL, revision, hash and
verified license metadata. The full card download stays in the private `.tmp`
cache, outside this distributable demo tree.

Attribution: *FLEURS: Few-shot Learning Evaluation of Universal Representations
of Speech* (2022), Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod,
Siddharth Dalmia, Jason Riesa, Clara Rivera and Ankur Bapna; distributed by
Google. [Paper](https://arxiv.org/abs/2205.12446).
[CC BY 4.0 terms](https://creativecommons.org/licenses/by/4.0/).
No provider or speaker endorsement is implied.

Each `audio/*.source.wav` is the exact original WAV member extracted from the
commit-pinned provider archive. It is not trimmed or enhanced. For browser
playback, `audio/*.playback.wav` is decoded to mono 16 kHz PCM16 using the same
audio preparation code that Orukeet uses. This format conversion is the only
audio change; no speech was generated. References are the provider's text.

The per-clip provenance files contain stable source URLs, the selected original
TSV row and its hash, the full source TSV hash, member name, audio hash,
attribution, and provider archive metadata. Only the selected member was copied from each streamed archive. The
full archive's advertised LFS hash is recorded but **was not locally verified**;
the extracted WAV itself is fully hashed. See
[French provenance](provenance/fr_fr.json),
[Spanish provenance](provenance/es_419.json), and
[Latvian provenance](provenance/lv_lv.json).

These source-audio rights are separate from the model's weight license and the
repository's code license. See the [model card](../../MODEL_CARD.md) for model
scope and rights. Keep this kit private until the release review authorizes its
publication.

## Reproduce the exact selections

From the model repository, use its Orukeet 0.1.0rc1 environment and the verified
Q8 model with the NVIDIA native runtime:

```sh
.venv/bin/python demos/multilingual/fetch_sources.py
.venv/bin/python demos/multilingual/reproduce.py \
  --model /path/to/orukeet-v0.1.0rc1-q8.gguf \
  --runtime /path/to/metal-runtime \
  --device metal \
  --output demos/multilingual/rerun.json
```

The first command uses public, revision-pinned provider URLs without Hub
credentials. It streams the source archives and saves only the selected WAV
members; the transferred archives can be much larger than these three clips.
Full TSV/card downloads go to `.tmp/multilingual-sources`, outside `demos/`;
only the selected three source rows are retained in the demo's provenance.
It checks a previously recorded audio hash if provenance already exists.

The second command validates the released model hash through the Orukeet API,
transcribes the original source WAVs once in the fixed order, and retains every
result. It writes `rerun.json` and `rerun.html`, preserving the original
[receipt](receipt.json) and [listening page](index.html). The receipt includes
a portable invocation, hardware, backend, package version, code and library
hashes, original audio hashes, playback conversion and complete model output.
Exact machine-local command paths are kept only in the private `.tmp` cache.

Call durations are diagnostic receipts, not benchmarks. There is one model
initialization and no warm-up; the first call may include backend first-use
work. Each call includes file decoding, audio preparation, IPC and inference.
It excludes initialization, playback, microphone capture, UI and application
interaction. No speed claim is made from these three calls.
