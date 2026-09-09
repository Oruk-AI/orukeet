# What useful open model releases ship

Research for Orukeet's private release preparation, September 2026.

The strongest lesson is practical: make the first useful result easy to get,
then make it easy to inspect. A good paper cannot compensate for a broken
install. A pleasant demo cannot compensate for an unexplained training set.
Orukeet needs both across ASR workloads. A single application integration
should not define the model release.

## How these references were chosen

We examined seven model projects or families, one widely adopted native port,
and our actual native runtime. The group includes small speech models,
multilingual recognizers, research-first language models and a compact TTS
release. That gives us different release problems to learn from.

The [GitHub snapshot](repository-snapshot.json) records primary-source interest
and reuse proxies at the time of research. Whisper had 108,539 stars,
whisper.cpp 53,467, Moonshine 11,013, OLMo 6,666, SmolLM 3,890,
DeepSeek-R1 92,022, Qwen3-ASR 3,476 and Kokoro 8,706. These counts are neither
unique users nor evidence that a particular README caused adoption. Older
projects have had more time to accumulate them. We selected recognizable,
used projects and inspected what they actually offer; this is a qualitative
comparison, not a causal growth study.

## The cases

| Reference | What the release puts in a user's hands | Decision for Orukeet |
| --- | --- | --- |
| [Whisper](https://github.com/openai/whisper) | A package, CLI and short Python example; model-size and memory tradeoffs; model card and paper; an explicit MIT license for code and weights. | Put a working local transcription command before the training story. State RAM, hardware and scope beside performance claims. Our training transparency must be assessed separately from Whisper's adoption. |
| [whisper.cpp](https://github.com/ggml-org/whisper.cpp) | Native builds, a sample recording, quantization, platform-specific acceleration, bindings and application examples. This is a port, not a new model release. | Treat native integration as a deliverable. Ship pinned binaries and checksums, exercise real worker pipes on each OS, and keep a short public-audio smoke test. |
| [Moonshine](https://github.com/moonshine-ai/moonshine) | A microphone quickstart, cross-platform bindings, streaming-oriented documentation and explicit exceptions to its model licensing. | Reduce setup friction and distinguish windowed preview from native streaming. Don't claim mobile or Raspberry Pi support merely because a compatible architecture exists. List model-specific terms instead of a vague “open” badge. |
| [OLMo](https://github.com/allenai/OLMo) / [OLMo-core](https://github.com/allenai/OLMo-core) | Training configurations, data mixtures, intermediate checkpoints, logs, inference examples and evaluation tools. The older repository clearly directs users to the current trainer. | Include intermediate Orukeet checkpoints and the actual launch arguments. Preserve failed experiments and separate historical instructions from the current path. Make the active entry point obvious. |
| [SmolLM](https://github.com/huggingface/smollm) | Compact models accompanied by public data-mixture and training details, a standard inference interface and linked research resources. | Explain the adaptation recipe without requiring readers to reconstruct it from a container. Keep inference dependencies small and training dependencies separate. Avoid declaring a result reproducible because a weight file is downloadable. |
| [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) | A technical report, downloadable full and distilled variants, evaluation settings, usage recommendations and explicit parent-model license notes. | State lineage prominently and make evaluation settings part of the claim. Orukeet has one validated deployment format; a large menu of speculative variants would only create maintenance work. |
| [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) | Offline and serving examples, a fine-tuning entry point, containers and evaluation settings including precision, decoding and language selection. | Publish an exact reference evaluator and a portable local API. A Docker image is useful for training, but should not be required for a desktop user to transcribe a file. |
| [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) | Audible samples, a notebook-sized example, versioned model facts and hashes, training-cost disclosure and a recognizable voice. | Give the report some personality and make artifacts easy to identify. Use real recordings with their rights documented. Don't borrow Kokoro's low-cost narrative without a complete Orukeet compute ledger. |
| [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp/tree/v0.1.0) | The runtime Orukeet actually uses, with model conversion and platform builds. It is much newer and smaller than the adoption references above. | Pin the source and archives, test the C ABI and preserve notices. Its existence is not evidence that our model works on every published target. |

## Decisions that follow from the research

**Lead with general ASR.** The first page should identify the Parakeet
adaptation and show a working transcription command. Then show how the same
model fits batch jobs, media pipelines and application workers. Its measured
average improvement is the evidence for calling it a better Parakeet. “Better at every accent” and
“fast on every computer” are unsupported and would make a poor first impression
once somebody tries a weak language or an ordinary CPU.

**Show a result that can be challenged.** Put the 7.70% relative WER reduction
beside its absolute values, evaluation size and scope. Show the 4.24% reduction
without Latvian nearby. Retain the original failed 10% promotion target. The
technical report should make an informed reader more confident in the
measurement process, not more impressed by adjectives.

**Keep the first install short.** The candidate has a Python package, an
explicit download command, local-file transcription and a context-managed API
that reuses the recognizer. A separate model/runtime path makes offline use
visible. Private authentication is a review-stage requirement; a future public
quickstart must be tested without an authorized account after publication.

**Ship the tools behind the graph.** Preserve model hashes, the sealed metric
registry, original and deployed aggregate results, paired-bootstrap code and
the figure builder. Use descriptive filenames. A reviewer should be able to
tell the source checkpoint from Q8, FP16, stage 2 and an abandoned conversion.
The mel-filterbank failure belongs in the report because it explains an
otherwise easy-to-repeat export mistake.

**Separate artifact availability from scientific reproducibility.** The code
and final weights are not the entire training story. Include data sources,
selection and filtering, dependency versions, launch arguments, initialization,
and known missing state. The OSI definition asks for data information, code and
parameters sufficient to study and modify the system; popularity does not
answer that question. Orukeet's upstream foundation pretraining remains a
boundary we must describe. [Open Source AI Definition](https://opensource.org/ai/open-source-ai-definition).

**Make metadata useful.** The model card should name the base model, supported
languages, artifact license, task and evaluation scope. Hugging Face parses
structured model metadata and evaluation results, but our custom macro must
not be presented as an official dataset leaderboard score. For this candidate,
the full custom evaluation lives in the report and JSON receipts.
[Model-card documentation](https://huggingface.co/docs/hub/model-cards).

**Invite reproducible bug reports.** Ask for OS, architecture, runtime, model
hash, a minimal audio example the reporter can share, and whether the failure
occurs after a reload. Audio is sensitive; a bug template should not demand
someone's private conversation. Provide a clear report path before inviting
wide use.

## A concrete launch package

| Item | Review artifact |
| --- | --- |
| First result | README install and local-file commands; Python API |
| Application use | General ASR usage guide; persistent worker API; separate private OpenWhispr integration |
| Weights | Pinned Q8, source and intermediate checkpoint references |
| Technical evidence | Report, figures, aggregate receipts, evaluation scripts |
| Training transparency | Historical recipes, actual arguments, source and filtering audit |
| Rights | Code license, proposed final-weight license, attribution and source terms |
| Support | Contribution guide and reproducible bug template |
| Announcement | A restrained draft with one accuracy claim and qualified speed numbers |

The review should happen in that order: install it, transcribe, inspect a
failure, reproduce a metric, then judge the announcement. No public launch,
post, repository visibility change or PR merge is part of this preparation.

## Draft announcement

> We're preparing Orukeet, a multilingual Parakeet fine-tune for general ASR. Its 714 MB native model reduces our 20-language macro WER from
> 16.58% to 15.30% on 23,038 previously evaluated recordings. Excluding Latvian,
> which contributes the largest gain, the relative reduction is 4.24%.
>
> In OpenWhisper, an 11-second clip takes 111 ms on an M5 Max with Metal and
> 52 ms on an A100 with CUDA, as warm transcription medians. CPU performance
> is less dramatic. The report includes it, along with the original failed
> promotion target, training recipe, model hashes and evaluation code.
>
> We wanted a model people could actually run and inspect. Orukeet is ready
> for private review; public release will follow that review.

Before using this draft publicly, replace the review-stage closing with the
approved release status and verified public links. Do not turn the GPU
microbenchmark into a claim about complete dictation latency.
