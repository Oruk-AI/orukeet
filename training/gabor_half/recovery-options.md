# Recovery options and outcomes

Research notes, 6 September 2026. R9 through R14 are complete and unqualified.
Executed outcomes are separated below from prepared alternatives.

The fixed constraint remains the same: preserve the 12,288 selected Gabor
functions and train every other model parameter. Kernel-fit error alone has
not predicted recognition recovery. R8 improves the larger English endpoint
but worsens the multilingual mean, so further work must check both endpoints
and the existing individual guards.

## Reconstruct the affected convolution branches

[Jiang et al. (Interspeech 2023)](https://www.isca-archive.org/interspeech_2023/jiang23d_interspeech.html)
combine structured Conformer pruning with layerwise distillation. This directly
motivates the existing layer anchors, but their pruning results do not establish
that frozen Gabor replacement preserves accuracy.

[BRECQ (ICLR 2021)](https://arxiv.org/abs/2102.05426v2) reconstructs blocks after
quantization. [AdaRound (ICML 2020)](https://proceedings.mlr.press/v119/nagel20a.html)
also illustrates why closeness in weight space need not minimize task loss.
Both papers study different interventions from this experiment.

An inference from these methods is to give the trainable neighbors of the
replacements more capacity to compensate: the pointwise projections, free
depthwise rows and BatchNorm scale/bias. The pinned NeMo convolution branch
orders pointwise projection, GLU, depthwise convolution, normalization,
activation and output projection. Its source is
`nemo/collections/asr/parts/submodules/conformer_modules.py` at the recorded
NeMo-Speech revision. A bounded comparison could raise learning rates for these
parameters while keeping positive learning rates and ASR gradients everywhere
else. Another option is a branch reconstruction loss using the same original
teacher inputs on both branches, separating local approximation error from
upstream drift. The larger-neighbor-update option was run in R10; the same-input
branch alternative has not been run. Running statistics must stay
fixed, and the teacher must remain outside the student optimizer and exports.

R10 used 2e-5 for the 120 trainable convolution tensors, with ASR + 5 block + 5
branch loss. Its development mean improved, but larger English WER increased
by 0.281055 points. It also missed the larger multilingual mean and individual
guards. R11 therefore repeats the same initialization, sampling seed, mixture,
learning rates and horizon with 0.1 ASR + 50 block + 25 branch loss. Stronger
teacher supervision is a hypothesis; R10's result does not establish it will work.

R11's final checkpoint reduced the larger English regression and passed every
larger individual guard. Its multilingual mean remains 0.007812 points above
the original, and its sole development failure is one excess Finnish word error.
The exact receipt is `results/full-r11-0400.json`; the candidate remains unqualified.

## Average compatible recovered weights

[Model soups (ICML 2022)](https://proceedings.mlr.press/v162/wortsman22a.html)
studies averaging nearby fine-tuned weights into one inference model.
[Sparse Model Soups (ICLR 2024)](https://arxiv.org/abs/2306.16788v3) studies
averaging after pruning with a shared connectivity pattern. Their results make
weight averaging a plausible comparison, not a demonstrated ASR improvement.

Our candidates share the exact same frozen functions. A declared small set of
averages could combine their remaining weights while explicitly copying those
functions unchanged, verifying matching tokenizer assets, tensor layouts and
normalization buffers. This would produce one ordinary dense model, with no
ensemble at inference. It would still need the full kernel audit, development
checks and larger comparison. An average cannot inherit any parent's measured
accuracy or speed.

The prepared `average_recovered.py` helper has passed ten CPU unit tests and
checks matching inference configuration, tokenizer assets, tensor layouts and
buffers. A1 tested three declared direct averages of R2-0100, R7-0300 and
R11-0400. The run script records all weights before evaluation, preserves exact
fixed rows and uses the original qualification gates. Every executed result is
retained, and unrun larger comparisons are named. See the A1 section in the
recovery README. Results cannot be inferred from a weighted average of parent WERs.

All three A1 development means beat the original, but each misses individual
language guards. The lowest mean, 9.954121%, belongs to a1-0350 and receives
the declared diagnostic larger comparison. None can qualify from that result.
That larger comparison improves the multilingual mean by 0.019052 points but
regresses English by 0.046018, while individual guards pass. R12 instead continues
from R11-0400 for 800 steps with its existing teacher objective and original
training mixture. It retains every gate and was launched only after completing
A1 without qualification.

R12 again passes the larger English mean and individual guards, but misses
the multilingual mean by 0.007930 points. Both development exports miss only
FLEURS Latvian, by one excess error beyond the guard. A2 tests three declared
averages of R11-0400 and R12-0800 with R12 weights 35%, 50% and 65%. R11's sole
development miss is Finnish. The nearby checkpoints may average usefully, but
the generator cannot infer WER from their weights; all original checks remain
required. The exact choices and stopping rule appear in the recovery README.

A2-0650 passes all development checks and the larger multilingual mean. Its
larger English mean alone fails, by 0.012449 points; every individual guard
passes. R13 therefore starts from that average, halves R12's peak rates, keeps
its strong teacher coefficients, and runs 300 steps on 80% original mixture
plus 20% original English CV/FLEURS. This retains all data and tests an English
sampling change rather than another average. Its candidate set and stopping
rule are declared before training. No qualification is assumed.

R13 passes development at step 300, but misses both larger means: +0.027881
multilingual and +0.005728 English. English emphasis alone did not recover the
two endpoints together.

## Match transducer decisions

[Panchapagesan et al.](https://arxiv.org/abs/2011.06110) distill RNN-T models
through label and blank probabilities on the output lattice, including sparse
students. [Zeineldeen et al.](https://arxiv.org/abs/2303.05958) study full-sum
sequence distillation and the difficulties of matching different transducer
alignments. Neither paper studies this TDT Gabor intervention.

These results motivate checking a decision-level teacher signal if hidden-state
matching stalls. Our existing R4 posterior helper samples token and duration
distributions at lattice positions; it is not either paper's exact loss. R4
failed qualification. Combining that signal with the stronger R11/R12 anchors
is the R14 experiment: one encoder teacher forward, one ASR term and separate
token/duration checks. Four NeMo-container CPU tests verify the combined loss,
gradient ownership, unchanged layer-only behavior and padding/partition logic.
R14 started from A2-0650 on the original mixture, with all fixed Gabor rows and
qualification criteria preserved. Step 200 passes development and improves the
larger multilingual mean by 0.022459 points, but misses English by 0.006812.
Step 400 fails three development guards. A3 tests R14-200 with conservative
15%, 25% and 35% shares of R7-300, whose English mean improved by 0.060395
points while multilingual regressed by 0.020564. It uses the already tested
parameter-average implementation and retains the original selection rules.

All three A3 averages fail different development guards. The 15% variant's
larger diagnostic improves multilingual WER by 0.020189 points but regresses
English by 0.010911, more than R14. R15 therefore returns to R14-200 for a
200-step continuation using the shared teacher and R13's audited English
sampling mixture. Encoder/conv rates are halved; decoder/joint rates and loss
weights stay at R14 values. This tests a sampling change after posterior
recovery, with all original gates retained.

## Decision discipline

Finish the active comparison before choosing another intervention. Record the source, recipe and
candidate set before evaluating it. Keep every result, preserve the existing
criteria and use only the audited training membership. These exposed regressions
support debugging; repeated selection does not turn them into independent
evidence of generalization.
