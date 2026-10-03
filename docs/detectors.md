# Detector reference

Every threshold in `rldoctor` is a module-level constant with a comment explaining the
choice. This page collects them in one place, along with the reasoning and the source.

If you disagree with a number, that is a feature: the intent is that you can argue with a
threshold rather than with the tool. Open an issue with the run that changed your mind.

**A note on all of them:** thresholds are calibrated against documented failure modes,
against the twelve simulated scenarios in `rldoctor/simulate.py`, and against five public
GRPO training logs (see the README). Five runs is enough to have found five real bugs and
is nowhere near enough to be a calibration set. They remain a starting point informed by
the literature, and the honest way to use them is as a prompt to look, not as a verdict to
act on blindly.

Real-run data is still the single most valuable thing this project could receive, and a
run where a verdict is *wrong* is worth more than ten where it is right.

---

## `advantage_collapse` — zero-variance groups

**Failure.** GRPO centres rewards inside each group of `G` rollouts from the same prompt.
If every rollout earns the same reward, the centred advantage is exactly zero for all `G`
of them and that group's contribution to the policy gradient is exactly zero.

**Signal.** Fraction of degenerate groups, `f`. TRL logs it directly as
`frac_reward_zero_std`. When it is absent, `rldoctor` estimates it from a bounded reward
series under a binomial model, `P(degenerate) = p^G + (1-p)^G`, and labels the estimate as
such. Because rollouts from one prompt are positively correlated, that estimate is a
*lower* bound — if it already looks bad, reality is worse.

**Thresholds.** Stated as the cost multiplier per effective rollout, `1/(1-f)`:

| `f` | you pay | severity |
|---|---|---|
| 0.30 | 1.4× | INFO |
| 0.50 | 2.0× | WARNING |
| 0.75 | 4.0× | CRITICAL |

**The part that matters.** The detector splits the diagnosis by mean pass rate:

- pass rate ≥ 0.85 → **saturated**: groups are all-correct, the data is too easy, fix is
  harder prompts and a curriculum targeting the 0.3–0.7 band where group variance peaks.
- pass rate ≤ 0.15 → **too hard**: groups are all-wrong, fix is easier data, a bigger
  group so rare successes can appear at all — and checking the verifier, because an
  over-strict or crashing grader produces the identical signature.

Same symptom, opposite fix. Reporting `f` alone is not actionable.

**Sources.** DAPO ([arXiv:2503.14476](https://arxiv.org/abs/2503.14476)) for dynamic
sampling; *Advantage Collapse in GRPO*
([arXiv:2605.21125](https://arxiv.org/abs/2605.21125)).

---

## `entropy_collapse` — exploration exhausted

**Failure.** Policy entropy decays monotonically by default. Some decay is healthy: the
policy is committing to what it learned. Collapse is when it keeps falling *after* reward
stops improving — at which point rollouts within a group are near-identical, GRPO has
nothing left to rank, and the remaining budget buys nothing.

**Why a decay threshold does not work.** Entropy always decays, so "entropy has fallen
X%" fires on every healthy run eventually. `rldoctor` is driven by *arrival time*
instead: how soon entropy reaches the floor, relative to the run length so far.

| condition | severity (reward flat) | severity (reward rising) |
|---|---|---|
| already below floor | CRITICAL | WARNING |
| arrives within 0.25× run length | CRITICAL | WARNING |
| arrives within 1.0× run length | WARNING | INFO |
| further out, or not declining | OK | OK |

The floor is 10% of the *initial* entropy, where "initial" is anchored to the first ≤25
observations. A fraction-of-window baseline drifts as a run grows, and in live monitoring
a baseline that follows the series it is anchoring cannot detect a decline against it.

**The projection.** Entropy decay is multiplicative. `stats.project_threshold` fits both
a linear and a log-linear model, keeps whichever has the smaller residual in the original
units, and reports which was used.

Measured on the bundled `entropy_collapse` scenario, where the floor (0.0516) is truly
crossed at step 161:

| asked at | log-linear fit | linear-only fit | truth |
|---|---|---|---|
| step 100 | 156 | 130 | 161 |
| step 150 | 158 | 157 | 161 |

Early in the decay — which is when a warning is still worth acting on — the linear
projection reports roughly half the remaining time. The two converge as the curve
flattens, which is exactly when the warning no longer helps.

**Cross-checks.** A detected level shift (detrended CUSUM, ≥3σ) is reported separately —
a cliff rather than a slope usually means a config change, a resumed checkpoint, or a
reward-function bug landing mid-run, and none of those is fixed by tuning an entropy
bonus.

**Sources.** Clip-Cov / KL-Cov (*The Entropy Mechanism of RL for Reasoning LMs*); DAPO
clip-higher; *Understanding and Preventing Entropy Collapse in RLVR*
([arXiv:2605.11491](https://arxiv.org/abs/2605.11491)).

---

## `reward_hacking` — reward/eval divergence

**Failure.** The training reward is a proxy; the held-out score is the goal. When the
proxy climbs and the held-out score stalls, the policy has found something the verifier
rewards and the task does not.

**Signal.** The *hacking gap*: both series anchored to their own starting level and
divided by their own robust spread, then subtracted. Raw differences are meaningless
because reward and accuracy live on different scales.

| condition | severity |
|---|---|
| reward rising, held-out score **falling** | CRITICAL |
| reward rising, held-out flat, gap ≥ 0.30 | CRITICAL |
| reward rising, held-out flat, gap ≥ 0.15 | WARNING |
| rank correlation < 0 over ≥6 evals | WARNING |

**Requires a held-out series.** Without one this check skips and says so. It does not
attempt to infer hacking from training metrics alone, because it cannot.

**Sources.** *Auditing Reward Hackability in Code RL Training Environments*
([arXiv:2606.16062](https://arxiv.org/abs/2606.16062)); *LLMs Gaming Verifiers*
([arXiv:2604.15149](https://arxiv.org/abs/2604.15149)).

---

## `length_pathology` — length hacking and truncation

Two failures share one series.

**Truncation poisoning.** Responses hitting `max_response_length` are usually scored as
failures, so the reward measures your length cap rather than your policy — and every
truncated sample injects a systematically wrong advantage. WARNING at 5%, CRITICAL at
15%. Fix is overlong filtering (mask them out of the loss) before raising the cap.

**Length hacking.** Length growth alone is *not* a pathology — reasoning models are
supposed to think for longer as they improve. The detector requires three things
together: growth ≥ 50%, length–reward rank correlation ≥ 0.60, and held-out gain that is
*disproportionate* — under 25% of the length gain.

That proportionality test replaced a binary "did the eval improve?" check, which was too
blunt: real length hacking usually comes with a small genuine gain alongside a large one
bought with tokens.

With no held-out series the detector caps itself at INFO and says that it cannot tell
earned reasoning from padding. It does not accuse you on training metrics alone.

**Sources.** DAPO overlong filtering; Singhal et al., *A Long Way to Go: Investigating
Length Correlations in RLHF* ([arXiv:2310.03716](https://arxiv.org/abs/2310.03716)).

---

## `clip_saturation` — trust region eating the update

A clipped token contributes no gradient, so the clip fraction is the share of your update
that was discarded. INFO at 10%, WARNING at 20%, CRITICAL at 40% — at which point your
effective learning rate bears no relation to the one in your config.

**Asymmetry** is the more interesting half. When lower-bound clipping exceeds upper-bound
clipping by ≥3× (and is itself ≥2%), the update suppresses tokens far more readily than
it promotes them. Low-probability tokens — exactly the ones exploration needs — are
clipped away before they can be reinforced. This is the mechanism DAPO's clip-higher
addresses, and it feeds directly into entropy collapse; when entropy is falling at the
same time, the detector says so.

---

## `kl_drift` — drift from the reference policy

Two opposite failures. **Runaway drift**: KL grows while the held-out score does not, and
you pay for capability regression off-task with nothing in return (INFO at 0.5, WARNING
when high and accelerating, CRITICAL at 2.0 with progress stalled). **A strangled
policy**: peak KL below 1e-4 means the coefficient is doing the training, and it is not
training — also worth a WARNING, and worth checking that your reference model is really
the SFT init and not a stale copy of the policy.

The reported quantity is KL *per unit of held-out progress*: how much drift you are
spending per point of score. Acceleration is measured by comparing the second-half slope
against the first.

---

## `gradient_pathology` — NaNs, dead steps, vanishing updates, spikes

**Non-finite values.** NaN or inf → CRITICAL; every step after the first one is wasted
wall-clock even if the loss curve kept plotting. Only values the log actually *carried*
as non-finite count. A `nan` in the aligned series usually means the metric was not logged
at that step — every run that evaluates on a different cadence from training is full of
them — and an earlier version of this detector counted the padding and declared a healthy
run numerically dead.

**Exactly-zero updates.** Steps whose gradient norm is not small but literally `0.0`.
INFO at 20% of steps, WARNING at 40%, CRITICAL at 70%. This is not a decay: those steps
did not move the policy at all, while their rollouts were generated and paid for. In GRPO
it is the fingerprint of degenerate groups, and it is visible even when
`frac_reward_zero_std` is not logged. On one public run it was 88% of steps, rising from
40% in the first quarter to 93% in the last. These steps are the only thing besides
degenerate groups that feeds the waste estimate, because they are the only other thing
that genuinely buys nothing.

**Vanishing gradient.** Below 5% of the anchored early-run value → WARNING, and if
degenerate groups explain it the detector says so and points you there instead — the
gradient is the symptom, not the disease.

**Spikes.** Three or more values that are *both* beyond 8 robust sigma *and* at least 3x
the median. Both conditions are needed: on a tightly clustered series the MAD is tiny, so
a value only twice the median can clear 8 sigma, and an 80-step real run was flagged for
six such "spikes" before the ratio condition was added. Detection uses median/MAD rather
than mean/std, since spikes would otherwise inflate the very scale used to find them.

---

## `reward_composition` — which term is actually training the policy

The policy does not optimise the sum of your reward functions; it optimises whichever
component has variance it can move. A component with a large mean and no variance is a
constant, and a constant is invisible to a policy gradient.

Shares are computed on **recent** data (last 50%) using MAD² rather than variance, since
the question is which term is driving the policy now, not which dominated during warm-up.

CRITICAL when an auxiliary term (`format`, `tag`, `xml`, `length`, `lang`, `style`,
`think`) holds ≥90% of recent reward variance; WARNING at ≥70%. A component below 2% is
reported as dead — WARNING if it looks like your task reward, INFO if it is a saturated
format reward, which is expected behaviour.

**Deliberate limitation.** Below ~150 logged points this check reports information rather
than a verdict. Measured across scenarios, the variance-share signature of a benign
warm-up transient (77%) and a genuine takeover (82–97%) overlap too much to separate
reliably on a short window. Rather than pick a heuristic that would misfire in one
direction or the other, the detector waits and says why.

---

## `plateau` — the run finished a while ago

Uses a non-parametric trend test over the recent 35% of the run rather than "did the last
value beat the best value", which on a noisy curve declares progress roughly half the time
by construction.

WARNING when the recent trend is flat or declining *and* the best checkpoint is ≥30% of
the run behind you. The report converts the post-peak steps into wall-clock hours and
GPU-hours, and the prescription is to roll back to the best checkpoint and stop on
evidence rather than on a step budget.

---

## Adding a detector

One file in `src/rldoctor/detectors/`, one entry in `BUILTIN_DETECTORS`, and one scenario
in `simulate.py` with its expected detection recorded in `EXPECTED_DETECTIONS`. If the
check came out of a real log, add its shape to `tests/test_real_log_shapes.py` too —
that file exists because real logs have shapes the simulator does not. The
scenario is what makes the check trustworthy: it proves the detector fires on the failure
it claims and — via the healthy scenario — that it stays quiet otherwise.

The base class handles input validation, so a detector body can assume its `requires`
fields exist. Return `Severity.SKIPPED` with a named missing field rather than guessing.
