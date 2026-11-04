# Report-risk stopping v1: prospective execution contract

2026-10-23. Written before new path collection, fitting and fresh evaluation.

## Mainline question

Can a separate current-report-risk / continuation-gain head preserve useful
component-set acquisition savings without the premature report errors found in
the previous frozen 32-family experiment? That experiment motivates this change
only; none of its outcomes select the new models or thresholds.

The algorithm module is planning/termination over perception outputs, not a new
registration backend or flight controller. Generic ridge regression, thresholds,
temporal features and dual heads alone are not claimed as novel.

## Data and acquisition freeze

- Freeze existing component-set rank scorer and all RGB-D/backend/evaluator code.
- Existing scene IDs 0..15 train and 16..31 calibration only. Do not use old DEV
  32..47 or previous fresh evaluation families for learning/selection.
- For each family collect two complete real acquisition/refit trajectories:
  frozen rank scorer ignoring STOP, and greedy coverage ignoring STOP.
- Budget H=4 including H0, C=2 additional to C0: five additional acquisitions,
  six observed states per path. 384 train/cal fits, 192 states per split.
- Persist observed graph, report and decision before the next acquisition.
  Evaluate all states only after the entire actor trajectory has finished.
- Anonymous H0 reference IDs align successive reports; never physical truth IDs.

## Model and targets

Fixed normalized targets r_t=L_t/8 and g_t=(L_t-L_5)/8. g is realized full-budget
continuation benefit under the corresponding frozen acquisition policy, NOT
optimal Q, a hindsight best stopping time, or a calibrated probability.

Static features aggregate current declared risk, support, identity margin,
reported labels, displacement and marginal SD, stage and observed action scores.
Temporal additions compare current/previous and current/initial reports using
reference-aligned displacement changes, label flips and uncertainty/support
changes. No evaluator labels or future states are features.

Two-output standardized ridge heads with intercept; train-only mean/std;
alpha in [0.01,0.1,1,10], chosen by joint calibration MSE on both policies and all
six states; ties prefer larger alpha. Static and temporal heads separately fit.
Current risk is clipped at zero; continuation gain retains its sign.

Candidate stops when predicted risk <= tau AND predicted continuation gain <=
0.05*(5-t)/8. Otherwise acquire the frozen rank-selected first action and refit.
Budget exhaustion always stops. Threshold grid is fixed in config, -1 is an
explicit full-budget fallback. Remaining-cost term is the fixed acquisition
price, not a claimed optimal stopping Bellman value or measured flight cost.

## Matched controls and selection

Nine methods: dual_temporal (candidate), dual_static, risk_temporal (remove gain
gate), declared_rank (ordinary current report-risk threshold), fixed_rank
(fixed stage 0..5), original_rank (old zero-STOP scoring), full_rank,
declared_coverage, full_coverage.

Threshold/fixed-stage selection on calibration only: selected terminal aggregate
FP and bad-motion <= corresponding full-policy baseline; qualified TP >= it;
total task loss <= it (tolerance 1e-9). Among feasible choices minimize mean
acquisitions, then report loss, then smaller threshold/stage. Full-budget fallback
always feasible. These are empirical aggregate constraints on 16 families, not
per-family safety certification or population guarantees. Head MSE calibration
uses both policies; rank thresholds use only rank paths, coverage threshold only
coverage paths. No fresh-evaluation tuning.

## New evaluation and causal sharing

Freeze model weights, selected thresholds, config, contract and actor sources
before generating 32 fresh seeds 22362026..22362057. First16 original, next16
actual foreground-occluder stress; all16 original factorial cells in each.
Run both full acquisition policies, 384 actual fresh backend fits. STOP-only
variants share exact causal prefixes. Their actor consumes current observations
lazily and ends at first STOP; later computed states and labels are not inputs.
Prefix costs include mandatory return from that prefix. Clearly report shared
paths, not nine independent reruns or 288 independent scenes. Validate one
fresh selected endpoint with a physical acquire/refit/stop rerun.

Report every method, both conditions, qualified changes, FP, wrong displacement,
unknowns, task loss, acquisitions and return-inclusive travel. Primary contrasts:
candidate vs declared_rank, full_rank, full_coverage, original_rank, dual_static,
risk_temporal, fixed_rank. Exploratory 5000 paired family bootstrap, stratified
by condition for all32; unadjusted95% intervals, no confirmatory significance.
All families retained. Preserve failure and null results. No posthoc model edits.

## Decision boundary

Mainline advance requires quality/resource improvement over ordinary threshold
and fixed-budget controls on fresh families. Cost saving with additional report
errors is a tradeoff, not a win. Collapse to full-budget is evidence that this
head/selection cannot yet establish efficiency. If temporal/gain ablations do
not show incremental benefit, do not claim those mechanisms as paper innovation.
Engineering audits are necessary correctness checks, not algorithm contributions.
