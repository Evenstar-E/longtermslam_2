# Optional continuation v1: prospective contract

Written before model fitting and generation of new evaluation seeds. This is a
bounded planning-module experiment, not a new registration/control backend.

Question: can continuation value permitting stopping after the next observation
improve resource/quality over the prior full-budget target and frozen rank STOP?
Known Bellman recursion/ridge/cross-fitting are not novelty claims by themselves.

## Fixed inputs and training

Reuse ONLY the completed actual paths for old families0..15 train /16..31cal in
results/report_risk_stopping_v1/training_*. Rank and coverage each six observed
states, 192 states per split but16 independent families. No new training fits.
Never use either prior fresh32 evaluation batches to fit or select anything.
Static29/temporal61 observed report features, action scorer, backend and risk head
stay fixed. Risk heads are the previous alpha10 fitted models. New continuation
heads also have alpha10 FIXED, not an expanded hyperparameter search.

Normalized risk R=L/8, acquisition cost c=.05/8. Five acquisition decisions0..4;
budget5 always terminates. Sampled terminal target at t4 is c+L5/8. For t<4:
Yt=c+min(Rteacher(Zt+1),Cteacher,t+1(Zt+1)). These are estimates inside the min,
never per-path future truth minima or true intermediate losses inside the min.
Five stagewise continuation regressors are fitted. Actor sees current/past only.

Four folds are family_id mod4; both policies/all states of a family stay together.
For each fold train risk and backward continuation teacher bank on other12
families. Its own backward targets may use in-sample child predictions within
those12. Produce targets for held4 families using that excluded-family teacher.
Final continuation heads fit all16 families' held-family targets (except direct
sampled terminal target). Save every fold membership/target. This is cross-fitted
teacher regression, not a theorem of unbiased optimal value estimation.

## Prespecified methods

1. optional_cf_temporal: candidate, excluded-family recursive targets, temporal61.
2. optional_in_temporal: same features/cost/risk, ordinary in-sample backward
   regression. Strong same-input ordinary finite-horizon control.
3. optional_cf_static: remove temporal drift, same recursion/cross-fitting.
4. one_step_temporal: same stagewise regression, target c+L(t+1)/8, no recursion.
5. full_tail_temporal: same stagewise regression, target (5-t)c+L5/8.
6. optional_plain_temporal: candidate value bank, ordinary R<=C stop, no risk cap.
7..15: all nine previous frozen STOP baselines, same frozen settings, evaluated
   on the SAME new scenes: dual_temporal, dual_static, risk_temporal,
   declared_rank, fixed_rank, original_rank, full_rank, declared_coverage,
   full_coverage. Do not hide original_rank if it wins.

Guarded new methods STOP iff Rhat<=Chat AND Rhat<=tau. Budget5 forces STOP.
Plain method only Rhat<=Chat, no tunable threshold. Tau grid fixed in config;
-1 full-budget fallback. Calibration family aggregate quality constraints identical
to previous round: FP/bad-motion/loss<=own full rank and qualified>=it; then
minimize acquisitions, then loss, then smaller threshold. No safety certification.
Candidates calibrate only on rank paths; all training banks use both policies.

## Freeze and evaluation

Freeze code/config/contract/models/thresholds before generating32 seeds
22382026..22382057. First16 original, next16 shifted real RGB-D foreground
occluder; each all16 factorial cells. No expanded renderer, no altered backend.
Two real full acquisition paths per family,384 new backend fits. Fifteen STOP
policies consume causal shared prefixes:480 endpoints, NOT480 independent scenes
or independently rerun policies. Mandatory prefix return accounted for.
Physically rerun candidate on scene0 to its chosen stop and compare report/cost.

Report all methods/conditions: qualified change, FP, incorrect motion, unknown,
loss, acquisitions, return-inclusive travel, scalar loss+.05*acquisitions. Use
5000 paired family bootstrap, all32 stratified by condition; unadjusted95%
exploratory intervals. Primary candidate contrasts: every other method (all14).
No exclusions or retuning after seeing new evaluation; retain all failures.

## Research decision

Quality/cost improvement over old rank STOP AND ordinary finite-horizon is needed
before claiming incremental mainline value. Cross-fitting/temporal ablations must
support any corresponding mechanism claim. Mixed/no gain means freeze the
negative result and end the present aggregate stopping-head innovation claim;
do not automatically sweep thresholds, add kernels or start a second retry.
Engineering correctness/audits are not paper contributions.
