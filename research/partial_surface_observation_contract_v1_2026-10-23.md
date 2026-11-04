# Prospective partial-surface observation mechanism contract

New branch, no tuning on old observed16. Freeze all sources before generation.
Question: can normal-direction surface constraints avoid partial-overlap ICP bias,
and is the improvement beyond ordinary trimmed / point-to-plane registration?
Normal/Hessian use is established prior art, NOT the novelty claim.

Observed-only capture stores reference and current world points, estimated normals,
normal validity, appearance group and pixel provenance from legal H0,H1,C0,C1.
No true shape/pose/ID or future image enters prepare/solve. Same deterministic192
reference-point cap and same acquired observations for every backend. Palette
segmentation and historical hard association limitations remain unchanged.

Backends: legacy one-sided robust ICP + old symmetric score; ordinary70%-retained
point-to-point translation ICP + one-sided score; normal-compatible (cos>.8)
point-to-plane translation ICP,70% retained plane residuals, >=6 plane samples,
8 pair iterations. These are translation-only local variants, not exact published
TrICP implementations or globally optimized registration. Point-to-plane only
uses observed local surfaces; NO ray-censoring/true occlusion model implemented.

For the surface fit, N rows are unit estimated reference normals; d solves
N d = n.(q-p). Its quadratic information proxy H=3 N^T N/(K sigma²), sigma=.06m.
Trace matches isotropic I/sigma²; unknown tangent directions have no added
measurement information. This is conditional on nearest surfaces/normals and
not calibrated covariance. Shape term is mean absolute retained plane residual.
Initial d keeps median difference in null directions; those directions are NOT
certified by H. Candidate shares t and per-asset delta across frames using
sum (d-t-delta)^T H (d-t-delta) + surface/.1 +4 per active delta
+ ||delta||²/.6² +12 per unmatched current component. Exact delta block, bounded
quadratic t block, Hungarian assignment, same objective descent checked. Zero seed,
one normal-information-weighted coordinate-median seed and up to23voted seeds,
30iterations. Both surface models share these EXACT seeds computed from measured
surface H, giving the ordinary isotropic control the same initialization information.
Surface_isotropic uses EXACT SAME surface fits but I/.06² in the same matrix solver
(ordinary isotropic shared-motion objective): this is the primary algorithm control.
Both surface methods report UNKNOWN if aggregate assigned surface information
min/max eigenvalue ratio<.02. This rank heuristic is not a95% coverage statement.
Initial assignments profile each pair's delta using its actual H; no isotropic
certainty is introduced during initialization. Trimmed_shared and legacy_shared
use the frozen shared-motion solver; legacy_robust
uses old strong Cauchy multistart. All report thresholds/prior/budgets unchanged.

Fresh seeds22462026..33:8 repeated-unanchored world families, each rendered twice
with foreground occluder absent/present. SAME geometry, changes, camera and noise
draws/nominal poses; only RGB-D visibility changes.0/2changes alternate per family.
Additional4 normal/unique-appearance control families seeds22462126..29. Total20
scenes/480RGBDframes/20actual fixed acquisition paths/100method outputs,12independent
world families, NOT20independent families or100independent trajectories. Each path
has actual four acquisitions and mandatory return; no efficiency claim.

Primary outcome: paired candidate minus surface_isotropic task_loss on8 occluded
families; qualified/FP/bad-motion/identity_not_ok/UNKNOWN all retained. Secondary:
same contrast minus clear counterpart (occlusion interaction), same contrasts
against trimmed and legacy strong controls.5000bootstrap resamples8 paired families;
exploratory unadjusted95% intervals, not independent-layout confirmation. Normal
controls descriptive. Verify paired true geometry/nominal poses/noise identities
after reports persist; truth NEVER used by actor. Save raw inputs/backends/solutions
before first evaluator call. Full observed backend reconstruction and solvers replay.

Decision: ordinary surface/trimmed gain only = common measurement/backend repair.
Directional gain must exceed same-surface isotropic control without hiding UNKNOWN
coverage; otherwise no new algorithm claim. Preserve all failures, no parameter
sweep/new seeds after seeing these results. If geometry is unobservable, acknowledge
it; don't fake certainty by increasing information eigenvalues. Adaptive/fullSE3,
external registration implementations and independent-layout confirmation remain
unimplemented. Old quality gates/results/source unchanged.

Primary-source boundary checked this round: Rusinkiewicz & Levoy, Efficient
Variants of ICP (2001), https://lightfield.stanford.edu/papers/fasticp/ ; Censi,
An accurate closed-form estimate of ICP's covariance (2007),
https://censi.science/pub/research/2007-icra-icpcov.pdf ; Bonnabel et al., On the
Covariance of ICP-based Scan-matching Techniques, https://arxiv.org/abs/1410.7632 .
These establish existing normal/surface/covariance/observability research; our H
is a heuristic local curvature proxy and does NOT implement Censi's covariance.
