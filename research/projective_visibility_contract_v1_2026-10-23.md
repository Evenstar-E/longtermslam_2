# Prospective actual-observation projective visibility mechanism pilot

Freeze before fresh seeds22482026..33 clear/occluded pairs +22482126..29 normal.
8paired worlds+4normal=12worlds20renderings480RGBDframes20mainpaths120outputs.
Same geometry/change/camera/noise draws in each clear/occluded pair. H0/H1/C0/C1
actual legal acquisition and return; identical input for all methods. No true
mesh/pose/ID/futureimage in prepare/solve. Old20 scenes are observed development,
not tuned/retested for performance here. No parameter sweep or extra samples.

Build finite triangle patches ONLY by connecting three adjacent observed history
depth pixels within same observed connected component, depth jump<.08m. No filled
holes, completed back faces or true shape. Same observed historical hard nearest
association as old reference map. Render known patches shifted by3D translation d
into current nominal camera; zbuffer with perspective-correct depth. Back-facing
patches culled by observed triangle orientation; unobserved reference pixels impose
no residual, so this is an incomplete finite surface model, not a watertight asset.

Known projected patch pixels overlapping observed current component use robust
depth log(1+((zhat-z)/.06)^2). Known pixels outside component use finite-support
distance log(1+(distance_to_component_mask/1.5px)^2). If measured surface belongs
outside this observed component AND z+.06<zhat, censored methods remove that
boundary contradiction as foreground occlusion. Same-component closer depths
remain depth residuals: otherwise a real movement toward camera could be hidden.
Pixels outside known reference patch are UNKNOWN, no reverse silhouette residual.
Mean cost normalized by known projected pixels, count of censored/visible recorded.
<12comparable pixels adds a declared failure penalty; independent missing frames
are dropped, joint may use >=12 across its two views. Counts are heuristic, not
occlusion probabilities or calibrated covariance. Unknown self-occlusion remains.

Shared initialization: original shared-motion stage infers current identities;
ALL new methods keep its actual correspondences (can drop, cannot permute them).
Each reference uses the same median pair-fit translation as start for candidate
and per-frame ordinary control. Optimization is deterministic3D coordinate search:
two sweeps per step [.12,.06,.03,.015,.0075],61scheduled evaluations maximum before
cache reuse, local bound +/-.6m around common start; weak initial prior .01*|d-start|²/.6².
Joint evaluation includes both frames; independent has same per-frame render budget.
These are local optimizers, not global completeness or a physical displacement cap.

Candidate joint_censored fits ONE physical translation directly against all acquired
current views' raw costs. Ordinary independent_censored has SAME patches, masks,
censoring, initialization and search resolution but fits each frame then fuses.
Joint_uncensored removes censoring; joint_depth_only removes finite-boundary term
but keeps censoring. Legacy shared-motion and Cauchy controls retained. Final common
frozen sparse shared-motion decomposition obtains global t/local delta from fitted
translations; S stays common legacy shape score. This is a TWO-STAGE backend test,
NOT a complete joint likelihood over A,T,delta; identity remains frozen, SE3/active
selection/uncertainty calibration remain missing. Known ingredients not novelty.

Primary paired task_loss candidate minus independent_censored on8occluded worlds,
secondary clear contrast/occlusion interaction, ablations and strongest legacy
control. Keep qualified<=.10m+identity,FP,bad-motion,UNKNOWN, costs.5000bootstrap
by8worlds exploratory unadjusted; clear/occluded not independent.0 exclusions,
all raw inputs/backends/reports persist BEFORE truth. Replay legalAPI, finitepatch
construction, local searches, finalsolvers, metrics; retain any failures.

Decision: all-method gains explained by ordinary projective registration = common
backend repair, NOT candidate innovation. Joint coupling must beat SAME-CENSOR
strong ordinary control, not only uncensored/depth-only ablations. Must retain task
coverage and all negative normal cases. No posthoc threshold changes or confirmation.
If failed, freeze this exact candidate instead of generic geometry/parameter loops.

Prior art checked primary sources: KinectFusion2011 already uses model-based depth
tracking/projective association (https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/ismar_2011.pdf);
Co-Fusion2017 tracks/reconstructs independent objects with associated RGBD pixels
(https://arxiv.org/abs/1706.06629). Our CPU translation search is NOT an exact
reproduction of these systems. Neither projective ICP, object tracking, multiple
views nor geometric occlusion censoring is claimed first. Specificincrement remains
unproven; closest full occlusion-aware registration comparison still required.
