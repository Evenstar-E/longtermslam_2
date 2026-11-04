# Prospective bounded motion-objective experiment

Freeze before generating 16 new seeds 22442026..22442041. Previous 16 scenes are
observed diagnosis only; no fitting, threshold search or selecting a winner on them.
Reuse the four-regime actual RGB-D renderer and fixed H0,H1,C0,C1 acquisitions.
The new scenes vary seeds within the SAME layout generator; not unseen-layout
confirmation. 16 paired families, 80 method outputs, not 80 independent scenes.

Observed pair translations d_fij and surface residual s_fij come from acquired
RGB-D only. A_f is a partial one-to-one assignment with cost 12 per unmatched
current component. Candidate minimizes the SAME objective in every update/score:

E(A,t,delta)=sum_(f,i,j in A_f) [||d_fij-t-delta_j||²/sigma²+s_fij/0.1]
            +4 sum_j 1[delta_j != 0]+sum_j ||delta_j||²/tau²+12 N_unmatched.

sigma=.06 m, tau=.6 m, public |t|_infinity<=1.2 m; these are design constants,
not estimated from new truth. tau is a SOFT scale, no hard rejection of genuine
large displacement. No claim of calibrated posterior, covariance or probabilities.
24 voted translation seeds plus zero, 30 block-coordinate iterations, same new
candidate/control seed budget. Exact conditional delta and clipped mean t updates,
Hungarian assignment, check nonincreasing full energy. Candidate keeps distinct
assignment signatures within energy .5, existing UNKNOWN report rule unchanged.
Finite multistart/block optimization is not a global optimum guarantee.

Primary ordinary control independent_profile profiles a separate delta per pair:
q(d)=min(||d||²/sigma²,4/F+||d||²/(sigma²+F*tau²)), F=#current frames.
For equal shifts and full support, activation/amplitude priors sum to the same
amount; independent nuisance shifts can absorb cross-view disagreement. Missing
support has different prior allocation, explicitly a limitation of this comparison.
It uses same data, translation bounds, seeds, iterations and branch UNKNOWN rule.
Report each asset's median profiled local shift. shared_no_amplitude removes only
the candidate amplitude cost, including its initialization. Frozen old Cauchy
robust_multistart and old capped joint_branches receive the SAME 30-iteration
budget here (old 6-iteration results remain untouched); no weak baseline needed.

Question: does explicit amplitude cost prevent wrong-identity-as-large-change,
and does cross-view shared delta improve task outcomes beyond independent profile?
Preserve task_loss/qualified_tp/FP/bad-motion/wrong-identity/UNKNOWN, all four regimes,
actual acquisition and return costs, branch counts and objective traces.
Primary paired task_loss difference candidate minus independent_profile; secondary
versus robust_multistart and amplitude ablation. 5000 family bootstrap resamples
within regimes, exploratory unadjusted intervals, no publication confirmation.
No cost-saving claim: all methods use the same four acquired frames.

Decision: if only amplitude ablation improves, classify as model repair. If ordinary
same-prior control matches/beats candidate, do NOT claim shared-motion innovation.
If candidate improves, still require different-layout/SE3/active-loop and external
baselines before full paper claim. No automatic parameter sweep or sample expansion.
Sparse priors, regularization and coordinate descent are known ingredients. This
round tests a mechanism, not novelty of those ingredients. Old quality gates stay.
