# Metric RGB-D material descriptors: first trained prototype

## Material Passport
- Stage: code execution and fixed protocol. Four training runs completed; downstream evaluation not inspected at protocol creation.
- Data: eight TRAIN scenes only for fitting, 704 frame-pair/appearance cases, 48 queries/case. Other four old scenes are already analyzed DEVELOPMENT scenes, not independent confirmation.
- Models: contrastive-only seed31; multi-task seeds31/47/73; all final fixed500-step weights. No validation/test checkpoint selection. Main end-to-end seed31 preselected; other seeds for training/descriptor variability.
- Provenance: original textured data and old algorithm seals preserved. New feature learning is a standard supervised prototype, not a mathematical originality claim.

## Measurement representation
Surface sample locations come only from depth-valid anonymous spatial groups. Image-independent farthest-point sampling receives an equal share of the 96-point/object budget per observed view, avoiding exclusive selection of high-resolution close-ups. Every current object competes with all24 historical references.

For measured point p, estimate n by local measured-depth covariance; tangent e1 is the projection of world x (fallback y), e2=n cross e1. At radii0.06/0.15m, project the 12x12 tangent grid p+r(a e1+b e2) into the ORIGINAL camera. Input contains normalized RGB, measured height discrepancy and valid same-spatial-group mask. Absolute world position, camera translation, spatial slot number, physical ID, texture seed, acquisition group and filename do not enter the network. World-axis tangent orientation is a reference choice: no SE(3)-equivariance claim.

Two small CNN branches encode six RGB channels and four height/mask channels. A linear fusion and unit normalization give z_i in R64. Candidate-independent global shape completion is not implemented.

## Distinguishing physical correspondence from appearance persistence
For source descriptors Z and target set W, s_ij=z_i dot w_j /0.12. A pair-conditioned head m_i uses the best target descriptor, product, top score, runner-up gap and normalized entropy. A separate pair head c_ij uses absolute descriptor difference and product to predict unchanged appearance.

The full model uses:

    L = CE([s_i1,...,s_iM,-m_i], target_i) + .25 BCE(m_i,v_i)
        + .25 * 1[v_i=1] BCE(c_i,truepositive,1-changed_i).

Visible physical points retain their positive target even if repainted. Occluded/outside points use a dustbin target. Near duplicate SAME-physical-instance material points within2cm are removed from negative denominators; other points of the same object remain negatives. Contrastive-only control trains visible correspondence CE without auxiliary heads/dustbin. Independent quarter turns augment tangent orientation. Fixed AdamW1e-3, clip5,500 steps, no schedule/parameter search.

Heads are uncalibrated first prototypes. They see descriptor context, not a known true pose; output is not a guaranteed visibility probability. Class imbalance and sampled-candidate versus real-scene mismatch remain risks to measure, not assumed solved.

## Downstream common-input controls
Metric-rectified RootSIFT; untrained identical CNN; contrastive-only; full-trained ordinary fusion; full-trained visibility; full-trained material-weighted fusion; official pretrained SuperPoint+LightGlue modules.

For learned fusion/visibility/material, the point samples, descriptors, mutual-NN correspondences, all24x24 identities and data-fitted continuous pose proposals are identical. Mutual-NN uses no ratio threshold (ratio1) for all non-LightGlue methods. LightGlue runs original9 layers on actual per-image pixel coordinates with native confidence0.1, no early pruning; SuperPoint grayscale/config utilities are locally ported, all network layers unchanged, no resize other than padding to multiples of8. This is a matched-input published COMPONENT control, not an author benchmark reproduction.

Fusion L_g+L_tex; visibility adds .5 L_ray with geometry coefficient STILL1; material replaces texture support weights by sigmoid(m_i)*(.5+.5 sigmoid(c_ij)), also adding .5 L_ray. Thus repaint never removes geometric evidence or automatically becomes physical absence. All three get the same positive-support normalization and joint pose combinations. Heads are soft evidence, not a hard identity gate.

All learned/full ablations see the same proposed poses (centroid, ICP, descriptor RANSAC, descriptor-start ICP). All methods use three-period approximate cyclic one-to-one assignment and G12=G2 G1^-1. No claim equal FLOPs across modalities. All poses/pools must be sealed before reading old held-out supervision.

## Endpoints and stopping rule
Same original definition: both current identities, fixed historical material point p95 displacement <=5cm for H->C1/H->C2/C1->C2, all24 objects. Save complete proposals, then inspect true-ID pool coverage post-seal. Common-correct-ID error, full denominators, visible recall, hidden false acceptance and visible-repaint recall must accompany conditional precision.

Do not call training loss improvement an end-to-end gain, geometric rectification/denser sampling a learned gain, or any neural fusion architecture a novel principle. If material heads underperform same-feature fusion, retain failure and diagnose calibration/labels instead of silently choosing a checkpoint or threshold on these scenes.
