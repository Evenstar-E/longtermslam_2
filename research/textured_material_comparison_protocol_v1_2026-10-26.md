# Geometry, texture and visibility-constrained persistent material matching: frozen V1

## Material Passport
- Type: executable experiment protocol; non-learned mechanism prototype.
- Dataset: `data/textured_material_v1`, 12 scenes, 24 objects/scene, 3 periods, four paired appearances.
- Status at protocol creation: train_000 inference smoke completed, no private-label scores inspected; validation/test not executed.
- Training: none. RootSIFT is hand-crafted, not a pretrained neural model. Published learned strong baselines remain pending.

## Question and controls
Does persistent local appearance help multi-object cross-period identity AND displacement of the same historical material point? Does conditional visibility improve this beyond ordinary geometry/texture fusion, without merely abstaining on more inputs?

Six methods: depth-only trimmed bidirectional ICP; RootSIFT mutual-ratio matching plus measured-depth rigid RANSAC; ordinary geometry+texture fusion; fusion+visibility; fusion+historical-anchor triplets; both additions. Ordinary fusion already gets all three periods and a one-to-one cyclic identity solver. All fusion methods have exactly the same per-pair proposed continuous poses: centroid translation, ICP, feature RANSAC if available, feature-start ICP. Same joint-pose combinations. Persistence adds assignment/score computation, so this is NOT equal elapsed time/FLOPs. There is no ground-truth pose catalogue and no identity top-k cropping. Texture-only emits identity pose if too few rigid matches; it does not fall back to surface ICP.

## Inputs
Only measured RGB, depth/depth_valid, K, camera-to-world C, anonymous spatial slots. Known cameras and nominal slot frontend are favorable assumptions. All 24 spatial groups and all observations in a period are used; near-view acquisition group, instance IDs, complete geometry, texture seeds, frame filename and material labels are excluded from model inputs. Historical identity is the historical observed slot reference; it is not given in current periods. Feature/point caps and all thresholds are fixed in config before held-out inference. No parameter search based on validation/test outcomes.

## Mathematical prototype
Let q_et be measured world points for an anonymous slot at period e. Historical slot i has assignments pi_1(i), pi_2(i), with both pi permutations. Let G_e map historical world coordinates to period e, G_0=I.

Geometry cost is mean symmetric capped nearest-surface distance, scale 0.04m. RootSIFT mutual matches M_ab have measured 3D points x_k,y_k and descriptor distance d_k. Positive texture support is

    S_ab(G) = sum_{k in M_ab} exp(-||G x_k-y_k||²/(2 sigma²)) exp(-d_k²/(2*0.5²)),
    L_tex = 1-min(1,S_ab/6), sigma=0.025m.

The fixed six-support normalization is an engineering design, not a calibrated likelihood. Standard fusion is L_g+L_tex. Assignment minimizes historical-current costs plus 0.5 times the current-current edge, with two common starts and five alternating Hungarian iterations. This is approximate, not a globally optimal three-way solver.

For a proposed world point p, each real camera predicts camera-z z and pixel u. If no valid in-frame depth exists it is unknown; if z>d(u)+0.015 it is behind the first return and censored; if z<d(u)-0.015 it contradicts measured free space; if |z-d(u)|<=0.015 and the measured spatial slot matches the proposed destination, it supplies positive support. Multiple views keep a supporting view if one exists. Visibility surface loss averages capped depth discrepancy over exposed/free-space rays and adds a positive-support deficit. All-censored configurations cannot get zero cost. A minimum supporting-view rule is an optimistic multi-view approximation, not a complete probabilistic ray model.

Visibility fusion uses 0.5 L_g+0.5 L_ray+L_tex, with positive texture support additionally requiring target-view visibility. It retains a geometric term rather than claiming all unknown material is reconstructed.

Persistent anchor triplets are descriptor matches (h,c1,c2) sharing the SAME historical descriptor index. In historical coordinates their robust discrepancy is

    r_h² = (||x_h-G_1^-1 y_h1||² + ||x_h-G_2^-1 y_h2||²
            + ||G_1^-1 y_h1-G_2^-1 y_h2||²)/3,
    L_persist = 1-min(1,sum_h exp(-r_h²/(2 sigma²))/6).

The two persistence methods add 0.5 L_persist to identity assignment and joint-pose selection. Joint-pose selection enforces G_12=G_2 G_1^-1. Both current transforms are selected from common data-fitted continuous candidates; no additional end-to-end continuous latent-surface optimization or learned feature training is claimed.

These equations combine established registration, censoring and cycle-consistency mechanisms. A win would establish a task mechanism signal, not mathematical originality. A null outcome is retained; no held-out threshold sweep will follow as proof of novelty.

## Frozen evaluation
All predictions for all scenes/variants/methods are saved and SHA256 sealed BEFORE the separate scorer reads supervision. Report train, validation, test separately; test has only 2 independent generated scenes (48 correlated objects). Four appearances are paired interventions, not independent data.

Primary per-object outputs: both current identities correct, fixed observed historical material-point p95 displacement error for H->C1, H->C2 and C1->C2; joint qualified requires both identities and <=5cm for ALL three intervals. Compare conditional error only on correct identities AND the common-correct-ID intersection between candidate and ordinary fusion. Denominators and coverage must accompany all conditional figures.

For all 264 provided frame pairs, evaluate pose-induced material correspondence from actual source depth: nearest measured target point in the predicted slot, accepted within 4cm. Visibility methods additionally require an uncensored first-return match in that particular target view. Correct acceptance requires true visible status, same physical identity and <=2cm canonical material-point error. Report visible recall, accepted precision, occluded/outside false-acceptance rates and repaint-visible recall. This is pose-induced correspondence quality, NOT raw SIFT descriptor precision. The two mask ablations and common-ID comparisons prevent labeling a coverage reduction as a displacement gain.

Failures/absent texture pose remain in full denominators. No timeout is scored as zero. Original 192-task results cannot be ranked against this different dataset. Raw scores, per-scene paired deltas and numerical/integrity checks are retained; with 2 test scenes no population significance claim.
