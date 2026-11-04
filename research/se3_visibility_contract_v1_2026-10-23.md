# Prospective full map-frame SE(3) / association / limited-history pilot

Freeze before fresh seeds22502026..37.12 world families:6 unique/no foreground,
6 repeated/foreground.0or2 physical translations alternate per family. Each world
has2 coordinate-frame variants G=I and unknown G inSE3; RGBD geometry/noise/current
physical camera path IDENTICAL, nominal current poses change by G, so frame variants
are NOT independent worlds. Inject yaw6..9deg, roll/pitch+/-2deg, translation+/-.65m;
public prior totalrotation<=15deg,translationinfinity<=1.2m. These are prospective
coordinate-frame test choices, NOT measured drone error statistics or new physical
object rotations. 3 currentviews0,1,2 give real0/30/60deg camera directions versus
history0,1. Repeated worlds additionally use onlyH0 underunknownG: evidence/history
retrieval limited to1, NOT literal deletion of archive files or loss of allassetIDs.

Total12worlds24renderings576storedRGBDframes30mainactualacquisitionpaths150outputs.
Repeated framevariants and historybudget conditions are correlated. Allmethods in
eachcondition get SAME allowed H and C0,C1,C2 / return, metadata, prior,initbudget.
No true pose/ID/mesh in inference. Geometric path lengths unchanged by rigid G;
navigationgraph is in currentlocalframe, evaluation maps physicalgeometry properly.

Model q_fi approximately G(x_Afi+delta_Afi); G mapsarchive tocurrentlocal, delta
must be reported in ARCHIVE frame. A_f partialone-to-one. Ordinary SE3 registration:
observed component-centre congruent triples generate up to256 bounded proposals;
rank+16starts, Cauchyweighted Kabsch5rounds then observed cloudICP6rounds, matching
recomputed. No oracle starts. Retain up to3 distinct pose/association branches after
refinement. These are standard robustregistration ingredients, NOT novel and NOT
TEASER++/CLIPPER published algorithm reproductions or certified optimal solutions.

For each pose G, correct CURRENT nominal cameras by inverse(G), preserve RGBD and
intrinsics, reopen correspondences from ordinary registered geometry, and use same
frozen knownpatch/projective censored PER-FRAME measurement backend. Final residual
translation t inarchive composes as G_final=G*Trans(t), archive delta unchanged.
This is a two-stage local inference pipeline, not globally optimized joint SE3/shape.

Strong ordinary se3_rescore uses best cloud registration pose with reopenedA and
projective readout. Exploratory se3_late_visibility evaluates retainedbranches then
ranks best.score+sum fitted rawprojectivecost. This heuristic mixed energy retains
visibility until after discrete/pose competition, not a posterior or novel invention.
se3_frozen_identity gets SAME bestSE3 but old nominaltranslation preselectedA;
se3_cloud_only same bestSE3/reopenedA withoutrawprojectivefits; translation_projective
is same finitepatch/projective control without SE3 correction. Do NOT claim new
algorithm only by beating translation-only or frozenidentity weak ablations.

Primary comparison exploratory latevisibility vs ordinarySE3_rescore on6repeated
unknownG/one-history worlds; secondaries pairedG equivariance and2vs1history. Report
qualified<=.10m+ID,FP,badmotion,UNKNOWN,identity,poseerror,actualcandidateAcounts,
retained/late-selectedbranches,costs.5000bootstrap by6repeatedworlds, exploratory
unadjusted; no30independentcase claim. All outputs BEFORE evaluatortruth. Freeze
all source/config beforegeneration, no grids/samples/thresholds after observing.

If fullSE3/reopenedA gain is explained by ordinary registration call baseline/scope
engineering, NOT paperinnovation. If lateheuristic does not improve primary task,
freeze it; no automatic rescoringweightv2. No qualitygate lowering/old256/STOP revival.
Need externalimplementation/realcrosssessiondata/independentlayouts beforeinnovation.

Primary literature boundary verified: CLIPPER2024 robustassociation withoutinitial
guess (https://arxiv.org/abs/2402.07284), TEASER/TEASER++2020 robustregistration
(https://arxiv.org/abs/2001.07715), Co-Fusion2017 objectRGBDmotion reconstruction
(https://arxiv.org/abs/1706.06629). Multihypotheses, robustSE3, objectmotion and
geometric re-association themselves are priorart; this pilot claims nofirstever.
