# Joint correspondence prototype / synchronous chain contract

Prospective fixed-parameter16-scene MECHANISM PILOT, not confirmation or a
novelty claim. Old datasets/models unchanged. Newactor actually changes discrete
correspondences, not just continuous initialization under a fixed association.
All solvers receive identical acquired RGB-D and appearance labels. No truth IDs.

Four regimes x4 seeds: unique appearance/small drift, repeated geometry with one
unique anchor, repeated geometry without unique anchor, repeated with occlusion.
All repeated scenes use identical geometric templates at distinct locations;
layout jitter/scale and0or2localtranslations vary. Current nominal poses have
unknown signed common x drift about0.9m; public bound1.2m is given to ALL solvers.
No claim repeat_unanchored is mathematically unobservable (finite boundaries may
disambiguate). Exact unobservable symmetry is a separate unit test.

Use existing renderer/observation API, H0,H1,C0,C1 fixed evidence. Allmethods pay
same legal acquisitions and mandatoryreturn. This experiment tests the inference
module, NOT yet an adaptive viewpoint-selection contribution.

Build observed pair translation fits using bidirectional cloud ICP. Common latent
translation and frame-wise one-to-one assignments; component localdisplacement
shared across current frames. Candidate scores capped stable residual plus sparse
change penalty, retains distinct actual assignment branches and reports UNKNOWN
when near-cost branches disagree on identity or >10cm displacement. Heuristic
energies NOT probabilities. Current global transform translation only,notfullSE3.

Controls: nearest withmedian drift correction; one-to-one Hungarian thenmedian;
ordinary multistartCauchy registration withsame seeds/data andposthoc displacement.
No claim this ordinary control equals CLIPPER/ROBIN or allsoftmatching. Candidate
is known ingredients; specificinnovation remainsunproved. Preserveallnegativecases.

Save observations/solutions/reports beforeevaluationtruth. Originalidentity+
change+<=10cmmetric/taskloss. New16familyresults16pairedunits,64methodoutcomes,
not64independentfamilies. Report FP,badmotion,unknown,poseerror,actualassignment
branchcount andcosts. Pilot helpsmechanismdecisiononly,no gate relaxation,
no old256confirmation. Freezeallsource/configbeforegeneratingpilot. No retuning
on pilot; next independentlayoutfamilyconfirmationrequiresseparatecontract.
