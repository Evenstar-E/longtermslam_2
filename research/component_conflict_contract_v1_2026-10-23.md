# Prospective component-conflict representation experiment

Written before recollection/training/new evaluation. Change observed representation,
not backend/action ranking/terminal reporting. Four existing displacement modes
share hard identity associations: NOT alternative identity posteriors. Energy
weights are heuristic, NOT probabilities. Observe matching competition separately.

Collect old16train/16cal families, frozen rank and coverage full paths,384fits.
Save only observed mode displacement/weights/sensor variance and assignment
geometry before next action. Labels evaluated after entire path. Component
losses exactly sum to frozen task loss; missing-owner penalty distributed equally
among reported references (explicit bookkeeping convention). H0 IDs stable.

Shared ridge readout alpha10 fixed: current component loss, and five stagewise
next-observation component losses. Sum predictions/8; add acquisition .05/8 once.
STOP iff predicted current<=next+price and current<=calibrated cap. No new Bellman
or STOP architecture. Same cap grid/aggregate calibration quality constraints as
previous experiments; no guarantee beyond16cal families.

Candidate uses global61 context +14 local report/temporal features +12 conflict
features. Controls: report-only component; replace local conflict by its component
mean (unpaired); zero first8 modal features; zero last4 association/source features;
global mean/max/std pool of identical local26 info +context (139dim). These isolate
component supervision, added information and binding separately. Six old frozen
controls include one-step, ordinary finite-horizon, oldrankSTOP, declaredrisk and
both fullbudget strategies. No claim ridge/multimode/component pooling itself novel.

Freeze models/config/actor before32freshseeds22402026..57;16original16actualshifted
occluder,balanced16cells each. Two full paths/family384freshfits,12STOPpolicies
share causal prefixes (384outcomes,32independent families). Prefix return costs.
All families/errors retained. Primary candidate contrasts all11;5000 pairedfamily
bootstrap,stratifiedconditions,exploratory unadjusted95%. No evaluation retuning.
Physically replay a first early-stop family if present; otherwise scene0budget.
Require benefit over same-info pooled and report-only AND ordinary strong controls
before innovation claim. Negative results freeze, not automatic feature/grid retry.
