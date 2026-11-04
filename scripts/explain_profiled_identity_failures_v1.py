"""Selected posthoc worked examples; no model, parameter or output changes."""
from pathlib import Path
import numpy as np
from inspection_decisions.projective_visibility_v1 import decode
from inspection_decisions.correspondence_joint_v1 import prepare
from comparative_evidence.estimation_v1 import world_components
from scripts.run_profiled_identity_actions_v1 import read, save, digest


def main():
    out = Path('results/profiled_identity_actions_v1_repaired')
    cfg = read(out/'frozen_contract.json')['config']
    wd = out/'0000'; fitted = read(wd/'endpoint_6_backend.json'); raw = read(wd/'endpoint_6_observed.json')
    rows = []
    for k, branch in enumerate(fitted['options']):
        history = [decode(o) for o in raw['history']]; current = [decode(o) for o in raw['current']]
        G = np.asarray(branch['initial_map_transform'])
        for o in current: o['nominal_T'] = np.linalg.inv(G)@o['nominal_T']
        data = prepare(history, current)
        t = np.asarray(branch['best']['translation']); delta = np.asarray(branch['best']['delta'])
        assert not np.any(delta), 'This worked static example must have no motion penalties'
        fits = {(r['frame'], r['component'], r['reference']): np.asarray(r['fit']['d']) for r in branch['projective_fits']}
        residual = 0.; shape = 0.; unmatched = 0.
        for f, pairs in enumerate(branch['best']['matches']):
            unmatched += 12*(len(data['frames'][f]['D'])-len(pairs))
            for i, j in pairs:
                residual += np.sum((fits[f, i, j]-t-delta[j])**2)/cfg['sigma']**2
                shape += data['frames'][f]['S'][i][j]/.1
        assert abs(residual+shape+unmatched-branch['best']['score']) < 1e-9
        rows.append(dict(branch=k, residual=float(residual), shape=float(shape), unmatched=unmatched,
            total=float(residual+shape+unmatched), reported_E=branch['best']['score'],
            matched_projective_loss=sum(r['fit']['loss'] for r in branch['projective_fits'])))
    save(out/'worked_energy_decomposition.json', dict(status='POSTHOC selected illustrative case, not aggregate attribution',
        source_sha256=digest(__file__), world=0, action=6, selected=fitted['selected'], branches=rows))
    raw = read(out/'0011/initial_observed.json'); obs = decode(raw['current'][0])
    with np.load(out/'scenes/0011/evaluation/current_00_owner.npz') as z:
        owner = z['owner'].ravel()
    fragments = []
    for i, c in enumerate(world_components(obs)):
        values = owner[c['pixels']]; values = values[values >= 0]
        counts = np.bincount(values, minlength=8)
        fragments.append(dict(component=i, owner=int(counts.argmax()), pixels=len(values),
            purity=float(counts.max()/max(len(values), 1)), centre=c['centre'].tolist()))
    save(out/'fragment_identity_diagnostic.json', dict(status='POSTHOC evaluator-only fragment inspection',
        source_sha256=digest(__file__), world=11, fragments=fragments,
        scope='A complete explanation assigning both same-owner fragments to that asset violates per-frame one-to-one; dropping a fragment remains possible, but was not retained.'))
    print('ENERGY', rows, 'FRAGMENTS', fragments, flush=True)


if __name__ == '__main__': main()
