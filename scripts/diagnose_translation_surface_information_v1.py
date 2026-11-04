"""Postseal, no-GT, full-scene local information diagnosis. Does not infer IDs."""
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
from scripts.run_rescan_reversible_surface_v2 import load, read, save, sha
from inspection_decisions.translation_surface_information_v1 import TranslationSurface, complete_profile


def main(output='results/translation_surface_information_v1', stable=False):
    root = Path(output)
    root.mkdir(exist_ok=False)
    old = Path('results/shared_surface_identity_likelihood_v3')
    cfg = read(old/'config.json')
    source_files = [Path(__file__), Path('inspection_decisions/translation_surface_information_v1.py'),
                    Path('research/translation_surface_information_contract_2026-10-24.md')]
    inputs = {}
    rows, checks = [], []
    start = time.perf_counter()
    for scene in cfg['scenes']:
        H, O, _, _, _ = load(scene, cfg)
        folder = old/scene/'shared_mle'
        for name in ['state.npz', 'model.npz']:
            inputs[str(folder/name)] = sha(folder/name)
        state = np.load(folder/'state.npz')
        stored = np.load(folder/'model.npz')
        categories = 1+max(max(int(o['class'].max()) for o in O), max(h['class'] for h in H))
        kh, ku = cfg['history_components'], cfg['new_components']
        weights = np.r_[np.full(kh, np.log((1-cfg['new_mass'])/kh)),
                        np.full(ku, np.log(cfg['new_mass']/ku))]+np.log((1-cfg['outlier_mass'])/len(H))
        sigma = np.array([cfg['position_sigma_m']]*3+[cfg['color_sigma']]*3)
        out = np.log(cfg['outlier_mass'])-np.log(np.prod(stored['high']-stored['low']))-np.log(categories)
        validate_uid = min(h['uid'] for h in H if h['uid']>0)
        matrices = {}
        for i, h in enumerate(H):
            other, semantic = [], []
            for t in [0, 1]:
                L = stored[f'L{t+1}'].copy()
                L[:, i] = -np.inf
                other.append(logsumexp(np.c_[L, np.full(len(L), out)], axis=1))
                prob = np.where(O[t]['class']==h['class'], cfg['semantic_correct'],
                                (1-cfg['semantic_correct'])/max(1, categories-1))
                semantic.append(np.log(prob))
            p = TranslationSurface(O, stored['mean'][i], state['poses'][:, i], other,
                                   weights, semantic, sigma, kh)
            g, observed, complete, missing = p.information()
            eig, vec = np.linalg.eigh(observed)
            profile = complete_profile(complete)
            value = p.delta if stable else p.value
            base = value(np.zeros(9))
            grad_step = .001*g/max(np.linalg.norm(g), 1e-300)
            curvature_step = .001*vec[:, 0]
            deltas = [value(grad_step)-base, value(curvature_step)-base, value(-curvature_step)-base]
            scale = max(1., float(np.linalg.eigvalsh(complete).max()))
            row = {'scene': scene, 'uid': h['uid'], 'gradient_norm': float(np.linalg.norm(g)),
                   'observed_min_eigenvalue': float(eig[0]), 'complete_max_eigenvalue': scale,
                   'negative_curvature_relative_1e-8': bool(eig[0]<-1e-8*scale),
                   'missing_min_eigenvalue': float(np.linalg.eigvalsh(missing).min()),
                   'profile_min_eigenvalue': float(np.linalg.eigvalsh(profile).min()),
                   'gradient_step_loglikelihood_delta': float(deltas[0]),
                   'min_eigenvector_plus_delta': float(deltas[1]), 'min_eigenvector_minus_delta': float(deltas[2]),
                   'any_predefined_step_improves_1e-5': bool(max(deltas)>1e-5)}
            rows.append(row)
            matrices[f'{i}_gradient'] = g
            for key, val in [('observed', observed), ('complete', complete), ('missing', missing), ('profile', profile)]:
                matrices[f'{i}_{key}'] = val
            assert np.linalg.eigvalsh(missing).min() > -1e-7*scale
            if h['uid']==validate_uid:
                rng = np.random.default_rng(20261024)
                for direction in range(4):
                    v = rng.normal(size=9); v /= np.linalg.norm(v)
                    for step in [1e-4, 5e-5]:
                        plus, minus = value(step*v), value(-step*v)
                        fd_g = (plus-minus)/(2*step)
                        fd_h = -(plus+minus-2*base)/step**2
                        exact_g, exact_h = float(g@v), float(v@observed@v)
                        err_g = abs(fd_g-exact_g)/max(1., abs(exact_g))
                        err_h = abs(fd_h-exact_h)/max(1., abs(exact_h))
                        checks.append({'scene': scene, 'uid': h['uid'], 'direction': direction, 'step': step,
                                       'analytic_gradient': exact_g, 'fd_gradient': fd_g,
                                       'analytic_curvature': exact_h, 'fd_curvature': fd_h,
                                       'gradient_relative_error': err_g, 'curvature_relative_error': err_h})
                        assert err_g<.002 and err_h<.002, checks[-1]
            save(root/'rows.json', rows)
            print(scene, h['uid'], 'min_eig', eig[0], 'deltas', deltas, flush=True)
        np.savez_compressed(root/f'{scene}_matrices.npz', **matrices)
    assert all(sha(p)==v for p, v in inputs.items())
    summary = {'assets': len(rows), 'negative_restricted_curvature': sum(r['negative_curvature_relative_1e-8'] for r in rows),
               'any_predefined_1mm_step_improves': sum(r['any_predefined_step_improves_1e-5'] for r in rows),
               'gradient_1mm_step_improves': sum(r['gradient_step_loglikelihood_delta']>1e-5 for r in rows),
               'finite_difference_checks': len(checks), 'elapsed_s': time.perf_counter()-start,
               'current_GT_read': False, 'new_identity_predictions': False,
               'meaning': 'Restricted translation/collective-shape local diagnosis, no originality or uncertainty calibration claim.'}
    save(root/'finite_difference.json', checks)
    save(root/'summary.json', summary)
    source_files += [Path('scripts/diagnose_translation_surface_information_v2.py'), Path('research/translation_surface_numerics_amendment_2026-10-24.md')]
    save(root/'manifest.json', {'inputs_unchanged': inputs, 'sources': {str(p): sha(p) for p in source_files}, 'stable_increment': stable})
    print(summary, flush=True)


if __name__=='__main__':
    main()
