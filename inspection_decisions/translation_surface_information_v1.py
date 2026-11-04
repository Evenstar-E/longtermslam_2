"""Exact restricted observed information, using Louis (1982); not new theory.

Nine parameters: two world translations and one collective canonical shift
of all newly learned centers. Colors, rotations and other assets stay fixed.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


class TranslationSurface:
    def __init__(self, observations, mean, poses, other, logweights, semantic,
                 sigma, history_count):
        self.O = observations
        self.mean = mean
        self.poses = poses
        self.other = other
        self.weights = logweights
        self.semantic = semantic
        self.sigma = sigma
        self.kh = history_count

    def batch(self, t, sl, theta):
        centers = self.mean.copy()
        centers[self.kh:, :3] += theta[6:9]
        world = centers.copy()
        R = self.poses[t, :3, :3]
        world[:, :3] = centers[:, :3] @ R.T + self.poses[t, :3, 3] + theta[3*t:3*t+3]
        o = self.O[t]
        features = np.c_[o['xyz'][sl], o['rgb'][sl]]
        lp = (-.5 * cdist(features/self.sigma, world/self.sigma, 'sqeuclidean')
              -3*np.log(2*np.pi)-np.log(self.sigma).sum()
              +self.weights[None, :]+self.semantic[t][sl, None])
        return lp, world

    def value(self, theta):
        total = 0.
        for t, o in enumerate(self.O):
            for start in range(0, len(o['xyz']), 2048):
                sl = slice(start, start+2048)
                lp, _ = self.batch(t, sl, theta)
                total += np.dot(o['mass'][sl], np.logaddexp(self.other[t][sl], logsumexp(lp, axis=1)))
        return float(total)

    def information(self):
        complete = np.zeros((9, 9))
        missing = np.zeros((9, 9))
        gradient = np.zeros(9)
        eye = np.eye(3)/self.sigma[0]**2
        zero = np.zeros(9)
        for t, o in enumerate(self.O):
            a = slice(3*t, 3*t+3)
            R = self.poses[t, :3, :3]
            for start in range(0, len(o['xyz']), 2048):
                sl = slice(start, start+2048)
                lp, world = self.batch(t, sl, zero)
                partition = np.logaddexp(self.other[t][sl], logsumexp(lp, axis=1))
                r = np.exp(lp-partition[:, None])
                m = o['mass'][sl]
                u = (o['xyz'][sl, None, :]-world[None, :, :3])/self.sigma[0]**2
                new = r[:, self.kh:]
                unew = u[:, self.kh:]
                n_all = float(np.sum(m[:, None]*r))
                n_new = float(np.sum(m[:, None]*new))
                complete[a, a] += n_all*eye
                complete[6:9, 6:9] += n_new*eye
                complete[a, 6:9] += n_new*eye@R
                complete[6:9, a] += n_new*R.T@eye
                score = np.zeros((len(m), 9))
                score[:, a] = np.einsum('nk,nkj->nj', r, u)
                score[:, 6:9] = np.einsum('nk,nkj->nj', new, unew)@R
                gradient += m@score
                # E[ss^T] minus E[s]E[s]^T, including zero score for other IDs.
                e_all = np.einsum('n,nk,nki,nkj->ij', m, r, u, u)
                e_new = np.einsum('n,nk,nki,nkj->ij', m, new, unew, unew)
                missing[a, a] += e_all
                missing[a, 6:9] += e_new@R
                missing[6:9, a] += R.T@e_new
                missing[6:9, 6:9] += R.T@e_new@R
                missing -= score.T@(m[:, None]*score)
        observed = complete-missing
        return gradient, (observed+observed.T)/2, complete, (missing+missing.T)/2

    def delta(self, theta):
        """Stable exact likelihood increment, avoiding total-likelihood cancellation."""
        total = 0.
        for t, o in enumerate(self.O):
            shift = np.tile(theta[3*t:3*t+3], (len(self.mean), 1))
            shift[self.kh:] += theta[6:9]@self.poses[t, :3, :3].T
            for start in range(0, len(o['xyz']), 2048):
                sl = slice(start, start+2048)
                lp, world = self.batch(t, sl, np.zeros(9))
                partition = np.logaddexp(self.other[t][sl], logsumexp(lp, axis=1))
                r = np.exp(lp-partition[:, None])
                residual = o['xyz'][sl, None, :]-world[None, :, :3]
                ratio = (np.einsum('nkj,kj->nk', residual, shift)-.5*np.sum(shift**2, axis=1))/self.sigma[0]**2
                total += np.dot(o['mass'][sl], np.log1p(np.sum(r*np.expm1(ratio), axis=1)))
        return float(total)


def complete_profile(complete):
    """Known-responsibility local curvature only; never a physical error bound."""
    A, B, C = complete[:6, :6], complete[:6, 6:], complete[6:, 6:]
    profile = A-B@np.linalg.pinv(C, rcond=1e-12)@B.T
    return (profile+profile.T)/2
