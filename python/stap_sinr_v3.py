#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        stap_sinr_v3.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       SINR-loss experiment for the matched and mismatched regimes.
#
#  Purpose
#    Measures how the average signal-to-interference-plus-noise ratio (SINR)
#    loss of an adaptive beamformer falls with training support, for five
#    covariance estimators, on an interference-limited uniform circular array.
#
#  Description
#    Builds an interference-plus-noise covariance from J equal-power jammers
#    placed on a rotation orbit of the array symmetry group, so the covariance
#    is exactly invariant under the dihedral subgroup D_J and is rank J. An
#    optional off-orbit jammer breaks the symmetry. Estimators compared: the
#    sample covariance; a diagonally loaded sample covariance whose loading
#    level is chosen by an oracle sweep at each snapshot count; the
#    forward-backward (persymmetric) average; the D_J group average; and the
#    convex shrinkage of the sample covariance toward the group average with
#    the intensity from the closed-form Frobenius plug-in of arXiv:2605.17111.
#    Also records the best member of an alpha grid as the shrinkage envelope.
#
#  Inputs
#    none; the scenario is generated in-script from fixed seeds
#
#  Outputs
#    stap_sinr.json  losses, intensities and scenario statistics
#
#  Usage
#    python3 stap_sinr_v3.py
#
#  Version
#    Created:       bundle v5, 2026-09-01
#    Last modified: bundle v7.2, 2026-09-01
#
#  Revision history
#    v5     2026-09-01   created; replaces stap_sinr_v2.py
#    v6     2026-09-01   interference-limited scenario; oracle loading
#                        baseline; published shrinkage plug-in; UCA
#                        reflection for the persymmetric baseline
#    v6     2026-09-01   records rogue_inr_db and trial count for
#                        provenance
#    v7.2   2026-09-01   source headers added
#
#  Author
#    Mitchell A. Thornton  <mitch@smu.edu>
#    ORCID 0000-0003-3559-9511
#
#  License
#    SPDX-License-Identifier: MIT
#    The MIT license grants copyright permissions only and grants no
#    rights under patents. See PATENTS.md.
#
#  Copyright (c) 2026 Mitchell A. Thornton
# ============================================================================

"""SINR loss versus training support on a uniform circular array, interference-limited.

Scenario. An M-element uniform circular array of radius r wavelengths observes J
equal-power jammers placed on a rotation orbit of the array symmetry group, at
azimuths 2*pi*k/J. The resulting interference-plus-noise covariance is exactly
invariant under the dihedral subgroup D_J of the array automorphism group D_M, is
rank J, and leaves the conventional beamformer tens of decibels from the
clairvoyant optimum, so adaptive nulling is the operative task. The matched group
is D_J, the largest subgroup of the array automorphism group that the field also
respects.

Mismatch regime. One additional jammer is placed off the orbit, so the covariance
carries a component outside the commutant and the pure projection is biased.

Estimators.
  SCM      sample covariance, defined only for K >= M
  LSMI*    diagonally loaded SCM with the loading level tuned by an oracle sweep
           at each K, so the curve upper-bounds any data-driven loading rule
  persym   forward-backward (persymmetric) average, the classical Z_2 exploit
  group    D_J group average, the Reynolds projection onto the commutant
  shrink   convex shrinkage of the SCM toward the group average, intensity from
           the closed-form Frobenius-MSE plug-in of arXiv:2605.17111 Eqs. (18)-(20)

v3 changes versus v2:
  - interference-limited scenario replaces the near-white random-Wishart covariance
  - diagonal loading promoted to a reported baseline with an oracle-tuned level
  - shrinkage intensity restricted to the perpendicular component (published plug-in)
"""
import json
import numpy as np

rng_global = np.random.default_rng(0)

# oracle grid for the diagonal-loading level, in units of tr(S)/M
DL_GRID = np.logspace(-3.0, 2.0, 26)


# ---------------------------------------------------------------- group helpers

def dihedral_perms(N):
    """Full dihedral group of the N-gon acting on sensor indices."""
    idx = np.arange(N)
    return [(idx + r) % N for r in range(N)] + [(r - idx) % N for r in range(N)]


def subgroup_perms(M, J, reflections=True):
    """D_J subgroup of D_M: rotations by multiples of M/J positions, plus reflections."""
    if M % J:
        raise ValueError("J must divide M")
    idx = np.arange(M)
    step = M // J
    ps = [(idx + r * step) % M for r in range(J)]
    if reflections:
        ps += [(r * step - idx) % M for r in range(J)]
    return ps


def perm_mat(p):
    Q = np.zeros((len(p), len(p)))
    Q[np.arange(len(p)), p] = 1.0
    return Q


def reynolds(A, Pm):
    return sum(P @ A @ P.T for P in Pm) / len(Pm)


def orbit_outer(x, perms):
    """(1/|G|) sum_g (P_g x)(P_g x)^H, formed from the index orbit."""
    Y = np.stack([x[p] for p in perms], axis=1)
    return (Y @ Y.conj().T) / len(perms)


def persym(A):
    """Forward-backward average over the array's reflection, the classical Z_2 exploit.

    On a uniform circular array with sensor m at azimuth 2*pi*m/M, the reflection
    that fixes the array is m -> (-m) mod M, so the exchange operator is the
    permutation matrix of that map rather than the anti-diagonal exchange matrix
    of a line array. This is the reflection whose invariant set is the
    centro-Hermitian covariances for this index convention.
    """
    M = A.shape[0]
    Q = perm_mat((-np.arange(M)) % M)
    return 0.5 * (A + Q @ A.conj() @ Q.T)


def commutant_dim(Pm, M):
    """dim of the commutant, as the rank of the Reynolds projector on M x M matrices."""
    S = sum(np.kron(P, P) for P in Pm) / len(Pm)
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    return int(round((w > 0.5).sum()))


def orbit_count(perms, M):
    """Independent cross-check: number of orbits of G on ordered index pairs."""
    seen = set()
    orbits = 0
    for i in range(M):
        for j in range(M):
            if (i, j) in seen:
                continue
            orbits += 1
            for p in perms:
                seen.add((int(p[i]), int(p[j])))
    return orbits


# ---------------------------------------------------------------- array and scenario

def uca_steer(M, az, r=1.5):
    n = np.arange(M)
    return np.exp(1j * 2 * np.pi * r * np.cos(az - 2 * np.pi * n / M))


def scenario_R(M, J, inr_db=30.0, radius=1.5, az0=0.0,
               rogue_az=None, rogue_inr_db=20.0):
    """J jammers on a D_J orbit, plus a unit white floor, plus an optional off-orbit jammer."""
    inr = 10.0 ** (inr_db / 10.0)
    Rint = np.zeros((M, M), complex)
    for k in range(J):
        a = uca_steer(M, az=az0 + 2 * np.pi * k / J, r=radius)
        Rint += np.outer(a, a.conj())
    Rint = Rint / np.trace(Rint) * M
    R = inr * Rint + np.eye(M)
    if rogue_az is not None:
        a = uca_steer(M, az=rogue_az, r=radius)
        A = np.outer(a, a.conj())
        R = R + (10.0 ** (rogue_inr_db / 10.0)) * A / np.trace(A) * M
    return 0.5 * (R + R.conj().T)


def scenario_stats(R, Pm, s):
    Rinv = np.linalg.inv(R)
    opt = np.real(s.conj() @ Rinv @ s)
    quiescent = np.abs(s.conj() @ s) ** 2 / np.real(s.conj() @ R @ s)
    w = np.linalg.eigvalsh(R)
    dev = max(np.linalg.norm(Q @ R @ Q.T - R) for Q in Pm) / np.linalg.norm(R)
    return {"cond": float(w[-1] / w[0]),
            "rank_above_floor": int(np.sum(w > 1.01 * w[0])),
            "invariance_rel_dev": float(dev),
            "adaptive_gain_db": float(-10 * np.log10(quiescent / opt))}


# ---------------------------------------------------------------- shrinkage

def shrink_intensity(X, S, T, perms):
    """Closed-form Frobenius-MSE plug-in, arXiv:2605.17111 Eqs. (18)-(20).

    Numerator is the variance of the component perpendicular to the commutant,
    Vperp = (1/K^2) sum_l || Pperp(x_l x_l^H) - Pperp(S) ||_F^2 with
    Pperp(A) = A - Phi_G(A); denominator is || S - Phi_G(S) ||_F^2 = Vperp + D.
    Restricting to the perpendicular component removes the in-commutant sample
    variance, which shrinkage toward the target cannot reduce.
    """
    K = X.shape[1]
    Sperp = S - T
    d2 = np.linalg.norm(Sperp, 'fro') ** 2
    if d2 <= 0:
        return 1.0
    v = 0.0
    for k in range(K):
        xk = X[:, k]
        Ok = np.outer(xk, xk.conj())
        v += np.linalg.norm((Ok - orbit_outer(xk, perms)) - Sperp, 'fro') ** 2
    return float(np.clip(v / K ** 2 / d2, 0.0, 1.0))


# ---------------------------------------------------------------- experiment

def run(M=16, J=4, inr_db=30.0, radius=1.5, rogue_az=None, rogue_inr_db=20.0,
        T=400, Ks=None, seed=7):
    rng = np.random.default_rng(seed)
    perms = subgroup_perms(M, J, reflections=True)
    Pm = [perm_mat(p) for p in perms]
    dimC = commutant_dim(Pm, M)
    R = scenario_R(M, J, inr_db, radius, 0.0, rogue_az, rogue_inr_db)
    s = uca_steer(M, az=2 * np.pi / J * 0.30, r=radius)
    stats = scenario_stats(R, Pm, s)
    Lc = np.linalg.cholesky(R)
    sinr_opt = np.real(s.conj() @ np.linalg.inv(R) @ s)

    def loss(Rhat):
        try:
            w = np.linalg.solve(Rhat, s)
        except np.linalg.LinAlgError:
            return np.nan
        return (np.abs(w.conj() @ s) ** 2 / np.real(w.conj() @ R @ w)) / sinr_opt

    Ks = Ks or [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    res = {k: [] for k in ['scm', 'dl', 'persym', 'group', 'shrink', 'shrink_oracle']}
    res['Ks'] = Ks
    res['dl_level'] = []
    res['rho'] = []
    res['rho_oracle'] = []
    rho_grid = np.linspace(0.0, 1.0, 21)
    for K in Ks:
        acc = {k: [] for k in ['scm', 'persym', 'group', 'shrink']}
        dl_acc = [[] for _ in DL_GRID]
        rho_grid_acc = [[] for _ in rho_grid]
        rho_acc = []
        for _ in range(T):
            Z = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)
            X = Lc @ Z
            Sh = (X @ X.conj().T) / K
            G = reynolds(Sh, Pm)
            acc['scm'].append(loss(Sh) if K >= M else np.nan)
            acc['persym'].append(loss(persym(Sh)))
            acc['group'].append(loss(G))
            rho = shrink_intensity(X, Sh, G, perms)
            rho_acc.append(rho)
            acc['shrink'].append(loss(rho * G + (1 - rho) * Sh))
            scale = np.real(np.trace(Sh)) / M
            for j, c in enumerate(DL_GRID):
                dl_acc[j].append(loss(Sh + c * scale * np.eye(M)))
            for j, g in enumerate(rho_grid):
                rho_grid_acc[j].append(loss(g * G + (1 - g) * Sh))

        def db(v):
            v = np.array(v)
            v = v[np.isfinite(v) & (v > 0)]
            return float(10 * np.log10(np.mean(v))) if v.size else None

        for k in acc:
            res[k].append(db(acc[k]))
        dl_db = [db(a) for a in dl_acc]
        best = int(np.nanargmax([-1e9 if v is None else v for v in dl_db]))
        res['dl'].append(dl_db[best])
        res['dl_level'].append(float(DL_GRID[best]))
        res['rho'].append(float(np.mean(rho_acc)))
        rg_db = [db(a) for a in rho_grid_acc]
        bg = int(np.nanargmax([-1e9 if v is None else v for v in rg_db]))
        res['shrink_oracle'].append(rg_db[bg])
        res['rho_oracle'].append(float(rho_grid[bg]))
    res.update({'M': M, 'J': J, 'dimC': dimC, 'deff': M * M / dimC,
                'inr_db': inr_db, 'radius': radius, 'trials': T,
                'rogue_az': rogue_az,
                'rogue_inr_db': (rogue_inr_db if rogue_az is not None else None),
                'stats': stats})
    return res


def _show(r, name):
    st = r['stats']
    print(f"--- {name} ---")
    print(f"    M={r['M']} J={r['J']} dim C={r['dimC']} d_eff={r['deff']:.2f} | "
          f"cond(R)={st['cond']:.0f} rank={st['rank_above_floor']} "
          f"adaptive gain={st['adaptive_gain_db']:.1f} dB "
          f"invariance dev={st['invariance_rel_dev']:.1e}")
    print(f"{'K':>4} {'SCM':>8} {'LSMI*':>8} {'persym':>8} {'group':>8} {'shrink':>8}"
          f" {'rho':>6} {'shrink*':>8} {'rho*':>6}")
    for i, K in enumerate(r['Ks']):
        def f(k):
            return f"{r[k][i]:7.2f}" if r[k][i] is not None else "    -  "
        print(f"{K:>4} {f('scm')} {f('dl')} {f('persym')} {f('group')} {f('shrink')}"
              f" {r['rho'][i]:6.3f} {f('shrink_oracle')} {r['rho_oracle'][i]:6.2f}")
    print()


if __name__ == "__main__":
    matched = run()
    mism = run(rogue_az=2 * np.pi / 4 * 0.62, rogue_inr_db=15.0)
    json.dump({'matched': matched, 'mismatch': mism}, open('stap_sinr.json', 'w'), indent=1)
    _show(matched, "MATCHED (four jammers on a D_4 orbit, INR 30 dB)")
    _show(mism, "MISMATCH (one additional off-orbit jammer at 15 dB)")
    print("LSMI* is diagonal loading with an oracle-tuned level at each K, so it")
    print("upper-bounds any data-driven loading rule.")
