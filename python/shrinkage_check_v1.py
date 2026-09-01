#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        shrinkage_check_v1.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Verifies the shrinkage intensity against the published plug-in.
#
#  Purpose
#    Confirms that the intensity used here is the published estimator and
#    quantifies what the earlier, incorrect form cost.
#
#  Description
#    The earlier form placed the full sample-covariance residual in the
#    numerator. The published plug-in restricts it to the component
#    perpendicular to the commutant, since the in-commutant sample variance is
#    not reduced by moving toward the target and so must not raise the
#    intensity. This script reports both intensities and the SINR loss each
#    produces.
#
#  Inputs
#    none; scenario generated in-script
#
#  Outputs
#    shrinkage_check.json
#
#  Usage
#    python3 shrinkage_check_v1.py
#
#  Version
#    Created:       bundle v5, 2026-09-01
#    Last modified: bundle v7.2, 2026-09-01
#
#  Revision history
#    v5     2026-09-01   created
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

"""Compare the v4.1 shrinkage intensity against the published AD MSE plug-in.

v4.1 used
    rho = min(1, [ (1/K^2) sum_l || x_l x_l^H - S ||_F^2 ] / || S - Phi_G(S) ||_F^2 )
which places the FULL sample-covariance variance in the numerator.

arXiv:2605.17111 Eqs. (18)-(20) define the closed-form Frobenius-MSE plug-in with
the numerator restricted to the component perpendicular to the commutant,
    Vperp   = (1/K^2) sum_l || Pperp(x_l x_l^H) - Pperp(S) ||_F^2
    Vperp+D = || S - Phi_G(S) ||_F^2
    alpha   = clip( Vperp / (Vperp+D), 0, 1 )
Since Pperp(S) = S - Phi_G(S), the correct numerator drops the in-commutant
variance, which is not reduced by moving toward the target and therefore should
not push the intensity up.

This script reports both intensities and the SINR loss each produces, in the
matched and mismatched regimes.
"""
import json
import numpy as np

rng = np.random.default_rng(0)


def dihedral_perms(N):
    idx = np.arange(N)
    return [(idx + r) % N for r in range(N)] + [(r - idx) % N for r in range(N)]


def perm_mat(p):
    Q = np.zeros((len(p), len(p)))
    Q[np.arange(len(p)), p] = 1.0
    return Q


def reynolds(A, Pm):
    return sum(P @ A @ P.T for P in Pm) / len(Pm)


def commutant_dim(Pm, M):
    S = sum(np.kron(P, P) for P in Pm) / len(Pm)
    w = np.linalg.eigvalsh(0.5 * (S + S.T))
    return int(round((w > 0.5).sum()))


def uca_steer(M, az, r=0.7):
    n = np.arange(M)
    return np.exp(1j * 2 * np.pi * r * np.cos(az - 2 * np.pi * n / M))


def rho_v41(X, S, T):
    """v4.1 intensity: full residual variance in the numerator."""
    K = X.shape[1]
    d2 = np.linalg.norm(S - T, 'fro') ** 2
    b2 = sum(np.linalg.norm(np.outer(X[:, k], X[:, k].conj()) - S, 'fro') ** 2
             for k in range(K)) / K ** 2
    return float(np.clip(b2 / d2 if d2 > 0 else 1.0, 0.0, 1.0))


def rho_plugin(X, S, T, Pm):
    """Published plug-in: perpendicular component only (arXiv:2605.17111 (18)-(20))."""
    K = X.shape[1]
    d2 = np.linalg.norm(S - T, 'fro') ** 2
    Sperp = S - T
    b2 = 0.0
    for k in range(K):
        Ok = np.outer(X[:, k], X[:, k].conj())
        b2 += np.linalg.norm((Ok - reynolds(Ok, Pm)) - Sperp, 'fro') ** 2
    b2 /= K ** 2
    return float(np.clip(b2 / d2 if d2 > 0 else 1.0, 0.0, 1.0))


def make_R(M, Dm, mismatch=0.0, seed=0):
    rg = np.random.default_rng(seed)
    H = rg.standard_normal((M, M)) + 1j * rg.standard_normal((M, M))
    Rint = reynolds(0.5 * (H @ H.conj().T), Dm)
    Rint = Rint / np.trace(Rint) * M
    R = 10.0 * Rint + np.eye(M)
    if mismatch > 0:
        a = uca_steer(M, az=1.1)
        A = np.outer(a, a.conj())
        R = R + mismatch * 10.0 * A / np.trace(A) * M
    return 0.5 * (R + R.conj().T)


def run(M=16, mismatch=0.0, T=400, Ks=None):
    Dm = [perm_mat(p) for p in dihedral_perms(M)]
    R = make_R(M, Dm, mismatch)
    Lc = np.linalg.cholesky(R)
    s = uca_steer(M, az=0.3)
    sinr_opt = np.real(s.conj() @ np.linalg.inv(R) @ s)

    def loss(Rhat):
        w = np.linalg.solve(Rhat, s)
        return (np.abs(w.conj() @ s) ** 2 / np.real(w.conj() @ R @ w)) / sinr_opt

    Ks = Ks or [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    out = {'Ks': Ks, 'rho_v41': [], 'rho_plugin': [],
           'loss_v41': [], 'loss_plugin': [], 'loss_group': []}
    for K in Ks:
        r41, rpi, l41, lpi, lg = [], [], [], [], []
        for _ in range(T):
            Z = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)
            X = Lc @ Z
            S = (X @ X.conj().T) / K
            G = reynolds(S, Dm)
            a1 = rho_v41(X, S, G)
            a2 = rho_plugin(X, S, G, Dm)
            r41.append(a1)
            rpi.append(a2)
            l41.append(loss(a1 * G + (1 - a1) * S))
            lpi.append(loss(a2 * G + (1 - a2) * S))
            lg.append(loss(G))
        out['rho_v41'].append(float(np.mean(r41)))
        out['rho_plugin'].append(float(np.mean(rpi)))
        out['loss_v41'].append(float(10 * np.log10(np.mean(l41))))
        out['loss_plugin'].append(float(10 * np.log10(np.mean(lpi))))
        out['loss_group'].append(float(10 * np.log10(np.mean(lg))))
    return out


if __name__ == "__main__":
    M = 16
    Dm = [perm_mat(p) for p in dihedral_perms(M)]
    dimC = commutant_dim(Dm, M)
    print(f"M={M}  dim C={dimC}  d_eff={M*M/dimC:.2f}\n")

    res = {}
    for name, mm in [('matched', 0.0), ('mismatch', 0.30)]:
        r = run(M=M, mismatch=mm)
        res[name] = r
        print(f"--- {name} (mismatch={mm}) ---")
        print(f"{'K':>4} {'rho_v41':>9} {'rho_plug':>9} "
              f"{'loss_v41':>9} {'loss_plug':>10} {'loss_grp':>9}")
        for i, K in enumerate(r['Ks']):
            print(f"{K:>4} {r['rho_v41'][i]:9.3f} {r['rho_plugin'][i]:9.3f} "
                  f"{r['loss_v41'][i]:9.2f} {r['loss_plugin'][i]:10.2f} "
                  f"{r['loss_group'][i]:9.2f}")
        print()
    json.dump(res, open('shrinkage_check.json', 'w'), indent=1)
