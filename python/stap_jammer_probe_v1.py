#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        stap_jammer_probe_v1.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Compares a near-white and an interference-limited scenario side by side.
#
#  Purpose
#    Documents why the interference-limited scenario is used: a covariance
#    drawn as a group average of a random Wishart matrix is nearly white, so
#    the conventional beamformer is already near the clairvoyant optimum and
#    the comparison does not exercise adaptive nulling.
#
#  Description
#    Reports conditioning, interference rank, group-invariance deviation and
#    the available adaptive gain of each scenario, then runs the full estimator
#    comparison on both.
#
#  Inputs
#    none; both scenarios generated in-script
#
#  Outputs
#    stap_jammer_probe.json
#
#  Usage
#    python3 stap_jammer_probe_v1.py
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

"""Interference-limited scenario probe.

The v4 experiment drew the interference covariance as a dihedral average of a
random Wishart matrix. That covariance is nearly white (condition number about
1.5 at M = 16), so the conventional beamformer already sits within 0.06 dB of
the clairvoyant optimum and every estimator that returns a well-conditioned
matrix scores well. It does not exercise adaptive nulling.

This probe builds an interference-limited alternative that is still exactly
group invariant: J strong jammers placed on a rotation orbit of the array
symmetry group. The covariance is then invariant under the D_J subgroup of the
array's D_M automorphism group, is rank J, and leaves the conventional
beamformer tens of decibels from optimum.

Reports, for both scenarios, the conditioning, the available adaptive gain, and
the SINR loss of every estimator including diagonal loading with an oracle-tuned
level at each K.
"""
import json
import numpy as np

import stap_sinr_v3 as S

DL_GRID = np.logspace(-3.0, 2.0, 26)


def subgroup_perms(M, J, reflections=True):
    """Rotations by multiples of M/J sensor positions, optionally with reflections."""
    idx = np.arange(M)
    step = M // J
    ps = [(idx + r * step) % M for r in range(J)]
    if reflections:
        ps += [(r * step - idx) % M for r in range(J)]
    return ps


def jammer_ring_R(M, J, inr_db=30.0, radius=1.5, az0=0.0):
    """J equal-power jammers on a D_J rotation orbit, plus a unit white floor."""
    inr = 10.0 ** (inr_db / 10.0)
    Rint = np.zeros((M, M), complex)
    for k in range(J):
        a = S.uca_steer(M, az=az0 + 2 * np.pi * k / J, r=radius)
        Rint += np.outer(a, a.conj())
    Rint = Rint / np.trace(Rint) * M
    R = inr * Rint + np.eye(M)
    return 0.5 * (R + R.conj().T)


def scenario_stats(R, Pm, s):
    Rinv = np.linalg.inv(R)
    opt = np.real(s.conj() @ Rinv @ s)
    quiescent = np.abs(s.conj() @ s) ** 2 / np.real(s.conj() @ R @ s)
    w = np.linalg.eigvalsh(R)
    dev = max(np.linalg.norm(Q @ R @ Q.T - R) for Q in Pm) / np.linalg.norm(R)
    return {
        "cond": float(w[-1] / w[0]),
        "rank_above_floor": int(np.sum(w > 1.01 * w[0])),
        "invariance_rel_dev": float(dev),
        "adaptive_gain_db": float(-10 * np.log10(quiescent / opt)),
    }


def compare(M, R, Pm, perms, s, T=400, Ks=None, seed=7):
    rng = np.random.default_rng(seed)
    Lc = np.linalg.cholesky(R)
    sinr_opt = np.real(s.conj() @ np.linalg.inv(R) @ s)
    dimC = S.commutant_dim(Pm, M)

    def loss(Rhat):
        try:
            w = np.linalg.solve(Rhat, s)
        except np.linalg.LinAlgError:
            return np.nan
        return (np.abs(w.conj() @ s) ** 2 / np.real(w.conj() @ R @ w)) / sinr_opt

    Ks = Ks or [2, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64]
    out = {k: [] for k in ['scm', 'dl', 'persym', 'group', 'shrink']}
    out['Ks'] = Ks
    out['dl_level'] = []
    out['rho'] = []
    for K in Ks:
        acc = {k: [] for k in ['scm', 'persym', 'group', 'shrink']}
        dl_acc = [[] for _ in DL_GRID]
        rho_acc = []
        for _ in range(T):
            Z = (rng.standard_normal((M, K)) + 1j * rng.standard_normal((M, K))) / np.sqrt(2)
            X = Lc @ Z
            Sh = (X @ X.conj().T) / K
            G = S.reynolds(Sh, Pm)
            acc['scm'].append(loss(Sh) if K >= M else np.nan)
            acc['persym'].append(loss(S.persym(Sh)))
            acc['group'].append(loss(G))
            rho = S.shrink_intensity(X, Sh, G, perms)
            rho_acc.append(rho)
            acc['shrink'].append(loss(rho * G + (1 - rho) * Sh))
            scale = np.real(np.trace(Sh)) / M
            for j, c in enumerate(DL_GRID):
                dl_acc[j].append(loss(Sh + c * scale * np.eye(M)))

        def db(v):
            v = np.array(v)
            v = v[np.isfinite(v) & (v > 0)]
            return float(10 * np.log10(np.mean(v))) if v.size else None

        for k in acc:
            out[k].append(db(acc[k]))
        dl_db = [db(a) for a in dl_acc]
        best = int(np.nanargmax([-1e9 if v is None else v for v in dl_db]))
        out['dl'].append(dl_db[best])
        out['dl_level'].append(float(DL_GRID[best]))
        out['rho'].append(float(np.mean(rho_acc)))
    out['M'] = M
    out['dimC'] = dimC
    out['deff'] = M * M / dimC
    return out


def _show(r, name, stats):
    print(f"--- {name} ---")
    print(f"    dim C = {r['dimC']}, d_eff = {r['deff']:.2f}; "
          f"cond(R) = {stats['cond']:.1f}, rank = {stats['rank_above_floor']}, "
          f"adaptive gain = {stats['adaptive_gain_db']:.1f} dB, "
          f"invariance dev = {stats['invariance_rel_dev']:.1e}")
    print(f"{'K':>4} {'SCM':>8} {'LSMI*':>8} {'persym':>8} {'group':>8} {'shrink':>8} {'rho':>6}")
    for i, K in enumerate(r['Ks']):
        def f(k):
            return f"{r[k][i]:7.2f}" if r[k][i] is not None else "    -  "
        print(f"{K:>4} {f('scm')} {f('dl')} {f('persym')} {f('group')} {f('shrink')}"
              f" {r['rho'][i]:6.3f}")
    print()


if __name__ == "__main__":
    M = 16
    out = {}

    # (A) v4 scenario: dihedral average of a random Wishart, full D_M matched group
    permsA = S.dihedral_perms(M)
    PmA = [S.perm_mat(p) for p in permsA]
    RA = S.make_R(M, PmA, 0.0)
    sA = S.uca_steer(M, az=0.3)
    stA = scenario_stats(RA, PmA, sA)
    rA = compare(M, RA, PmA, permsA, sA)
    out['diffuse_DM'] = {'stats': stA, 'res': rA}

    # (B) interference-limited: 4 jammers on a D_4 orbit, matched group D_4
    J = 4
    permsB = subgroup_perms(M, J, reflections=True)
    PmB = [S.perm_mat(p) for p in permsB]
    RB = jammer_ring_R(M, J, inr_db=30.0, radius=1.5, az0=0.0)
    sB = S.uca_steer(M, az=2 * np.pi / J * 0.30, r=1.5)
    stB = scenario_stats(RB, PmB, sB)
    rB = compare(M, RB, PmB, permsB, sB)
    out['jammers_D4'] = {'stats': stB, 'res': rB}

    _show(rA, "(A) v4 scenario: near-white G-invariant covariance, full D_16", stA)
    _show(rB, "(B) interference-limited: four jammers on a D_4 orbit, INR 30 dB", stB)
    json.dump(out, open('stap_jammer_probe.json', 'w'), indent=1)
    print("LSMI* is diagonal loading with an oracle-tuned level at each K.")
