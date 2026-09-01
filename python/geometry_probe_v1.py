#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        geometry_probe_v1.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Geometry and group helper: coordinates to point group, commutant, d_eff.
#
#  Purpose
#    Supplies the shared geometry routines: building an array from
#    coordinates, finding its sensor-permutation automorphism group, and
#    computing the commutant dimension and effective dimension.
#
#  Description
#    The commutant dimension is computed two ways, by the rank of the Reynolds
#    projector and by counting orbits on ordered index pairs, so the two act
#    as a check on each other. Also models a position-jittered circular array
#    for graceful-degradation studies.
#
#  Inputs
#    array coordinates supplied by the caller
#
#  Outputs
#    geometry_results.json when run directly
#
#  Usage
#    imported as a helper, or python3 geometry_probe_v1.py
#
#  Version
#    Created:       bundle v3
#    Last modified: bundle v7.2, 2026-09-01
#
#  Revision history
#    v3                  created
#    v4                  retained as the shared helper
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

"""Probe: does the array-symmetry STAP estimator extend to arbitrary geometries?

Part 1. For each geometry, find the sensor-permutation automorphism group directly from the
coordinates (all orthogonal maps about the centroid that permute the sensors), then compute the
commutant dimension and d_eff = M^2/dim(C). An independent orbit-on-ordered-pairs count is printed
as a cross-check. A random (asymmetric) array is included as the honest negative control.

Part 2. A near-symmetric circular array: an isotropic-diffuse interference covariance
R[m,n] = INR * J0(2*pi*||p_m - p_n||) + I is exactly dihedral-invariant for the ideal UCA. Sensor
positions are then jittered, breaking the symmetry. We report the explained fraction e(D_N) versus
jitter (how much symmetry survives) and the SINR loss of the group average versus its shrinkage
form at one jitter level.

All numbers are produced here; nothing is asserted that this code does not compute.
"""
import numpy as np
from scipy.special import j0

import stap_sinr_v2 as S


# ----------------------------- geometries -----------------------------
def uca(N, spacing=0.5):
    R = spacing / (2 * np.sin(np.pi / N))
    n = np.arange(N)
    return np.c_[R * np.cos(2 * np.pi * n / N), R * np.sin(2 * np.pi * n / N)]


def ura(P, Q, d=0.5):
    xs, ys = np.meshgrid(np.arange(P) * d, np.arange(Q) * d)
    return np.c_[xs.ravel(), ys.ravel()]


def hexagon_ring(M=6, spacing=0.5):
    # M points on a regular polygon (M=6 gives a hexagon ring), dihedral D_M
    R = spacing / (2 * np.sin(np.pi / M))
    n = np.arange(M)
    return np.c_[R * np.cos(2 * np.pi * n / M), R * np.sin(2 * np.pi * n / M)]


def hex_lattice_19(d=0.5):
    # centered hexagonal cluster: center + ring of 6 + ring of 12 (19 points), point group D_6
    pts = [(0.0, 0.0)]
    for ring, count, r in [(1, 6, d), (2, 12, None)]:
        if ring == 1:
            for k in range(6):
                a = np.pi / 3 * k
                pts.append((d * np.cos(a), d * np.sin(a)))
        else:
            # second shell of a triangular lattice: 6 vertices at 2d and 6 edge-midpoints at sqrt(3)d
            for k in range(6):
                a = np.pi / 3 * k
                pts.append((2 * d * np.cos(a), 2 * d * np.sin(a)))
            for k in range(6):
                a = np.pi / 3 * k + np.pi / 6
                pts.append((np.sqrt(3) * d * np.cos(a), np.sqrt(3) * d * np.sin(a)))
    return np.array(pts)


def random_array(M=16, seed=7):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, 2))


# ----------------------- symmetry group from coordinates -----------------------
def find_point_group(pos, tol=1e-6):
    pos = pos - pos.mean(0)
    M = len(pos)
    r = np.hypot(pos[:, 0], pos[:, 1])
    scale = max(r.max(), 1e-9)
    i0 = int(np.argmax(r))
    if r[i0] < tol * scale:
        return [np.arange(M)]
    ang = np.arctan2(pos[:, 1], pos[:, 0])
    cands = []
    for j in range(M):
        if abs(r[j] - r[i0]) > tol * scale:
            continue
        da = ang[j] - ang[i0]
        c, s = np.cos(da), np.sin(da)
        cands.append(np.array([[c, -s], [s, c]]))              # rotation v0 -> vj
        phi = 0.5 * (ang[i0] + ang[j])
        c2, s2 = np.cos(2 * phi), np.sin(2 * phi)
        cands.append(np.array([[c2, s2], [s2, -c2]]))          # reflection v0 -> vj
    perms, seen = [], set()
    for O in cands:
        mapped = pos @ O.T
        sigma = np.full(M, -1, int)
        used, ok = set(), True
        for i in range(M):
            d = np.hypot(pos[:, 0] - mapped[i, 0], pos[:, 1] - mapped[i, 1])
            k = int(np.argmin(d))
            if d[k] > tol * scale or k in used:
                ok = False
                break
            sigma[i] = k
            used.add(k)
        if ok:
            key = tuple(sigma)
            if key not in seen:
                seen.add(key)
                perms.append(sigma)
    return perms


def num_orbitals(perms, M):
    seen, c = set(), 0
    for i in range(M):
        for j in range(M):
            if (i, j) in seen:
                continue
            c += 1
            for s in perms:
                seen.add((int(s[i]), int(s[j])))
    return c


def gain_row(name, pos):
    M = len(pos)
    perms = find_point_group(pos)
    Pm = [S.perm_mat(p) for p in perms]
    dimC = S.commutant_dim(Pm, M)
    orb = num_orbitals(perms, M)
    deff = M * M / dimC
    return name, M, len(perms), dimC, orb, deff


# ----------------------- near-symmetric degradation -----------------------
def diffuse_R(pos, inr_db=10.0):
    k = 2 * np.pi
    D = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    C = j0(k * D)
    R = (10 ** (inr_db / 10.0)) * C + np.eye(len(pos))
    return 0.5 * (R + R.T)


def steer(pos, theta):
    k = 2 * np.pi
    return np.exp(1j * k * (pos[:, 0] * np.cos(theta) + pos[:, 1] * np.sin(theta)))


def dihedral_Pm(N):
    return [S.perm_mat(p) for p in S.dihedral_perms(N)]


def explained_fraction_vs_jitter(N=16, jitters=(0.0, 0.02, 0.05, 0.1, 0.2), seed=11):
    rng = np.random.default_rng(seed)
    Pm = dihedral_Pm(N)
    base = uca(N)
    out = []
    for jit in jitters:
        # average explained fraction over a few jitter draws
        es = []
        for _ in range(20 if jit > 0 else 1):
            pos = base + (jit * rng.standard_normal(base.shape) if jit > 0 else 0.0)
            R = diffuse_R(pos)
            G = S.reynolds(R, Pm)
            es.append(np.linalg.norm(G, "fro") ** 2 / np.linalg.norm(R, "fro") ** 2)
        out.append((jit, float(np.mean(es))))
    return out


def sinr_vs_K(N=16, jitter=0.05, T=300, Ks=(2, 4, 8, 16, 32, 64), seed=13):
    rng = np.random.default_rng(seed)
    Pm = dihedral_Pm(N)
    pos = uca(N) + jitter * np.random.default_rng(99).standard_normal((N, 2))  # fixed jittered array
    R = diffuse_R(pos)
    e_dn = float(np.linalg.norm(S.reynolds(R, Pm), "fro") ** 2 / np.linalg.norm(R, "fro") ** 2)
    Lc = np.linalg.cholesky(R + 1e-9 * np.eye(N))
    s = steer(pos, 0.3)
    Rinv = np.linalg.inv(R)
    sinr_opt = np.real(s.conj() @ Rinv @ s)

    def loss(Rhat):
        try:
            w = np.linalg.solve(Rhat, s)
        except np.linalg.LinAlgError:
            return np.nan
        return (np.abs(w.conj() @ s) ** 2 / np.real(w.conj() @ R @ w)) / sinr_opt

    res = {k: [] for k in ["scm", "persym", "group", "shrink"]}
    for K in Ks:
        acc = {k: [] for k in res}
        for _ in range(T):
            Z = (rng.standard_normal((N, K)) + 1j * rng.standard_normal((N, K))) / np.sqrt(2)
            X = Lc @ Z
            Sh = (X @ X.conj().T) / K
            G = S.reynolds(Sh, Pm)
            acc["scm"].append(loss(Sh) if K >= N else np.nan)
            acc["persym"].append(loss(S.persym(Sh)))
            acc["group"].append(loss(G))
            Rs, _ = S.lw_shrink_to_target(X, G)
            acc["shrink"].append(loss(Rs))
        for k in acc:
            v = np.array(acc[k])
            v = v[np.isfinite(v) & (v > 0)]
            res[k].append(10 * np.log10(np.mean(v)) if v.size else None)
    return e_dn, Ks, res


if __name__ == "__main__":
    import json
    print("PART 1: symmetry gain by geometry (d_eff = M^2 / dim C)\n")
    specs = [
        ("UCA N=16 (baseline)", uca(16)),
        ("UCA N=12", uca(12)),
        ("URA 4x4 (square)", ura(4, 4)),
        ("URA 2x8 (rectangle)", ura(2, 8)),
        ("Hexagon ring (6)", hexagon_ring(6)),
        ("Hex lattice (19)", hex_lattice_19()),
        ("Random array (M=16)", random_array(16)),
    ]
    rows = [gain_row(nm, pos) for nm, pos in specs]
    print("  %-22s %3s %5s %6s %7s %8s" % ("geometry", "M", "|G|", "dimC", "orbits", "d_eff"))
    for nm, M, g, dC, orb, de in rows:
        flag = "  <- cross-check OK" if dC == orb else "  <- MISMATCH"
        print("  %-22s %3d %5d %6d %7d %8.2f%s" % (nm, M, g, dC, orb, de, flag))

    print("\nPART 2: near-symmetric UCA (isotropic-diffuse interference), N=16\n")
    print("  explained fraction e(D_N) vs position jitter (fraction of wavelength):")
    evj = explained_fraction_vs_jitter()
    for jit, e in evj:
        print("    jitter=%.2f  e(D_N)=%.4f" % (jit, e))

    e_dn, Ks, res = sinr_vs_K(jitter=0.05)
    print("\n  SINR loss (dB) at jitter=0.05 lambda  [e(D_N)=%.4f]:" % e_dn)
    print("    %4s %8s %8s %8s %8s" % ("K", "SCM", "persym", "group", "shrink"))
    for i, K in enumerate(Ks):
        def f(k):
            return "%7.2f" % res[k][i] if res[k][i] is not None else "    -  "
        print("    %4d %s %s %s %s" % (K, f("scm"), f("persym"), f("group"), f("shrink")))

    json.dump({
        "geometry": [{"name": nm, "M": M, "G": g, "dimC": dC, "orbits": orb, "deff": de}
                     for nm, M, g, dC, orb, de in rows],
        "jitter_efrac": [{"jitter": j, "e_DN": e} for j, e in evj],
        "jitter_sinr": {"e_DN": e_dn, "Ks": list(Ks), **res},
    }, open("geometry_results.json", "w"), indent=1)
