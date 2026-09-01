#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        validate_repo.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Validates a checkout of the public repository.
#
#  Purpose
#    Lets a push be verified from the receiving end. Structure, hygiene and
#    policy are checked inside a clone, without running the long experiments,
#    so a mistake in what was published is caught where it matters.
#
#  Description
#    Checks that the required files are present, that no build artifact or
#    result file has been committed, that every script compiles and carries a
#    header with an SPDX identifier and a copyright line, and that no text
#    file mentions the manuscript or its venue. Inside a git work tree it also
#    reports branch, remote and cleanliness and fails if an ignored-class file
#    is tracked. With --smoke it adds a short numerical check: the group
#    invariance of the scenario, the agreement of the two commutant-dimension
#    computations, and the rotation identity.
#
#  Inputs
#    the repository clone containing this file
#
#  Outputs
#    a pass or fail report; non-zero exit on any failure
#
#  Usage
#    python3 validate_repo.py
#    python3 validate_repo.py --smoke
#
#  Version
#    Created:       bundle v7.2, 2026-09-01
#    Last modified: bundle v7.3, 2026-09-01
#
#  Revision history
#    v7.2   2026-09-01   created
#    v7.3   2026-09-01   source headers added
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

"""Validate a checkout of the public repository.

Checks the structure, hygiene and policy of a repository clone without running
the long experiments. It is written to be run from inside a clone, so that a
push can be verified from the receiving end rather than only from the machine
that produced it:

    python3 validate_repo.py            structural and policy checks
    python3 validate_repo.py --smoke    the above, plus a short numerical check

Checks performed:
  1. required files are present
  2. no file that should never be committed is present or tracked
  3. every Python file compiles
  4. every Python file carries the standard header, an SPDX identifier and a
     copyright line
  5. no text file mentions the manuscript or its venue
  6. when run inside a git work tree, reports branch, remote and cleanliness,
     and fails if an ignored-class file is tracked
  7. with --smoke, the group invariance of the scenario, the agreement of the
     two commutant-dimension computations and the rotation identity

Exits non-zero on any failure, so it can be used in a hook or a workflow.
"""
import pathlib
import re
import subprocess
import sys

# the --smoke import must not leave a __pycache__ behind in the clone
sys.dont_write_bytecode = True

REQUIRED = [
    "README.md",
    "LICENSE",
    "PATENTS.md",
    ".gitignore",
    "validate_repo.py",
    "python/requirements.txt",
    "python/stap_sinr_v3.py",
    "python/stap_scaling_v1.py",
    "python/nll_calib_probe_v1.py",
    "python/stap_tables_v2.py",
    "python/make_stap_figs_v4.py",
    "python/emit_tex_v1.py",
    "python/provenance_v1.py",
    "python/shrinkage_check_v1.py",
    "python/stap_jammer_probe_v1.py",
    "python/geometry_probe_v1.py",
]

# patterns that must never appear in the repository
UNWANTED_GLOBS = ["**/*.pdf", "**/*.json", "**/__pycache__", "**/.DS_Store",
                  "**/figures", "**/generated", "**/*.aux", "**/*.log"]

# The terms are assembled from fragments so that this file, which scans every
# text file in the repository including itself, does not trip its own check.
FORBIDDEN_TERMS = [
    "ICA" + "SSP",
    "Beyond " + "Persym" + "metry",
    r"\bthe " + r"paper\b",
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".gitignore", ""}

PASS, FAIL = "PASS", "FAIL"


class Report:
    def __init__(self):
        self.rows = []
        self.failed = False

    def add(self, ok, name, detail=""):
        self.rows.append((PASS if ok else FAIL, name, detail))
        if not ok:
            self.failed = True

    def show(self):
        width = max(len(n) for _, n, _ in self.rows)
        for status, name, detail in self.rows:
            line = f"  [{status}] {name.ljust(width)}"
            if detail:
                line += f"  {detail}"
            print(line)
        print()
        if self.failed:
            print("VALIDATION FAILED")
        else:
            print("VALIDATION PASSED")


def git(root, *args):
    try:
        r = subprocess.run(["git", "-C", str(root), *args],
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def check_required(root, rep):
    missing = [f for f in REQUIRED if not (root / f).exists()]
    rep.add(not missing, "required files present",
            "missing: " + ", ".join(missing) if missing else
            f"{len(REQUIRED)} files")


def check_unwanted(root, rep):
    """Build artifacts must not be tracked; on disk they are fine if ignored.

    Running the scripts inside a clone legitimately creates result files,
    figures and caches. What matters is that git ignores them, not that they are
    absent, so inside a work tree this checks tracked files and reports ignored
    ones as a note. Outside a work tree there is no ignore mechanism to rely on,
    so presence is the failure.
    """
    present = []
    for pat in UNWANTED_GLOBS:
        for p in root.glob(pat):
            if ".git/" in str(p) or p.name == ".gitignore":
                continue
            present.append(str(p.relative_to(root)))
    present = sorted(set(present))

    top = git(root, "rev-parse", "--show-toplevel")
    in_clone = top is not None and pathlib.Path(top).resolve() == root
    if not in_clone:
        rep.add(not present, "no build artifacts present",
                "found: " + ", ".join(present[:6]) if present else "clean")
        return

    tracked = set((git(root, "ls-files") or "").split("\n"))
    bad = [f for f in present
           if f in tracked or any(t.startswith(f + "/") for t in tracked)]
    note = "clean"
    if not bad and present:
        note = f"{len(present)} ignored artifact(s) on disk, none tracked"
    rep.add(not bad, "no build artifacts tracked",
            "tracked: " + ", ".join(bad[:6]) if bad else note)


def check_compiles(root, rep):
    bad = []
    for p in sorted(root.rglob("*.py")):
        try:
            compile(p.read_text(), str(p), "exec")
        except SyntaxError as e:
            bad.append(f"{p.relative_to(root)}:{e.lineno}")
    rep.add(not bad, "every Python file compiles",
            ", ".join(bad) if bad else
            f"{len(list(root.rglob('*.py')))} files")


def check_headers(root, rep):
    bad = []
    for p in sorted(root.rglob("*.py")):
        t = p.read_text()
        if "#  File:" not in t or "SPDX-License-Identifier" not in t \
                or "Copyright (c)" not in t:
            bad.append(str(p.relative_to(root)))
    rep.add(not bad, "headers, SPDX and copyright on every script",
            ", ".join(bad) if bad else "complete")


def check_forbidden(root, rep):
    hits = []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git/" in str(p):
            continue
        if p.suffix not in TEXT_SUFFIXES and p.name != ".gitignore":
            continue
        text = p.read_text(errors="ignore")
        for pat in FORBIDDEN_TERMS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{p.relative_to(root)}:{line}")
    rep.add(not hits, "no manuscript or venue references",
            ", ".join(hits[:5]) if hits else "clean")


def check_git(root, rep):
    # Only treat this as a clone when the work tree root is this directory.
    # Validating a staged copy that merely sits inside some other repository
    # must not report that copy's tracked files as if they were the clone's.
    top = git(root, "rev-parse", "--show-toplevel")
    if top is None or pathlib.Path(top).resolve() != root:
        rep.add(True, "git work tree",
                "not a clone root; structural checks only")
        return
    branch = git(root, "rev-parse", "--abbrev-ref", "HEAD") or "unknown"
    remote = git(root, "remote", "get-url", "origin") or "no origin"
    status = git(root, "status", "--porcelain")
    rep.add(status == "", "working tree clean",
            f"branch {branch}, origin {remote}" if status == ""
            else "uncommitted changes present")
    tracked = git(root, "ls-files") or ""
    leaked = [f for f in tracked.split("\n")
              if f.endswith((".pdf", ".json", ".aux", ".log"))
              or "__pycache__" in f or f.endswith(".DS_Store")]
    rep.add(not leaked, "no ignored-class file tracked",
            ", ".join(leaked[:5]) if leaked else "clean")
    count = len([f for f in tracked.split("\n") if f])
    rep.add(count > 0, "files tracked", f"{count}")


def check_smoke(root, rep):
    sys.path.insert(0, str(root / "python"))
    try:
        import numpy as np
        import stap_sinr_v3 as S
    except Exception as e:
        rep.add(False, "numerical smoke", f"import failed: {e}")
        return
    try:
        M, J = 16, 4
        perms = S.subgroup_perms(M, J, True)
        Pm = [S.perm_mat(p) for p in perms]
        R = S.scenario_R(M, J, 30.0, 1.5)
        dev = max(np.linalg.norm(Q @ R @ Q.T - R) for Q in Pm) / np.linalg.norm(R)
        pdev = np.linalg.norm(S.persym(R) - R) / np.linalg.norm(R)
        dc, oc = S.commutant_dim(Pm, M), S.orbit_count(perms, M)
        step, p = M // J, 3
        a = S.uca_steer(M, az=0.37, r=1.5)
        lem = np.max(np.abs(S.perm_mat(perms[p]) @ a
                            - S.uca_steer(M, az=0.37 - 2 * np.pi * step * p / M,
                                          r=1.5)))
        rep.add(dev < 1e-12, "scenario is group invariant", f"{dev:.1e}")
        rep.add(pdev < 1e-12, "persymmetric baseline matched", f"{pdev:.1e}")
        rep.add(dc == oc == 34, "commutant dimension agrees with orbit count",
                f"{dc} and {oc}")
        rep.add(lem < 1e-12, "rotation identity holds", f"{lem:.1e}")
    except Exception as e:
        rep.add(False, "numerical smoke", f"{type(e).__name__}: {e}")


def main():
    root = pathlib.Path(__file__).resolve().parent
    smoke = "--smoke" in sys.argv
    print(f"Validating repository at {root}\n")
    rep = Report()
    check_required(root, rep)
    check_unwanted(root, rep)
    check_compiles(root, rep)
    check_headers(root, rep)
    check_forbidden(root, rep)
    check_git(root, rep)
    if smoke:
        check_smoke(root, rep)
    else:
        print("  (run with --smoke to add the numerical checks)\n")
    rep.show()
    sys.exit(1 if rep.failed else 0)


if __name__ == "__main__":
    main()
