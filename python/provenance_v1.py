#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================================
#  File:        provenance_v1.py
#  Project:     Symmetry-Group Covariance Estimation for Adaptive Arrays
#  Brief:       Records which machine produced the current results.
#
#  Purpose
#    Makes the provenance of the reported numbers checkable after the fact,
#    since the rendered document carries whatever machine last ran the
#    experiments.
#
#  Description
#    Captures host, platform, CPU string, interpreter and library versions,
#    and the first sixteen hex digits of the SHA-256 digest of each experiment
#    script and each result file.
#
#  Inputs
#    the experiment scripts and the four result files
#
#  Outputs
#    PROVENANCE.md, provenance.json
#
#  Usage
#    python3 scripts/provenance_v1.py   # from the bundle root
#
#  Version
#    Created:       bundle v6, 2026-09-01
#    Last modified: bundle v7.2, 2026-09-01
#
#  Revision history
#    v6     2026-09-01   created
#    v7     2026-09-01   tracks make_stap_figs_v4.py
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

"""Record which machine produced the current set of results.

Every experimental value is a macro emitted by emit_tex_v1.py from the result
JSON, so a rendered document carries whatever machine last ran the experiments.
This script writes that fact down so the provenance is checkable afterwards.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

SCRIPTS = ["stap_sinr_v3.py", "stap_tables_v2.py", "stap_scaling_v1.py",
           "nll_calib_probe_v1.py", "make_stap_figs_v4.py", "emit_tex_v1.py",
           "apply_headers.py", "sync_github_image.py", "validate_repo.py"]


def digest(path):
    if not os.path.exists(path):
        return "missing"
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    try:
        import numpy
        npv = numpy.__version__
    except Exception:
        npv = "unavailable"
    try:
        import scipy
        spv = scipy.__version__
    except Exception:
        spv = "unavailable"
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        cpu = ""
    if not cpu:
        cpu = platform.processor() or "unknown"

    rec = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "python": sys.version.split()[0],
        "numpy": npv,
        "scipy": spv,
        "script_sha256_16": {s: digest(os.path.join(here, s)) for s in SCRIPTS},
        "results_sha256_16": {j: digest(os.path.join(root, j)) for j in
                              ("stap_sinr.json", "stap_tables.json",
                               "stap_scaling.json", "nll_calib.json")},
    }
    json.dump(rec, open(os.path.join(root, "provenance.json"), "w"), indent=1)

    md = ["# Provenance",
          "",
          "Every experimental number is emitted from the result JSON by",
          "`emit_tex_v1.py`, so a rendered document reports whatever machine last",
          "ran the experiments. This file records that machine.",
          "",
          f"- generated: {rec['generated_utc']}",
          f"- host: {rec['hostname']}",
          f"- platform: {rec['platform']}",
          f"- cpu: {rec['cpu']}",
          f"- python {rec['python']}, numpy {rec['numpy']}, scipy {rec['scipy']}",
          "",
          "## Script digests (sha256, first 16 hex)",
          ""]
    for k, v in rec["script_sha256_16"].items():
        md.append(f"- `{k}`  {v}")
    md += ["", "## Result digests (sha256, first 16 hex)", ""]
    for k, v in rec["results_sha256_16"].items():
        md.append(f"- `{k}`  {v}")
    md.append("")
    open(os.path.join(root, "PROVENANCE.md"), "w").write("\n".join(md))
    print(f"provenance: {rec['hostname']} / {rec['platform']} / "
          f"python {rec['python']} numpy {rec['numpy']}")


if __name__ == "__main__":
    main()
