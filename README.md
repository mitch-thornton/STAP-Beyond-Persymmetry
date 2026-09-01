# Symmetry-Group Covariance Estimation for Adaptive Arrays

Reference implementation of a symmetry-matched covariance estimator for adaptive
beamforming, and of the experiments that measure its training requirement.

Persymmetric (forward-backward) processing exploits one reflection symmetry of an array
and halves the training requirement, a fixed factor of two regardless of aperture. A
symmetric array carries a whole group of sensor permutations, and an interference field
that respects a subgroup of it produces a covariance lying in that subgroup's commutant.
Averaging over the matched subgroup is the orthogonal projection onto the commutant, and
the training support it needs is

```
K_3dB  ~  2M / d_eff        d_eff = M^2 / dim(commutant)
```

the classical Reed-Mallett-Brennan requirement divided by the effective dimension, which is
computed from the array geometry and the field symmetry before any data are taken. The
sample covariance is the case `d_eff = 1` and persymmetry the case `d_eff = 2`.

## Contents
- `python/stap_sinr_v3.py`         SINR-loss experiment, matched and mismatch regimes,
- `python/stap_scaling_v1.py`      aperture sweep for a fixed and a growing symmetry order,
- `python/nll_calib_probe_v1.py`   Frobenius plug-in versus held-out NLL calibration,
- `python/stap_tables_v2.py`       variance law, subgroup ladder, invertibility thresholds,
- `python/make_stap_figs_v4.py`    the three result figures,
- `python/emit_tex_v1.py`          results JSON to LaTeX macros and table floats,
- `python/provenance_v1.py`        machine and digest record,
- `python/shrinkage_check_v1.py`   shrinkage intensity against the published plug-in,
- `python/stap_jammer_probe_v1.py` probe comparing a near-white and an interference-limited scenario,
- `python/geometry_probe_v1.py`    geometry/group helper,
- `python/requirements.txt`.

## Quick start
```
pip install -r python/requirements.txt
cd python
python stap_sinr_v3.py        # writes stap_sinr.json
python stap_tables_v2.py      # writes stap_tables.json
python stap_scaling_v1.py     # writes stap_scaling.json   (the long one, ~8 min)
python nll_calib_probe_v1.py  # writes nll_calib.json      (~4 min)
python make_stap_figs_v4.py   # writes figures/fig_{matched,calib,scaling}.{pdf,png}
python emit_tex_v1.py         # writes generated/ LaTeX macros and table floats
```
Seeds are fixed, so results reproduce up to Monte Carlo error at the shipped trial counts.
`emit_tex_v1.py` turns the result files into the macros and tables that a manuscript can
input directly, so no experimental value has to be transcribed by hand.

## Repository layout
```
.
├── README.md
├── LICENSE
├── PATENTS.md
└── python/
    ├── stap_sinr_v3.py
    ├── stap_scaling_v1.py
    ├── nll_calib_probe_v1.py
    ├── stap_tables_v2.py
    ├── make_stap_figs_v4.py
    ├── emit_tex_v1.py
    ├── provenance_v1.py
    ├── shrinkage_check_v1.py
    ├── stap_jammer_probe_v1.py
    ├── geometry_probe_v1.py
    └── requirements.txt
```

## What it shows
A uniform circular array faces `J` equal-power jammers placed on a rotation orbit of the
array symmetry group, so the interference-plus-noise covariance is exactly invariant under
the corresponding dihedral subgroup, is rank `J`, and leaves the conventional beamformer
tens of decibels from the clairvoyant optimum.

With the symmetry order growing with the aperture (`J = M/4`) the group average holds three
decibels of SINR loss from a constant 3.9 snapshots at M = 16 and 3.8 at M = 48, while the
sample covariance needs 29.5 and 93.9, the persymmetric estimator 15.6 and 47.5, and
oracle-tuned diagonal loading 7.4 and 23.9. Oracle-tuned loading is found to need about twice
the interference rank, so it scales with rank while the group average scales with `M/d_eff`.

The limits are measured alongside. The effective dimension is the exact Frobenius
variance-reduction factor only for an identity-scaled covariance (measured ratio 7.52 to 7.60
against `d_eff = 7.53` there, but 2.6 to 2.8 on the steeply colored jammer covariance) while
remaining the correct predictor of the training requirement. With a fixed symmetry order the
group advantage over diagonal loading closes as the aperture grows. And when a
symmetry-breaking source is present the projection carries a bias floor, so above about half
the sensor count a loaded sample covariance is preferable; below it the group family is far
ahead (6.40 dB against 18.33 dB at K = 4).

## Methods compared
Sample covariance; diagonally loaded sample covariance with the loading level chosen by an
oracle sweep at each snapshot count, so the baseline upper-bounds any data-driven loading
rule; the forward-backward (persymmetric) average using the reflection that fixes a circular
array; the matched-subgroup average; and convex shrinkage toward the group average under two
calibrations, a closed-form Frobenius plug-in and a held-out Gaussian likelihood criterion.

The group average coincides with the maximum-likelihood estimate in Andersson's invariant
normal models and with the group-symmetric covariance regularization of Shah and
Chandrasekaran; both shrinkage calibrations are from arXiv:2605.17111.

## License and patents
Code is MIT licensed (`LICENSE`), which grants copyright permissions only. See `PATENTS.md`.

## Contact
Mitchell A. Thornton, `mitch@smu.edu`.
