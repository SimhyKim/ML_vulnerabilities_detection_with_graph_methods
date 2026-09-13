# Wild OOD C/C++ probe (VulProtoCL)

Function-level C/C++ snippets for an **out-of-distribution** run of VulProtoCL.
Labels come from NVD CVE records mapped to GitHub (or kernel.org→GitHub) fix commits.
A function is **vulnerable** if its body overlaps the patch hunk; **non-vulnerable**
functions are in the same files.

## Constraints

- Language: C/C++ only (tree-sitter `cpp`), function-level.
- CVE **published ≥ 2024-01-01** so the window sits after BigVul and the PrimeVul collection era.
- Chromium / FFmpeg / QEMU / Wireshark repos blocked (ReVeal Chrome + Devign projects).


## Counts

- Vulnerable functions: **110**
- Non-vulnerable functions: **1000**
- Distinct CVEs: **68**
- Distinct GitHub repos: **15**
- Publication year in this snapshot: **2026 only**.

## CWE (vulnerable set)

- CWE-787: 23
- CWE-125: 19
- CWE-416: 13
- CWE-121: 12
- CWE-122: 9
- CWE-476: 8
- CWE-78: 8
- CWE-190: 7
- CWE-193: 4
- CWE-825: 4
- CWE-191: 3
- CWE-415: 3
- CWE-617: 3
- CWE-908: 3
- CWE-120: 2
- CWE-189: 2
- CWE-362: 2
- CWE-134: 1
- CWE-22: 1
- CWE-252: 1
- CWE-400: 1
- CWE-401: 1
- CWE-59: 1
- CWE-693: 1
- CWE-770: 1
- CWE-917: 1

## Projects (vulnerable set)

- `vim/vim`: 16
- `freerdp/freerdp`: 14
- `openssl/openssl`: 14
- `torvalds/linux`: 14
- `python/cpython`: 9
- `radareorg/radare2`: 9
- `hashcat/hashcat`: 6
- `libevent/libevent`: 6
- `gnome/libxml2`: 5
- `libssh2/libssh2`: 5
- `imagemagick/imagemagick`: 4
- `pnggroup/libpng`: 3
- `opensc/opensc`: 2
- `tukaani-project/xz`: 2
- `redis/redis`: 1

## Files

- `vulnerables.json` / `non-vulnerables.json` — ReVeal-compatible `code` plus provenance fields.
- `manifest.jsonl` — metadata only (CVE, CWE, commit, file, function).

## How to run inference (no retraining)

Uses the published ReVeal checkpoints in `results_publish/seed_*/protocl_best.pt`. Alpha and threshold stay frozen from ReVeal validation.

```powershell
python -u run_wild_ood_protocl.py --seed 42
python -u run_wild_ood_protocl.py --seeds 42,43,44
```

Metrics land in `results/wild_ood/seed_<N>/metrics.csv` and `results_wild_ood/wild_ood_seeds.csv`.


